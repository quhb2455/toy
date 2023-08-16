from trainer import Trainer
from predictor import Predictor
from datasets import DatasetCreater
from models import BaseModel, DivBaseModel, MultiHeadBaseModel
from loss_fn import FocalLoss, TripletMargingLoss
from utils import save_config, mixup, cutmix, score, get_loss_weight, set_seed, cal_cls_report, MultiHead_cutmix, MultiHead_mixup

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
# from pytorch_metric_learning import miners, losses

import cv2
import numpy as np
from tqdm import tqdm

class BaseMain(Trainer, Predictor, DatasetCreater) :
    def __init__(self, **cfg) -> None:
        super().__init__()
        # self.model = BaseModel(**cfg).to(cfg["device"])
        # self.model = DivBaseModel(**cfg).to(cfg["device"])
        self.model = MultiHeadBaseModel(**cfg).to(cfg["device"])
        self.optimizer = Adam(self.model.parameters(), lr=cfg["learning_rate"])
        # self.criterion = FocalLoss(alpha=cfg["focal_alpha"],gamma=cfg["focal_gamma"]).to(cfg["device"])
        self.criterion = nn.CrossEntropyLoss().to(cfg["device"])
        self.metric_criterion = TripletMargingLoss().to(cfg["device"])
        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=60, eta_min=5e-4)
        
        if cfg["mode"] == 'train' :
            self.train_loader, self.valid_loader = self.create_dataloader([self.get_transform('train', **cfg), 
                                                                           self.get_transform('valid', **cfg)], **cfg)
            
        elif cfg["mode"] == 'infer' :
            self.test_loader = self.create_dataloader(self.get_transform('infer', **cfg), **cfg)
            
    def train(self, **cfg) :
        self.run(**cfg)
        
    def train_on_batch(self, img, label, step, **cfg) :
        self.optimizer.zero_grad()

        img = img.to(cfg["device"])
        label_daily = label[0].to(cfg["device"])
        label_gender = label[1].to(cfg["device"])
        label_embel = label[2].to(cfg["device"])

        # img, lam, label_a, label_b = MultiHead_cutmix(img, label)
        # img, lam, label_a, label_b = cutmix(img, label)
        img, lam, label_a, label_b = MultiHead_mixup(img, label)
        # img, lam, label_a, label_b = mixup(img, label)

        label_daily_b = label_b[0].to(cfg["device"])
        label_gender_b = label_b[1].to(cfg["device"])
        label_embel_b = label_b[2].to(cfg["device"])
        
        d_output, g_output, e_output = self.model(img)
        # loss = lam * self.criterion(output, label_a) + (1 - lam) * self.criterion(output, label_b)#+ self.metric_criterion(emb, label)
        daily_loss = lam * self.criterion(d_output, label_daily) + (1 - lam) * self.criterion(d_output, label_daily_b)
        gender_loss = lam * self.criterion(g_output, label_gender) + (1 - lam) * self.criterion(g_output, label_gender_b)
        embel_loss = lam * self.criterion(e_output, label_embel) + (1 - lam) * self.criterion(e_output, label_embel_b)
        # metric_loss = self.metric_criterion(emb, label)
        
        # daily_loss = self.criterion(d_output, label_daily) 
        # gender_loss = self.criterion(g_output, label_gender)
        # embel_loss = self.criterion(e_output, label_embel)
        
        # loss = self.criterion(output, label.type(torch.float32))
        loss = daily_loss + gender_loss + embel_loss
        loss.backward()
        self.optimizer.step()
        
        # acc = score(mixup_label, output)
        d_acc = score(label_daily, d_output)
        g_acc = score(label_gender, g_output)
        e_acc = score(label_embel, e_output)
        acc = (d_acc + g_acc + e_acc) / 3
        # acc = score(torch.argmax(label, dim=1), output)

        
        batch_metric = {
            "acc" : acc,
            "loss" : loss.item(),
            'd_loss' : daily_loss.item(),
            'g_loss' : gender_loss.item(),
            'e_loss' : embel_loss.item(),
            'd_acc' : d_acc,
            'g_acc' : g_acc,
            'e_acc' : e_acc,
        }
        
        return batch_metric 
    
    def train_on_epoch(self, epoch, **cfg):
        self.model.train()
        train_acc, train_loss = [], []
        train_d_loss, train_g_loss, train_e_loss = [], [], []
        train_d_acc, train_g_acc, train_e_acc = [], [], []
        tqdm_train = tqdm(self.train_loader)
        for step, (img, label) in enumerate(tqdm_train) :
            batch_res = self.train_on_batch(img, label, step, **cfg)
            
            train_acc.append(batch_res["acc"])
            train_loss.append(batch_res["loss"])
            train_d_loss.append(batch_res["d_loss"])
            train_g_loss.append(batch_res["g_loss"])
            train_e_loss.append(batch_res["e_loss"])
            train_d_acc.append(batch_res["d_acc"])
            train_g_acc.append(batch_res["g_acc"])
            train_e_acc.append(batch_res["e_acc"])
            
            log = {
                "Epoch" : epoch,
                "Training Acc" : np.mean(train_acc),
                "Training Loss" : np.mean(train_loss),
                "Training D Acc" : np.mean(train_d_acc),
                "Training G Acc" : np.mean(train_g_acc),
                "Training E Acc" : np.mean(train_e_acc),
                "Training D Loss" : np.mean(train_d_loss),
                "Training G Loss" : np.mean(train_g_loss),
                "Training E Loss" : np.mean(train_e_loss),
                
            }
            tqdm_train.set_postfix(log)
        
        self.logging(log, epoch)    
        self.scheduler_step()
    
    def valid_on_batch(self, img, label, step, **cfg):
        img = img.to(cfg["device"])
        # label = label.to(cfg["device"])
        label_daily = label[0].to(cfg["device"])
        label_gender = label[1].to(cfg["device"])
        label_embel = label[2].to(cfg["device"])
        
        if cfg["binary_mode"] :
            mixup_label = torch.argmax(label, dim=1)
        
            output = self.model(img)
            loss = self.criterion(output, label.type(torch.float32))
            
            acc = score(mixup_label, output)
        else :        
            # output = self.model(img)
            # loss = self.criterion(output, label)
            d_output, g_output, e_output = self.model(img)
            
            daily_loss = self.criterion(d_output, label_daily)
            gender_loss = self.criterion(g_output, label_gender)
            embel_loss = self.criterion(e_output, label_embel)
            loss = daily_loss + gender_loss + embel_loss
            
            # acc = score(label, output)
            d_acc = score(label_daily, d_output)
            g_acc = score(label_gender, g_output)
            e_acc = score(label_embel, e_output)
            acc = (d_acc + g_acc + e_acc) / 3
            
        batch_metric = {
            "acc" : acc,
            "loss" : loss.item(),
            'd_loss' : daily_loss.item(),
            'g_loss' : gender_loss.item(),
            'e_loss' : embel_loss.item(),
            # "labelAcc" : cls_report
            'd_acc' : d_acc,
            'g_acc' : g_acc,
            'e_acc' : e_acc,
            
        }
        
        # self.logging({'LabelAcc' : cls_report}, step, mode='multi')
        return batch_metric
    
    def valid_on_epoch(self, epoch, **cfg):
        self.model.eval()
        valid_acc, valid_loss, valid_output = [], [], [[], []]
        valid_d_loss, valid_g_loss, valid_e_loss = [], [], []
        valid_d_acc, valid_g_acc, valid_e_acc = [], [], []
        tqdm_valid = tqdm(self.valid_loader)
        for step, (img, label) in enumerate(tqdm_valid) :
            batch_res = self.valid_on_batch(img, label, step, **cfg)
            
            valid_acc.append(batch_res["acc"])
            valid_loss.append(batch_res["loss"])
            valid_d_loss.append(batch_res["d_loss"])
            valid_g_loss.append(batch_res["g_loss"])
            valid_e_loss.append(batch_res["e_loss"])
            valid_d_acc.append(batch_res["d_acc"])
            valid_g_acc.append(batch_res["g_acc"])
            valid_e_acc.append(batch_res["e_acc"])
            # valid_output[0].extend(batch_res['labelAcc'][0])
            # valid_output[1].extend(batch_res['labelAcc'][1])
            log = {
                "Epoch" : epoch,
                "Validation Acc" : np.mean(valid_acc),
                "Validation Loss" : np.mean(valid_loss),
                "Validation D Acc" : np.mean(valid_d_acc),
                "Validation G Acc" : np.mean(valid_g_acc),
                "Validation E Acc" : np.mean(valid_e_acc),
                "Validation D Loss" : np.mean(valid_d_loss),
                "Validation G Loss" : np.mean(valid_g_loss),
                "Validation E Loss" : np.mean(valid_e_loss),
            }
            tqdm_valid.set_postfix(log)
            
        # self.logging({"LabelAcc" : cal_cls_report(valid_output[0], valid_output[1])}, epoch, mode='multi')
        self.logging(log, epoch)    
        return np.mean(valid_acc), np.mean(valid_loss)
    
    def infer(self, **cfg) :
        return self.prediction(**cfg)
    
    
    def get_transform(self, _mode, **cfg) :
        resize = cfg["resize"]
        if _mode == 'train' :
            return A.Compose([
                A.Resize(resize, resize),
                A.OneOf([
                    A.RandomBrightness((-0.2, 0.2), p=1),
                    A.RandomToneCurve(scale=0.5, p=1),    
                ], p=0.3),
                
                A.CoarseDropout(min_holes=10, max_holes=20, max_height=20, min_height=20 ,p=0.4),
                A.OneOf([
                    A.GlassBlur(sigma=1, max_delta=2,p=1),
                    A.RandomGridShuffle(grid=(2,2), p=1)
                ], p =0.3),
                A.OneOf([
                    A.Affine(rotate=(-180, 180), fit_output=False, mask_interpolation=1, mode=3),
                    # A.OpticalDistortion(border_mode=1,
                    #         distort_limit=1, shift_limit=1),    
                    A.ElasticTransform(p=1, alpha=10, sigma=10 * 0.25, alpha_affine=10 * 0.25),
                    A.GridDistortion(p=1)
                ], p=0.3),
                
                A.OneOf([
                    A.RandomRain(slant_lower=-10, slant_upper=10,  brightness_coefficient=1, drop_color=(10, 10, 10), p=1),
                    A.RandomSnow(snow_point_lower= 0.05, snow_point_upper= 0.3, brightness_coeff= 3.5, p=1),
                    A.RandomSunFlare(src_radius=25, p=1),
                ], p=0.3),
                A.Normalize(),
                # A.Normalize(mean=(0.548172032,0.467046563,0.434142448),
                #             std=(0.12784231,0.122905336,0.119736256)),
                ToTensorV2()
            ])
        elif _mode == 'valid' :
            return A.Compose([
                A.Resize(resize, resize),
                # A.RandomBrightness((0.2, 0.3), p =1),
                A.Normalize(),
                # A.Normalize(mean=(0.548172032,0.467046563,0.434142448),
                #             std=(0.12784231,0.122905336,0.119736256)),
                ToTensorV2()
            ])
        elif _mode == 'infer' : 
            return A.Compose([
                A.Resize(resize, resize),
                A.Normalize(),
                # A.Normalize(mean=(0.548172032,0.467046563,0.434142448),
                #             std=(0.12784231,0.122905336,0.119736256)),
                ToTensorV2()
            ])

    
if __name__ == "__main__" :
    
    cfg = {
        "mode" : "train", #train, #infer
        
        "model_name" : "tf_efficientnetv2_s.in21k", #"tf_efficientnetv2_m.in21k", #"swinv2_base_window12to16_192to256_22kft1k",
        #"tf_efficientnetv2_s.in21k",#"eva_large_patch14_196.in22k_ft_in1k",#"beit_base_patch16_224.in22k_ft_in22k", #"convnextv2_base.fcmae_ft_in1k"
        "num_classes" : 86,
        # daily - 6
        # Gender - 5
        # Embellishment - 3
        
        "learning_rate" : 5e-4,
        "focal_alpha" : 2,
        "focal_gamma" : 2,
        "resize" : 224,
        
        "data_train_path" : "./sub-task1/Dataset/Train",
        "data_train_csv_path" : "./sub-task1/Dataset/info_etri20_emotion_train.csv",
        "data_valid_path" : "./sub-task1/Dataset/Validation",
        "data_valid_csv_path" : "./sub-task1/Dataset/info_etri20_emotion_validation.csv",
        
        "data_infer_path" : "/aif/Dataset/Test/",
        "data_infer_csv_path" : "/aif/Dataset/info_etri20_color_test.csv",
        # "data_infer_path" : "./sub-task2/Dataset/Test_sample",
        # "data_infer_csv_path" : "./sub-task2/Dataset/info_etri20_color_test_sample.csv",
        
        "epochs" : 80,
        "batch_size" : 32,
        "num_worker" : 4,
        "early_stop_patient" : 10,
        
        "reuse" : False, #True, #False
        "weight_path" : "./sub-task1/ckpt/tf_efficientnetv2_s.in21k/multi_head_MoreAug/6E-val0.18936908293169813-tf_efficientnetv2_m.in21k.pth",
        
        "save_path" : "./sub-task1/ckpt/tf_efficientnetv2_s.in21k/multi_head_MoreAug",
        "output_path" : "./sub-task1/output/tf_efficientnetv2_s.in21k/multi_head_MoreAug",
        "log_path" : "./sub-task1/logging/multi_head_MoreAug",
        "device" : "cuda",
        
        "binary_mode" : False,
        "seed": 2455,
        "note" : ["More Aug, delete OpticalDistortion", "multi_head 학습", "Cutmix&MixUp X", "CE Loss", "Adam Optim 사용"]
    }        
    
    if cfg["mode"] == "train" :
        cfg["shuffle"] = True
    elif cfg["mode"] == "infer" :
        cfg["shuffle"] = False
    
    set_seed(cfg["seed"])
    save_config(cfg, cfg["save_path"], save_name=cfg["mode"]+"_config")
    
    base_main = BaseMain(**cfg)
    
    if cfg["mode"] == "train" :
        base_main.train(**cfg)
    elif cfg["mode"] == "infer" :
        base_main.infer(**cfg)
    
    