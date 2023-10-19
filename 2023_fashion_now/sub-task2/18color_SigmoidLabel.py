from trainer import Trainer
from predictor import Predictor
from datasets import DatasetCreater
from models import BaseModel, DivBaseModel, MultiHeadBaseModel, ColorClassifierHead, BinaryClassifierHead
from loss_fn import FocalLoss, TripletMargingLoss, BCELoss, AsymmetricLoss
from optim_fn import SAM
from utils import save_config, mixup, cutmix, score, get_loss_weight, set_seed, cal_cls_report, Multi_cutmix

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
# from pytorch_metric_learning import miners, losses

import cv2
import numpy as np
from tqdm import tqdm

class CreateMeanTileImage(ImageOnlyTransform):
    def __init__(
        self,
        min_size=10,
        max_size=15,
        num=10, 
        always_apply=False,
        p=1
    ):
        super(CreateMeanTileImage, self).__init__(always_apply, p)
        self.min_size=min_size
        self.max_size=max_size
        self.num=num
        
    def apply(self, img, **params):
        return self.create_mean_image(img)
    
    def create_mean_image(self, img) :
        rgb = [int(img[:, :, i].mean()) for i in range(3)]
        img_h, img_w = img.shape[:2]
        for _ in range(self.num) :
            tile_w = np.random.randint(self.min_size, self.max_size)
            tile_h = np.random.randint(self.min_size, self.max_size)
            y1 = np.random.randint(1, img_h)
            x1 = np.random.randint(1, img_w)
            y2 = y1 + tile_h
            x2 = x1 + tile_w
            img[y1 : y2, x1: x2, :] = rgb
        return np.uint8(img)
    
    def get_transform_init_args_names(self):
        return ("min_size", "max_size", "num")
    
class BaseMain(Trainer, Predictor, DatasetCreater) :
    def __init__(self, **cfg) -> None:
        super().__init__()
        self.model = DivBaseModel(**cfg).to(cfg["device"])        
        self.optimizer = Adam(self.model.parameters(), lr=cfg["learning_rate"])
        # self.ce_criterion = BCELoss().to(cfg["device"])
        self.ce_criterion = AsymmetricLoss().to(cfg["device"])
        # self.metric_criterion = TripletMargingLoss().to(cfg["device"])
        # self.scheduler = CosineAnnealingLR(self.optimizer.base_optimizer, T_max=cfg['epochs'], eta_min=0.008)
        
        if cfg["mode"] == 'train' :
            self.train_loader, self.valid_loader = self.create_dataloader([self.get_transform('train', **cfg), 
                                                                           self.get_transform('valid', **cfg)], **cfg)
            
        elif cfg["mode"] == 'infer' :
            self.test_loader = self.create_dataloader(self.get_transform('infer', **cfg), **cfg)
            
    def train(self, **cfg) :
        self.run(**cfg)
        
    def train_on_epoch(self, epoch, **cfg):
        self.model.train()
        train_acc, train_loss = [], []
        
        tqdm_train = tqdm(self.train_loader)
        for step, (img, label) in enumerate(tqdm_train) :
            batch_res = self.train_on_batch(img, label, step, **cfg)
            
            train_acc.append(batch_res["acc"])
            train_loss.append(batch_res["loss"])
            
            log = {
                "Epoch" : epoch,
                "Training Acc" : np.mean(train_acc),
                "Training Loss" : np.mean(train_loss),
            }
            tqdm_train.set_postfix(log)
        
        self.logging(log, epoch)    
        self.scheduler_step()
        
    def train_on_batch(self, img, label, step, **cfg) :
        self.optimizer.zero_grad()

        img = img.to(cfg["device"])
        label = label.to(cfg["device"])
        
        # img, lam, label_a, label_b = cutmix(img, label) 
        output = self.model(img)
        
        # ce_loss = lam * (self.ce_criterion(output, label_a)) + (1-lam) * (self.ce_criterion(output, label_b))
        # m_loss = lam * self.metric_criterion(emb, label_a) + (1-lam) * self.metric_criterion(emb, label_b)
        loss = self.ce_criterion(output, label)
        # m_loss = lam * self.metric_criterion(emb, label_a) + (1-lam) * self.metric_criterion(emb, label_b)
        
        # loss = ce_loss + m_loss
        
        loss.backward()
        self.optimizer.step()
        
        acc = score(label, output, binary=cfg['binary_mode'])
        
        batch_metric = {
            "acc" : acc,
            "loss" : loss.item(),
        }
        
        return batch_metric 

    def valid_on_epoch(self, epoch, **cfg):
        self.model.eval()
        valid_acc, valid_loss, valid_output = [], [], [[], []]
        tqdm_valid = tqdm(self.valid_loader)
        for step, (img, label) in enumerate(tqdm_valid) :
            batch_res = self.valid_on_batch(img, label, step, **cfg)
            
            valid_acc.append(batch_res["acc"])
            valid_loss.append(batch_res["loss"])
            valid_output[0].extend(batch_res['labelAcc'][0])
            valid_output[1].extend(batch_res['labelAcc'][1])
            
            log = {
                "Epoch" : epoch,
                "Validation Acc" : np.mean(valid_acc),
                "Validation Loss" : np.mean(valid_loss),
            }
            tqdm_valid.set_postfix(log)
            
        self.logging({"LabelAcc" : cal_cls_report(valid_output[0], valid_output[1])}, epoch, mode='multi')
        self.logging(log, epoch)    
        return np.mean(valid_acc), np.mean(valid_loss)
    
    def valid_on_batch(self, img, label,  step, **cfg):
        img = img.to(cfg["device"])
        label = label.to(cfg["device"])
        
        if cfg["binary_mode"] :
            # mixup_label = torch.argmax(label, dim=1)
        
            output = self.model(img)
            loss = self.ce_criterion(output, label)
            
            acc, cls_report = score(label, output, mode="valid", binary=cfg['binary_mode'])
        else :        
            output = self.model(img)
            loss = self.ce_criterion(output, label)
            
            acc, cls_report = score(label, output, mode="valid", binary=cfg['binary_mode'])
            
        batch_metric = {
            "acc" : acc,
            "loss" : loss.item(),
            "labelAcc" : cls_report
        }
        
        return batch_metric
    
       
    def infer(self, **cfg) :
        return self.prediction(**cfg)
    
    
    def prediction(self, **cfg) :
        self.pred_weight_load(cfg["weight_path"],cfg["device"])
        self.model.eval()
        model_preds = []
        with torch.no_grad() :
            for img in tqdm(self.test_loader) :
                model_preds += self.predict_on_batch(img, **cfg)
        
        return self.save_to_csv(model_preds, **cfg)
        
    
    def predict_on_batch(self, img, **cfg) :
        img = img.to(cfg["device"])
        output = self.model(img)
        return output.argmax(1).detach().cpu().numpy().tolist()
    
    
    def get_transform(self, _mode, **cfg) :
        resize = cfg["resize"]
        if _mode == 'train' :
            return A.Compose([
                A.Resize(resize, resize),
                # A.RandomBrightness((0.2, 0.3), p =1),
                A.OneOf([
                    A.CoarseDropout(min_holes=5, max_holes=10, max_height=8, min_height=8),
                    A.GlassBlur(sigma=1, max_delta=2),        
                    A.RandomRain(slant_lower=-10, slant_upper=10,  brightness_coefficient=1, drop_color=(10, 10, 10), p=1),
                    A.RingingOvershoot(blur_limit=(5, 5), p=1),
                    
                ], p=0.3),
                
                A.OneOf([
                    A.OpticalDistortion(p=1, border_mode=1,
                        distort_limit=3, shift_limit=1),
                    A.Affine(rotate=(-180, 180), fit_output=False, mask_interpolation=1, mode=3, p=0.3),
                ], p=0.3),
                CreateMeanTileImage(min_size=10, max_size=15, num=15,p=0.3),
                A.RandomGridShuffle(grid=(3, 3), p=0.3),
                A.ElasticTransform(alpha=10, sigma=10 * 0.25, alpha_affine=10 * 0.25, p=0.2), #>>
                A.Normalize(mean=(0.7, 0.3, 0.3)),
                ToTensorV2()
            ])
        elif _mode == 'valid' :
            return A.Compose([
                A.Resize(resize, resize),
                A.Normalize(mean=(0.7, 0.3, 0.3)),
                ToTensorV2()
            ])
        elif _mode == 'infer' : 
            return A.Compose([
                A.Resize(resize, resize),
                A.Normalize(),
                ToTensorV2()
            ])

    
if __name__ == "__main__" :
    
    cfg = {
        "mode" : "train", #train, #infer
        
        "model_name" : "seresnext50_32x4d", #"tf_efficientnetv2_m.in21k", #"swinv2_base_window12to16_192to256_22kft1k",
        # "tf_efficientnet_b4", #"resnetv2_101x1_bit", #"resnetv2_152x2_bit", #"resnetv2_50x1_bit",
        #"tf_efficientnetv2_s.in21k",#"eva_large_patch14_196.in22k_ft_in1k",, #"wide_resnet101_2", #"seresnext50_32x4d"
        #"beit_base_patch16_224.in22k_ft_in22k", #"convnextv2_base.fcmae_ft_in1k", #"seresnet101", #"seresnext101_64x4d"
        #"nf_resnet50",#"nfnet_f1"
        "num_classes" : 18,
        
        "learning_rate" : 5e-4,
        "focal_alpha" : 2,
        "focal_gamma" : 2,
        "resize" : 224,
        
        "data_train_path" : "./sub-task2/Dataset/Train",
        "data_train_csv_path" : "./sub-task2/Dataset/cosine_label_aug_info_etri20_color_train.csv",
        "data_valid_path" : "./sub-task2/Dataset/Validation",
        "data_valid_csv_path" : "./sub-task2/Dataset/cosine_label_aug_info_etri20_color_validation.csv",
        
        "data_infer_path" : "/aif/Dataset/Test/",
        "data_infer_csv_path" : "/aif/Dataset/info_etri20_color_test.csv",
        # "data_infer_path" : "./sub-task2/Dataset/Test_sample",
        # "data_infer_csv_path" : "./sub-task2/Dataset/info_etri20_color_test_sample.csv",
        
        "epochs" : 80,
        "batch_size" : 32,
        "num_worker" : 4,
        "early_stop_patient" : 30,
        
        "reuse" : False, #True, #False
        "weight_path" : "./sub-task2/ckpt/seresnext50_32x4d/SigmoidLabel_RGBNormalize/12E-val0.5823576946932211-resnetv2_152d.pth",
        
        "save_path" : "./sub-task2/ckpt/seresnext50_32x4d/SigmoidLabel_RGBNormalize/step1",
        "output_path" : "./sub-task2/output/seresnext50_32x4d/SigmoidLabel_RGBNormalize/step1",
        "log_path" : "./sub-task2/logging/SigmoidLabel_RGBNormalize/step1",
        "device" : "cuda",
        
        "binary_mode" : True,
        
        "loss_weight" : [0.8622, 1.2872, 1.0935, 0.9228, 0.9122, 1.0983, 1.1635, 1.0886,
                        1.015 , 1.1258, 0.9061, 0.9446, 0.9728, 0.8595, 0.9743, 0.9163,
                        0.9292, 0.9281],
        
        
        "seed": 2455,
        "note" : ["offline aug for distance smapling data",
                "Cutmix 사용", "CE Loss 사용", "Adam Optim 사용"]
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
    
    