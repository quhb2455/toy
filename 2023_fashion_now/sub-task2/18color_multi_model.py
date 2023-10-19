from trainer import Trainer
from predictor import Predictor
from datasets import DatasetCreater
from models import BaseModel, DivBaseModel, MultiHeadBaseModel, ColorClassifierHead, BinaryClassifierHead
from loss_fn import FocalLoss, TripletMargingLoss
from optim_fn import SAM
from utils import save_config, mixup, cutmix, score, get_loss_weight, set_seed, cal_cls_report, read_json

import torch
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
# from pytorch_metric_learning import miners, losses

import cv2
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from copy import deepcopy
import json

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
        self.ce_criterion = nn.CrossEntropyLoss().to(cfg["device"])
        self.metric_criterion = TripletMargingLoss().to(cfg["device"])
        # self.scheduler = CosineAnnealingLR(self.optimizer.base_optimizer, T_max=cfg['epochs'], eta_min=0.008)
        
        self.valid_output_chk = []
        self.valid_gt_chk = []
        
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
       
        img, lam, label_a, label_b = cutmix(img, label) 
        emb, output = self.model(img, div=True)
        
        ce_loss = lam * (self.ce_criterion(output, label_a)) + (1-lam) * (self.ce_criterion(output, label_b))
        m_loss = lam * self.metric_criterion(emb, label_a) + (1-lam) * self.metric_criterion(emb, label_b)
        loss = ce_loss + m_loss
        
        loss.backward()
        self.optimizer.step()
        
        acc = score(label, output)
               
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
        
        # z = pd.DataFrame({"GT" : self.valid_gt_chk, "Output" : self.valid_output_chk})
        # z.to_csv("./valid_chk.csv", index=False) 
        return np.mean(valid_acc), np.mean(valid_loss)
    
    def valid_on_batch(self, img, label,  step, **cfg):
        img = img.to(cfg["device"])
        label = label.to(cfg["device"])
        
        if cfg["binary_mode"] :
            mixup_label = torch.argmax(label, dim=1)
        
            output = self.model(img)
            loss = self.criterion(output, label.type(torch.float32))
            
            acc = score(mixup_label, output)
        else :        
            output = self.model(img)
            loss = self.ce_criterion(output, label)
            # self.valid_output_chk.extend(output.argmax(1).detach().cpu().numpy().tolist()) 
            # self.valid_gt_chk.extend(label.detach().cpu().numpy().tolist())
            
            acc, cls_report = score(label, output, mode="valid")
            
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
        
        # return self.save_to_csv(model_preds, **cfg).to_csv(cfg["output_path"])
        with open(cfg["output_path"], 'w') as f:
            json.dump(model_preds, f, indent="\t")
        
    
    def predict_on_batch(self, img, **cfg) :
        img = img.to(cfg["device"])
        output = self.model(img)
        # return output.argmax(1).detach().cpu().numpy().tolist()
        return output.detach().cpu().numpy().tolist()
    
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
                A.Normalize(),
                # A.Normalize(mean=(0.7, 0.5, 0.1)),
                # A.Normalize(mean=(0.548172032,0.467046563,0.434142448),
                #             std=(0.12784231,0.122905336,0.119736256)),
                ToTensorV2()
            ])
        elif _mode == 'valid' :
            return A.Compose([
                A.Resize(resize, resize),
                # CreateMeanTileImage(min_size=10, max_size=15, num=30,p=1),
                # A.RandomGridShuffle(grid=(3, 3), p=1),
                # A.RandomBrightness((0.2, 0.3), p =1),
                # A.Normalize(mean=(0.7, 0.5, 0.1)),
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

    def merge_csv(self, **cfg) :
        label_dec = [0, 2, 3, 8, 10, 14, 17,   1, 4, 5, 7, 11, 16,   6, 9, 12, 13, 15]
        df0 = np.array(read_json(cfg["group1_path"]))
        df1 = np.array(read_json(cfg["group2_path"]))
        df2 = np.array(read_json(cfg["group3_path"]))
        
        # new_df = pd.concat([df0, df1, df2], axis=1, ignore_index=True)
        pred_list = np.concatenate([df0, df1, df2], axis=1).argmax(1).tolist()
        pred_list = [label_dec[i] for i in pred_list]

        # pred_list = []
        # for pred in new_df.iloc :
        #     pred_list.append(label_dec[torch.tensor([v for v in pred.values()]).argmax().item()])
        
        return self.save_to_csv(pred_list, **cfg).to_csv(cfg["output_path"])
    
    
if __name__ == "__main__" :
    
    cfg = {
        "mode" : "train", #train, #infer
        
        "model_name" : "wide_resnet101_2", #"tf_efficientnetv2_m.in21k", #"swinv2_base_window12to16_192to256_22kft1k",
        # "tf_efficientnet_b4", #"resnetv2_101x1_bit", #"resnetv2_152x2_bit", #"resnetv2_50x1_bit",
        #"tf_efficientnetv2_s.in21k",#"eva_large_patch14_196.in22k_ft_in1k",, #"wide_resnet101_2", #"seresnext50_32x4d"
        #"beit_base_patch16_224.in22k_ft_in22k", #"convnextv2_base.fcmae_ft_in1k", #"seresnet101", #"seresnext101_64x4d"
        #"nf_resnet50",#"nfnet_f1" #tf_efficientnet_b0 #skresnext50_32x4d
        "num_classes" : 5, # 7, 6, 5
        
        "learning_rate" : 5e-4,
        "focal_alpha" : 2,
        "focal_gamma" : 2,
        "resize" : 224,
        
        "data_train_path" : "./sub-task2/Dataset/Train",
        "data_train_csv_path" : "./sub-task2/Dataset/aug_info_train_2.csv",
        "data_valid_path" : "./sub-task2/Dataset/Validation",
        "data_valid_csv_path" : "./sub-task2/Dataset/aug_info_valid_2.csv",
        
        # "data_infer_path" : "/aif/Dataset/Test/",
        # "data_infer_csv_path" : "/aif/Dataset/info_etri20_color_test.csv",
        "data_infer_path" : "./sub-task2/Dataset/Test_sample",
        "data_infer_csv_path" : "./sub-task2/Dataset/info_etri20_color_test_sample.csv",
        
        "epochs" : 80,
        "batch_size" : 32,
        "num_worker" : 4,
        "early_stop_patient" : 30,
        
        "reuse" : False, #True, #False
        "weight_path" : "./sub-task2/ckpt/nf_seresnet101/YCrCb_3_group/63E-val0.6227062319989951-seresnext50_32x4d.pth",
        
        "save_path" : "./sub-task2/ckpt/nf_seresnet101/YCrCb_3_group",
        "output_path" : "./sub-task2/output/nf_seresnet101/YCrCb_3_group",
        "log_path" : "./sub-task2/logging/YCrCb_3_group",
        "device" : "cuda",
        
        "binary_mode" : False,
        
        "loss_weight" : [0.04182298179797138, 0.1192079207920792, 0.5833333333333334, 0.6417910447761194, 
                         0.7506234413965087, 0.9121212121212121, 0.9741100323624595, 1.0],
        # "loss_weight" : [0.03382999511480215, 0.21193573068094873, 0.3176605504587156, 0.3768707482993197, 
        #                  0.8052325581395349, 0.8523076923076923, 1.0],
        # "loss_weight" : [0.03455401216852256, 0.3013013013013013, 0.3340732519422864, 0.36887254901960786, 
        #                  0.940625, 1.0],
        
        
        "seed": 2455,
        "note" : ["offline aug for distance smapling data",
                "Cutmix 사용", "CE Loss 사용", "Adam Optim 사용"]
    }        
    
    if cfg["mode"] == "train" :
        cfg["shuffle"] = True
    elif cfg["mode"] == "infer" :
        cfg["shuffle"] = False
    
    set_seed(cfg["seed"])
    # save_config(cfg, cfg["save_path"], save_name=cfg["mode"]+"_config")
    
    
    
    if cfg["mode"] == "train" :
        cfg_1 = deepcopy(cfg)
        cfg_1["num_classes"] = 7
        cfg_1["data_train_csv_path"] = "./sub-task2/Dataset/aug_info_train_0.csv"
        cfg_1["data_valid_csv_path"] = "./sub-task2/Dataset/aug_info_valid_0.csv"
        cfg_1["save_path"] = "./sub-task2/ckpt/wide_resnet101_2/YCrCb_1_group/"
        cfg_1["log_path"] = "./sub-task2/logging/YCrCb_1_group"
        base_main = BaseMain(**cfg_1)
        base_main.train(**cfg_1)
        
        cfg_2 = deepcopy(cfg)
        cfg_2["num_classes"] = 6
        cfg_2["data_train_csv_path"] = "./sub-task2/Dataset/aug_info_train_1.csv"
        cfg_2["data_valid_csv_path"] = "./sub-task2/Dataset/aug_info_valid_1.csv"
        cfg_2["save_path"] = "./sub-task2/ckpt/wide_resnet101_2/YCrCb_2_group/"
        cfg_2["log_path"] = "./sub-task2/logging/YCrCb_2_group"
        base_main = BaseMain(**cfg_2)
        base_main.train(**cfg_2)
        
        cfg_3 = deepcopy(cfg)
        cfg_3["num_classes"] = 5
        cfg_3["data_train_csv_path"] = "./sub-task2/Dataset/aug_info_train_2.csv"
        cfg_3["data_valid_csv_path"] = "./sub-task2/Dataset/aug_info_valid_2.csv"
        cfg_3["save_path"] = "./sub-task2/ckpt/wide_resnet101_2/YCrCb_3_group/"
        cfg_3["log_path"] = "./sub-task2/logging/YCrCb_3_group"
        base_main = BaseMain(**cfg_3)
        base_main.train(**cfg_3)

        
    elif cfg["mode"] == "infer" :
        cfg_1 = deepcopy(cfg)
        cfg_1["num_classes"] = 7
        cfg_1["weight_path"] = "./sub-task2/ckpt/seresnext50_32x4d/YCrCb_1_group/7E-val0.8495624725570149-seresnext50_32x4d.pth"
        cfg_1["output_path"] = "./sub-task2/output/seresnext50_32x4d/YCrCb_group/result_1.csv"
        cfg_1["group"] = 1
        base_main = BaseMain(**cfg_1)
        base_main.infer(**cfg_1)
        
        cfg_2 = deepcopy(cfg)
        cfg_2["num_classes"] = 6
        cfg_2["weight_path"] = "./sub-task2/ckpt/seresnext50_32x4d/YCrCb_2_group/7E-val0.8260962968913973-seresnext50_32x4d.pth"
        cfg_2["output_path"] = "./sub-task2/output/seresnext50_32x4d/YCrCb_group/result_2.csv"
        cfg_2["group"] = 2
        base_main = BaseMain(**cfg_2)
        base_main.infer(**cfg_2)
        
        cfg_3 = deepcopy(cfg)
        cfg_3["num_classes"] = 5
        cfg_3["weight_path"] = "./sub-task2/ckpt/seresnext50_32x4d/YCrCb_3_group/13E-val0.8702208287760302-seresnext50_32x4d.pth"
        cfg_3["output_path"] = "./sub-task2/output/seresnext50_32x4d/YCrCb_group/result_3.csv"
        cfg_3["group"] = 3
        base_main = BaseMain(**cfg_3)
        base_main.infer(**cfg_3)
    
        cfg_merge = deepcopy(cfg)
        cfg_merge["group1_path"] = "./sub-task2/output/seresnext50_32x4d/YCrCb_group/result_1.csv"
        cfg_merge["group2_path"] = "./sub-task2/output/seresnext50_32x4d/YCrCb_group/result_2.csv"
        cfg_merge["group3_path"] = "./sub-task2/output/seresnext50_32x4d/YCrCb_group/result_3.csv"
        cfg_merge["output_path"] = "./sub-task2/output/seresnext50_32x4d/YCrCb_group/result.csv"
        base_main = BaseMain(**cfg_merge)
        base_main.merge_csv(**cfg_merge)
