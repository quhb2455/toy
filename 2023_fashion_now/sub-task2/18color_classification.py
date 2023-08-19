from trainer import Trainer
from predictor import Predictor
from datasets import DatasetCreater
from models import BaseModel, DivBaseModel
from loss_fn import FocalLoss, TripletMargingLoss
from optim_fn import SAM
from utils import save_config, mixup, cutmix, score, get_loss_weight, set_seed, enable_running_stats, disable_running_stats

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

class BaseMain(Trainer, Predictor, DatasetCreater) :
    def __init__(self, **cfg) -> None:
        super().__init__()
        # self.model = BaseModel(**cfg).to(cfg["device"])
        self.model = DivBaseModel(**cfg).to(cfg["device"])
        self.optimizer = Adam(self.model.parameters(), lr=cfg["learning_rate"])
        # self.optimizer = SAM(self.model.parameters(), SGD, weight_decay=0.0001, momentum=0.9,lr=cfg["learning_rate"])
        # self.criterion = FocalLoss(alpha=cfg["focal_alpha"],gamma=cfg["focal_gamma"]).to(cfg["device"])
        self.criterion = nn.CrossEntropyLoss().to(cfg["device"])
        self.metric_criterion = TripletMargingLoss().to(cfg["device"])
        # self.scheduler = CosineAnnealingLR(self.optimizer.base_optimizer, T_max=cfg['epochs'], eta_min=0.008)
        
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
        label = label.to(cfg["device"])

        img, lam, label_a, label_b = cutmix(img, label) 
        
        emb, output = self.model(img, div=True)
        loss = lam * self.criterion(output, label_a) + (1 - lam) * self.criterion(output, label_b) + lam * self.metric_criterion(emb, label_a) + (1 - lam) * self.metric_criterion(emb, label_b)
        
        # loss = self.criterion(output, label.type(torch.float32))
        loss.backward()
        self.optimizer.step()
        
        # acc = score(mixup_label, output)
        acc = score(label, output)
        # acc = score(torch.argmax(label, dim=1), output)

        
        batch_metric = {
            "acc" : acc,
            "loss" : loss.item()
        }
        
        return batch_metric 

    def valid_on_batch(self, img, label, step, **cfg):
        img = img.to(cfg["device"])
        label = label.to(cfg["device"])
        
        if cfg["binary_mode"] :
            mixup_label = torch.argmax(label, dim=1)
        
            output = self.model(img)
            loss = self.criterion(output, label.type(torch.float32))
            
            acc = score(mixup_label, output)
        else :        
            output = self.model(img)
            loss = self.criterion(output, label)
            
            acc, cls_report = score(label, output, mode="valid")
            
        batch_metric = {
            "acc" : acc,
            "loss" : loss.item(),
            "labelAcc" : cls_report
        }
        
        # self.logging({'LabelAcc' : cls_report}, step, mode='multi')
        return batch_metric
       
    def infer(self, **cfg) :
        return self.prediction(**cfg)
    
    
    def get_transform(self, _mode, **cfg) :
        resize = cfg["resize"]
        if _mode == 'train' :
            return A.Compose([
                A.Resize(resize, resize),
                # A.RandomBrightness((0.2, 0.3), p =1),
                # A.CoarseDropout(min_holes=10, max_holes=20, max_height=20, min_height=20 ,p=0.2),
                A.GlassBlur(sigma=1, max_delta=2,p=0.3),
                # A.RandomGridShuffle(grid=(4,4), p=0.5),
                A.OneOf([
                    A.Affine(rotate=(-180, 180), fit_output=False, mask_interpolation=1, mode=3),
                    A.OpticalDistortion(border_mode=1,
                            distort_limit=1, shift_limit=1),    
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
        
        "model_name" : "resnetv2_101x1_bit", #"tf_efficientnetv2_m.in21k", #"swinv2_base_window12to16_192to256_22kft1k",
        # "tf_efficientnet_b4", #"resnetv2_101x1_bit", #"resnetv2_152x2_bit",
        #"tf_efficientnetv2_s.in21k",#"eva_large_patch14_196.in22k_ft_in1k",
        #"beit_base_patch16_224.in22k_ft_in22k", #"convnextv2_base.fcmae_ft_in1k"
        "num_classes" : 18,
        
        "learning_rate" : 5e-4,
        "focal_alpha" : 2,
        "focal_gamma" : 2,
        "resize" : 224,
        
        "data_train_path" : "./sub-task2/Dataset/Train",
        "data_train_csv_path" : "./sub-task2/Dataset/sampling_aug_info_etri20_color_train.csv",
        "data_valid_path" : "./sub-task2/Dataset/Validation",
        "data_valid_csv_path" : "./sub-task2/Dataset/info_etri20_color_validation.csv",
        
        "data_infer_path" : "/aif/Dataset/Test/",
        "data_infer_csv_path" : "/aif/Dataset/info_etri20_color_test.csv",
        # "data_infer_path" : "./sub-task2/Dataset/Test_sample",
        # "data_infer_csv_path" : "./sub-task2/Dataset/info_etri20_color_test_sample.csv",
        
        "epochs" : 80,
        "batch_size" : 48,
        "num_worker" : 4,
        "early_stop_patient" : 10,
        
        "reuse" : False, #True, #False
        "weight_path" : "./sub-task2/ckpt/resnetv2_101x1_bit/Sampling_lowSpatialAugPercent/12E-val0.5823576946932211-resnetv2_152d.pth",
        
        "save_path" : "./sub-task2/ckpt/resnetv2_101x1_bit/Sampling_lowSpatialAugPercent",
        "output_path" : "./sub-task2/output/resnetv2_101x1_bit/Sampling_lowSpatialAugPercent",
        "log_path" : "./sub-task2/logging/Sampling_lowSpatialAugPercent",
        "device" : "cuda",
        
        "binary_mode" : False,
        "seed": 2455,
        "note" : ["Low Distance Sampling", "spatial aug percent 1 -> 0.3", "Metric Learning label_b 도 이용", "Label1,2,8,9에 Offline Aug 사용", 
                "CenterCrop 사용하지 않음", 
                "Aug 정도 하향조정", 
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
    
    