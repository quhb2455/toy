from trainer import Trainer
from predictor import Predictor
from datasets import DatasetCreater
from models import BaseModel
from loss_fn import FocalLoss, AsymmetricLoss, AsymmetricLossSingleLabel
from utils import save_config, mixup, cutmix, score, get_loss_weight

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from pytorch_metric_learning import miners, losses

import cv2
import numpy as np
from tqdm import tqdm
    
    
class BaseMain(Trainer, Predictor, DatasetCreater) :
    def __init__(self, **cfg) -> None:
        super().__init__()
        self.model = BaseModel(**cfg).to(cfg["device"])
        self.optimizer = Adam(self.model.parameters(), lr=cfg["learning_rate"])
        self.criterion = nn.CrossEntropyLoss().to("cuda")        
        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=60, eta_min=5e-4)
        
        if cfg["mode"] == 'train' :
            self.train_loader, self.valid_loader = self.create_dataloader([self.get_transform('train', **cfg), 
                                                                           self.get_transform('valid', **cfg)], **cfg)
        elif cfg["mode"] == 'infer' :
            self.test_loader = self.create_dataloader(self.get_transform('infer', **cfg), **cfg)
            
    def train(self, **cfg) :
        self.run(**cfg)
        
    def train_on_batch(self, img, label, **cfg) :
        self.optimizer.zero_grad()

        img = img.to(cfg["device"])
        label = label.to(cfg["device"])
    
        output = self.model(img)
        # loss = lam * self.criterion(output, label_a) + (1 - lam) * self.criterion(output, label_b)
        loss = self.criterion(output, label.type(torch.float32))
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

    def valid_on_batch(self, img, label, **cfg):
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
            
            acc = score(label, output)
        batch_metric = {
            "acc" : acc,
            "loss" : loss.item()
        }
        
        return batch_metric
       
    def infer(self, **cfg) :
        self.prediction(**cfg)
    
    
    def get_transform(self, _mode, **cfg) :
        resize = cfg["resize"]
        if _mode == 'train' :
            return A.Compose([
                A.Resize(resize, resize),
                A.Normalize(),
                ToTensorV2()
            ])
        elif _mode == 'valid' :
            return A.Compose([
                A.Resize(resize, resize),
                # A.Emboss(p=1),
                # A.Sharpen(p=1), 
                # A.FancyPCA(p=1),
                A.Normalize(),
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
        "mode" : "infer", #train, #infer
        
        "model_name" : "tf_efficientnetv2_s.in21k", #"tf_efficientnetv2_m.in21k", #"swinv2_base_window12to16_192to256_22kft1k",
        #"tf_efficientnetv2_s.in21k",#"eva_large_patch14_196.in22k_ft_in1k",#"beit_base_patch16_224.in22k_ft_in22k",
        "num_classes" : 18,
        
        "learning_rate" : 1e-4,
        "focal_alpha" : 2,
        "focal_gamma" : 2,
        "resize" : 300,
        
        "data_path" : "./data/train",
        "epochs" : 80,
        "batch_size" : 16,
        "num_worker" : 1,
        "early_stop_patient" : 10,
        
        "reuse" : False, #True, #False
        "weight_path" : "./ckpt/tf_efficientnetv2_s.in21k/sigmoid_labeling_scratch_asyloss_step5_OverSampling_ModifiedAug2/24E-val0.8545489091407458-tf_efficientnetv2_s.in21k.pth",
        
        "save_path" : "./ckpt/tf_efficientnetv2_s.in21k/sigmoid_labeling_scratch_asyloss_step5_OverSampling_ModifiedAug2",
        "output_path" : "./output/tf_efficientnetv2_s.in21k/sigmoid_labeling_scratch_asyloss_step5_OverSampling_ModifiedAug2",
        "log_path" : "./logging/",
        "device" : "cuda",
        
    }        
    
    if cfg["mode"] == "train" :
        cfg["shuffle"] = True
    elif cfg["mode"] == "infer" :
        cfg["shuffle"] = False
    
    save_config(cfg, cfg["save_path"], save_name=cfg["mode"]+"_config")
    
    base_main = BaseMain(**cfg)
    
    if cfg["mode"] == "train" :
        base_main.train(**cfg)
    elif cfg["mode"] == "infer" :
        base_main.infer(**cfg)
    
    