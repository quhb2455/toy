from typing import Tuple
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
    
class BackgroundRemove(ImageOnlyTransform):
    def __init__(self,
                 threshold=215, 
                 always_apply: bool = False, p: float = 0.5):
        super(BackgroundRemove, self).__init__(always_apply, p)
        self.threshold = threshold
    
    def apply(self, img, **params):
        return self.remove_background(img)
    
    def remove_background(self, img):
        img[np.where((img > [self.threshold, self.threshold, self.threshold]).all(axis = 2))] = [0,0,0]
        return img
    
    def get_transform_init_args_names(self):
        return ("threshold",)
    
class BaseMain(Trainer, Predictor, DatasetCreater) :
    def __init__(self, **cfg) -> None:
        super().__init__()
        DatasetCreater.__init__(self)
        
        self.model = BaseModel(**cfg).to(cfg["device"])
        self.optimizer = Adam(self.model.parameters(), lr=cfg["learning_rate"])
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg['label_weight'])).to("cuda")        
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=60, eta_min=5e-4)
        
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
        loss = self.criterion(output, label)
        loss.backward()
        self.optimizer.step()
        
        acc = score(label, output)

        
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
    
    # def predict_on_batch(self, img, **cfg) :
        
    #     img = img.to(cfg["device"])
    #     output = self.model(img)
        
    #     # output = output * cfg["label_weight"]
    #     output = output.detach().cpu() * np.array([[cfg["label_weight"]] * output.shape[0]])[0]

    #     # binary_ = torch.softmax(output, dim=1)
    #     # binary_[binary_  > 0.9] = 1 
    #     # binary_[binary_  <= 0.9] = 0 
    #     # return output.argmax(1).detach().cpu().numpy().tolist()
        
    #     return output.argmax(1).numpy().tolist()
    
    def get_transform(self, _mode, **cfg) :
        resize = cfg["resize"]
        if _mode == 'train' :
            return A.Compose([
                A.Resize(resize, resize),
                BackgroundRemove(always_apply=True),
                # MixedEdgeImage(alpha=0.15, beta=0.85, p=0.5),
                A.OneOf([
                    A.CLAHE(p=1),
                    A.ImageCompression(p=1),
                ],p=1),
                A.RandomBrightnessContrast(brightness_limit=(-0.3, -0.1), p=1),
                A.OneOf([
                    A.GridDistortion(p=1, 
                        always_apply=False, 
                        num_steps=1, 
                        distort_limit=(-0.1, 0.3), 
                        interpolation=2, 
                        border_mode=2, 
                        value=(0, 0, 0), 
                        mask_value=None),
                    A.OpticalDistortion(p=1,
                        distort_limit=0.4, shift_limit=0.04),    
                ],p=0.5),                
                A.ElasticTransform(p=0.7, 
                    alpha=120, sigma=120 * 0.1, alpha_affine=120 * 0.1),
                A.RandomGridShuffle((3, 3), p=0.4),
                A.Normalize(),
                ToTensorV2()
            ])
        elif _mode == 'valid' :
            return A.Compose([
                A.Resize(resize, resize),
                BackgroundRemove(always_apply=True),
                A.Normalize(),
                ToTensorV2()
            ])
        elif _mode == 'infer' : 
            return A.Compose([
                A.Resize(resize, resize),
                BackgroundRemove(always_apply=True),
                A.Normalize(),
                ToTensorV2()
            ])

    
if __name__ == "__main__" :
    
    cfg = {
        "mode" : "train", #train, #infer
        
        "model_name" : "tf_efficientnetv2_s.in21k", #"tf_efficientnetv2_m.in21k", #"swinv2_base_window12to16_192to256_22kft1k",
        #"tf_efficientnetv2_s.in21k",#"eva_large_patch14_196.in22k_ft_in1k",#"beit_base_patch16_224.in22k_ft_in22k",
        "num_classes" : 5,
        
        "learning_rate" : 1e-4,
        "focal_alpha" : 2,
        "focal_gamma" : 2,
        "resize" : 512,
        
        "data_path" : "./data/train_thumbnails", #"./data/test_thumbnails", #"./data/train_thumbnails",
        "csv_path" : "./data/train.csv",
        "epochs" : 80,
        "batch_size" : 4,
        "num_worker" : 4,
        "early_stop_patient" : 10,
        
        "binary_mode" : False,
        "reuse" : False, #True, #False
        "weight_path" : "./ckpt/tf_efficientnetv2_s.in21k/rmbg_lossweight_effiv2s_512/11E-val0.5294871794871795-tf_efficientnetv2_s.in21k.pth",
        
        "save_path" : "./ckpt/tf_efficientnetv2_s.in21k/rmbg_lossweight_effiv2s_512",
        "output_path" : "./output/tf_efficientnetv2_s.in21k/rmbg_lossweight_effiv2s_512",
        "log_path" : "./logging",
        "device" : "cuda",
        "label_name" :["HGSC", "LGSC", "EC", "CC", "MC"],
        
        "label_weight" : [0.4646, 0.3709, 0.2072, 0.9787, 1.0]
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
    
    