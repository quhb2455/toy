from trainer import Trainer
from predictor import Predictor
from datasets import DatasetCreater
from models import BaseModel
from loss_fn import FocalLoss, AsymmetricLoss, AsymmetricLossSingleLabel
from utils import save_config, mixup, cutmix, score, get_loss_weight, label_dec, label_enc

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from pytorch_metric_learning import miners, losses

import os
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
import pandas as pd
    
class BaseMain(Trainer, Predictor, DatasetCreater) :
    def __init__(self, **cfg) -> None:
        super().__init__()
        self.model = BaseModel(**cfg).to(cfg["device"])
        self.optimizer = Adam(self.model.parameters(), lr=cfg["learning_rate"])
        self.criterion = AsymmetricLoss().to(cfg["device"])
        
        
        if cfg["mode"] == 'train' :
            self.train_loader, self.valid_loader = self.create_dataloader([self.get_transform('train', **cfg), 
                                                                           self.get_transform('valid', **cfg)], **cfg)
        elif cfg["mode"] == 'infer' :
            self.test_loader = self.create_dataloader(self.get_transform('infer', **cfg), **cfg)
            
    def train(self, **cfg) :
        self.run(**cfg)
           
       
    def infer(self, **cfg) :
        self.prediction(**cfg)
    
    def predict_on_batch(self, img, **cfg) :
        img = img.to(cfg["device"])
        return torch.sigmoid(self.model(img)).detach().cpu().numpy().tolist()
    
    def save_to_csv(self, results, **cfg) :
        _label_dec = label_dec(cfg["label_name"])
        _label_enc = label_enc(cfg["label_name"])

        df_label_score_list = {n:[] for idx, n in enumerate(cfg["label_name"])}
        df_label_score_list['id'] = [p for p in glob(os.path.join(cfg["data_path"], "*"))]
        
        for idx, res in enumerate(results) :            
            for i in range(len(cfg['label_name'])) :
                df_label_score_list[_label_dec[i]].append(res[i])
                
        df = pd.DataFrame(df_label_score_list)
        df.to_csv(os.path.join(cfg["output_path"], "ensemble_sigmoid_score.csv"), index=False, encoding="utf-8")
    
    def get_transform(self, _mode, **cfg) :
        resize = cfg["resize"]
        if _mode == 'train' :
            return A.Compose([
                A.Resize(resize, resize),
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
        "num_classes" : 19,
        
        "learning_rate" : 1e-4,
        "focal_alpha" : 2,
        "focal_gamma" : 2,
        "resize" : 300,
        
        "data_path" : "./data/test",#"./data/more_first_aug_train", #"./data/combine_train",#"./data/train",#"./data/test",
        "epochs" : 80,
        "batch_size" : 16,
        "num_worker" : 1,
        "early_stop_patient" : 10,
        
        "binary_mode" : False,
        "reuse" : False, #True, #False
        "weight_path" : "./ckpt/tf_efficientnetv2_s.in21k/resize300_mixup_CEloss/22E-val0.8811827300038849-tf_efficientnetv2_s.in21k.pth",
        
        "save_path" : "./ckpt/tf_efficientnetv2_s.in21k/resize300_mixup_CEloss",
        "output_path" : "./output/tf_efficientnetv2_s.in21k/resize300_mixup_CEloss",
        
        "device" : "cuda",
        "label_name" : ["가구수정", "걸레받이수정", "곰팡이", "꼬임", "녹오염", "들뜸",
                        "면불량", "몰딩수정", "반점", "석고수정", "오염", "오타공", "울음",
                        "이음부불량", "창틀,문틀수정", "터짐", "틈새과다", "피스", "훼손",],
        
        
        "label_weight" : [0.7666686290865917, 0.7261767277215512,0.738474800154709, 0.7536669073754095, 0.7848826747173203, 
                          0.7657658643255456, 0.7158896209947423, 0.7260579484233907, 0.7485242309606435, 0.6422744210327779, 
                          0.552108503121636, 0.6986621176537868, 0.6401662469548162,  0.7378511773074832,  0.768388257281936,
                          0.7482707985553708, 0.8101942504483788, 0.7820480751627472, 0.39420923444967515]
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
    
    