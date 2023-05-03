from trainer import Trainer
from predictor import Predictor
from datasets import DatasetCreater
from models import BaseModel
from loss_fn import FocalLoss
from utils import save_config, mixup, cutmix, score, get_loss_weight, label_dec

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from pytorch_metric_learning import miners, losses

import os
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob

class BaseMain(Trainer, Predictor, DatasetCreater) :
    def __init__(self, **cfg) -> None:
        super().__init__()
        self.model = BaseModel(**cfg).to(cfg["device"])
        self.optimizer = Adam(self.model.parameters(), lr=cfg["learning_rate"])
        self.criterion = nn.CrossEntropyLoss().to("cuda")

        
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
        
        if cfg["error_rate"] :
            return self.model(img).argmax(1).detach().cpu().numpy().tolist()

        if cfg["sigmoid_labeling"] :
            return torch.sigmoid(self.model(img)).detach().cpu().numpy().tolist()
    
    def save_to_csv(self, results, **cfg) :
        
        if cfg["error_rate"] :
            _label_dec = label_dec(cfg["label_name"])

            img_name_list = [p for p in glob(os.path.join(cfg["data_path"], "*"))]
            res_label_list = [_label_dec[i] for i in results]        
            
            df = pd.DataFrame({"id" : img_name_list, "label":res_label_list})
            df.to_csv(os.path.join(cfg["output_path"], "train_dataset_error_rate.csv"), index=False)
        
        if cfg["sigmoid_labeling"] :
            _label_dec = label_dec(cfg["label_name"])
            df_label_score_list = {n:[] for idx, n in enumerate(cfg["label_name"])}
            df_label_score_list['id'] = [p for p in glob(os.path.join(cfg["data_path"], "*"))]
            
            for res in results :
                for i in range(len(cfg['label_name'])) :
                    df_label_score_list[_label_dec[i]].append(res[i])
                    
            df = pd.DataFrame(df_label_score_list)
            df.to_csv(os.path.join(cfg["output_path"], "sigmoid_labeling.csv"), index=False)
            
            
    def get_data(self, **cfg) :
        data_path = cfg["data_path"]
        sig_path = cfg["sigmoid_labeling_path"] if cfg["sigmoid_labeling"] else None
        
        mode = cfg["mode"]
        
        img_path_list = []
        label_list = []
        
        if mode == "infer" :
            for p in glob(os.path.join(data_path, "*")) :
                img_path_list.append(p)
            return img_path_list, None
        
        else :
            sig_label = pd.read_csv(sig_path)
            
            
            for p in glob(os.path.join(data_path, "*", "*")) :
                img_path_list.append(p)
                label_list.append(label_name)

            train_img, valid_img, train_label, valid_label = train_test_split(img_path_list, 
                                                                              label_list, 
                                                                              test_size=0.2, 
                                                                              stratify=label_list, 
                                                                              random_state=2455)
            if cfg["sigmoid_labeling"] :
                train_sig_label = []
                for p in train_img:
                    label_name = p.split("\\")[-2]
                    original_filename = p.split("\\")[-1].split("_")[0] + ".png"
                    sig_id = os.path.join(data_path, label_name, original_filename)
                    
                    train_sig_label.append([int(sig_label[sig_label['id'] == sig_id][n].item())]for n in cfg["label_name"])
                
                valid_sig_label = []
                for p in valid_img:
                    label_name = p.split("\\")[-2]
                    original_filename = p.split("\\")[-1].split("_")[0] + ".png"
                    sig_id = os.path.join(data_path, label_name, original_filename)
                    
                    valid_sig_label.append([int(sig_label[sig_label['id'] == sig_id][n].item()) for n in cfg["label_name"]])
                return [train_img, valid_img], [train_sig_label, valid_sig_label]
            
            else :
                return [train_img, valid_img], [train_label, valid_label]
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
        "error_rate" : True,
        "sigmoid_labeling" : False, 
        
        "model_name" : "tf_efficientnetv2_s.in21k", #"tf_efficientnetv2_m.in21k", #"swinv2_base_window12to16_192to256_22kft1k",
        #"tf_efficientnetv2_s.in21k",#"eva_large_patch14_196.in22k_ft_in1k",#"beit_base_patch16_224.in22k_ft_in22k",
        "num_classes" : 19,
        
        "learning_rate" : 1e-3,
        "focal_alpha" : 2,
        "focal_gamma" : 2,
        "resize" : 300,
        
        "sigmoid_labeling_path" : "./output/tf_efficientnetv2_s.in21k/sigmoid_labeling_step2/sigmoid_labeling.csv",
        "data_path" : "./data/noaug_ori_train/*", #"./data/combine_train",#"./data/train",#"./data/test",
        "epochs" : 60,
        "batch_size" : 16,
        "num_worker" : 3,
        "early_stop_patient" : 10,
                
        "reuse" : False, #True, #False
        "weight_path" : "./ckpt/tf_efficientnetv2_s.in21k/sigmoid_labeling_scratch_asyloss_step5/31E-val0.8926313482028264-tf_efficientnetv2_s.in21k.pth",
        
        "save_path" : "./ckpt/tf_efficientnetv2_s.in21k/sigmoid_labeling_scratch_asyloss_step5",
        "output_path" : "./output/tf_efficientnetv2_s.in21k/sigmoid_labeling_scratch_asyloss_step5",
        
        "device" : "cuda",
        "label_name" : ["가구수정", "걸레받이수정", "곰팡이", "꼬임", "녹오염", "들뜸",
                        "면불량", "몰딩수정", "반점", "석고수정", "오염", "오타공", "울음",
                        "이음부불량", "창틀,문틀수정", "터짐", "틈새과다", "피스", "훼손",]
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
    
    