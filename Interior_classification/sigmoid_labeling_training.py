from trainer import Trainer
from predictor import Predictor
from datasets import DatasetCreater, SigmoidCustomDataset
from models import BaseModel
from loss_fn import FocalLoss, AsymmetricLoss
from utils import save_config, mixup, cutmix, score, get_loss_weight, label_dec, label_enc

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from pytorch_metric_learning import miners, losses
from sklearn.model_selection import train_test_split

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
        # self.criterion = nn.CrossEntropyLoss().to("cuda")
        # self.criterion=nn.BCEWithLogitsLoss().to("cuda")
        # self.miner = miners.MultiSimilarityMiner()
        # self.criterion = losses.TripletMarginLoss()
        # self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(get_loss_weight(cfg["data_path"])).to("cuda"))
        #FocalLoss(alpha=cfg["focal_alpha"], gamma=cfg["focal_gamma"])
        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=60, eta_min=5e-4)
        
        if cfg["mode"] == 'train' :
            self.train_loader, self.valid_loader = self.create_dataloader([self.get_transform('train', **cfg), 
                                                                           self.get_transform('valid', **cfg)], **cfg)
        elif cfg["mode"] == 'infer' :
            self.test_loader = self.create_dataloader(self.get_transform('infer', **cfg), **cfg)
    
    def create_dataset(self, transform, **cfg) :
        img_path, _, sig_label_list = self.get_data(**cfg)        
        
        if cfg["mode"] == "train" :
            save_config(transform[0].to_dict(), cfg["save_path"], save_name="train_transform")
            save_config(transform[1].to_dict(), cfg["save_path"], save_name="valid_transform")
            
            return [SigmoidCustomDataset(img_path[0], sig_label_list[0], transform=transform[0]), 
                    SigmoidCustomDataset(img_path[1], sig_label_list[1], transform=transform[1])]
            
        elif cfg["mode"] == 'infer' :
            save_config(transform.to_dict(), cfg["output_path"], save_name="infer_transform")
            return SigmoidCustomDataset(img_path, sig_label_list, transform=transform)
        
    def train(self, **cfg) :
        self.run(**cfg)
        
    def train_on_batch(self, img, label, **cfg) :
        self.optimizer.zero_grad()

        img = img.to(cfg["device"])
        label = label.to(cfg["device"])
        
        if cfg["binary_mode"] :
            mixup_label = torch.argmax(label, dim=1)
            # if np.random.binomial(n=1, p=0.5) :
            img, lam, label_a, label_b = mixup(img, mixup_label)
            # img, lam, label_a, label_b = mixup(img, label)
        

            label_a = torch.FloatTensor([[1 if i == a else 0 for i in range(len(cfg["label_name"]))] for a in label_a]).to(cfg["device"])
            label_b = torch.FloatTensor([[1 if i == b else 0 for i in range(len(cfg["label_name"]))] for b in label_b]).to(cfg["device"])
        
        
        else :
            img, lam, label_a, label_b = mixup(img, label)
    
        output = self.model(img)
        # loss = lam * self.criterion(output, label_a) + (1 - lam) * self.criterion(output, label_b)
        loss = lam * self.criterion(output, label_a) + (1 - lam) * self.criterion(output, label_b)
                
        loss.backward()
        self.optimizer.step()
        
        # acc = score(mixup_label, output)
        # acc = score(label, output)
        acc = score(torch.argmax(label, dim=1), output)
        
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
            
            # acc = score(label, output)
            acc = score(torch.argmax(label, dim=1), output)
            
        batch_metric = {
            "acc" : acc,
            "loss" : loss.item()
        }
        
        return batch_metric
       
    def infer(self, **cfg) :
        self.prediction(**cfg)
        
    def predict_on_batch(self, img, **cfg) :
        img = img.to(cfg["device"])
        
        if cfg["error_rate"] :
            return self.model(img).argmax(1).detach().cpu().numpy().tolist()

        if cfg["sigmoid_labeling"] :
            return torch.sigmoid(self.model(img)).detach().cpu().numpy().tolist()
        
        
    def get_data(self, **cfg) :
        data_path = cfg["data_path"]
        sig_path = cfg["sigmoid_labeling_path"] if cfg["sigmoid_labeling"] else None
        
        mode = cfg["mode"]
        
        img_path_list = []
        label_list = []
        
        if mode == "infer" :
            for p in glob(os.path.join(data_path, "*")) :
                img_path_list.append(p)
            return img_path_list, None, None
        
        else :
            sig_label = pd.read_csv(sig_path)
            
            
            for p in glob(os.path.join(data_path, "*", "*")) :
                img_path_list.append(p)
                label_list.append(p.split("\\")[-2])

            train_img, valid_img, train_label, valid_label = train_test_split(img_path_list, 
                                                                              label_list, 
                                                                              test_size=0.2, 
                                                                              stratify=label_list, 
                                                                              random_state=2455)
            if cfg["sigmoid_labeling"] :
                train_sig_label = []
                for p in tqdm(train_img):
                    label_name = p.split("\\")[-2]
                    original_filename = p.split("\\")[-1].split("_")[-1]  if "_" in  p.split("\\")[-1] else  p.split("\\")[-1]
                    sig_id = os.path.join("./data/noaug_ori_train", label_name, original_filename)
                    
                    train_sig_label.append([round(sig_label[sig_label['id'] == sig_id][n].item(), 4) for n in cfg["label_name"]])
                
                valid_sig_label = []
                for p in tqdm(valid_img):
                    label_name = p.split("\\")[-2]
                    original_filename = p.split("\\")[-1].split("_")[-1]  if "_" in  p.split("\\")[-1] else  p.split("\\")[-1]
                    sig_id = os.path.join("./data/noaug_ori_train", label_name, original_filename)
                    
                    valid_sig_label.append([round(sig_label[sig_label['id'] == sig_id][n].item(), 4) for n in cfg["label_name"]])
                return [train_img, valid_img], [train_label, valid_label], [train_sig_label, valid_sig_label]
            
            else :
                return [train_img, valid_img], [train_label, valid_label]
    
    def save_to_csv(self, results, **cfg) :
        
        if cfg["error_rate"] :
            _label_dec = label_dec(cfg["label_name"])
            img_name_list = [p for p in glob(os.path.join(cfg["data_path"], "*"))]
            res_label_list = [_label_dec[i] for i in results]        
            
            df = pd.DataFrame({"id" : img_name_list, "label":res_label_list})
            df.to_csv(os.path.join(cfg["output_path"], "train_dataset_error_rate.csv"), index=False)
        
        if cfg["sigmoid_labeling"] :
            _label_dec = label_dec(cfg["label_name"])
            _label_enc = label_enc(cfg["label_name"])

            df_label_score_list = {n:[] for idx, n in enumerate(cfg["label_name"])}
            df_label_score_list['id'] = [p for p in glob(os.path.join(cfg["data_path"], "*"))]
            
            for idx, res in enumerate(results) :
                gt = _label_enc[df_label_score_list['id'][idx].split("\\")[-2]]
                res = np.array(res)
                res *= cfg["sigmoid_norm"]
                res[gt] = 1.0
                
                for i in range(len(cfg['label_name'])) :
                    df_label_score_list[_label_dec[i]].append(res[i])
                    
            df = pd.DataFrame(df_label_score_list)
            df.to_csv(os.path.join(cfg["output_path"], "sigmoid_labeling.csv"), index=False, encoding="utf-8")
        
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
                # A.RandomGridShuffle((3, 3), p=0.4),
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
        "mode" : "train", #train, #infer
        
        "error_rate" : False,
        "sigmoid_labeling" : True,
        "sigmoid_labeling_path" : "./output/tf_efficientnetv2_s.in21k/sigmoid_labeling_scratch_asyloss_step4/sigmoid_labeling.csv",
        "sigmoid_norm" : 0.5,
        
        "model_name" : "tf_efficientnetv2_s.in21k", #"tf_efficientnetv2_m.in21k", #"swinv2_base_window12to16_192to256_22kft1k",
        #"tf_efficientnetv2_s.in21k",#"eva_large_patch14_196.in22k_ft_in1k",#"beit_base_patch16_224.in22k_ft_in22k",
        "num_classes" : 19,
        
        "learning_rate" : 1e-4,
        "focal_alpha" : 2,
        "focal_gamma" : 2,
        "resize" : 300,
        
        "data_path" : "./data/train",#"./data/noaug_ori_train/*", #"./data/combine_train",#"./data/train",#"./data/test",
        "epochs" : 140,
        "batch_size" : 16,
        "num_worker" : 2,
        "early_stop_patient" : 10,
        
        "binary_mode" : False,
        "reuse" : False, #True, #False
        "weight_path" : "./ckpt/tf_efficientnetv2_s.in21k/sigmoid_labeling_scratch_asyloss_step5/26E-val0.8973153669801178-tf_efficientnetv2_s.in21k.pth",
        
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
    
    