from trainer import Trainer
from predictor import Predictor
from datasets import DatasetCreater
from models import BaseModel, DivBaseModel
from loss_fn import FocalLoss, TripletMargingLoss, RGBDistanceCELoss
from utils import save_config, mixup, cutmix, score, get_loss_weight, set_seed, distance_score

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
        self.model = DivBaseModel(**cfg).to(cfg["device"])
        # self.cls_head = ClassifierHead(**cfg).to(cfg["device"])
        self.optimizer = Adam(self.model.parameters(), lr=cfg["learning_rate"])
        # self.optimizer = Adam(
        #     [{'params': self.model.parameters()},
        #     {'params': self.cls_head.parameters()}], lr=cfg["learning_rate"])
        # self.criterion = FocalLoss(alpha=cfg["focal_alpha"],gamma=cfg["focal_gamma"]).to(cfg["device"])
        self.criterion = nn.CrossEntropyLoss().to(cfg["device"])
        # self.criterion = RGBDistanceCELoss(color_mean=torch.tensor(cfg['mean']), device=cfg["device"])#.to(cfg["device"])
        # self.criterion = nn.MSELoss().to(cfg['device'])
        self.metric_criterion = TripletMargingLoss().to(cfg["device"])
        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=60, eta_min=5e-4)
        
        self.color_mean = torch.tensor(cfg['mean']).to(cfg["device"])
        self.cosine_dist = nn.CosineSimilarity()
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
        
        # mean_label = torch.tensor(cfg["mean"])[label].to(cfg["device"])
        
        img, lam, label_a, label_b = cutmix(img, label)

        emb, output = self.model(img, div=True)
        color_dist = torch.stack([self.cosine_dist(output, self.color_mean[i]) for i in range(18)], dim= 1)
        ce_loss = lam * self.criterion(color_dist, label_a) + (1 - lam) * self.criterion(color_dist, label_b)
        m_loss = lam * self.metric_criterion(emb, label_a) + (1 - lam) * self.metric_criterion(emb, label_b)
        
        # output = self.cls_head(mse_output)
        ## loss = lam * self.criterion(output, label_a) + (1 - lam) * self.criterion(output, label_b) + self.metric_criterion(emb, label)
        # loss = self.criterion(output, label.type(torch.float32))
        
        # loss = self.criterion(mse_output, mean_label)  + self.cls_criterion(output, label)
        loss = ce_loss + m_loss
        
        loss.backward()
        self.optimizer.step()
        
        # acc = score(mixup_label, output)
        # acc = score(label, output)
        acc = score(label, color_dist)
        # acc = score(torch.argmax(label, dim=1), output)
        # acc = distance_score(output, cfg["mean"], label)
        
        batch_metric = {
            "acc" : acc,
            "loss" : loss.item()
        }
        
        return batch_metric 

    def valid_on_batch(self, img, label, step, **cfg):
        img = img.to(cfg["device"])
        
        # mean_label = torch.tensor(cfg["mean"])[label].to(cfg["device"])
        label = label.to(cfg["device"])
        
        if cfg["binary_mode"] :
            mixup_label = torch.argmax(label, dim=1)
        
            output = self.model(img)
            loss = self.criterion(output, label.type(torch.float32))
            
            acc = score(mixup_label, output)
        else :        
            # mse_output = self.model(img)
            # output = self.cls_head(mse_output)
            output = self.model(img)
            
            color_dist = torch.stack([self.cosine_dist(output, self.color_mean[i]) for i in range(18)], dim= 1)

            # loss = self.criterion(output, label)
            loss = self.criterion(color_dist, label)  + self.metric_criterion(output, label)
            
            acc, cls_report = score(label, color_dist, mode="valid")
            # acc, cls_report = score(label, output, mode="valid")
            # acc, cls_report = distance_score(output, cfg["mean"], label, mode="valid")
            
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
        
        "model_name" : "wide_resnet101_2", #"tf_efficientnetv2_m.in21k", #"swinv2_base_window12to16_192to256_22kft1k",
        #"tf_efficientnetv2_s.in21k",#"eva_large_patch14_196.in22k_ft_in1k",#"beit_base_patch16_224.in22k_ft_in22k", #"convnextv2_base.fcmae_ft_in1k"
        "num_classes" : 3,#18,
        
        "learning_rate" : 1e-3,
        "focal_alpha" : 2,
        "focal_gamma" : 2,
        "resize" : 112,
        
        "data_train_path" : "./sub-task2/Dataset/Train",
        "data_train_csv_path" : "./sub-task2/Dataset/aug_info_etri20_color_train.csv",
        "data_valid_path" : "./sub-task2/Dataset/Validation",
        "data_valid_csv_path" : "./sub-task2/Dataset/info_etri20_color_validation.csv",
        
        "data_infer_path" : "/aif/Dataset/Test/",
        "data_infer_csv_path" : "/aif/Dataset/info_etri20_color_test.csv",
        # "data_infer_path" : "./sub-task2/Dataset/Test_sample",
        # "data_infer_csv_path" : "./sub-task2/Dataset/info_etri20_color_test_sample.csv",
        
        "epochs" : 80,
        "batch_size" : 128,
        "num_worker" : 4,
        "early_stop_patient" : 30,
        
        "reuse" : False, #True, #False
        "weight_path" : "./sub-task2/ckpt/wide_resnet101_2/color_mse/57E-val0.4005303784909048-tf_efficientnetv2_s.in21k.pth",
        
        "save_path" : "./sub-task2/ckpt/wide_resnet101_2/color_CosSim_CELoss",
        "output_path" : "./sub-task2/output/wide_resnet101_2/color_CosSim_CELoss",
        "log_path" : "./sub-task2/logging/color_CosSim_CELoss",
        "device" : "cuda",
        
        "mean" : [[0.5929038396997668, 0.3447893774705255, 0.3534702696658963],
                [0.7665971665100372, 0.5314981737576343, 0.4956528644167255],
                [0.7309501628007073, 0.5093203567310064, 0.40060355200704784],
                [0.7787287993428619, 0.6552336182558675, 0.6522036959181674],
                [0.5589467650404378, 0.49574030928901575, 0.5568065255342798],
                [0.5447731371879583, 0.44650972469321293, 0.3918042830671195],
                [0.7149737882347199, 0.6360402982412814, 0.564551075328806],
                [0.7828125076920801, 0.7402873690555676, 0.6970276026861575],
                [0.7647062373363801, 0.6932244844060559, 0.5163739765777509],
                [0.7000854542618209, 0.5555028407266888, 0.3702079283344232],
                [0.6021083419661138, 0.6571667375890373, 0.7043396523777105],
                [0.3873693275064437, 0.44309869447395145, 0.54100329972712],
                [0.35859169981022515, 0.3576853452716988, 0.388269571394032],
                [0.42615141236897475, 0.487827988048575, 0.45356261225350003],
                [0.4960811126398995, 0.4775583252393666, 0.42478018575630516],
                [0.7879098283532181, 0.7655871917324902, 0.7553171471311435],
                [0.5414178383405477, 0.5272657009473801, 0.5259978119992829],
                [0.34385191904078777, 0.331707936345238, 0.3298444517893208]],
        "binary_mode" : False,
        "seed": 2455,
        "note" : ["cosine Sim + CE Loss with color label", "Metric Learning", "Label1,2,8,9에 Offline Aug 사용", 
                "CenterCrop 사용하지 않음", "Aug 정도 하향조정",
                "Cutmix 사용",  "Adam Optim 사용"]
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
    
    