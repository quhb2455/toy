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

class MixedEdgeImage(ImageOnlyTransform):
    def __init__(
        self,
        alpha=0.3,
        beta=0.7,
        always_apply=False,
        p=1
    ):
        super(MixedEdgeImage, self).__init__(always_apply, p)
        self.alpha = alpha
        self.beta = beta
        
    def apply(self, img, **params):
        return self.mixed_edge_image(img)
    
    def mixed_edge_image(self, img) :
        canny_img = cv2.Canny(img, 100, 200, apertureSize=7)
        empty_img = np.zeros((canny_img.shape[0], canny_img.shape[1], 3))
        for i in range(3):
            empty_img[:,:,i] = canny_img
        return np.uint8(empty_img *self.alpha + img * self.beta)
    
    def get_transform_init_args_names(self):
        return ("alpha", "beta")
    
    
class BaseMain(Trainer, Predictor, DatasetCreater) :
    def __init__(self, **cfg) -> None:
        super().__init__()
        self.model = BaseModel(**cfg).to(cfg["device"])
        self.optimizer = Adam(self.model.parameters(), lr=cfg["learning_rate"])
        self.criterion = nn.CrossEntropyLoss().to("cuda")        
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
        "weight_path" : "./ckpt/tf_efficientnetv2_s.in21k/sigmoid_labeling_scratch_asyloss_step5_OverSampling_ModifiedAug2/24E-val0.8545489091407458-tf_efficientnetv2_s.in21k.pth",
        
        "save_path" : "./ckpt/tf_efficientnetv2_s.in21k/sigmoid_labeling_scratch_asyloss_step5_OverSampling_ModifiedAug2",
        "output_path" : "./output/tf_efficientnetv2_s.in21k/sigmoid_labeling_scratch_asyloss_step5_OverSampling_ModifiedAug2",
        
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
    
    