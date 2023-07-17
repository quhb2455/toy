from trainer import Trainer
from predictor import Predictor
from datasets import DatasetCreater, MultiHeadCustomDataset
from models import BaseModel, MultiHeadBaseModel, ClassifierHead_1, ClassifierHead_2, ClassifierHead_3, ClassifierHead_4, ClassifierHead_5, ClassifierHead_6
from loss_fn import FocalLoss, AsymmetricLoss, AsymmetricLossSingleLabel
from utils import save_config, mixup, cutmix, score, get_loss_weight
from optim_fn import SAM

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

class BaseMain(Trainer, Predictor, DatasetCreater) :
    def __init__(self, **cfg) -> None:
        super().__init__()
        self.base_model = MultiHeadBaseModel(**cfg).to(cfg["device"])

        self.head_1 = ClassifierHead_1(base_dim=cfg["base_dim"], mid_dim=512).to(cfg["device"])
        self.head_2 = ClassifierHead_2(base_dim=cfg["base_dim"], mid_dim=512).to(cfg["device"])
        self.head_3 = ClassifierHead_3(base_dim=cfg["base_dim"], mid_dim=512).to(cfg["device"])
        self.head_4 = ClassifierHead_4(base_dim=cfg["base_dim"], mid_dim=512).to(cfg["device"])
        self.head_5 = ClassifierHead_5(base_dim=cfg["base_dim"], mid_dim=512).to(cfg["device"])
        self.head_6 = ClassifierHead_6(base_dim=cfg["base_dim"], mid_dim=512).to(cfg["device"])
        
        # self.optimizer = Adam([{'params': self.base_model.parameters()}, 
        #                         {'params': self.head_1.parameters()}, 
        #                         {'params': self.head_2.parameters()}, 
        #                         {'params': self.head_3.parameters()}, 
        #                         {'params': self.head_4.parameters()}, 
        #                         {'params': self.head_5.parameters()}, 
        #                         {'params': self.head_6.parameters()}], lr=cfg["learning_rate"])
        self.criterion = AsymmetricLoss().to(cfg["device"])
        # self.criterion = AsymmetricLossSingleLabel().to(cfg["device"])


        # self.base_optimizer = Adam
        self.optimizer = SAM([{'params': self.base_model.parameters()}, 
                                {'params': self.head_1.parameters()}, 
                                {'params': self.head_2.parameters()}, 
                                {'params': self.head_3.parameters()}, 
                                {'params': self.head_4.parameters()}, 
                                {'params': self.head_5.parameters()}, 
                                {'params': self.head_6.parameters()}], Adam, lr=cfg["learning_rate"]) # momentum=0.9
        
        if cfg["mode"] == 'train' :
            self.train_loader, self.valid_loader = self.create_dataloader([self.get_transform('train', **cfg), 
                                                                           self.get_transform('valid', **cfg)], **cfg)
        elif cfg["mode"] == 'infer' :
            self.test_loader = self.create_dataloader(self.get_transform('infer', **cfg), **cfg)
    
    def create_dataset(self, transform, **cfg) :
        img_path, label_list = self.get_data(**cfg)        
        
        if cfg["mode"] == "train" :
            save_config(transform[0].to_dict(), cfg["save_path"], save_name="train_transform")
            save_config(transform[1].to_dict(), cfg["save_path"], save_name="valid_transform")
            
            return [MultiHeadCustomDataset(img_path[0], label_list[0], transform=transform[0], senario=cfg["num_senario"]), 
                    MultiHeadCustomDataset(img_path[1], label_list[1], transform=transform[1], senario=cfg["num_senario"])]
            
        elif cfg["mode"] == 'infer' :
            save_config(transform.to_dict(), cfg["output_path"], save_name="infer_transform")
            return MultiHeadCustomDataset(img_path, label_list, transform=transform)
        
    def train(self, **cfg) :
        self.run(**cfg)
        
    def train_on_epoch(self, epoch, **cfg):
        self.base_model.train()
        self.head_1.train()
        self.head_2.train()
        self.head_3.train()
        self.head_4.train()
        self.head_5.train()
        self.head_6.train()
        train_acc, train_loss = [], []
        tqdm_train = tqdm(self.train_loader)
        for img, label, first_group, second_group in tqdm_train :
            batch_res = self.train_on_batch(img, label, first_group, second_group, **cfg)
            
            train_acc.append(batch_res["acc"])
            train_loss.append(batch_res["loss"])
            
            tqdm_train.set_postfix({
                "Epoch" : epoch,
                "Training Acc" : np.mean(train_acc),
                "Training Loss" : np.mean(train_loss),
            })
            
    def train_on_batch(self, img, label, first_grouping, second_grouping, **cfg) :
        self.optimizer.zero_grad()

        img = img.to(cfg["device"])
        label = label.to(cfg["device"])
        first_grouping = first_grouping.to(cfg["device"])
        second_grouping = second_grouping.to(cfg["device"])

        img, lam, label_a, label_b = mixup(img, label)

        fm = self.base_model(img)
        output_1 = self.head_1(fm, mask=second_grouping[:, :4])
        output_2 = self.head_2(fm, mask=second_grouping[:, 4:9])
        output_3 = self.head_3(fm, mask=second_grouping[:, 9:14])
        output_4 = self.head_4(fm, mask=second_grouping[:, 14:16])
        output_5 = self.head_5(fm, mask=second_grouping[:, 16:18])
        output_6 = self.head_6(fm, mask=second_grouping[:, 18])
        
        loss_1 = lam * self.criterion(output_1, label_a[:, :4]) + (1 - lam) * self.criterion(output_1, label_b[:, :4])
        loss_2 = lam * self.criterion(output_2, label_a[:, 4:9]) + (1 - lam) * self.criterion(output_2, label_b[:, 4:9])
        loss_3 = lam * self.criterion(output_3, label_a[:, 9:14]) + (1 - lam) * self.criterion(output_3, label_b[:, 9:14])
        loss_4 = lam * self.criterion(output_4, label_a[:, 14:16]) + (1 - lam) * self.criterion(output_4, label_b[:, 14:16])
        loss_5 = lam * self.criterion(output_5, label_a[:, 16:18]) + (1 - lam) * self.criterion(output_5, label_b[:, 16:18])
        loss_6 = lam * self.criterion(output_6, label_a[:, 18]) + (1 - lam) * self.criterion(output_6, label_b[:, 18])
        
        # loss = self.criterion(output_1, label[:, :4]) + self.criterion(output_2, label[:, 4:9]) \
        #         + self.criterion(output_3, label[:, 9:14]) + self.criterion(output_4, label[:, 14:16]) \
        #         + self.criterion(output_5, label[:, 16:18]) + self.criterion(output_6, label[:, 18])
        loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6
        loss.backward()
        # self.optimizer.step()
        self.optimizer.first_step(zero_grad=True)
        
        fm = self.base_model(img)
        output_1 = self.head_1(fm, mask=second_grouping[:, :4])
        output_2 = self.head_2(fm, mask=second_grouping[:, 4:9])
        output_3 = self.head_3(fm, mask=second_grouping[:, 9:14])
        output_4 = self.head_4(fm, mask=second_grouping[:, 14:16])
        output_5 = self.head_5(fm, mask=second_grouping[:, 16:18])
        output_6 = self.head_6(fm, mask=second_grouping[:, 18])
        
        loss_1 = lam * self.criterion(output_1, label_a[:, :4]) + (1 - lam) * self.criterion(output_1, label_b[:, :4])
        loss_2 = lam * self.criterion(output_2, label_a[:, 4:9]) + (1 - lam) * self.criterion(output_2, label_b[:, 4:9])
        loss_3 = lam * self.criterion(output_3, label_a[:, 9:14]) + (1 - lam) * self.criterion(output_3, label_b[:, 9:14])
        loss_4 = lam * self.criterion(output_4, label_a[:, 14:16]) + (1 - lam) * self.criterion(output_4, label_b[:, 14:16])
        loss_5 = lam * self.criterion(output_5, label_a[:, 16:18]) + (1 - lam) * self.criterion(output_5, label_b[:, 16:18])
        loss_6 = lam * self.criterion(output_6, label_a[:, 18]) + (1 - lam) * self.criterion(output_6, label_b[:, 18])
        
        loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6
        loss.backward()
        self.optimizer.second_step(zero_grad=True)
        
        
        
        output = torch.cat([output_1, output_2, output_3, output_4, output_5, output_6], dim=1)
        acc = score(torch.argmax(label, dim=1), output)

        
        batch_metric = {
            "acc" : acc,
            "loss" : loss.item()
        }
        
        return batch_metric 
    
    def valid_on_epoch(self, epoch, **cfg):
        self.base_model.eval()
        self.head_1.eval()
        self.head_2.eval()
        self.head_3.eval()
        self.head_4.eval()
        self.head_5.eval()
        self.head_6.eval()
        valid_acc, valid_loss = [], []
        tqdm_valid = tqdm(self.valid_loader)
        for img, label, first_group, second_group in tqdm_valid :
            batch_res = self.valid_on_batch(img, label, first_group, second_group, **cfg)
            
            valid_acc.append(batch_res["acc"])
            valid_loss.append(batch_res["loss"])
            tqdm_valid.set_postfix({
                "Epoch" : epoch,
                "Validation Acc" : np.mean(valid_acc),
                "Validation Loss" : np.mean(valid_loss),
            })
        
        return np.mean(valid_acc)
    
    def valid_on_batch(self, img, label, first_grouping, second_grouping, **cfg):
        img = img.to(cfg["device"])
        label = label.to(cfg["device"])    
        first_grouping = first_grouping.to(cfg["device"])
        second_grouping = second_grouping.to(cfg["device"])
    
        # output = self.model(img)
        # loss = self.criterion(output, label)
        
        fm = self.base_model(img)
        output_1 = self.head_1(fm, mask=second_grouping[:, :4])
        output_2 = self.head_2(fm, mask=second_grouping[:, 4:9])
        output_3 = self.head_3(fm, mask=second_grouping[:, 9:14])
        output_4 = self.head_4(fm, mask=second_grouping[:, 14:16])
        output_5 = self.head_5(fm, mask=second_grouping[:, 16:18])
        output_6 = self.head_6(fm, mask=second_grouping[:, 18])
        
        loss = self.criterion(output_1, label[:, :4]) + self.criterion(output_2, label[:, 4:9]) \
                + self.criterion(output_3, label[:, 9:14]) + self.criterion(output_4, label[:, 14:16]) \
                + self.criterion(output_5, label[:, 16:18]) + self.criterion(output_6, label[:, 18])
                
        
        output = torch.cat([output_1, output_2, output_3, output_4, output_5, output_6], dim=1)
        acc = score(torch.argmax(label, dim=1), output)
        batch_metric = {
            "acc" : acc,
            "loss" : loss.item()
        }
        
        return batch_metric
    
    def save_checkpoint(self, epoch, val_acc, **cfg) :
        if self.best_score < val_acc:
            self.best_score = val_acc
            torch.save({
                "epoch": epoch,
                "base_model_state_dict": self.base_model.state_dict(),
                "head_1_state_dict": self.head_1.state_dict(),
                "head_2_state_dict": self.head_2.state_dict(),
                "head_3_state_dict": self.head_3.state_dict(),
                "head_4_state_dict": self.head_4.state_dict(),
                "head_5_state_dict": self.head_5.state_dict(),
                "head_6_state_dict": self.head_6.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()
            }, os.path.join(cfg["save_path"], str(epoch) + 'E-val' + str(self.best_score) + '-' + cfg["model_name"] + '.pth'))
            self.early_stop_cnt = 0 
        else : 
            self.early_stop_cnt += 1
            
    def infer(self, **cfg) :
        self.prediction(**cfg)
        
    def prediction(self, **cfg) :
        self.pred_weight_load(cfg["weight_path"])
        self.base_model.eval()
        self.head_1.eval()
        self.head_2.eval()
        self.head_3.eval()
        self.head_4.eval()
        self.head_5.eval()
        self.head_6.eval()
             
        model_preds = []
        with torch.no_grad() :
            for img in tqdm(self.test_loader) :
                model_preds += self.predict_on_batch(img, **cfg)
        
        self.save_to_csv(model_preds, **cfg)
        
    def predict_on_batch(self, img, **cfg) :
        
        img = img.to(cfg["device"])
        
        fm = self.base_model(img)
        output_1 = self.head_1(fm, None)
        output_2 = self.head_2(fm, None)
        output_3 = self.head_3(fm, None)
        output_4 = self.head_4(fm, None)
        output_5 = self.head_5(fm, None)
        output_6 = self.head_6(fm, None)
        
        output = torch.cat([output_1, output_2, output_3, output_4, output_5, output_6], dim=1)
        
        # output = output * cfg["label_weight"]
        # output = output.detach().cpu() * np.array([[cfg["label_weight"]] * output.shape[0]])[0]

        # binary_ = torch.softmax(output, dim=1)
        # binary_[binary_  > 0.9] = 1 
        # binary_[binary_  <= 0.9] = 0 
        # return output.argmax(1).detach().cpu().numpy().tolist()
        
        return output.argmax(1).detach().cpu().numpy().tolist()
    
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
        
        "model_name" : "resnetv2_101x1_bit.goog_in21k", #"tf_efficientnetv2_m.in21k", #"swinv2_base_window12to16_192to256_22kft1k",
        #"tf_efficientnetv2_s.in21k",#"eva_large_patch14_196.in22k_ft_in1k",#"beit_base_patch16_224.in22k_ft_in22k",
        "num_classes" : 19,
        
        "learning_rate" : 1e-4,
        "focal_alpha" : 2,
        "focal_gamma" : 2,
        "resize" : 448,
        
        "data_path" : "./data/train",#"./data/more_first_aug_train", #"./data/combine_train",#"./data/train",#"./data/test",
        "epochs" : 80,
        "batch_size" : 16,
        "num_worker" : 1,
        "early_stop_patient" : 10,
        
        "binary_mode" : False,
        "reuse" : False, #True, #False
        "weight_path" : "./ckpt/resnetv2_101x1_bit.goog_in21k/multi_head_training_senario_3_SAM/17E-val0.8993648513045062-tf_efficientnetv2_s.in21k.pth",
        
        "save_path" : "./ckpt/resnetv2_101x1_bit.goog_in21k/multi_head_training_senario_3_SAM",
        "output_path" : "./output/resnetv2_101x1_bit.goog_in21k/multi_head_training_senario_3_SAM",
        
        "device" : "cuda",
        "label_name" : ["가구수정", "걸레받이수정", "곰팡이", "꼬임", "녹오염", "들뜸",
                        "면불량", "몰딩수정", "반점", "석고수정", "오염", "오타공", "울음",
                        "이음부불량", "창틀,문틀수정", "터짐", "틈새과다", "피스", "훼손",],
        
        "num_senario" : 2,
        "base_dim" : 2048,
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
    
    