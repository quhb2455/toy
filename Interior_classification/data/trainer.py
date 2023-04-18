
from utils import score

import torch
import numpy as np
import os
from tqdm import tqdm

class Trainer() :
    def __init__(self) -> None:
        # self.train_loader = None
        # self.valid_loader = None
        # self.criterion = None
        # self.optimizer = None
        # self.model = None
        
        self.best_score = 0
        self.early_stop_cnt = 0
        
    def run(self, **cfg) :
        for e in cfg["epochs"] :
            self.train_on_epoch(e, **cfg)
            valid_acc = self.valid_on_epoch(e, **cfg)
            
            self.save_checkpoint(e, valid_acc)
            
            if self.early_stop_cnt == cfg["early_stop_patient"] :
                print("=== EARLY STOP ===")
                break
            
    def train_on_epoch(self, epoch, **cfg):
        train_acc, train_loss = [], []
        tqdm_train = tqdm(self.train_loader)
        for img, label in tqdm_train :
            batch_res = self.train_on_batch(img, label, **cfg)
            
            tqdm_train.set_postfix({
                "Epoch" : epoch,
                "Training Acc" : np.mean(train_acc.append(batch_res["acc"])),
                "Training Loss" : np.mean(train_loss.append(batch_res["loss"])),
            })
                            
    def train_on_batch(self, img, label, **cfg) :
        self.optimizer.zero_grad()

        img = img.to(cfg["device"])
        label = label.to(dfg["device"])
        
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
    
    def valid_on_epoch(self, epoch, **cfg):
        valid_acc, valid_loss = [], []
        tqdm_valid = tqdm(self.valid_loader)
        for img, label in tqdm_valid :
            batch_res = self.valid_on_batch(img, label, **cfg)
            
            tqdm_valid.set_postfix({
                "Epoch" : epoch,
                "Validation Acc" : np.mean(valid_acc.append(batch_res["acc"])),
                "Validation Loss" : np.mean(valid_loss.append(batch_res["loss"])),
            })
        
        return np.mean(valid_acc)
    
    def valid_on_batch(self, img, label, **cfg):
        img = img.to(cfg["device"])
        label = label.to(dfg["device"])
        
        output = self.model(img)
        loss = self.criterion(output, label)
        
        acc = score(label, output)
        
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
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()
            }, os.path.join(cfg["save_path"], str(epoch) + 'E-val' + str(self.best_score) + '-' + cfg["model_name"] + '.pth'))
            self.early_stop_cnt = 0
        else :
            self.early_stop_cnt += 1