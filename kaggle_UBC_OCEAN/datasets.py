from utils import read_json, save_config, load_img, label_enc
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from glob import glob
import pandas as pd
import os
import cv2
import json
   

class CustomDataset(Dataset):
    def __init__(self, imgs, labels, transform=None, binary_mode=False):
        self.imgs = imgs
        self.labels = labels
        self.label_enc = label_enc(sorted(set(labels))) if labels != None else None
        self.transform = transform
        self.binary_mode = binary_mode
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        
        image = load_img(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform :
            image = self.transform(image=image)['image']

        if self.labels is not None:
            if self.binary_mode :
                label = torch.tensor(self.binary_encoder(self.label_enc[self.labels[index]]), dtype=torch.long)
            else :
                label = self.label_enc[self.labels[index]]
            
            return image, label

        else:
            return image
    
    def binary_encoder(self, label):
        total_label_len = len(self.label_enc.keys())
        return [1 if i == label else 0 for i in range(total_label_len) ]
    
class DatasetCreater() :
    def __init__(self) :
        self.base_filename = "_thumbnail.png"
    
    def create_dataset(self, transform, **cfg) :
        img_path, label_list = self.get_data(**cfg)        
        
        if cfg["mode"] == "train" :
            save_config(transform[0].to_dict(), cfg["save_path"], save_name="train_transform")
            save_config(transform[1].to_dict(), cfg["save_path"], save_name="valid_transform")
            
            return [CustomDataset(img_path[0], label_list[0], transform=transform[0], binary_mode=cfg["binary_mode"]), 
                    CustomDataset(img_path[1], label_list[1], transform=transform[1], binary_mode=cfg["binary_mode"])]
            
        elif cfg["mode"] == 'infer' :
            save_config(transform.to_dict(), cfg["output_path"], save_name="infer_transform")
            return CustomDataset(img_path, label_list, transform=transform)
    
    
    def create_dataloader(self, transform, **cfg) :
        ds = self.create_dataset(transform, **cfg)
        
        if isinstance(ds, list) :
            return (DataLoader(ds[0], batch_size=cfg["batch_size"], shuffle=cfg["shuffle"], num_workers=cfg["num_worker"]), 
                    DataLoader(ds[1], batch_size=cfg["batch_size"], shuffle=cfg["shuffle"], num_workers=cfg["num_worker"]))
        else :
            return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=cfg["shuffle"], num_workers=cfg["num_worker"])
        
        
    def get_data(self, **cfg) :
        data_path = cfg["data_path"]
        csv_file = pd.read_csv(cfg["csv_path"])
        mode = cfg["mode"]
        
        img_path_list = []
        label_list = []
        
        if mode == "infer" :
            for df in csv_file.iloc :
                img_path_list.append(os.path.join(data_path, str(df['image_id'])+self.base_filename))
                # label_list.append(df['label'])
            return img_path_list, None
        
        else :
            for df in csv_file.iloc :
                if df['is_tma'] :
                    continue
                    # img_path_list.append(os.path.join(data_path, "train_images", str(df['image_id'])+".png"))
                else :
                    img_path_list.append(os.path.join(data_path, str(df['image_id'])+self.base_filename))
                label_list.append(df['label'])
    
            train_img, valid_img, train_label, valid_label = train_test_split(img_path_list, 
                                                                              label_list, 
                                                                              test_size=0.1, 
                                                                              stratify=label_list, 
                                                                              random_state=2455)
            return [train_img, valid_img], [train_label, valid_label]
    
