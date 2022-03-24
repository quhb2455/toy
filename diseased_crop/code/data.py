import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import json
import cv2

from glob import glob
import os

class CustomDataset(Dataset):
    def __init__(self,
                 files,
                 transforms,
                 label_decoder,
                 opt,
                 mode='train'):

        if opt.use_kfold:
            self.files = self.kfold_files(files, opt)
        else:
            self.files = files

        self.mode = mode
        self.label_decoder = label_decoder  # label_encoder
        self.csv_feature_dict = opt.csv_feature_dict
        self.max_len = opt.max_len
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        file = self.files[i]

        # CSV
        csv_data, seq_len = self.csv_preprocessing(file)

        # image
        img = self.img_preprocessing(file)

        if self.mode == 'train':
            # Label
            label = self.label_preprocessing(file)

            return {
                'img': torch.tensor(img, dtype=torch.float32),
                'label': torch.tensor(self.label_decoder[label], dtype=torch.long),
                'csv_feature': torch.tensor(csv_data, dtype=torch.float32),
                'seq_len': seq_len
            }

        else:
            return {
                'img': torch.tensor(img, dtype=torch.float32),
                'csv_feature': torch.tensor(csv_data, dtype=torch.float32),
                'seq_len': seq_len
            }

    def kfold_files(self, data_index, opt):
        file_list = glob(os.path.join(opt.dataset_path, "*/*.jpg"))
        return [file_list[idx] for idx in data_index]

    def csv_preprocessing(self, file):
        # CSV
        csv_path = file.replace("jpg", "csv")
        df = pd.read_csv(csv_path)[self.csv_feature_dict.keys()]
        df = df.replace('-', 0)

        # MinMax scaling
        for col in df.columns:
            df[col] = df[col].astype(float) - self.csv_feature_dict[col][0]
            df[col] = df[col] / (self.csv_feature_dict[col][1] - self.csv_feature_dict[col][0])

        # pack_padded_sequence 하기 위한 len 추가
        seq_len = len(df)

        df_np = df.to_numpy()
        df_len, df_features = df_np.shape

        csv_data = np.zeros([self.max_len, df_features])
        csv_data[0:df_len, :] = df_np

        return csv_data, seq_len

    def img_preprocessing(self, file):
        image_path = file
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(image=img)["image"]
        img = img.transpose(2, 0, 1)

        return img

    def label_preprocessing(self, file):
        json_path = file.replace("jpg", "json")
        with open(json_path, 'r') as f:
            json_file = json.load(f)

        crop = json_file['annotations']['crop']
        disease = json_file['annotations']['disease']
        risk = json_file['annotations']['risk']

        return f'{crop}_{disease}_{risk}'