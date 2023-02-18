import cv2
import os
from glob import glob
import numpy as np
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, _df, labels, transform=None):
        self._df = _df
        self.labels = labels

        self.transform = transform

    def __len__(self):
        return len(self._df)

    def __getitem__(self, index):
        img_path = self._df['img_path'].iloc[index]
        
        # print(img_path)
        # image = cv2.imread(os.path.join('./data', *img_path.split('/')[2:]))
        image = cv2.imread(img_path)
        # h, w = image.shape[:2]
        # upper = int(h/2) - int(h * 0.2)
        # bottom = int(h/2) + int(h * 0.4)
        
        # image_2 = image[upper:bottom, :, :]
        
        if self.transform :
            image = self.transform(image=image)['image']
            #image_2 = self.transform(image=image_2)['image']
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.labels is not None:
            label = self.labels[index]
            return image, label

        else:
            return image#, image_2
    
def collate_fn(batch):
    img_stack = []
    label_stack = []

    for idx, (img, label) in enumerate(batch):
        if idx == 0 :
            img_stack = img
            label_stack.append(label)
        else :
            img_stack = torch.cat([img_stack, img], dim=0)

    return img_stack.unsqueeze(0), torch.tensor(label_stack, dtype=torch.long)

def divided_train_val(df, stack=False):
    if stack :
        # 원본 csv
        train_list = []
        valid_list = []
        _df = pd.read_csv('./data/train.csv')
        sampled = _df['sample_id'][np.random.choice(len(_df), int(len(_df) * 0.2))].values.tolist()
        for idx, si in enumerate(df['sample_id']) :
            if "_".join(si.split("_")[:2]) in sampled :
                valid_list.append(idx)
            else :
                train_list.append(idx)

        return df.iloc[train_list].reset_index(drop=True), \
            df.iloc[valid_list].reset_index(drop=True), \
            df['label'].iloc[train_list].tolist(), \
            df['label'].iloc[valid_list].tolist()
        

    else :
        train_df, val_df, train_labels, val_labels = train_test_split(
            df.drop(columns=['sample_id']),
            df['label'],
            test_size=0.2,
            random_state=2455
        )
        return train_df, val_df, train_labels.tolist(), val_labels.tolist()

def transform_parser(grid_shuffle_p=0.8, data_type='train') :
    if data_type == 'train' :
        return A.Compose([
            
            # ego+crash mosaic 핛브용
            A.Resize(512, 512),
            A.OneOf([
                A.CLAHE(p=0.5),
                A.ImageCompression(p=0.5),
                A.RandomBrightnessContrast(p=0.5)
            ],p=1),
            

            # weather 학습 용
            # A.Resize(384, 384),
            # A.OneOf([
            #     A.Blur(blur_limit=(3, 3)),
            # ], p=1),
            # A.Spatter(p=0.5, mode=['rain', 'mude']),
            # A.RandomGridShuffle(p=0.6, grid=(7, 7)),

            A.Normalize(),
            ToTensorV2()
        ])
    elif data_type == 'valid' :
        return A.Compose([
            A.Resize(512, 512),
            # A.Rotate(limit=(45), p=1),
            # A.RandomGridShuffle(p=grid_shuffle_p, grid=(2,2)),
            A.Normalize(),
            ToTensorV2()
        ])


def img_parser(data_path, div, training=True):
    path = sorted(glob(data_path), key=lambda x: int(x.split('\\')[-1].split('.')[0]))

    if training:
        return path[:div], path[div:]
    else:
        return path


def image_label_dataset(df_path, img_path, div=0.8, grid_shuffle_p=0.8, training=True):
    all_df = pd.read_csv(df_path)
    transform = transform_parser(grid_shuffle_p=grid_shuffle_p, data_type='valid')

    if training:
        train_df = all_df.iloc[:int(len(all_df) * div)]
        val_df = all_df.iloc[int(len(all_df) * div):]

        train_img, valid_img = img_parser(img_path, int(len(all_df) * div), training=training)
        return (train_img, valid_img), (train_df['label'].values, val_df['label'].values), transform

    else:
        img = img_parser(img_path, div=None, training=training)
        return img, all_df, transform


def custom_dataload(df_set, label_set, batch_size, data_type, shuffle, stack) :
    transform = transform_parser(data_type=data_type)
    ds = CustomDataset(df_set, label_set, transform)
    if stack :
        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=6)
    else :
        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=6)
    return dl


def train_and_valid_dataload(df_set, label_set, batch_size=16, shuffle=True, stack=False) :
    train_loader = custom_dataload(df_set[0], label_set[0], batch_size, 'train', shuffle, stack)
    val_loader = custom_dataload(df_set[1], label_set[1], batch_size, 'valid', False, stack)
    return train_loader, val_loader
