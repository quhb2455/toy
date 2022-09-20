import cv2
from glob import glob
import numpy as np
import pandas as pd

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, img_list, label_list=None, transforms=None):
        self.img_list = img_list
        self.label_list = label_list
        self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            img = self.transforms(image=img)['image']

        # training
        if self.label_list is not None:
            label = self.label_list[idx]
            return img, torch.tensor(label)

        # test
        else:
            return img


def transform_parser(grid_shuffle_p=0.8) :
    return A.Compose([
        A.Rotate(limit=(45), p=1),
        A.RandomGridShuffle(p=grid_shuffle_p, grid=(2,2)),
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
    transform = transform_parser(grid_shuffle_p=grid_shuffle_p)

    if training:
        train_df = all_df.iloc[:int(len(all_df) * div)]
        val_df = all_df.iloc[int(len(all_df) * div):]

        train_img, valid_img = img_parser(img_path, int(len(all_df) * div), training=training)
        return (train_img, valid_img), (train_df['label'].values, val_df['label'].values), transform

    else:
        img = img_parser(img_path, div=None, training=training)
        return img, all_df, transform


def custom_dataload(img_set, label_set, batch_size, transform, shuffle) :
    ds = CustomDataset(img_set, label_set, transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    return dl


def train_and_valid_dataload(img_set, label_set, transform, batch_size=16) :
    train_loader = custom_dataload(img_set[0], label_set[0], batch_size, transform, True)
    val_loader = custom_dataload(img_set[1], label_set[1], batch_size, transform, False)
    return train_loader, val_loader
