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
    def __init__(self, medical_df, labels):
        self.medical_df = medical_df
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.medical_df)

    def __getitem__(self, index):
        img_path = self.medical_df['img_path'].iloc[index]

        image = cv2.imread(os.path.join('./data', *img_path.split('/')[1:]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.labels is not None:
            label = self.labels[index]
            return image, label

        else:
            return image

def collate_fn(batch):
    img_stack = []
    label_stack = []
    _transforms = A.Compose([
        A.Resize(384, 384),
        # A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False,
                    p=1.0),
    ])

    for img, label in batch:
        # img 나누기
        w_ratio = int(img.shape[1] / 16)
        h_ratio = int(img.shape[0] / 8) # 16

        label_stack.append(label)
        for w in range(16):
            for h in range(8): # 16
                cropped_img = _transforms(image=img[h * h_ratio: (h + 1) * h_ratio, w * w_ratio: (w + 1) * w_ratio])['image'].transpose(2, 0, 1)
                img_stack.append(cropped_img.tolist())

    return torch.Tensor(img_stack), torch.Tensor(label_stack)

def divided_train_val(df):
    train_df, val_df, train_labels, val_labels = train_test_split(
        df.drop(columns=['N_category']),
        df['N_category'],
        test_size=0.2,
        random_state=2455
    )
    return train_df, val_df, train_labels, val_labels

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


def custom_dataload(df_set, label_set, batch_size, shuffle) :
    ds = CustomDataset(df_set, label_set)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=6)
    return dl


def train_and_valid_dataload(df_set, label_set, batch_size=16) :
    train_loader = custom_dataload(df_set[0], label_set[0], batch_size, True)
    val_loader = custom_dataload(df_set[1], label_set[1], batch_size, False)
    return train_loader, val_loader
