import cv2
import os
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
# from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

def collate_fn(batch):
    img_stack = []
    label_stack = []

    for idx, (img, label) in enumerate(batch):
        if idx == 0 :
            img_stack = img
            label_stack.append(label)
        else :
            img_stack = torch.cat([img_stack, img], dim=0)

    return img_stack.unsqueeze(0), torch.Tensor(label_stack)


class _ChannelStackDataset(Dataset):
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
        
        if self.transform :
            image = self.transform(image=image)['image']
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.labels is not None:
            label = self.labels[index]
            return image, label

        else:
            return image

if __name__ == '__main__':
    csv_path = "./data/50frame_train.csv"

    df = pd.read_csv(csv_path)
    label_set = df['label']
    transform = A.Compose([
                    A.Resize(224, 224),
                    # A.Rotate(limit=(45), p=1),
                    # A.RandomGridShuffle(p=grid_shuffle_p, grid=(2,2)),
                    A.Normalize(),
                    ToTensorV2()
                ])
    ds = _ChannelStackDataset(df, label_set, transform)
    dl23 = DataLoader(ds, batch_size=50, shuffle=False, collate_fn=collate_fn, num_workers=6)

    for img, label in tqdm(dl23) :
        print(img.shape)
        print(label.shape)
        break