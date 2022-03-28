import torch
from torch.utils.data import Dataset

import albumentations as A
from sklearn.model_selection import train_test_split, StratifiedKFold

import cv2
import os
import numpy as np


class DataParser() :
    def __init__(self, data_path, class_encoder, random_seed):
        self.img_list, self.label_list, self.class_count = self.get_data(data_path, class_encoder)
        self.weights = self.cal_weight(self.class_count)
        self.random_seed = random_seed


    def cal_weight(self, class_count):
        return torch.tensor(np.max(class_count) / class_count)

    def get_fold_data(self, fold_datalist):
        imgs = [self.img_list[i] for i in fold_datalist]
        labels = [self.label_list[i] for i in fold_datalist]

        return imgs, labels

    def get_data(self, data_path, class_encoder):
        classes = os.listdir(data_path)
        class_path_list = [os.path.join(data_path, class_name) for class_name in classes]

        class_count = []
        img_list = []
        label_list = []
        for class_path in class_path_list:
            data_list = list(map(lambda x: os.path.join(class_path, x), os.listdir(class_path)))
            labels = [class_encoder[class_path.split('\\')[-1]]] * len(data_list)
            class_count.append(len(data_list))

            img_list.extend(data_list)
            label_list.extend(labels)

        return img_list, label_list, class_count



    def get_transforms(self):
        train_transfors = A.Compose([
            A.Resize(224,224),
            A.Rotate(),
            A.HorizontalFlip(),
            A.ColorJitter(),
            A.Normalize()
        ])

        valid_transforms = A.Compose([
            A.Resize(224, 224),
            A.Normalize()
        ])

        return train_transfors, valid_transforms


    def DatasetParsing(self, fold_train=None, fold_valid=None):
        if fold_train is None :
            train_imgs, valid_imgs, train_labels, valid_labels = train_test_split(self.img_list,
                                                                                  self.label_list,
                                                                                  train_size=0.8,
                                                                                  shuffle=True,
                                                                                  random_state=self.random_seed,
                                                                                  stratify=self.label_list)
        else :
            train_imgs, train_labels = self.get_fold_data(fold_train)
            valid_imgs, valid_labels = self.get_fold_data(fold_valid)

        train_transforms, valid_transforms = self.get_transforms()

        train_dataset = TrainDataset(img_list=train_imgs, label_list=train_labels, transforms=train_transforms)
        valid_dataset = TrainDataset(img_list=valid_imgs, label_list=valid_labels, transforms=valid_transforms)

        return train_dataset, valid_dataset


    def Test_DatasetParsing(self):
        _, test_transforms = self.get_transforms()
        test_dataset = TestDataset(img_list=self.img_list, transforms=test_transforms)

        return test_dataset



class TrainDataset(Dataset):
    def __init__(self, img_list, label_list, transforms=None):
        self.img_list = img_list
        self.label_list = label_list
        self.transforms = transforms

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms :
            img = self.transforms(image=img)["image"]

        img = img.transpose(2, 0, 1)
        label = self.label_list[idx]

        img = torch.tensor(img, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)

        return img, label

    def __len__(self):
        assert len(self.img_list) == len(self.label_list)
        return len(self.img_list)



class TestDataset(Dataset) :
    def __init__(self, img_list, transforms=None):
        self.img_list = img_list
        self.transforms = transforms

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms :
            img = self.transforms(image=img)["image"]

        img = img.transpose(2, 0, 1)

        img = torch.tensor(img, dtype=torch.float)

        return img

    def __len__(self):
        return len(self.img_list)