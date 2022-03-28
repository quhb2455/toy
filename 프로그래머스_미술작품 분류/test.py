import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import numpy as np
from easydict import EasyDict
import timm
import random
from utils import DataParser, TrainDataset

class_encoder = {
    "dog": 0,
    "elephant": 1,
    "giraffe": 2,
    "guitar": 3,
    "horse": 4,
    "house": 5,
    "person": 6
}

kfold = StratifiedKFold(n_splits=6, random_state=11, shuffle=True)
# x = [0,1,2,3,4,5]
# y = [0,0,0,1,1,2]
#
#
# for k, (fold_train, fold_valid) in enumerate(kfold.split(x, y), 1):
#     print(fold_train)
#     print(fold_valid)
Datasets = DataParser('./data/train', class_encoder, 11)
# a = []
# b = []
#
print(Datasets.img_list)
print(Datasets.label_list)
for k, (fold_train, fold_valid) in enumerate(kfold.split(Datasets.img_list, Datasets.label_list), 1):
    print(len(fold_train))
    print(len(fold_valid))
