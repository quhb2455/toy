from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime

import os
import json
import cv2
import numpy as np
import torch

def score(true_labels, model_preds, threshold=None) :
    model_preds = model_preds.argmax(1).detach().cpu().numpy().tolist()
    true_labels = true_labels.detach().cpu().numpy().tolist()
    return f1_score(true_labels, model_preds, average='macro')

def save_config(config, save_path, save_name="") :
    os.makedirs(save_path, exist_ok=True)
    cfg_save_time = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    with open(os.path.join(save_path, f"{save_name}_{cfg_save_time}.json"), 'w') as f:
        json.dump(config, f, indent="\t")

def save_img(path, img, extension=".png") :
    result, encoded_img = cv2.imencode(extension, img)
    if result:
        with open(path, mode='w+b') as f:
            encoded_img.tofile(f)

def load_img(path) :
    img_array = np.fromfile(path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img

def label_enc(label_name) : 
    return {n:idx for idx, n in enumerate(label_name)}

def label_dec(label_name) : 
    return {idx:n for idx, n in enumerate(label_name)}

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix(imgs, labels):
    lam = np.random.beta(1.0, 1.0)
    rand_index = torch.randperm(imgs.size()[0]).cuda()
    target_a = labels
    target_b = labels[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
    imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))

    return imgs, lam, target_a, target_b

def mixup(imgs, labels):
    lam = np.random.beta(1.0, 1.0)
    rand_index = torch.randperm(imgs.size()[0]).cuda()
    mixed_imgs = lam * imgs + (1 - lam) * imgs[rand_index, :]
    target_a, target_b = labels, labels[rand_index]

    return mixed_imgs, lam, target_a, target_b