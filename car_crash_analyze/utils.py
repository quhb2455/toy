import os
from datetime import datetime
import json
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
from tqdm import tqdm

import torch.nn.functional as F
import torch

def weight_freeze(model) :
    for i, child in enumerate(model.children()) :
        for n, p in child.named_modules() :
            if n != 'classifier' :
                for param in p.parameters():
                    param.requires_grad = False
            elif n == 'classifier' :
                for param in p.parameters():
                    param.requires_grad = True
    return model


def weight_load(model, optimizer, ckpt, training=True):
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])

    if training :
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer, checkpoint['epoch']

    else :
        return model

def get_models(model, checkpoint):
    models = []
    for path in checkpoint:
        models.append(weight_load(model, None, path, training=False))
        print(f"MODEL LOAD ... from {path}")

    if len(checkpoint) == 1:
        return models[0]
    else:
        return models

def get_each_frame_infer_result(pred_value, pred_index) :
    df = pd.DataFrame({'pred_value' : pred_value, 'pred_index' : pred_index})
    pred_list = [0 for i in range(1800)]
    cnt = 0
    for j in tqdm(range(0, 90000, 50)) :    
        best_score = 0
        best_cls = 0
        for i in range(13) :
            mean_score = sum(df['pred_value'][j : j + 49][df['pred_index'] == i].values)/(len(df['pred_value'][j : j + 50][df['pred_index'] == i].values.tolist()) + +0.000001)
            if mean_score >= best_score  :
                best_score = mean_score
                best_cls = i

        pred_list[cnt] = best_cls
        cnt += 1
    return pred_list


def save_to_csv(df, preds, save_path):
    df['label'] = preds
    df.to_csv(save_path, index=False)

def save_config(cfg, output) :
    cfg_save_name = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    with open(os.path.join(output, f"{cfg_save_name}.json"), 'w') as f:
        json.dump(cfg, f, indent="\t")

def read_csv(path) :
    return pd.read_csv(path)

def logging(logger, data, step) :
    for i, (k,v) in enumerate(data.items()) :
        logger.add_scalar(k, v, step)

def batch_score(true_labels, model_preds, threshold) :
    # model_preds = model_preds.argmax(1).detach().cpu().numpy().tolist()
    t = model_preds.shape[0]
    model_preds = sigmoid2binary(torch.sigmoid(model_preds.detach().cpu()), threshold)
    true_labels = true_labels.detach().cpu().numpy().tolist() * t
    # return accuracy_score(true_labels, model_preds)
    return f1_score(true_labels, model_preds, average='macro')

def new_batch_score(true_labels, model_preds, threshold) :
    # model_preds = model_preds.argmax(1).detach().cpu().numpy().tolist()
    true_labels = true_labels.detach().cpu().numpy().tolist()# * model_preds.shape[0]
    model_preds = sigmoid2binary(torch.sigmoid(model_preds.detach().cpu()), threshold, mode='tensor')

    if model_preds[model_preds == 1].shape[0] > model_preds[model_preds == 0].shape[0] :
        model_pred = 1
    else :
        model_pred = 0

    if model_pred == true_labels[0] :
         return 1
    else :
        return 0
    # return accuracy_score(true_labels, model_preds)
    # return f1_score(true_labels, model_preds, average='macro')

def score(true_labels, model_preds, threshold=None) :
    # timnming
    # model_preds = sigmoid2binary(torch.sigmoid(model_preds.detach().cpu()), threshold)
    # print("model_preds : ",model_preds)

    # model_preds = torch.sigmoid(model_preds.detach().cpu())
    # weather
    model_preds = model_preds.argmax(1).detach().cpu().numpy().tolist()
    
    true_labels = true_labels.detach().cpu().numpy().tolist()

    # print("true_labels : ",true_labels)
    # print("model_preds : ",model_preds)
    return f1_score(true_labels, model_preds, average='macro')#1 - abs(true_labels[0] - model_preds[0])
    # if model_preds == true_labels :
    #     return 1
    # else :
    #     return 0

def sigmoid2binary(sigmoid_preds, threshold, mode=None) :
    sigmoid_preds[sigmoid_preds > threshold] = 1
    sigmoid_preds[sigmoid_preds <= threshold] = 0
    if mode == None :
        return sigmoid_preds.detach().cpu().numpy().tolist()
    if mode == 'tensor' :
        return sigmoid_preds.detach().cpu()

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

