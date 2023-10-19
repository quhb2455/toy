from typing import Any
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from scipy.spatial import distance
from datetime import datetime
from glob import glob
from tqdm import tqdm
import os
import json
import cv2
import random
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# CAM
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm

import torch
from torch.utils.tensorboard import SummaryWriter

def set_seed(seed) :
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def get_loss_weight(data_path):
    num_data_samples = []
    for p in sorted(glob(os.path.join(data_path, "*"))) :
        num_data_samples.append(len(os.listdir(p)))
    # return [1 - (x / sum(num_data_samples)) for x in num_data_samples]
    return [sum(num_data_samples) / (x * len(glob(os.path.join(data_path, "*")))) for x in num_data_samples]

def score(true_labels, model_preds, mode=None, binary=False) :
    model_preds = model_preds.argmax(1).detach().cpu().numpy().tolist()
    if binary :
        true_labels = true_labels.argmax(1).detach().cpu().numpy().tolist()
    else :
        true_labels = true_labels.detach().cpu().numpy().tolist()
    if mode == None :
        return f1_score(true_labels, model_preds, average='weighted')
    else :
        f1score = f1_score(true_labels, model_preds, average='weighted')
        # cls_report = confusion_matrix(true_labels, model_preds)
        return f1score, [true_labels, model_preds]
 
def distance_score(model_preds, mean_labels, true_labels, mode=None):
    preds = model_preds.detach().cpu().numpy().tolist()
    # mean_labels = mean_labels.detach().cpu().numpy().tolist()
    true_labels = true_labels.numpy().tolist()
    dist_label_list = []
    for p in preds :
        dist_label_list.append(np.argmin([distance.euclidean(p, m) for m in mean_labels]))
    if mode == None :
        return f1_score(true_labels, dist_label_list, average='weighted')
    else :
        f1score = f1_score(true_labels, dist_label_list, average='weighted')
        return f1score, [true_labels, dist_label_list]
 
def cal_cls_report(true_labels, model_preds) :
    rpt = classification_report(true_labels, model_preds, zero_division=0.0, output_dict=True)
    del rpt['accuracy']
    del rpt['macro avg']
    del rpt['weighted avg']
    return {str(k) : v['f1-score'] for k, v in rpt.items()}
    
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

def read_json(path):
    with open(path, "r") as j:
        meta = json.load(j)
    return meta

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

def Multi_cutmix(imgs, labels, upper, lower) :
    lam = np.random.beta(1.0, 1.0)
    rand_index = torch.randperm(imgs.size()[0]).cuda()
    target_a = labels
    target_b = labels[rand_index]
    upper_a = upper
    upper_b = upper[rand_index]
    lower_a = lower
    lower_b = lower[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
    imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))

    return imgs, lam, target_a, target_b, upper_a, upper_b, lower_a, lower_b

def mixup(imgs, labels):
    lam = np.random.beta(1.0, 1.0)
    rand_index = torch.randperm(imgs.size()[0]).cuda()
    mixed_imgs = lam * imgs + (1 - lam) * imgs[rand_index, :]
    target_a, target_b = labels, labels[rand_index]

    return mixed_imgs, lam, target_a, target_b

def LSBswap(imgs, k=25) : 
    maximum = imgs[0].shape[0] if imgs[0].shape[0] > imgs[0].shape[1] else imgs[0].shape[1]
    rand_location = np.random.randint(maximum, size=(k, 2))
    # rand_index = torch.randperm(imgs.size()[0]).cuda()
    rand_pick = np.random.choice(4, size=1, replace=False)
    
    for rand_pick in rand_location :
        x,y =  rand_pick
        
        for c in range(0, 3):
            bin_1 = bin(imgs[:, c, y, x])
            bin_2 = bin(imgs[rand_pick, c, y, x])
        
            tmp = bin_1[-2:]
            bin_1[-2:] = bin_2[-2:] 
            bin_2[-2:] = tmp
            
            imgs[:, c, y, x] = int(bin_1[-2:], 2)
            imgs[rand_pick, c, y, x] = int(bin_2[-2:], 2)
    return imgs

def returnCAM(feature_conv, weight_softmax, class_idx):
    # https://github.com/chaeyoung-lee/pytorch-CAM/blob/master/update.py
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def get_CAM(net, final_conv_name, img, label_name):
    transforms = A.Compose([
        A.Resize(300, 300),
        A.Normalize(),
        ToTensorV2()
    ])
    
    features_blobs = []

    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())

    net._modules.get(final_conv_name).register_forward_hook(hook_feature)
    
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
    
    img_tensor = transforms(image=img)["image"]
    img_variable = Variable(img_tensor.unsqueeze(0)).to("cuda")
    logit = net(img_variable.type('torch.cuda.FloatTensor'))
    
    classes = label_dec(label_name)
    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    # for i in range(0, 19):
    #     line = '{:.3f} -> {}'.format(probs[i], classes[idx[i].item()])
    #     print(line)

    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0].item()])
    
    # print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0].item()])
    # img = cv2.imread("./test.png")
    # img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    height, width, _ = img.shape

    CAM = cv2.resize(CAMs[0], (width, height))

    heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap * 0.3 + img * 0.5, classes[idx[0].item()]

def logging(path):
    os.makedirs(path, exist_ok=True)
    logger = SummaryWriter(path)
    return logger

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)