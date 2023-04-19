from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime

import os
import json
import cv2
import numpy as np

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