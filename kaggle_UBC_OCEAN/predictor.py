
from utils import score, label_dec

import torch
import numpy as np
import os
from tqdm import tqdm
from glob import glob

import pandas as pd

class Predictor() :
    def __init__(self) -> None:
        pass
        
    def prediction(self, **cfg) :
        self.pred_weight_load(cfg["weight_path"])
        self.model.eval()        
        model_preds = []
        with torch.no_grad() :
            for img in tqdm(self.test_loader) :
                model_preds += self.predict_on_batch(img, **cfg)
        
        self.save_to_csv(model_preds, **cfg)
        
    
    def predict_on_batch(self, img, **cfg) :
        img = img.to(cfg["device"])
        return self.model(img).argmax(1).detach().cpu().numpy().tolist()
        
    def save_to_csv(self, results, **cfg) :
        _label_dec = label_dec(cfg["label_name"])

        img_name_list = [os.path.basename(p).split("_")[0] for p in glob(os.path.join(cfg["data_path"], "*"))]
        res_label_list = [_label_dec[i] for i in results]        
        
        df = pd.DataFrame({"image_id" : img_name_list, "label":res_label_list})
        df.to_csv(os.path.join(cfg["output_path"], "submission.csv"), index=False)
    
    def pred_weight_load(self, weight_path) :
        checkpoint = torch.load(weight_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    # def pred_weight_load(self, weight_path) :
    #     checkpoint = torch.load(weight_path)
    #     # self.model.load_state_dict(checkpoint['model_state_dict'])
    #     self.base_model.load_state_dict(checkpoint["base_model_state_dict"]),
    #     self.head_1.load_state_dict(checkpoint["head_1_state_dict"]),
    #     self.head_2.load_state_dict(checkpoint["head_2_state_dict"]),
    #     self.head_3.load_state_dict(checkpoint["head_3_state_dict"]),
    #     self.head_4.load_state_dict(checkpoint["head_4_state_dict"]),
    #     self.head_5.load_state_dict(checkpoint["head_5_state_dict"]),
    #     self.head_6.load_state_dict(checkpoint["head_6_state_dict"]),