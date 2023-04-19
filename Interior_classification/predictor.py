
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
        self.weight_load(cfg["weight_path"])
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

        img_name_list = [os.path.splitext(os.path.basename(p))[0] for p in glob(os.path.join(cfg["data_path"], "*"))]
        res_label_list = [_label_dec[i] for i in results]        
        
        df = pd.DataFrame({"id" : img_name_list, "label":res_label_list})
        df.to_csv(cfg["output_path"], index=False)
    
    def weight_load(self, weight_path) :
        checkpoint = torch.load(weight_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])