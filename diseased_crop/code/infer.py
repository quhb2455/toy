import os
from glob import glob
from tqdm import tqdm
import numpy as np
import time

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import albumentations as A

from model import CNN2RNN
from utils import label_preprocessing, csv_feature_dict, data_split, transform, softvoting, hardvoting
from data import CustomDataset
from options import options

def submission(preds, label_encoder, opt) :
    preds_answ = np.array([label_encoder[int(val)] for val in preds])
    submission_csv = pd.read_csv(opt.csv)
    submission_csv['label'] = preds_answ
    submission_csv.to_csv(os.path.join(opt.save_path, f'{int(time.time())}.csv'), index=False)

def model_loads(path, opt) :
    model = CNN2RNN(opt).to(opt.device)
    model.load_state_dict(torch.load(path, map_location=opt.device))
    return model.to(opt.device).eval()

def predict(test_loader, models, opt):
    tqdm_dataset = tqdm(test_loader)
    results = []
    for batch, batch_item in enumerate(tqdm_dataset):

        if opt.voting == "soft":
            predictions = softvoting(models, batch_item, opt)

        elif opt.voting == "hard":
            predictions = hardvoting(models, batch_item, opt)

        else:
            # single model
            img = batch_item['img'].to(opt.device)
            csv_feature = batch_item['csv_feature'].to(opt.device)
            seq_len = batch_item['seq_len'].to(opt.device)

            predictions = models[0](img, csv_feature, seq_len)
            predictions = F.softmax(predictions.cpu(), dim=1)

        batch_result = [int(torch.argmax(prediction)) for prediction in predictions]
        results.extend(batch_result)

    return results

if __name__ == "__main__" :

    opt, label_options = options()
    opt.csv_feature_dict = csv_feature_dict(opt.dataset_path, label_options.csv_features)

    # label_enc, dec 및 trasnforms 설정
    label_encoder, label_decoder = label_preprocessing(opt.label_path)
    _, valid_transforms = transform()

    test = data_split(opt.dataset_path, label_decoder, mode="test")
    test_dataset = CustomDataset(test, valid_transforms, label_decoder, opt, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

    # option 출력
    print("\n<< option >>")
    print(*["{} : {}".format(k, v) for k, v in vars(opt).items()], sep='\n')

    models_path = glob(opt.model_path)

    models = []
    if len(models_path) != 1:
        for idx, kfold_model_path in enumerate(models_path):
            models.append(model_loads(kfold_model_path, opt))

    elif len(models_path) == 1:
        models.append(model_loads(models_path[0], opt))

    preds = predict(test_loader, models, opt)

    print(preds)
    submission(preds, label_encoder, opt)