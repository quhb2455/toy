import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from utils import TestDataset

import albumentations as A

import cv2
import os
import numpy as np
from tqdm import tqdm

from easydict import EasyDict
import pandas as pd


def get_models(model_list) :
    return [torch.load(model).eval() for model in model_list]


def softvoting(models, img) :

    predicts = torch.zeros(img.size(0), 7)
    for model in models :
        output = model(img)
        output = F.softmax(output.cpu(), dim=1)
        predicts += output

    # 둘다 값은 똑같이 나옴.
    # pred_avg = predicts / len(models)
    # answer = pred_avg.argmax(dim=-1)
    # _, answer2 = torch.max(pred_avg, 1)

    return predicts / len(models)


def Test_DatasetParsing(img_list, transforms):
    test_dataset = TestDataset(img_list=img_list, transforms=transforms)
    return test_dataset


def inference(model, img_list ,transforms, args) :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = Test_DatasetParsing(img_list, transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    results = []

    for test_img in tqdm(test_loader):
        test_img = test_img.to(device=device)

        if args.softvoting :
            preds = softvoting(model, test_img)

        else :
            preds = model(test_img)

        answers = preds.argmax(dim=-1)
        results.extend(answers.cpu().numpy())

    # submission_result = {img: answer for img, answer in zip(img_list, results)}

    submission_path = os.path.join(args.export, 'submission_really.csv')
    submission_df = pd.DataFrame({'answer_value': results})
    submission_df.to_csv(submission_path)


def main(args) :

    class_path = os.listdir(args.test_path)[0]
    img_path = os.path.join(args.test_path, class_path)
    img_list = [ os.path.join(img_path, img) for img in os.listdir(img_path)]

    transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize()
    ])


    if args.softvoting :
        model = get_models(args.models)

    else:
        model = torch.load(args.model_name).eval()


    inference(model, img_list, transforms, args)



if __name__ == "__main__" :
    config ={
        "model_name" : "./data/model/0.9522004450575878_9E_best_model.pt",
        "models" : ["Kfold_0_0.9617_12E_best_model.pt",
                    "Kfold_2_0.9683_11E_best_model.pt", "Kfold_3_0.9509_11E_best_model.pt"],
        'export' : './result',
        'batch_size' : 4,
        'test_path' : './data/test',
        'softvoting' : True
    }

    args = EasyDict(config)

    main(args)