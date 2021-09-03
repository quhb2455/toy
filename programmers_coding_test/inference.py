import torch
from torch.utils.data import DataLoader
from utils import TestDataset

import albumentations as A

import cv2
import os
import numpy as np
from tqdm import tqdm

from easydict import EasyDict
import pandas as pd

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

        model.eval()
        preds = model(test_img)

        answers = preds.argmax(dim=-1)
        print(answers)
        results.extend(answers.cpu().numpy())
    print(results)

    submission_result = {img: answer for img, answer in zip(img_list, results)}

    submission_path = os.path.join(args.export, 'submission_really.csv')
    submission_df = pd.DataFrame({'answer_value': results})
    submission_df.to_csv(submission_path)



def main(args) :

    model = torch.load(args.model_name)

    class_path = os.listdir(args.test_path)[0]
    img_path = os.path.join(args.test_path, class_path)
    img_list = [ os.path.join(img_path, img) for img in os.listdir(img_path)]

    transforms = A.Compose([
        A.Resize(224, 224),
        A.Normalize()
    ])

    inference(model, img_list ,transforms, args)





if __name__ == "__main__" :
    config ={
        "model_name" : "./data/model/0.9522004450575878_9E_best_model.pt",
        'export' : './result',
        'batch_size' : 16,
        'test_path' : './data/test'
    }
    args = EasyDict(config)
    main(args)