import h5py
import random
import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import gc
from sklearn.decomposition import PCA

import argparse


def img_save(point, path, size=50):
    fig = plt.figure(figsize=(2, 2))

    x = point[:, 0]
    y = point[:, 1]

    plt.axis('off')

    plt.scatter(x, y, s=0.1)

    plt.savefig(path, dpi=size)
    plt.close(fig)

def converting(args) :
    all_points = h5py.File(args.data, 'r')
    pca = PCA(n_components=2)

    if args.mode == 'train' :
        for i in tqdm(range(0, 50000)):
            point = np.array(all_points[str(i)])
            pt_pca = pca.fit_transform(point)
            img_save(pt_pca, os.path.join(args.output,f'{str(i)}.jpg'), size=112)

    else :
        for i  in tqdm(range(50000, 90000)) :
            point = np.array(all_points[str(i)])
            pt_pca = pca.fit_transform(point)
            img_save(pt_pca, os.path.join(args.output,f'{str(i)}.jpg'), size=112)

if __name__ == '__main__' :
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--data", type=str, default='./data/train.h5')
    parser.add_argument("--output", type=str, default='./data/train_img/')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    converting(args)
