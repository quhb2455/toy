import os
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import albumentations as A

from model import CNN
from utils import VillageData_split, VillageLabel_preprocessing, transform, rand_bbox
from data import VillageDataset
from options import options


def accuracy_function(real, pred):
    real = real.cpu()
    pred = torch.argmax(pred, dim=1).cpu()
    score = f1_score(real, pred, average='macro')
    return score


def run(train_loader, valid_loader, opt):

    model = CNN(opt.model_name, opt.num_classes).to(opt.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate)
    criterion = nn.CrossEntropyLoss()

    early_stopping_step = 0
    best_loss = 10
    for epoch in range(opt.epochs):

        # training
        tqdm_train = tqdm(train_loader)
        train_loss, train_macro_f1 = 0, 0
        for batch, batch_item in enumerate(tqdm_train):
            model.train()
            img = batch_item[0].to(opt.device)
            label = batch_item[1].to(opt.device)

            lam = np.random.beta(1.0, 1.0)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                # add - cutmix
                rand_index = torch.randperm(img.size()[0])
                target_a = label
                target_b = label[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
                img[:, :, bbx1:bbx2, bby1:bby2] = img[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))

                output = model(img)
                loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)

            loss.backward()
            optimizer.step()
            score = accuracy_function(label, output)

            train_loss += loss
            train_macro_f1 += score

            tqdm_train.set_postfix({"Epoch": epoch + 1,
                                    "train_loss": train_loss / (batch + 1),
                                    "train_f1": train_macro_f1 / (batch + 1)})

        # validation
        tqdm_valid = tqdm(valid_loader)
        mean_valid_loss, valid_loss, valid_macro_f1 = 0, 0, 0
        for batch, batch_item in enumerate(tqdm_valid):
            img = batch_item[0].to(opt.device)
            label = batch_item[1].to(opt.device)

            model.eval()
            with torch.no_grad():
                output = model(img)
                loss = criterion(output, label)
            score = accuracy_function(label, output)

            valid_loss += loss
            valid_macro_f1 += score

            mean_valid_loss = valid_loss / (batch + 1)
            tqdm_valid.set_postfix({"valid_loss": mean_valid_loss,
                                    "valid_f1": valid_macro_f1 / (batch + 1)})


        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            os.makedirs(opt.save_path, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(opt.save_path, f'{epoch}E_{mean_valid_loss:0.4f}_{opt.model_name}.pt'))
        else:
            early_stopping_step += 1
            print(f"Early Stopping Step : [{early_stopping_step} / {opt.early_stopping}]")

        if early_stopping_step == opt.early_stopping:
            print("=== Early Stop ===")
            break

if __name__ == "__main__" :
    opt, label_options = options()

    # label_enc, dec 및 trasnforms 설정
    label_encoder, label_decoder = VillageLabel_preprocessing(opt.label_path)
    train_transforms, valid_transforms = transform(size=opt.resize)

    # option 출력
    print("\n<< option >>")
    print(*["{} : {}".format(k, v) for k, v in vars(opt).items()], sep='\n')

    # data split with stratify
    train, valid = VillageData_split(opt.dataset_path, label_decoder)

    train_dataset = VillageDataset(train, label_decoder, train_transforms)
    valid_dataset = VillageDataset(valid, label_decoder, valid_transforms)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=opt.batch_size, shuffle=False)

    run(train_loader, valid_loader, opt)
