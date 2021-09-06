import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

import numpy as np
from easydict import EasyDict
import timm
import random
from utils import DataParser, TrainDataset


class Train():
    def __init__(self, args):
        self.model_name = args.model_name
        self.export = args.export

        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.lr = args.learning_rate

        self.k_fold_n = args.k_fold_n
        self.early_stop = args.early_stop

        self.train_path = args.train_path
        self.test_path = args.test_path

        self.random_seed = args.random_seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.CUDA_NUMPY_SETTING(random_seed=args.random_seed)

    def create_model(self, model_name, num_classes):
        model = timm.create_model(model_name,pretrained=True, num_classes=num_classes).to(device=self.device)
        return model


    def CUDA_NUMPY_SETTING(self, random_seed=11):
        torch.manual_seed(random_seed)
        torch.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        np.random.seed(random_seed)
        random.seed(random_seed)

        print("==" * 30)
        print("USING CUDA is ",torch.cuda.is_available())
        print("DETECTED GPU NUMNER : ", torch.cuda.current_device())
        print("Using ", torch.cuda.device_count()," GPUs")
        for i in range(torch.cuda.device_count()) :
            print("GPU name is ", torch.cuda.get_device_name(i))
        print("==" * 30)

    def calc_ACC(self, pred, label):
        preds = pred.argmax(dim=-1)
        comp_list = (preds==label).cpu().tolist()

        cnt = 0
        for i in comp_list :
            if i == True : cnt += 1

        return cnt / len(preds)

    def train(self):

        class_encoder = {
            "dog": 0,
            "elephant": 1,
            "giraffe": 2,
            "guitar": 3,
            "horse": 4,
            "house": 5,
            "person": 6
        }

        Datasets = DataParser(self.train_path, class_encoder, self.random_seed)

        model = self.create_model(self.model_name, num_classes=7)
        criterion = nn.CrossEntropyLoss(weight=(Datasets.weights).to(device=self.device, dtype=torch.float))
        optimizer = AdamW(params=model.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)


        if self.k_fold_n == False :

            train_dataset, valid_dataset = Datasets.DatasetParsing()

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)

            results = {
                'train_acc' : [],
                'train_loss' : [],
                'valid_loss' : [],
                'valid_acc' : [],
                'valid_f1' : []
            }

            best_snapshot = {
                'best_epoch' : 0,
                'best_f1' : 0,
                'best_model' : None
            }

            # train
            for E in range(1, self.epoch + 1) :
                model.train()

                iter_results = {
                    'train_acc' : [],
                    'train_loss': [],
                    'valid_loss': [],
                    'valid_acc': [],
                    'valid_f1': []
                }

                # iter
                for iter, (batch_img, batch_label) in enumerate(train_loader, start=1) :

                    imgs = batch_img.to(device=self.device, dtype=torch.float)
                    labels = batch_label.to(device=self.device)

                    optimizer.zero_grad()

                    pred = model(imgs)
                    loss = criterion(pred, labels)

                    loss.backward()
                    optimizer.step()

                    iter_results['train_acc'].append(self.calc_ACC(pred, labels))
                    iter_results['train_loss'].append(loss.cpu().item())

                    print(f"Epoch [{E} / {self.epoch}]      Iter [{iter} / {len(train_loader)}]" , end="\r")

                # eval
                with torch.no_grad() :
                    for iter, (batch_img, batch_label) in enumerate(valid_loader, start=1) :
                        model.eval()

                        imgs = batch_img.to(device=self.device, dtype=torch.float)
                        labels = batch_label.to(device=self.device)

                        val_pred = model(imgs)
                        val_pred_argmax = val_pred.argmax(dim=-1)
                        val_loss = criterion(val_pred, labels)

                        iter_results['valid_acc'].append(self.calc_ACC(val_pred, labels))
                        iter_results['valid_loss'].append(val_loss.cpu().item())
                        iter_results['valid_f1'].append(f1_score(y_true=labels.cpu().numpy(),
                                                                 y_pred=val_pred_argmax.cpu().numpy(),
                                                                 average="macro"))

                        print(f"Epoch [{E} / {self.epoch}]      valid_Iter [{iter} / {len(valid_loader)}]" , end="\r")

                # print(iter_results['valid_acc'])
                # print(type(iter_results['valid_acc']))
                results['train_acc'].append(np.mean(iter_results['train_acc']) * 100)
                results['train_loss'].append(np.mean(iter_results['train_loss']))
                results['valid_acc'].append(np.mean(iter_results['valid_acc']) * 100)
                results['valid_loss'].append(np.mean(iter_results['valid_loss']))
                results['valid_f1'].append(np.mean(iter_results['valid_f1']))

                scheduler.step()

                print(
                    f"[Epoch {E} / {self.epoch}] "
                    f"train_acc : {results['train_acc'][-1]:.4f} | "
                    f"train_loss : {results['train_loss'][-1]:.4f} | "
                    f"valid_acc : {results['valid_acc'][-1]:.4f} | "
                    f"valid_loss : {results['valid_loss'][-1]:.2f} | "
                    f"valid_f1 : {results['valid_f1'][-1]:.4f}"
                )

                if results['valid_f1'][-1] > best_snapshot['best_f1'] :
                    best_snapshot['best_f1'] = results['valid_f1'][-1]
                    best_snapshot['best_model'] = model
                    best_snapshot['best_epoch'] = E
                    early_stop_cout = 0

                else :
                    print(f"early_stop_couter : {early_stop_cout} / {self.early_stop}")
                    early_stop_cout += 1

                if early_stop_cout >= self.early_stop :
                    print()
                    print("*" * 20)
                    print("!! EARLY STOP !!")
                    print("*" * 20)
                    print()
                    break

            return results, best_snapshot

        else :
            # StratifiedKFold가 class 간의  balance를 맞춰주기 때문에 weights는 필요 없음
            kfold = StratifiedKFold(n_splits=self.k_fold_n, random_state=self.random_seed, shuffle=True)
            print(len(Datasets.img_list))
            print(len(Datasets.label_list))
            kfold_result = {"results": [], "best_snapshot":[]}
            for k, (fold_train, fold_valid) in enumerate(kfold.split(Datasets.img_list, Datasets.label_list), 1) :
                train_imgs, train_labels = [[Datasets.img_list[i] for i in fold_train]] * 2
                valid_img, valid_labels = [[Datasets.label_list[i] for i in fold_valid]] * 2

                train_transforms, valid_transforms = Datasets.get_transforms()
                fold_train_dataset = TrainDataset(img_list=train_imgs, label_list=train_labels, transforms=train_transforms)
                fold_valid_dataset = TrainDataset(img_list=valid_img, label_list=valid_labels, transforms=valid_transforms)


                # Datasets.img_list = fold_img
                # Datasets.label_list = fold_label
                # fold_train_dataset, fold_valid_dataset = Datasets.DatasetParsing()

                fold_train_loader = DataLoader(fold_train_dataset, batch_size=self.batch_size, shuffle=True)
                fold_valid_loader = DataLoader(fold_valid_dataset, batch_size=self.batch_size, shuffle=True)

                results = {
                    'train_acc': [],
                    'train_loss': [],
                    'valid_acc': [],
                    'valid_loss': [],
                    'valid_f1': []
                }

                best_snapshot = {
                    'best_epoch': 0,
                    'best_f1': 0,
                    'best_model': None
                }
                early_stop_count = 0
                for E in range(self.epoch + 1) :
                    model.train()

                    iter_result = {
                        'iter_train_acc': [],
                        'iter_train_loss': [],
                        'iter_valid_acc': [],
                        'iter_valid_loss': [],
                        'iter_valid_f1': []
                    }
                    for iter, (fold_img, fold_label) in enumerate(fold_train_loader, 1) :

                        fold_img = fold_img.to(device=self.device, dtype=torch.float)
                        fold_label = fold_label.to(device=self.device)

                        optimizer.zero_grad()

                        pred = model(fold_img)
                        loss = criterion(pred, fold_label)

                        loss.backward()
                        optimizer.step()

                        iter_result['iter_train_acc'].append(self.calc_ACC(pred,fold_label))
                        iter_result['iter_train_loss'].append(loss.cpu().item())

                        print(f"Epoch [{E} / {self.epoch}]      Iter [{iter} / {len(fold_train_loader)}]" , end="\r")

                    with torch.no_grad() :
                        model.eval()

                        for iter, (fold_img, fold_label) in enumerate(fold_valid_loader, 1) :

                            fold_img = fold_img.to(device=self.device, dtype=torch.float)
                            fold_label = fold_label.to(device=self.device)

                            pred = model(fold_img)
                            loss = criterion(pred, fold_label)

                            pred_argmax = pred.argmax(dim=-1)


                            iter_result['iter_valid_acc'].append(self.calc_ACC(pred, fold_label))
                            iter_result['iter_valid_loss'].append(loss.cpu().item())
                            iter_result['iter_valid_f1'].append(f1_score(y_true=fold_label.cpu().numpy(),
                                                                 y_pred=pred_argmax.cpu().numpy(),
                                                                 average="macro"))

                            print(f"Epoch [{E} / {self.epoch}]      Iter [{iter} / {len(fold_valid_loader)}]", end="\r")

                    results['train_acc'].append(np.mean(iter_result['iter_train_acc']) * 100)
                    results['train_loss'].append(np.mean(iter_result['iter_train_loss']))
                    results['valid_acc'].append(np.mean(iter_result['iter_valid_acc']) * 100)
                    results['valid_loss'].append(np.mean(iter_result['iter_valid_loss']))
                    results['valid_f1'].append(np.mean(iter_result['iter_valid_f1']))

                    scheduler.step()

                    print(
                        f"[Epoch {E} / {self.epoch}] "
                        f"train_acc : {results['train_acc'][-1]:.4f} | "
                        f"train_loss : {results['train_loss'][-1]:.4f} | "
                        f"valid_acc : {results['valid_acc'][-1]:.4f} | "
                        f"valid_loss : {results['valid_loss'][-1]:.2f} | "
                        f"valid_f1 : {results['valid_f1'][-1]:.4f}"
                    )

                    if results['valid_f1'][-1] >= best_snapshot['best_f1'] :
                        best_snapshot['best_f1'] = results['valid_f1'][-1]
                        best_snapshot['best_epoch'] = E
                        best_snapshot['best_model'] = model
                        early_stop_count = 0
                    else:
                        early_stop_count += 1

                    if early_stop_count >= self.early_stop :
                        break

                # kfold_result['results'].append(results) # >> ?? 굳이 저장을 해야하나?
                kfold_result['best_snapshot'].append(best_snapshot)

                print(f"Fold [{k} / {self.k_fold_n}]  Best-F1 [{kfold_result['best_snapshot'][k]['best_f1']}]")

            return kfold_result

def main(args):

    training = Train(args)
    if args.k_fold_n == False :
        results, best_snapshot = training.train()
        torch.save(best_snapshot['best_model'],
                   './' + str(best_snapshot['best_f1']) + '_' + str(best_snapshot['best_epoch']) + 'E_best_model.pt')

    else :
        kfold_best_snapshot = training.train()
        for idx, bs in enumerate(kfold_best_snapshot) :
            torch.save(bs['best_model'],
                   './Kfold_' + str(idx) +'_'+ str(bs['best_f1']) + '_' + str(bs['best_epoch']) + 'E_best_model.pt')


    print(results)

    print("DONE")

    # return "hello"


if __name__ == "__main__" :
    config={
        'model_name' : 'vit_base_patch16_224',
        'export' : './data/model',
        'batch_size' : 16,
        'epoch' : 15,
        'k_fold_n': 5,
        'learning_rate' : 1e-4,
        'early_stop' : 5,
        'random_seed' : 11,
        'train_path' : './data/train',
        'test_path' : './data/test'

    }

    args = EasyDict(config)

    main(args)