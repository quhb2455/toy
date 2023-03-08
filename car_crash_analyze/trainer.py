import torch
import os
import argparse
from tqdm import tqdm

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter
import torch
from datasets import *
from utils import *
from models import NN, simple_NN
from loss_fn import BCELoss

class Trainer() :
    def __init__ (self, model, optimizer, criterion, device, args) :
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-4)
        self.args = args
        # train_df, val_df, train_labels, val_labels = divided_train_val(read_csv('./data/train.csv'))
        self.train_loader ,self.val_loader = self.get_dataloader(
            args.CSV_PATH, 
            args.BATCH_SIZE, 
            args.SHUFFLE, 
            args.STACK,
            args.RESIZE)
        
        self.log_writter = SummaryWriter(args.LOG)
        self.save_path = args.OUTPUT

        self.best_score = 0
        self.device = device
        self.early_stop_cnt = 0
    def run(self):
        if self.args.REUSE :
            self.model, self.optimizer, self.args.START_EPOCH = weight_load(self.model,
                                                                            self.optimizer,
                                                                            self.args.CHECKPOINT)
            
        for epoch in range(self.args.START_EPOCH + 1, self.args.EPOCHS + 1) :
            # control RandomGridShuffle, Mixup and WeightFreeze
            # self.training_controller(epoch)

            # training
            self.training(epoch)

            # validation
            self.validation(epoch)
            self.scheduler.step()
            
            # if self.early_stop_cnt == 5 :
            #     break

    def training(self, epoch):
        self.model.train()
        tqdm_train = tqdm(self.train_loader)
        train_acc, train_loss = [], []
        for batch, (img, label) in enumerate(tqdm_train, start=1):

            self.optimizer.zero_grad()
            img = img.to(self.device)
            # img = [i.to(self.device) for i in img]
            label = label.to(self.device)

            if self.args.APPLY_CUTMIX:
                img, lam, label_a, label_b = cutmix(img, label)
                # output, batch_output = self.model(img)
                output = self.model(img)
                # output = output.squeeze(1)
                # label_a = label_a.float()
                # label_b = label_b.float()
                
                loss = lam * self.criterion(output, label_a) + (1 - lam) * self.criterion(output, label_b)
            
            elif self.args.APPLY_MIXUP:
                img, lam, label_a, label_b = mixup(img, label)
                # output, batch_output = self.model(img)
                output = self.model(img)
                # output = output.squeeze(1)
                loss = lam * self.criterion(output, label_a) + (1 - lam) * self.criterion(output, label_b)
                
            else:
                # output, batch_output = self.model(img)
                output = self.model(img)
                loss = self.criterion(output, label)

            loss.backward()
            self.optimizer.step()

            # acc = new_batch_score(label, output, self.args.THRESHOLD)
            # print(batch_acc)
            acc = score(label, output, self.args.THRESHOLD)
            # print(acc)

            train_acc.append(acc)
            train_loss.append(loss.item())

            tqdm_train.set_postfix({
                'Epoch': epoch,
                'Training Acc': np.mean(train_acc),
                'Training Loss': np.mean(train_loss)
            })

            data = {
                'training loss': loss.item(),
                'training acc': acc
            }
            logging(self.log_writter, data, epoch * len(self.train_loader) + batch)
        
        
    def validation(self, epoch):
        self.model.eval()
        val_acc, val_loss = [], []
        tqdm_valid = tqdm(self.val_loader)
        with torch.no_grad():
            for batch, (img, label) in enumerate(tqdm_valid):
                img = img.to(self.device)
                # img = [i.to(self.device) for i in img]
                label = label.to(self.device)

                # output, batch_output = self.model(img)
                output = self.model(img)
                
                # output = output.squeeze(1)
                # label = label.float()
                
                loss = self.criterion(output, label)

                # acc = new_batch_score(label, output, self.args.THRESHOLD)
                acc = score(label, output, self.args.THRESHOLD)

                val_acc.append(acc)
                val_loss.append(loss.item())

                tqdm_valid.set_postfix({
                    'Epoch' : epoch,
                    'Valid Acc': np.mean(val_acc),
                    'Valid Loss': np.mean(val_loss)
                })

                data = {
                    'validation loss': loss.item(),
                    'validation acc': acc
                }
                logging(self.log_writter, data, epoch * len(self.val_loader) + batch)
                
        self.model_save(epoch, np.mean(val_acc))

    def kfold_setup(self, model, optimizer, criterion, train_ind, valid_ind, kfold):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader, self.val_loader = train_and_valid_dataload((self.img_set[train_ind], self.img_set[valid_ind]),
                                                                      (self.label_set[train_ind], self.label_set[valid_ind]),
                                                                      self.transform,
                                                                      batch_size=self.args.BATCH_SIZE)
        self.log_writter = SummaryWriter(os.path.join(self.args.LOG , str(kfold)))
        self.save_path = os.path.join(self.args.OUTPUT, str(kfold) + self.args.MODEL_NAME)

    def training_controller(self, epoch):
        # Turn off RandomGridShuffle and Turn on Mixup
        if epoch == self.args.CTL_STEP[0]:
            self.train_loader, self.val_loader = train_and_valid_dataload(self.img_set,
                                                                          self.label_set,
                                                                          transform_parser(grid_shuffle_p=0),
                                                                          self.args.BATCH_SIZE)
            self.APPLY_MIXUP = True

        # Turn off Mixup and freeze classifier
        elif epoch == self.args.CTL_STEP[1]:
            self.APPLY_MIXUP = False
            self.model = weight_freeze(self.model)


    def get_dataloader(self, csv_path, batch_size, shuffle=True, stack=False, resize=384):
        train_df, val_df, train_labels, val_labels = divided_train_val(read_csv(csv_path), stack=stack)
        return train_and_valid_dataload((train_df, val_df), (train_labels, val_labels), 
                                        batch_size=batch_size, 
                                        shuffle=shuffle, stack=stack, 
                                        resize=resize)
        

    def model_save(self, epoch, val_acc):
        if self.best_score < val_acc:
            self.best_score = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()
            }, os.path.join(self.save_path, str(epoch) + 'E-val' + str(self.best_score) + '-' + self.args.MODEL_NAME + '.pth'))
            self.early_stop_cnt = 0
        else :
            self.early_stop_cnt += 1

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--BATCH_SIZE", type=int, default=1)
    parser.add_argument("--LEARNING_RATE", type=float, default=1e-2)
    parser.add_argument("--EPOCHS", type=int, default=70)
    parser.add_argument("--RESIZE", type=int, default=384)
    parser.add_argument("--FOCAL_GAMMA", type=int, default=2)
    parser.add_argument("--FOCAL_ALPHA", type=int, default=2)
    parser.add_argument("--THRESHOLD", type=float, default=0.5)
    parser.add_argument("--APPLY_CUTMIX", type=bool, default=False)

    parser.add_argument("--MODEL_NAME", type=str, default='efficientnetv2_rw_s') #swin_base_patch4_window7_224_in22k
    parser.add_argument("--KFOLD", type=int, default=0)

    parser.add_argument("--CSV_PATH", type=str, default="./data/train.csv")

    parser.add_argument("--OUTPUT", type=str, default='./ckpt')
    parser.add_argument("--LOG", type=str, default='./tensorboard/PCA_img/test')

    parser.add_argument("--REUSE", type=bool, default=False)
    parser.add_argument("--CHECKPOINT", type=str, default='./ckpt/3E-val0.8645-efficientnet_b0.pth')

    parser.add_argument("--START_EPOCH", type=int, default=0)


    args = parser.parse_args()
    os.makedirs(args.OUTPUT, exist_ok=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = NN(model_name=args.MODEL_NAME, num_classes=1, hid_bs=1792, bs=16).to(device) #simple_NN(model_name=args.MODEL_NAME, num_classes=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.LEARNING_RATE)
    criterion = BCELoss()

    if args.KFOLD == 0 :
        trainer = Trainer(model, optimizer, criterion, device, args)
        trainer.run()

    elif args.KFOLD > 0 :
        kfold = StratifiedKFold(n_splits=args.KFOLD, shuffle=True)
        trainer = Trainer(model, optimizer, criterion, device, args)
        for k, (train_ind, valid_ind) in enumerate(kfold.split(trainer.img_set, trainer.label_set)) :
            trainer.kfold_setup(model, optimizer, criterion, train_ind, valid_ind, k)
            trainer.run()