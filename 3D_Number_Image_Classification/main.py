import os
import argparse
import torch

from trainer import Trainer
from predictor import Predictor

from utils import *
from models import CNN
from loss_fn import FocalLoss

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = CNN(args.MODEL_NAME).to(device)

    if args.MODE == 'train' :
        os.makedirs(args.OUTPUT, exist_ok=True)
        save_config(vars(args), args.OUTPUT)

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.LEARNING_RATE)
        criterion = FocalLoss(args.FOCAL_GAMMA, args.FOCAL_ALPHA)
        if args.KFOLD == 0:
            trainer = Trainer(model, optimizer, criterion, device, args)
            trainer.run()

        elif args.KFOLD > 0:
            kfold = StratifiedKFold(n_splits=args.KFOLD, shuffle=True)
            trainer = Trainer(model, optimizer, criterion, device, args)
            for k, (train_ind, valid_ind) in enumerate(kfold.split(trainer.img_set, trainer.label_set)):
                trainer.kfold_setup(model, optimizer, criterion, train_ind, valid_ind, k)
                trainer.run()

    elif args.MODE == 'test' :

        predictor = Predictor(model, device, args)

        preds = predictor.run()
        save_to_csv(predictor.df, preds, args.OUTPUT)




if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="")

    subparsers = parser.add_subparsers(dest='MODE')


    # common
    parser.add_argument("--BATCH_SIZE", type=int, default=64)
    parser.add_argument("--MODEL_NAME", type=str, default='efficientnet_b0')
    parser.add_argument("--IMG_PATH", type=str, default="./data/img/224img_train/*")
    parser.add_argument("--CSV_PATH", type=str, default="./data/train.csv",
                        help='For test, using sample_submission.csv file')
    parser.add_argument("--OUTPUT", type=str, default='./ckpt',
                        help='For train, checkpoint save path \\'
                             'For test, predicted label save path')

    # train
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument("--LEARNING_RATE", type=float, default=1e-4)
    train_parser.add_argument("--EPOCHS", type=int, default=70)
    train_parser.add_argument("--CTL_STEP", nargs="+", type=int, default=[36, 61])
    train_parser.add_argument("--FOCAL_GAMMA", type=int, default=2)
    train_parser.add_argument("--FOCAL_ALPHA", type=int, default=2)
    train_parser.add_argument("--KFOLD", type=int, default=5)
    train_parser.add_argument("--LOG", type=str, default='./tensorboard/PCA_img/test')
    train_parser.add_argument("--REUSE", type=bool, default=False)
    train_parser.add_argument("--CHECKPOINT", type=str, default='./ckpt/3E-val0.8645-efficientnet_b0.pth')
    train_parser.add_argument("--START_EPOCH", type=int, default=0)

    # test
    test_parser = subparsers.add_parser('test')
    test_parser.add_argument("--ENSEMBLE", type=str, default=None)
    test_parser.add_argument("--CHECKPOINT",  nargs="+", type=str,
                        default=['./ckpt/58E-val0.957-efficientnet_b0.pth',
                                 './ckpt/55E-val0.9542-efficientnet_b0.pth',
                                 './ckpt/53E-val0.9537-efficientnet_b0.pth',
                                 './ckpt/49E-val0.953-efficientnet_b0.pth',
                                 './ckpt/45E-val0.9521-efficientnet_b0.pth'])


    args = parser.parse_args()
    # print(vars(args))
    main(args)