import torch
import os
import argparse
from tqdm import tqdm

import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter

from datasets import *
from utils import *
from models import simple_NN

class Predictor() :
    def __init__ (self, model, device, args) :
        self.model = get_models(model, args.CHECKPOINT)
        self.test_loader, self.df = self.get_dataloader(args.CSV_PATH, args.IMG_PATH, args.BATCH_SIZE)
        self.ensemble = args.ENSEMBLE
        self.save_path = args.OUTPUT
        self.batch_size = args.BATCH_SIZE
        self.device = device
        self.type = args.TYPE
    def run(self):
        if self.ensemble == 'soft':
            return self.ensemble_predict_softVoting()

        elif self.ensemble == 'hard':
            return self.ensemble_predict_hardVoting()

        elif self.ensemble is None:
            return self.predict()


    def predict(self):
        self.model.eval()
        model_preds = []
        model_preds_ind = []
        with torch.no_grad() :
            for img in tqdm(self.test_loader) :
                img = img.to(self.device)
                preds = self.model(img)
                
                
                
                
                if self.type =='each' :
                    pred_v, pred_ind = preds.max(1)

                    model_preds += pred_v.detach().cpu().numpy().tolist()
                    model_preds_ind += pred_ind.detach().cpu().numpy().tolist()
                
                else :
                    model_preds += preds.argmax(1).detach().cpu().numpy().tolist()
                

        return model_preds, model_preds_ind

    def ensemble_predict_softVoting(self):
        model_preds = []
        with torch.no_grad():
            for img in tqdm(self.test_loader):
                img = img.to(self.device)
                batch_preds_score = []
                for m in self.model:
                    m.eval()
                    pred = m(img)
                    batch_preds_score.append(pred.detach().cpu().numpy())

                batch_preds_score = np.mean(np.array(batch_preds_score), axis=0)
                best_score_ind = np.argmax(batch_preds_score, axis=1)
                model_preds += best_score_ind.tolist()
        return model_preds

    def ensemble_predict_hardVoting(self):
        model_preds = []
        with torch.no_grad():
            for img in tqdm(self.test_loader):
                img = img.to(self.device)

                batch_len = [i for i in range(self.batch_size)]
                batch_preds_score = []
                batch_preds_label = []
                for m in self.model:
                    m.eval()
                    pred = m(img)

                    pred = pred.max(1)
                    batch_preds_score.append(pred[0].detach().cpu().numpy())
                    batch_preds_label.append(pred[1].detach().cpu().numpy())

                best_score_ind = np.argmax(batch_preds_score, axis=0)
                batch_preds_label = np.array(batch_preds_label)

                model_preds += batch_preds_label[best_score_ind[batch_len], batch_len].tolist()
        return model_preds

    def get_dataloader(self, csv_path, img_path, batch_size):
        # transform = transform_parser()
        # img_set = glob(img_path)
        df = pd.read_csv(csv_path)
        stack = True if self.type == 'stack' else False
        # img_set, df, transform = image_label_dataset(csv_path, img_path,
        #                                              grid_shuffle_p=0, training=False)
        return custom_dataload(df, None, batch_size, data_type='valid', shuffle=False, stack=stack), df



if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--BATCH_SIZE", type=int, default=64)
    parser.add_argument("--MODEL_NAME", type=str, default='efficientnet_b0')
    parser.add_argument("--ENSEMBLE", type=str, default=None)
    parser.add_argument("--TYPE", type=str, default='each', 
        help='each = 50개(1 video = 50frame) frame을 평균내서 가장 높은 값을 가진 클래스를 할당\
            stack = 50개 frame을 채널 단위로 겹쳐서 추론')

    parser.add_argument("--IMG_PATH", type=str, default="./data/new_train/*")
    parser.add_argument("--CSV_PATH", type=str, default="./data/new_test.csv")
    parser.add_argument("--SUBMIT",type=str,default="./data/sample_submission.csv")
    parser.add_argument("--OUTPUT", type=str, default='./results/div_2width_effib0_224.csv')
    parser.add_argument("--CHECKPOINT",  nargs="+", type=str, 
                        default=["./ckpt/div_2width_effib0_224/1E-val1.0-efficientnet_b0.pth"])                        # default=['./ckpt/58E-val0.957-efficientnet_b0.pth',
                        #          './ckpt/55E-val0.9542-efficientnet_b0.pth',
                        #          './ckpt/53E-val0.9537-efficientnet_b0.pth',
                        #          './ckpt/49E-val0.953-efficientnet_b0.pth',
                        #          './ckpt/45E-val0.9521-efficientnet_b0.pth'])


    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = simple_NN(args.MODEL_NAME, num_classes=13).to(device)

    predictor = Predictor(model, device, args)

    pred_value, pred_index = predictor.run()
    
    save_to_csv(
        read_csv(args.SUBMIT), 
        get_each_frame_infer_result(pred_value, pred_index), 
        args.OUTPUT)