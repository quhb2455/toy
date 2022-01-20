# 작물 병해 분류 AI 경진대회

Skills: Pytorch
진행 기간: 2021/10/13 → 2021/10/22
한 줄 소개: Dacon 경진대회

![Untitled](image/Untitled.png)

**주최 : 한국지능정보사회진흥원(NIA)**

**주관 : 팜한농**

**운영 : 데이콘**

**리더보드 : [https://dacon.io/competitions/official/235842/leaderboard](https://dacon.io/competitions/official/235842/leaderboard)**

**URL : [https://dacon.io/competitions/official/235842/overview/description](https://dacon.io/competitions/official/235842/overview/description)**

![Untitled](image/Untitled%201.png)

## INDEX

1. [결과]()
2. [평가 방식]()
3. [데이터셋]()
4. [사용 모델]()
5. [사용 기법]()
6. [전체코드]()
    1. [Data Parsing]()
    2. [CutMix Random Box]()
    3. [Training Define]()
        1. [Train 클래스 정의]()
        2. [CUDA setting]()
        3. [Model define using `timm`]()
        4. [Accuracy calculation]()
        5. [Class weight calculation]()
        6. [Training]()
    4. [Configuration]()
    5. [Main function and Model save]()
7. [자체 피드백]()

# 결과

---

- Public Score
    - 전체 등수 : 18 / 170
    - 점수 : 0.98565
- Private Score
    - 전체 등수 : 18 / 170
    - 점수 : 0.97979
    

# 평가 방식

---

- 평가 산식 : Macro - f1
- Public score : 전체 test 셋의 33%
- Private score : 전체 test 셋의 67%

⇒ Private Score로 최종 등수 산출

# 데이터셋

---

### Train set

![Untitled](image/Untitled%202.png)

- 총 개수 : 250장
- uid : 데이터 고유 아이디
- img_path : 이미지 데이터 경로
- disease : 병해정보
- disease_code : 병해 코드
    
    ![Untitled](image/Untitled%203.png)
    
    ![Untitled](image/Untitled%204.png)
    

### Test set

![Untitled](image/Untitled%205.png)

- 총 개수 : 4750장
- uid : 데이터 고유 아이디
- img_path : 이미지 데이터 경로

# 사용 모델

---

- Vision Transformer
    - 현재 Classification에서 인기가 많은 모델
    - `ImageNet`으로 pre-training 됨
    - `timm` 모델 명
        - vit_base_patch16_224
        
- NFNet
    - 경험에 기반한 선택
    - [미술 작품 분류 문제](https://www.notion.so/25232ddb6d7444798333724024c0222b)에서 성능이 좋았음
    - `ImageNet`으로 pre-training 됨
    - `timm` 모델 명
        - nfnet_l0
        
- Swin Transformer
    - 경험에 기반한 선택
    - [미술 작품 분류 문제](https://www.notion.so/25232ddb6d7444798333724024c0222b)에서 성능이 좋았음
    - `ImageNet`으로 pre-training 됨
    - `timm` 모델 명
        - swin_large_patch4_window7_224

# 사용 기법

---

- **Image Augmentation**
    - Resize
        - Random Crop을 고려하여 사용
    - RandomCrop
        - 사용 모델 입력 사이즈에 맞춰서 사용
    - Rotate
    - HorizontalFlip
    - Normalize
    - Cutmix
        - 분류 문제에 효율적임
        
- **Model Ensemble**
    - ViT, NFNet, Swin_Transformer 중에서 2개 혹은 3개를 앙상블함
    
- **Stratified K-fold 검증**
    - class 별 개수를 맞추기 위해 사용
    
- **Class 별 loss Weight  부여**
    - Data Imbalance 문제 해결을 위해 사용
    

# 전체 코드

---

### Index

1. [Data Parsing]()
2. [CutMix Random Box]()
3. [Training Define]()
    1. [Train 클래스 정의]()
    2. [CUDA setting]()
    3. [Model define using `timm`]()
    4. [Accuracy calculation]()
    5. [Class weight calculation]()
    6. [Training]()
4. [Configuration]()
5. [Main function and Model save]()

### Data Parsing

- code 보기
    
    ```python
    import torch
    from torch.utils.data import Dataset
    
    import albumentations as A
    from sklearn.model_selection import train_test_split, StratifiedKFold
    
    import cv2
    import os
    import numpy as np
    from glob import glob
    
    import pandas as pd
    
    class TrainDataset(Dataset):
        def __init__(self, img_list, label_list, transforms=None):
            self.img_list = img_list
            self.label_list = label_list
            self.transforms = transforms
    
        def __getitem__(self, idx):
            img = cv2.imread(self.img_list[idx], cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
            if self.transforms :
                img = self.transforms(image=img)["image"]
    
            img = img.transpose(2, 0, 1)
            label = self.label_list[idx]
    
            img = torch.tensor(img, dtype=torch.float)
            label = torch.tensor(label, dtype=torch.long)
    
            return img, label
    
        def __len__(self):
            assert len(self.img_list) == len(self.label_list)
            return len(self.img_list)
    
    class DataParser() :
        def __init__(self, data_path, random_seed, resize):
            self.img_list, self.label_list = self.get_data(data_path)
            self.random_seed = random_seed
            self.resize = resize
    
        def get_data(self, data_path):
            
            data = pd.read_csv(data_path)
    
            upper_path = data_path.split('/')[-1]
    
            img_list = []
            label_list = []
            for i in range(len(data['img_path'])) :
                img_list.append(data_path.replace(upper_path, data['img_path'][i]))
                label_list.append(data['disease_code'][i])
        
            return img_list, label_list
    
        
        def get_fold_data(self, fold_datalist):
            imgs = [self.img_list[i] for i in fold_datalist]
            labels = [self.label_list[i] for i in fold_datalist]
    
            return imgs, labels
            
            
            
        def get_transforms(self, train=True):
            if train :
                transforms = A.Compose([
                    A.Resize(self.resize + 20,self.resize + 20),
                    A.RandomCrop(self.resize, self.resize),
                    A.Rotate(),
                    A.HorizontalFlip(),
    #                 A.ColorJitter(),
                    A.Normalize()
                ])
            else :
                transforms = A.Compose([
                    A.Resize(224, 224),
                    A.Normalize()
                ])
    
            return transforms
        
        
        def DatasetParsing(self, fold_train=None, fold_valid=None):
            if fold_train is None :
                train_imgs, valid_imgs, train_labels, valid_labels = train_test_split(self.img_list,
                                                                                      self.label_list,
                                                                                      train_size=0.9,
                                                                                      shuffle=True,
                                                                                      random_state=self.random_seed,
                                                                                      stratify=self.label_list)
            else :
                train_imgs, train_labels = self.get_fold_data(fold_train)
                valid_imgs, valid_labels = self.get_fold_data(fold_valid)
    
            train_transforms = self.get_transforms()
            valid_transforms = self.get_transforms(train=False)
    
            train_dataset = TrainDataset(img_list=train_imgs, label_list=train_labels, transforms=train_transforms)
            valid_dataset = TrainDataset(img_list=valid_imgs, label_list=valid_labels, transforms=valid_transforms)
    
            return train_dataset, valid_dataset
    
    ```
    

### CutMix Random Box

- code 보기
    
    ```python
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
    
        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)
    
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
    
        return bbx1, bby1, bbx2, bby2
    ```
    

### Training Define

- code 보기
    - Train 클래스 정의
        - 아래에 나오는 모든 함수들은 Train 클래스에 포함 되는 것임
        
        ```python
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from torch.utils.data import DataLoader
        
        from sklearn.metrics import f1_score
        from sklearn.model_selection import StratifiedKFold
        
        import numpy as np
        from easydict import EasyDict
        import timm
        import random
        
        import wandb
        from time import time
        
        class Train():
            def __init__(self, args):
                self.model_name = args.model_name
                self.export = args.export
        
                self.epoch = args.epoch
                self.batch_size = args.batch_size
                self.lr = args.learning_rate
        
                self.k_fold_n = args.k_fold_n
                self.early_stop = args.early_stop
                self.resize = args.resize
        
                self.cosine_lr_Tmax = args.cosine_lr_Tmax
                self.cosine_lr_eta_min = args.cosine_lr_eta_min
                
                self.train_path = args.train_path
                self.test_path = args.test_path
                self.num_classes = args.num_classes
        
                self.random_seed = args.random_seed
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
                self.weights = self.cal_weights(self.train_path, self.num_classes) if args.weights else None
        
                self.args = args
                # self.CUDA_NUMPY_SETTING(random_seed=args.random_seed)
        ```
        
    - CUDA Setting - 정의는 했지만 사용 안함
        
        ```python
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
        ```
        
    - Model define using `timm`
        
        ```python
        def create_model(self, model_name, num_classes):
                model = timm.create_model(model_name,pretrained=True, num_classes=num_classes).to(device=self.device)
                return model
        
        def model(self,datasetWeights=None):
                model = self.create_model(self.model_name, num_classes=self.num_classes)
                criterion = nn.CrossEntropyLoss(weight=(datasetWeights).to(device=self.device, dtype=torch.float)
                                                if datasetWeights is not None
                                                else None)
                optimizer = AdamW(params=model.parameters(), lr=self.lr)
                scheduler = CosineAnnealingLR(optimizer, T_max=self.cosine_lr_Tmax, eta_min=self.cosine_lr_eta_min)
        
                return model, criterion, optimizer, scheduler
        ```
        
    - Accuracy calculation
        
        ```python
        def cal_ACC(self, pred, label):
                preds = pred.argmax(dim=-1)
                comp_list = (preds==label).cpu().tolist()
        
                cnt = 0
                for i in comp_list :
                    if i == True : cnt += 1
        
                return cnt / len(preds)
        ```
        
    - Class weight calculation
        
        ```python
         def cal_weights(self, data_path, num_classes) :
                train_total = pd.read_csv(data_path)
                num_label = np.array([pd.value_counts(train_total['disease_code'].values)[i] for i in range(num_classes)])
                return torch.tensor(np.max(num_label)/num_label).to(self.device, dtype=torch.float)
        ```
        
    - Training code
        
        ```python
        def run(self, train_loader, valid_loader, weights):
                
                results = {
                    'train_acc': [],
                    'train_loss': [],
                    'valid_loss': [],
                    'valid_acc': [],
                    'valid_f1': []
                }
        
                best_snapshot = {
                    'best_epoch': 0,
                    'best_f1': 0,
                    'best_model': None
                }
        
                early_stop_cout = 0
        
                model, criterion, optimizer, scheduler = self.model(weights)
        
                wandb.init(project='vit', entity='quhb2455',  config=self.args)
        #         wandb.watch(model, criterion, log="all", log_freq=1)
                
                for E in range(1, self.epoch + 1):
                    model.train()
        
                    iter_results = {
                        'train_acc': [],
                        'train_loss': [],
                        'valid_loss': [],
                        'valid_acc': [],
                        'valid_f1': []
                    }
                    start_t = time()
                    # iter
                    for iter, (batch_img, batch_label) in enumerate(train_loader, start=1):
                        imgs = batch_img.to(device=self.device, dtype=torch.float)
                        labels = batch_label.to(device=self.device)
                        
                        optimizer.zero_grad()
                        
        								# Cutmix - random box 적용 
                        lam = np.random.beta(1.0, 1.0)
                        rand_index = torch.randperm(imgs.size()[0])
                        target_a = labels
                        target_b = labels[rand_index]
                        bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
                        imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
                        
                        # adjust lambda to exactly match pixel ratio
                        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))
                              
                        pred = model(imgs)
        								# Cutmix - loss 적용 
                        loss = criterion(pred, target_a) * lam + criterion(pred, target_b) * (1. - lam)
        
                        loss.backward()
                        optimizer.step()
                        
                        iter_results['train_acc'].append(self.cal_ACC(pred, labels))
                        iter_results['train_loss'].append(loss.cpu().item())
        
                        print(f"Epoch [{E} / {self.epoch}]      Iter [{iter} / {len(train_loader)}]", end="\r")
        
                        
                        
                    # eval
                    with torch.no_grad():
                        for iter, (batch_img, batch_label) in enumerate(valid_loader, start=1):
                            model.eval()
        
                            imgs = batch_img.to(device=self.device, dtype=torch.float)
                            labels = batch_label.to(device=self.device)
        
                            val_pred = model(imgs)
                            val_pred_argmax = val_pred.argmax(dim=-1)
                            val_loss = criterion(val_pred, labels)
        
                            iter_results['valid_acc'].append(self.cal_ACC(val_pred, labels))
                            iter_results['valid_loss'].append(val_loss.cpu().item())
                            iter_results['valid_f1'].append(f1_score(y_true=labels.cpu().numpy(),
                                                                     y_pred=val_pred_argmax.cpu().numpy(),
                                                                     average="macro"))
        
                            print(f"Epoch [{E} / {self.epoch}]      valid_Iter [{iter} / {len(valid_loader)}]", end="\r")
        
                    results['train_acc'].append(np.mean(iter_results['train_acc']))
                    results['train_loss'].append(np.mean(iter_results['train_loss']))
                    results['valid_acc'].append(np.mean(iter_results['valid_acc']))
                    results['valid_loss'].append(np.mean(iter_results['valid_loss']))
                    results['valid_f1'].append(np.mean(iter_results['valid_f1']))
        
                    scheduler.step()
                    
                    wandb.log({
                        'Epochs' : E,
                        'train_acc' : results['train_acc'][-1].item(),
                        'train_loss' : results['train_loss'][-1].item(),
                        'valid_acc' : results['valid_acc'][-1].item(),
                        'valid_loss' : results['valid_loss'][-1].item(),
                        'valid_f1' : results['valid_f1'][-1].item()                
                    })
                    end_t = time() - start_t
                    print(
                        f"[Epoch {E} / {self.epoch}] "
                        f"Time : {end_t:.4f} s | "
                        f"train_acc : {results['train_acc'][-1]:.4f} | "
                        f"train_loss : {results['train_loss'][-1]:.4f} | "
                        f"valid_acc : {results['valid_acc'][-1]:.4f} | "
                        f"valid_loss : {results['valid_loss'][-1]:.4f} | "
                        f"valid_f1 : {results['valid_f1'][-1]:.4f}"
                    )
        
                    if results['valid_f1'][-1] > best_snapshot['best_f1']:
                        best_snapshot['best_f1'] = results['valid_f1'][-1]
                        best_snapshot['best_model'] = model
                        best_snapshot['best_epoch'] = E
                        early_stop_cout = 0
        
                    else:
                        early_stop_cout += 1
                        print(f"early_stop_couter : {early_stop_cout} / {self.early_stop}")
        
                    if early_stop_cout >= self.early_stop:
                        print()
                        print("*" * 20)
                        print("!! EARLY STOP !!")
                        print("*" * 20)
                        print()
                        break
        
                return results, best_snapshot
        
        		def train(self):
        		
        		        Datasets = DataParser(self.train_path, self.random_seed, self.resize)
        		        
        		        if self.k_fold_n == False :
        		
        		            train_dataset, valid_dataset = Datasets.DatasetParsing()
        		
        		            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        		            valid_loader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=True)
        		            
        		            results, best_snapshot = self.run(train_loader, valid_loader, self.weights)
        		
        		            return results, best_snapshot
        						else :
        								print(f"Training Start [ {self.k_fold_n} ] times")
        		            
        		            kfold = StratifiedKFold(n_splits=self.k_fold_n, random_state=self.random_seed, shuffle=True)
        
        		            kfold_result = []
        								for k, (fold_train, fold_valid) in enumerate(kfold.split(Datasets.img_list, Datasets.label_list), 1) :
        										fold_train_dataset, fold_valid_dataset = Datasets.DatasetParsing(fold_train, fold_valid)
        										fold_train_loader = DataLoader(fold_train_dataset, batch_size=self.batch_size, shuffle=True)
        										fold_valid_loader = DataLoader(fold_valid_dataset, batch_size=self.batch_size, shuffle=True)
        
                        results, best_snapshot = self.run(fold_train_loader, fold_valid_loader, None)
        
                        
                        kfold_result.append(best_snapshot)
        
                        print(f"\n==== Fold [{k} / {self.k_fold_n}]  Best-F1 [{kfold_result[k-1]['best_f1']}] ====\n")
        
                    return kfold_result					
        
        ```
        

### Configuration

- code 보기
    - config는 모델 별로 따로 관리했음.
    
    ```python
    config={
        'model_name' : 'vit_base_patch16_224', # 사용하려는 timm 모델 
        'export' : './data/model', # 학습 후 모델이 저장될 장소
        'batch_size' : 16, 
        'epoch' : 20,
        'k_fold_n': 3, 
        'resize' : 224,
        'learning_rate' : 1e-4,
        'cosine_lr_Tmax' : 15,
        'cosine_lr_eta_min' : 1e-5,
        
        'early_stop' : 4,
        'random_seed' : 11, # dataset 셔플용
        'train_path' : './data/train.csv',
        'test_path' : './data/test.csv', # 안씀
        'num_classes' : 7, 
        'weights' : True # dataset 불균형 때문에 넣어봄, loss 계산 시 label 갯수를 기반으로 가중치 추가됨.
    }
    
    args = EasyDict(config)
    ```
    

### Main Function and Model Save

- code 보기
    
    ```python
    print("="*20)
    print(args)
    print("=" * 20)
    
    make_new_dir(args.export)
    
    training = Train(args)
    
    if args.k_fold_n == False :
        results, best_snapshot = training.train()
        
        print("Training DONE")
        
        torch.save(best_snapshot['best_model'],
                   args.export + '/' + str(best_snapshot['best_f1']) + '_' + str(best_snapshot['best_epoch']) + 'E_best_model.pt')
    
    else :
        kfold_best_snapshot = training.train()
        
        print("Training DONE")
        
        for idx, bs in enumerate(kfold_best_snapshot) :
            torch.save(bs['best_model'],
                   args.export + '/Kfold_' + str(idx) +'_'+ str(bs['best_f1']) + '_' + str(bs['best_epoch']) + 'E_best_model.pt')
    
    print("SUCCESS to saving the model")
    ```
    

# 자체 피드백

---

- StratifiedKFold 에 대한 잘못된 이해
    
    **⇒ 각 fold에 class 별 갯수를 맞춰주는 것으로 이해했어서 weight loss를 따로 입력해주지 않음.  weight loss 를 넣어주면 성능 증가 가능성 있음.**
    
    - 이전에는 fold 별로 데이터셋을 자를 때 class 별 개수도 맞춰주는 것으로 이해함
        - example
            - 0번 클래스 : 20장
            1번 클래스 : 28장
            2번 클래스 : 40장
            —> fold 3
            ⇒  fold #1
            0번 클래스 : 6장
            1번 클래스 : 6장
            2번 클래스 : 6장
            ...
    - 사용 후에는  fold 별로 각 class의 개수만 맞춰주는 것으로 다시 이해함
        - example
            - 0번 클래스 : 20장
            1번 클래스  : 28장
            2번 클래스 : 40장
            —> fold 4
            ⇒  fold #1
            0번 클래스 : 5장
            1번 클래스 : 7장
            2번 클래스 : 10장
            ...
- wandb을 제대로 사용 못했음.
    
    ⇒ wandb에 대한 이해도가 거의 없었고, 한번 써본다는 생각으로 사용한거라 제대로 못쓴듯.
    
    **⇒ learning rate 분석이나 여러 hyperparameter등 분석할 때 용이하다고 함. 다음 대회에 사용하여 parameter tuning을 한다면 성능 증가 가능성 있음.**
    
- 여러가지 분류 모델에 대한 이해도 및 시도 횟수 부족
    
    ⇒ 사용한 NFNet이나 Swin Transformer에 대한 이해도가 전혀 없었음. 그냥 이전에 사용했던 모델이라 그대로 사용함.
    
    **⇒ 풀고자 하는 문제에 최적화 되어있는 모델을 사용한다면 성능 증가 가능성 있음. 대신 모델에 대한 이해도 뿐만 아니라 딥러닝 전반적인 이해도를 높여야하지 않을까 싶음.**
    

- 평가방식에 대한 이해 부족
    
    ⇒ 평가 방식이 Macro - F1 인데, 별 생각 없이 진행함. 대회가 끝나고 찾아보니 Marco - f1은 모든 클래스에 대해서 평균적으로 얼마나 잘 작동하는지를 확인할 수 있는 지표라고 함. 자세한 부분은 따로 정리할 예정([이 곳](https://www.notion.so/AI-4b0b463045f54c919fbc71ba22362a50))
    
    ⇒ 전체적으로 일반화가 잘 되어 있는 모델을 만들어야 하는데 나는 여러 모델 앙상블을 통해서 점수 끌어올리기에 바빴음. 모델에 대한 이해도나 앙상블에 대한 이해가 없어서가 아닐까 싶음. 이 또한 모델에 대한 이해도가 올라가면 자동적으로 해소될 문제.
    
- Learning rate scheduler에 대한 이해 부족
    
    ⇒ `CosineAnnealingLR` 만 사용했었고 `CosineAnnealingLR`에 대한 이해도가 높질 않음. 그냥 저번에 썼으니까 경험적으로 쓴다는 느낌이 강함. 
    
    ⇒ wandb나 tensorboard 등의 훈련 시각화 툴을 사용해서 분석하면서 정해야할듯,  경험과 지식이 좀 더 필요함.
    
- Test Dataset Pseudo  Labeling 진행
    
    ⇒ 대회가 끝난 후 순위권 참가자의 코드를 보니 Test set으로 Pseudo Labeling을 진행했음.
    
    **⇒ 규칙에 따로 명시되어 있지 않았지만, Pseudo  Labeling을 하지않음. Pseudo  Labeling을 쓰게 되면 성능 증가 가능성이 있음.**
    
- 다양한 Optimizer 시도
    
    ⇒ `AdamW`만 시도 했음. 여러개 테스트 하면서 어떤 optimizer가 가장 좋은 성능을 뽑아 내는지 확인 했어야 했음.
    
    ⇒ 이 부분은 순위권 참가자가 공유 해준 코드를 보고 리뷰 할 예정