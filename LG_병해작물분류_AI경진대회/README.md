![Untitled](img/Untitled.png)

### 주최 : **LG AI Research**

### 주관 & 운영 : Dacon

### 목적 : “작물 환경 데이터”와 “작물 병해 이미지”를 이용해 “작물의 종류”, “병해의 종류”, “병해의 진행 정도”를 진단하는 AI 모델 개발

### URL : [**https://dacon.io/competitions/official/235870/overview/description**](https://dacon.io/competitions/official/235870/overview/description)

### 자세한 설명 : [https://charmed-creek-53c.notion.site/LG-AI-Research-AI-27d845b76f35404d959839753c401fa4](https://www.notion.so/LG-AI-Research-AI-27d845b76f35404d959839753c401fa4)

## Environments

- Window 10
- Anaconda
- Pytorch == 1.10

## Requirements

```bash
pip install -r requirements.txt
```

- 이미 구성되어 있는 가상환경에서 진행 했기 때문에 정확하지 않음

## 사용 모델

![Untitled](img/Untitled%201.png)

- EfficientNet-V2
- mLSTM-FCNs

## Public Dataset

- PlantVillage
    - Download URL
        - [https://www.kaggle.com/abdallahalidev/plantvillage-dataset](https://www.kaggle.com/abdallahalidev/plantvillage-dataset)
    - Dataset Structure
        
        ```
        archive
        ├ color
        	├ Apple__Apple_scab
        	...
        	└ Tomato___Tomato_Yellow_Leaf_Curl_Virus
        ├ grayscale
        └ segmented
        ```
        
        - color 만 이용하여 학습 진행
        - 각 폴더의 이름은 Label임

## Pretraining With Public Dataset

```bash
python village_train.py --dataset_path ../archive/color --label_path ../archive/color --num_classes 38 --batch_size 64
```

- kfold training은 사용하지 않음.
- `label_path` : 폴더 명이 label 이기 때문에 `dataset_path`와 동일한 경로 입력

## Training

- normal training
    
    ```bash
    python train.py --dataset_path ../data/train --label_path ../data/train.csv --num_classes 25
    ```
    
- training with pretrained model trained by Public dataset
    
    ```bash
    python train.py --dataset_path ../data/train --label_path ../data/train.csv --num_classes 25 --pretrained_path ../pretrained/model.pt
    ```
    
- kfold training with pretrained model trained by Public dataset
    
    ```bash
    python train.py --dataset_path ../data/train --label_path ../data/train.csv --num_classes 25 --use_kfold=True --kfold_splits 4 --pretrained_path ../pretrained/model.pt
    ```
    
- 이 외의 Option들은 `options.py` 를 참고

## Inference

```bash
python train.py 
--dataset_path ../data/test
--label_path ../data/train.csv 
--save_path ./
--num_classes 25 
--pretrained_path ../pretrained/model.pt
--voting softvoting
--csv ../data/sample_submission.csv
--model_path ./saved_model
```

- 다른 hyperparameters는 학습과 동일하게 가져가야함.
    - `dataset_path` : test 폴더로 변경
    - `save_path` : submission.csv가 저장될 위치 입력
- 추가할 hyperparameters는 `csv`, `voting`, `model_path`
    - `csv` : sample_submission.csv 경로
    - `voting` : **softvoting**과 **hardvoting**으로 입력 가능
        - single model로 하는 경우 입력 하지 않음
    - `model_path` : model들의 경로 입력
        - single model 일 경우 해당 모델의 경로 입력
            - ex) path/to/model.pt

## 결과

- Public Score : 전체 Test data 중 33 %
    - 전체 등수 : 68 / 344
    - 점수 : 0.93894
- Private Score : 전체 Test data 중 67%
    - 전체 등수 : 56 / 344
    - 점수 : 0.94434
