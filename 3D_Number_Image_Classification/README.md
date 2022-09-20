## Environment

- Window 10
- Anaconda
- Pytorch - 1.11

## **Description**

- VoxelNet을 통해 학습 시도 → 0.80점을 넘기기 힘들었음
- [코드 공유 게시판 글](https://dacon.io/competitions/official/235951/codeshare/6476?page=1&dtype=recent)을 참고하여 Point Cloud에 PCA를 적용하여 Image로 만들어서 학습 진행
- `EfficientNet B0` 사용
    - 색상이 복잡하지 않고 모양도 간단한 Image 데이터였기 때문에 더 큰 모델은 시도하지 않음
- `Rotate`, `RandomGridShuffle`, `Mixup`, `Backbone Freeze` 사용
    - Rotate는 학습 동안에 항시 적용했고 Test에서도 사용함
    - RandomGridShuffle을 통해서 Image가 가지는 Hight Level Feature를 학습
        - 2와 5, 6과 9는 회전 할 시 구분이 어렵기 때문에 Hight Level Feature 학습이 필요하다 생각함
        - [JigSaw puzzle](https://arxiv.org/pdf/1603.09246.pdf)에서 아이디어를 얻음
    - Mixup은 모델이 모호한 Image에 대해서도 잘 동작하기 위해 사용
    - 마지막 10 Epoch이 남았을 때, 모델의 Backbone을 Freeze하고 Rotate를 제외한 다른 Augmentation 없이 Classifier만 학습
        - TTA를 쓸 생각이 없었기 때문에 Augmentation이 적용되지 않은 데이터에 대해서도 잘 동작하길 바랬음
        - 적용 후 실제로 점수가 상승함
- `RandomGridShuffle` 35E 학습 → `Mixup` 25E 학습 → `Backbone Freeze` 10E 학습
    - RandomGridShuffle 과 Mixup을 동시에 사용하면 학습이 너무 어려워지고 원하는 방향으로 학습이 안될 것 같아서 따로 사용함
- `k-fold`, `Ensemble`, `Soft Voting` 사용
    - 5 fold로 학습
    - k-fold로 생성된 모델의 output을 Ensemble 함
    - Soft Voting과 Hard Voting을 각각 적용해 본 결과, Soft Voting의 점수가 더 높았음

## Get Started

- Conda  가상환경 설치
    
    ```
    conda env create --file environment.yaml
    ```
    
- Point Cloud Dataset을 Image Dataset 으로 변환
    
    ```
    python pc2img.py --mode train --data ./data/train.h5 --output ./data/img/train/
    ```
    

## EDA

- `EDA.ipynb` 참고
- 정리가 제대로 안되서 헷갈릴 수 있음

## Train

```
python main.py train --CTL_STEP 36 61
```

- `CTL_STEP` : RandomGridShuffle 과 Mixup 그리고 Backbone freeze를 적용할 Epoch
- 자세한 argument는 `main.py` 참고

## Test

```
python main.py test --ENSEMBLE soft --CHECKPOINT ./ckpt/model1.pth ./ckpt/model2.pth
```

- `ENSEMBLE` : soft voting 시 soft,  hard voting 시 hard라 입력, Ensemble을 하지 않을 시 입력 X
- 자세한 argument는 `main.py` 참고