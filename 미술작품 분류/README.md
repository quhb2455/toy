# 미술 작품 분류

Skills: Pytorch, Tensorflow
진행 기간: 2021/08/23 → 2021/08/23
한 줄 소개: 프로그래머스 머신러닝 문제

![Untitled](image/Untitled.png)

**주최 : 프로그래머스**

**URL : [https://programmers.co.kr/skill_check_assignments/133](https://programmers.co.kr/skill_check_assignments/133)**

## INDEX

1. [평가 방식]()
2. [데이터셋]()
3. [사용 모델]()
4. [사용 기법]()
5. [자체 피드백]()

# 평가방식

---

- Test set 정확도 기준으로 채점

# 데이터셋

---

- Art Painting 이미지로 객체가 가지고 있는 일반적인 모습이 아니고 색깔 또한 많이 다름
- 클래스 별 이미지 예시
    
    ![Untitled](image/Untitled%201.png)
    

- 클래스 별 데이터 개수
    
    ![Untitled](image/Untitled%202.png)
    
- 총 데이터 개수
    - Training : 1698장
    - Test : 350장

# 사용 모델

---

- Vision Transformer
    - 현재 Classification에서 인기가 많은 모델
    - `ImageNet`으로 pre-training 됨
    - `timm` 모델 명
        - vit_base_patch16_224
        
- EfficientNet-B0, B1, B2
    - 사용중인 노트북에 GPU가 없음. 그래서 CPU로 충분히 학습할 수 있는 모델이 필요했음
    - 적은 리소스로 최대 효율을 뽑을 수 있을 것이라 생각함.
    - `ImageNet`으로 pre-training 됨
    - Tensorflow 모델 사용
    
- NFNet
    - 프로그래머스에서 배포한 베스트 코드에서 사용한 모델이라 적용해봄
    - `ImageNet`으로 pre-training 됨
    - `timm` 모델 명
        - nfnet_l0
        
- Swin Transformer
    - 프로그래머스에서 배포한 베스트 코드에서 사용한 모델이라 적용해봄
    - `ImageNet`으로 pre-training 됨
    - `timm` 모델 명
        - swin_large_patch4_window7_224

# 사용 기법

---

- **Image Augmentation**
    - Resize
    - Rotate
    - HorizontalFlip
    - ColorJitter
    - Normalize
    
- **K-fold 검증**

# 자체 피드백

---

- 데이터셋에 대한 이해를 하지 않음
    
    ⇒ 현재까지 공부했던 방식이 데이터를 그냥 모델에 넣고 정확도가 잘나오길 바라기만 했음
    
    ⇒ 데이터에 대한 이해가 전혀 없어서 Image Augmentation이나 Imbalance 와 같은 데이터가 가지고 있는 근본적인 특성을 알아채지 못함
    

- Image Augmentation을 사용할 때 데이터셋의 특성에 맞게 사용하지 않음
    
    ⇒ `guitar` 클래스를 제외한 나머지 클래스들은 위아래가 명확하기 때문에 HorizontalFlip는 사용하면 안됐고 VerticalFlip을 사용했어야 했음.
    
    ⇒ Rotate는 좁은 범위 내에서만 사용했어야 했음.
    
- Data Imbalance 문제 해결 못함
    
    ⇒ class별 데이터 개수에 맞춰서 loss weight를 줬어야했음
    
- EfficientNet는 성능이 좋지 못했음
    
    ⇒ B0 전체를 학습할 때는 성능이 좋지 못했음 (정확도 70% 초반)
    
    ⇒ Backbone 4 stage 까지 freeze 시키고 5 stage 부터 학습 할 때 성능이 더 잘 나왔음
    (정확도 70% 후반)
    
- 학습 진행과정을 분석하지 않음
    
    ⇒ Tensorboard와 같은 학습 시각화 툴을 사용하지 않고 학습 중 출력되는 acc와 loss 만 관찰함
    
    ⇒ 시각화 툴이 없으니 hyper-parameter가 학습에 어떤 영향을 끼치는지 알 수 없었음.
    

- learning rate 값이나 learning rate scheduler 혹은 optimizer 와 같은 중요 parameter를 거의 고정해서 썼음
    
    ⇒ 시각화 툴을 이용해서 분석했어야함