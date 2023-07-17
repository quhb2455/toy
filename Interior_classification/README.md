<aside>
📌 5월에 대회가 끝났고 7월에 정리했기 때문에 코드가 동작하지 않을 수 있음

</aside>

### 주최  / 주관

![주최 : 한솔데코](image/Untitled.png)

주최 : 한솔데코

![주관 : 데이콘](image/Untitled%201.png)

주관 : 데이콘

### 주제

도배 하자의 유형 분류 AI 모델 개발

### 설명

총 19가지의 도배 하자 유형을 분류하는 AI 모델을 개발****

### 데이터 특징

```jsx
반점 : 3
틈새과다 : 5
가구수정 : 12
녹오염 : 14
이음부불량 : 17
울음 : 22
창틀,문틀수정 : 27
피스 : 51
들뜸 : 54
석고수정 : 57
면불량 : 99
몰딩수정 : 130
오타공 : 142
곰팡이 : 145
터짐 : 162
꼬임 : 210
걸레받이수정 : 307
오염 : 595
훼손 : 1405
```

- Class Imbalance 심각
- 이미지의 특징이 확실하지 않고, 클래스간 유사도가 높음
    - 전문가가 아니면 구분하기 어려움
- 전체 이미지에서 특징 부분이 아주 작은 경우 존재
- 특징 부분이 이미지 중앙에 있지 않고 잘려있거나 흔들려 찍힌 경우 존재
- 회전 시 클래스가 변경되는 경우 존재

### 시도 해봤던 것들

1. Zero-shot Classification
2. Edge detection을 이용한 Augmentation
    
    https://github.com/shomnathsomu/crack-detection-opencv
    
    - Canny filter
    - Median filter
    - 성능 변화 없음
        - Edge detection이 잘 작용하는 데이터가 있고 반대인 데이터도 있음
3. Crack Segmentation 사용
    
    https://github.com/khanhha/crack_segmentation
    
    - 성능이 너무 랜덤임, 크랙은 좀 찾는데 면불량은 못찾음
4. Prompt Engineering 관점에서 AutoEncoder model 사용
5. Offline Augmentation
6. LSB swap Augmentation
7. Segment Anything Model (SAM) 을 이용한 Augmentation
8. Ensemble
9. Multi Backbone
    - CNN + Transformer
10. Multi Head Classification
    - Backbone + class 개수 만큼 Classfier
11. Sigmoid Labeling

### 진행 방식

1. 클래스의 특징을 고려해 Offline - Augmentation 진행
    - 학습 전 데이터 벨런스를 맞춰줌
2. 이미 데이터에 Augmentation이 적용되어 있기 때문에 학습 중에는 약한 augmentation만 진행
    - 강한 Augmentation을 줄 시 Augmentation이 겹치면서 데이터가 뭉개지고 모델이 학습하지 못함
3. EfficientNet-B4 모델 학습
4. 학습된 모델로 학습 데이터 라벨링 진행
    - GT를 1로 두고 모델 prediction에 sigmoid를 적용한 sigmoid score로 나머지 데이터에 라벨링 진행
        
        ex) GT → [0, 0, 1, 0] 
        
        Pred → [0.1, 0.2, 0.4, 0.3]
        
        sigmoid → [0.5250, 0.5498, 0.5987, 0.5744] 
        
        x 0.5 → [0.2625, 0.2749, 0.3450, 0.2872]
        
        sigmoid label →[0.2625, 0.2749, 1, 0.2872]
        
5. sigmoid labeling이 된 데이터로 다시 Scratch training 진행
    - 학습된 모델에 이어서 학습하게되면 overtfitting됨
6. 4, 5번 step을 5번 반복

### Best commit id

`dcaa4d1184c8195313a31c3c58bfca2ee92ae7e0`

### Offline Augmentation

`augmentation_experiments.ipynb`에서 **`Offline Augmentations`** cell을 실행

### Sigmoid Labeling Training

```jsx
python sigmoid_labeling_training.py
```

- 코드 내부에 있는 config 값 변경 후 진행

### 개인적 생각

- Scratch 학습인데 왜 성능이 조금씩 증가하는지 모르겠음
    - step을 반복해서 진행하다보면 Sigmoid score가 개선되는 걸 볼 수 있었음
- 이미지 내의 특징점이 여러 곳으로 퍼져있을 때 어떤 식으로 Augmentation을 진행해야할지 모르겠음