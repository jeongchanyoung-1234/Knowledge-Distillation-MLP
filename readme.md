# Knowledge Distillation
- MNIST 데이터셋에 대해 MLP를 이용하여 정보 증류를 적용해보는 실험이 가능합니다.
- 정보 증류란 큰 모델의 출력을 soft label로 받아 실제 라벨(hard label)과 함께 학습하는 것을 의미합니다. 이런 식으로 큰 모델의 지식을 상대적으로 작은 모델에 전수할 수 있습니다.

## Quick Start
```buildoutcfg
python train.py [--params]
```

1. batch_size: 미니배치 사이즈 (default=128)
2. hidden_size: 은닉층의 크기, 기본적으로 2 layered Perceptron입니다. (default=1200) 
3. dropout_p: default=.8)
4. alpha: alpha가 작을수록 hard label에, 클수록 soft label에 집중합니다. 0.5 이하를 권합니다. (default=.1)
5. device: cuda 또는 cpu를 사용합니다. (default='cpu')
6. temperature: KDLoss에서 T를 설정합니다. 높은 값일수록 soft label의 엔트로피가 높아집니다. (default=20)
7. lr: 학습율 (default=1e-2)
8. weight_decay: l2 가중치 감쇄를 적용합니다. 사실 드랍아웃만으로 충분합니다. (default=0)
9. epoch: 학습 에포크 (default=10)
10. teacher_model_pth: teacher model을 학습하여 저장한 경로를 입력하면, kd 학습이 시작됩니다. None일 시 단일 모델학습입니다.(default=None)
11. save_model: 입력하면 훈련이 끝난 모델을 저장합니다.



## Experiment Result
- Teacher model의 파라미터는 다음과 같습니다. (대부분 논문의 수치를 사용했습니다.)
- *batch_size=128, dropout=0.8, epoch=10, hidden_size=1200, lr=0.01*
- optimizer는 SGD에 momentum 0.9를 주어 유지합니다.
- 이때 논문에서 제시한 정확도는 약 98.3이며, 실험에서도 98.29로 비슷한 수치가 나왔습니다.


- Student model의 파라미터는 다음과 같습니다.
- *batch_size=128, dropout=0.8, epoch=10, hidden_size=800, lr=0.1*
- KD을 사용하지 않았을 때 논문에서 제시한 정확도는 97.95이며 저는 약간 더 낮은 97.78이 나왔습니다. 조금 더 학습을 돌리면 올라갈 것 같긴 하지만 10에포크에서 멈추었습니다.


- 반면 KD를 사용했을 때의 훈련 파라미터는 다음과 같으며
- *alpha=0.1, temperature=20*
- 논문에서 약 98.5를 제시하였고 저는 98.23의 정확도가 나왔습니다. 세팅이 다른 모양인데 우선 small model으로
big model과 거의 비슷한 수준의 정확도를 이끌어내는 점은 확인했습니다.
