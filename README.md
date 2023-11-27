# Stock_Prediction

## Introduction
주식데이터를 여러가지 방법으로 분석하고 예측하는 프로젝트입니다. `hugging-face`에서 data를 전처리하고 관리하며,여러 데이터를 활용하여 훈련시키고 , 여러 모델을 사용할 수  있는 파이프라인을 만들려고 합니다.


## Data
1. `yfiaince`를  사용하여 주식가격 데이터를 모집합니다.
2. 전처리한 데이터는 `hugging-face`에 업로드합니다.
```
from datasets import load_dataset
dataset = load_dataset("SEHYUN66/data")
```

## Model
1. `pytorch`를 사용하여 모델을 구현합니다.<br/>
2. `hugging-face`에 업로드한 데이터를 사용하여 모델을 훈련시킵니다.

## Trainer
0. `hugging-face`에서 업로드된 데이터에서 input data setting에 맞게 데이터를  불러와 window를 생성합니다.
1. `accelerate`를 사용하여 모델을 훈련시킵니다.<br/>
2. 모델의 훈련과정과 학습평가 파라미터를 `wandb`에 업로드합니다. <br/>

```
./scirpt train_model.sh
```
## Setting
python 3.10 기준의 환경입니다.<br/>
먼저 pytorch gpu를 쓴다면 이거 아래를 명령어로 cuda용 pytorch를 설치합니다.
```
pip3 install torch
```

google tpu 기준
```
pip install torch~=2.1.0 torch_xla[tpu]~=2.1.0 -f https://storage.googleapis.com/libtpu-releases/index.html

```


이후 requirements.txt에 있는 것들 설치하기
```
pip install -r requirements.txt
```

# weight & bias login
```
wandb login 토큰
```