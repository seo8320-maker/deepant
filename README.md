# DeepAnt 오픈소스 코드 분석

## 1. 선택 논문

- 논문명: DeepAnT: A Deep Learning Approach for Unsupervised Anomaly Detection in Time Series
- 저자: Mohsin Munir, Shoaib Ahmed Siddiqui, Andreas Dengel, Sheraz Ahmed
- 학술지: IEEE Access, Volume 7, pp. 1991–2005, 2019
- DOI: 10.1109/ACCESS.2018.2886457

## 2. 원본 오픈소스

- Original GitHub: https://github.com/datacubeR/DeepAnt
- 본 저장소는 위 오픈소스를 기반으로 직접 실행 및 코드 분석한 내용을 정리한 것입니다.

## 3. 실행 환경

본 코드는 Google Colab에서 실행하였습니다.

- Python 3
- PyTorch
- PyTorch Lightning
- torchinfo
- pandas
- matplotlib

## 4. Google Colab 실행 방법

Google Colab에서 아래 명령어를 먼저 실행합니다.

```python
!git clone https://github.com/seo8320-maker/deepant.git
%cd deepant/DeepAnt-master
!pip install pytorch-lightning torchinfo
```

그다음 `DeepAnt.ipynb` 파일을 위에서부터 순서대로 실행합니다.

주의할 점은 반드시 `pytorch_lightning`과 `torchinfo`를 설치한 뒤 import 셀을 실행해야 한다는 것입니다.

## 5. VS Code / 로컬 실행 방법

VS Code에서 실행할 경우, 기존 Python 환경에서 PyTorch 관련 DLL 오류가 발생할 수 있으므로 새 conda 환경을 만드는 것을 권장합니다.

```powershell
conda create -n deepant python=3.10 -y
conda activate deepant
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install pytorch-lightning torchinfo pandas matplotlib scikit-learn jupyter ipykernel
python -m ipykernel install --user --name deepant --display-name "deepant"
```

이후 VS Code에서 `DeepAnt-master/DeepAnt.ipynb` 파일을 열고, 커널을 `deepant`로 선택한 뒤 위에서부터 순서대로 실행합니다.

실행 전 현재 작업 폴더에 아래 파일들이 있는지 확인합니다.

```python
import os
print(os.getcwd())
print(os.listdir())
```

출력 목록에 다음 파일들이 보여야 합니다.

```text
DeepAnt.ipynb
deepant.py
utils.py
data
checkpoints
```

## 6. 실행 시 발생한 오류 및 수정사항

본 오픈소스를 실행하는 과정에서 몇 가지 환경 관련 오류가 발생하였습니다.  
아래 수정사항은 모델 구조나 알고리즘을 변경한 것이 아니라, 실행 환경에 맞추기 위한 설정 변경입니다.

### 1) `ModuleNotFoundError: No module named 'pytorch_lightning'`

Colab 기본 환경에 `pytorch_lightning`이 설치되어 있지 않아 발생한 오류입니다.

해결 방법:

```python
!pip install pytorch-lightning torchinfo
```

설치가 끝난 뒤 import 셀을 다시 실행하면 됩니다.

### 2) `ModuleNotFoundError: No module named 'deepant'`

`deepant.py` 파일이 있는 폴더에서 실행하지 않아 발생한 오류입니다.  
GitHub 저장소를 clone한 뒤 반드시 `deepant.py`가 있는 폴더로 이동해야 합니다.

본인 저장소를 clone한 경우:

```python
%cd deepant/DeepAnt-master
```

현재 폴더 확인 방법:

```python
import os
print(os.listdir())
```

출력 목록에 `deepant.py`가 있어야 합니다.

### 3) `[WinError 126] torch_python.dll`

Windows Anaconda 환경에서 다음과 같은 오류가 발생할 수 있습니다.

```text
[WinError 126] 지정된 모듈을 찾을 수 없습니다.
Error loading ... torch\lib\torch_python.dll
```

이는 DeepAnt 코드 문제가 아니라 로컬 Windows/Anaconda 환경의 PyTorch 설치 문제입니다.  
본 분석에서는 Google Colab에서 정상 실행을 확인하였으며, 로컬 실행 시에는 새 conda 환경을 만들고 CPU 버전 PyTorch를 재설치하는 것을 권장합니다.

### 4) `MisconfigurationException: No supported gpu backend found!`

Colab에서 GPU가 활성화되지 않은 상태에서 원본 코드가 `accelerator="gpu"`로 설정되어 있어 발생한 오류입니다.

원본 코드:

```python
trainer = pl.Trainer(
    max_epochs=30,
    accelerator="gpu",
    devices=1,
    callbacks=[mc]
)
```

GPU를 사용할 경우 Colab에서 다음과 같이 설정합니다.

```text
런타임 → 런타임 유형 변경 → 하드웨어 가속기 → T4 GPU
```

GPU를 사용하지 않을 경우 아래처럼 CPU로 수정하여 실행할 수 있습니다.

```python
trainer = pl.Trainer(
    max_epochs=30,
    accelerator="cpu",
    devices=1,
    callbacks=[mc]
)
```

또는 실행 환경에 따라 자동 선택되도록 다음과 같이 수정할 수 있습니다.

```python
trainer = pl.Trainer(
    max_epochs=30,
    accelerator="auto",
    devices=1,
    callbacks=[mc]
)
```

## 7. 주요 코드 실행 흐름

1. GitHub 저장소 clone
2. 라이브러리 및 내부 모듈 import
3. 데이터 로딩
4. `TrafficDataset`으로 시계열 window 생성
5. `DeepAnt` 모델 생성
6. `AnomalyDetector`와 `DataModule` 구성
7. `Trainer`로 모델 학습
8. Checkpoint 저장 및 불러오기
9. Prediction loss 계산
10. Threshold 기반 이상탐지
11. 결과 시각화

## 8. 코드 분석 요약

DeepAnt는 과거 10개 시점의 시계열 값을 입력으로 받아 다음 시점 값을 예측하는 CNN 기반 모델입니다.

모델은 과거 window 데이터를 입력으로 받아 다음 시점 값을 예측하고, 실제값과 예측값의 차이인 `prediction loss`를 계산합니다. 정상 구간에서는 예측값과 실제값의 차이가 작지만, 이상 구간에서는 정상 패턴에서 벗어나기 때문에 prediction loss가 커집니다.

본 분석에서는 이 prediction loss를 anomaly score로 사용하고, threshold를 초과하는 시점을 이상 후보로 판단하였습니다.

## 9. 주요 실행 결과

원본 시계열 데이터 확인

아래 그래프는 DeepAnt 예제 데이터인 `Travel Time` 시계열을 시각화한 결과입니다.

- x축은 `timestamp`, y축은 `Travel Time` 값을 의미합니다.
- 전체적으로 낮은 값이 반복되지만, 특정 시점에서 값이 급격히 증가하는 spike가 나타납니다.
- DeepAnt는 과거 10개 시점의 값을 입력으로 사용하여 다음 시점 값을 예측합니다.
- 이후 실제값과 예측값의 차이인 `prediction loss`가 크게 발생하는 구간을 이상 후보로 판단합니다.
- 
<img width="1227" height="531" alt="image" src="https://github.com/user-attachments/assets/4fd01e73-3dd3-494c-8b2c-3167e4cb2b4a" />


 DeepAnt 모델 구조 확인

아래 출력은 `summary(model)`을 통해 DeepAnt 모델 구조를 확인한 결과입니다.

- 모델은 2개의 `Conv1d` 기반 블록으로 시간축 특징을 추출합니다.
- 각 Convolution 블록은 `Conv1d → ReLU → MaxPool1d` 구조로 구성되어 있습니다.
- 이후 `Flatten`을 통해 특징을 1차원으로 변환합니다.
- `Linear → ReLU → Dropout → Linear` 구조를 거쳐 다음 시점 값을 예측합니다.
- 전체 학습 가능한 파라미터 수는 4,593개입니다.

즉, DeepAnt는 과거 시계열 window를 입력으로 받아 1D CNN을 통해 시간 패턴을 추출하고, 최종적으로 다음 시점의 값을 예측하는 구조입니다.


<img width="781" height="637" alt="image" src="https://github.com/user-attachments/assets/0f6b4d93-804b-464f-be7c-e65110d4596e" />



학습 설정 확인

아래 출력은 PyTorch Lightning `Trainer`로 학습을 시작할 때 모델과 loss function 구성을 확인한 결과입니다.

- 학습 대상 모델은 `DeepAnt`입니다.
- 손실 함수는 `L1Loss`를 사용합니다.
- 학습 가능한 파라미터 수는 약 4.6K입니다.
- `model`과 `criterion`이 모두 train mode로 설정되어 있음을 확인할 수 있습니다.

즉, DeepAnt 모델은 입력 window를 바탕으로 다음 시점 값을 예측하고, 예측값과 실제값의 차이를 `L1Loss`로 계산하여 학습됩니다.


<img width="693" height="344" alt="image" src="https://github.com/user-attachments/assets/88c452c0-52ef-4282-a73e-f106d67fcd97" />


예측 및 Prediction Loss 계산 확인

아래 출력은 학습된 DeepAnt 모델을 이용하여 전체 데이터에 대해 예측을 수행한 결과입니다.

- `trainer.predict(anomaly_detector, dm)`을 통해 전체 데이터에 대한 예측을 수행합니다.
- 출력의 `item[1]` 값을 추출하여 각 시점의 `prediction loss`를 계산합니다.
- `preds_losses`는 시간 index를 가진 Series 형태로 저장되며, 이후 anomaly score로 사용됩니다.
- `Predicting 2152/2152`는 전체 2152개 window 샘플에 대한 예측이 완료되었음을 의미합니다.

즉, 이 단계에서는 학습된 모델이 각 시점의 다음 값을 예측하고, 실제값과의 차이인 prediction loss를 계산합니다. 이 prediction loss가 threshold 기반 이상탐지의 기준값으로 사용됩니다.


<img width="1148" height="338" alt="image" src="https://github.com/user-attachments/assets/99055bf9-1d24-4ade-8563-35fe00217bb5" />



 Loss Distribution 및 Threshold 설정 확인

아래 그래프는 전체 데이터에 대해 계산된 `prediction loss`의 분포를 나타낸 결과입니다.

- x축은 `prediction loss` 값, y축은 해당 loss 값이 나타난 빈도입니다.
- 대부분의 loss는 0 근처에 집중되어 있어, 모델이 대부분의 정상 패턴을 잘 예측했음을 확인할 수 있습니다.
- 오른쪽으로 길게 늘어진 구간은 예측 오차가 상대적으로 큰 시점들을 의미합니다.
- 빨간 점선은 선택한 threshold를 의미합니다.
- threshold보다 큰 prediction loss는 정상 패턴에서 벗어난 값으로 판단하여 이상 후보로 분류합니다.

즉, 이 그래프는 prediction loss의 전체 분포를 확인하고, threshold를 기준으로 이상 후보를 구분하는 과정을 보여줍니다.

<img width="1589" height="812" alt="image" src="https://github.com/user-attachments/assets/cc6240ac-6bec-4fcd-b461-2c94db22ad0d" />


 시간별 Prediction Loss와 Threshold 확인

아래 그래프는 시간에 따른 `prediction loss` 변화와 threshold를 함께 시각화한 결과입니다.

- 파란 선은 각 시점의 `prediction loss`를 의미합니다.
- 빨간 점선은 이상 판단 기준인 threshold를 의미합니다.
- 대부분의 구간에서는 prediction loss가 낮게 유지됩니다.
- 일부 시점에서는 loss가 급격히 증가하는 spike가 발생합니다.
- prediction loss가 threshold를 초과하는 시점은 이상 후보로 판단합니다.

즉, 이 그래프는 DeepAnt가 계산한 anomaly score가 시간에 따라 어떻게 변화하는지 보여주며, threshold를 기준으로 이상 구간을 탐지하는 과정을 나타냅니다.

<img width="1214" height="648" alt="image" src="https://github.com/user-attachments/assets/25c8d755-4c12-4626-a070-dc164fe21c36" />



 최종 이상탐지 결과 시각화

아래 그래프는 원본 `Travel Time` 시계열 위에 DeepAnt가 이상으로 탐지한 지점을 표시한 결과입니다.

- 파란 선은 원본 `Travel Time` 시계열 데이터입니다.
- 빨간 점은 DeepAnt가 이상으로 판단한 시점입니다.
- prediction loss가 threshold를 초과한 시점이 빨간 점으로 표시됩니다.
- 주로 travel time 값이 급격히 증가하는 spike 구간에서 이상이 탐지되었습니다.

즉, DeepAnt는 원본 시계열의 정상 패턴을 학습한 뒤, 예측 오차가 크게 발생한 시점을 이상 후보로 시각화합니다.

<img width="1614" height="682" alt="image" src="https://github.com/user-attachments/assets/16079de6-cacd-437f-adbd-efc98a288ef4" />



## 10. 캡스톤디자인과의 연관성

DeepAnt 예제 데이터는 실제 스마트팩토리 데이터는 아니지만, 알고리즘 구조는 모터 전류, 진동, 온도, 압력, PLC 센서값과 같은 단일 센서 시계열 이상탐지에 적용할 수 있습니다.

스마트팩토리에서는 설비의 정상 운전 데이터가 시간에 따라 연속적으로 수집됩니다. DeepAnt 방식은 정상 패턴을 학습한 뒤, 실제 센서값과 예측값의 차이가 커지는 시점을 이상 징후로 판단할 수 있습니다. 따라서 스마트팩토리 설비 상태 모니터링 및 예지보전의 기초 모델로 활용 가능성이 있습니다.

## 11. 한계 및 개선 방향

본 오픈소스는 실행과 코드 분석이 비교적 용이하다는 장점이 있지만, 다음과 같은 한계가 있습니다.

- 예제 데이터가 실제 스마트팩토리 또는 PLC 공정 데이터는 아닙니다.
- 단변량 시계열 데이터 중심의 구현입니다.
- 실제 공정에서는 여러 센서 간 상관관계가 중요하므로 다변량 확장이 필요합니다.
- Threshold 설정에 따라 오탐과 미탐이 발생할 수 있습니다.

향후 개선 방향은 다음과 같습니다.

- 다변량 센서 데이터로 확장
- PLC 상태값 및 제어 신호 추가
- Threshold 자동 튜닝
- 실제 스마트팩토리 설비 데이터 기반 검증
