# 2025 Fall Deep Learning Project: Crypto Price Prediction

## 📊 데이터셋 (Dataset)

### 1. 유니버스 선정 (`universe_builder.py`)
- **기준**: USDT 기준 거래대금(Quote Volume) 상위 N개 종목 (예: Top 50).
- **필터링**: 데이터 누락이 심하거나 특정 조건(예: 스테이블 코인 등)에 해당하는 종목은 자동으로 제외됩니다.
- **출력**: 날짜별로 학습 대상이 되는 종목 리스트가 담긴 JSON 파일.

### 2. 입력 변수 (`x_generator.py`) - Input Features
모델은 다양한 시간 단위(1분, 5분, 15분, 30분, 60분 등)의 과거 데이터를 요약한 **윈도우 기반 Feature**를 사용합니다. 모든 Feature는 특정 시점(t)의 전체 종목에 대해 Z-score 정규화(Neutralization)를 수행하여, 시장 전체의 등락 영향을 배제하고 종목 간의 상대적 우위를 학습합니다.

#### 기본 지표 (Price Ratios)
- **O2C / O2H / O2L**: 시가 대비 종가/고가/저가 비율
- **H2C / L2C**: 고가/저가 대비 종가 위치 비율
- **H2L_Vol**: 고가와 저가의 차이 (변동성 크기, 수축/확장 탐지)

#### 고급 지표 (Market Microstructure & Volatility)

| 지표명 (Feature) | 설명 (Description) | 활용 로직 (Logic) |
| :--- | :--- | :--- |
| **RVol** (Realized Volatility) | **실현 변동성** | 로그 수익률의 표준편차를 통해 실제 시장의 리스크 수준을 측정합니다. |
| **EffRatio** (Efficiency Ratio) | **효율성 지수** | 추세가 얼마나 직선적인지(순도)를 측정합니다. (1에 가까울수록 노이즈 없는 추세) |
| **OI_P_Corr** (OI-Price Correlation) | **OI-가격 상관계수** | 가격 상승 시 미결제약정(OI)이 동반 상승하는지 확인하여 추세의 진정성을 판단합니다. |
| **Force** (Force Index) | **포스 인덱스** | 가격 변동분에 거래량을 가중하여 매수/매도 세력의 실질적인 힘을 측정합니다. |
| **PctB** (Bollinger %B) | **볼린저 %B** | 볼린저 밴드 내에서 현재 가격의 상대적 위치를 나타내어 데이터의 정상성(Stationarity)을 확보합니다. |

- **NetTaker (순매수 체결 강도)**: 전체 거래량 중 시장가 매수(공격적 매수)의 우위 비율 (-1 ~ +1)
- **AvgTrade (평균 체결 금액)**: 주문 건당 평균 거래 규모 (큰손/고래의 시장 진입 감지)
- **OI_Chg (미결제약정 변동률)**: 시장에 깔린 판돈(Open Interest)의 증감 (자금 유입=추세지속, 유출=추세반전)
- **WhaleGap (스마트머니 괴리)**: 고래(Top Trader)와 개미(Global)의 롱/숏 비율 차이 (세력 추종 지표)
- **C2VWAP (VWAP 괴리율)**: 거래량 가중 평균가(세력 평단) 대비 현재가의 이격도 (지지/저항 및 과매수 판별)
- **Premium (프리미엄 추세)**: 현물 대비 선물 가격의 괴리 정도 (펀딩비 기반의 시장 과열/침체 판단)

#### 윈도우별 고급 엔지니어링 피처 (Window-Specific Engineered Features)

각 윈도우 크기(1분, 5분, 15분, 30분, 60분 등)에 대해 다음의 feature들을 생성합니다.

| 지표명 | 수식 | 설명(Description) | 활용 로직(Logic) |
| :--- | :--- | :--- | :--- |
| **`Close_Diff_Rate`** | $\frac{Close_t - Close_{t-w}}{Close_{t-w}}$ | **윈도우별 가격 변화율** | 해당 윈도우 기간 동안의 수익률을 측정하여 단기/중기/장기 모멘텀을 구분 |
| **`RatioSkew`** | $LS_{top} - LS_{global}$ | **포지션 집중도 왜도** | 고래(상위 트레이더)와 일반 투자자의 롱숏 비율 차이. 값이 클수록 스마트머니가 한쪽으로 쏠림 |
| **`RatioSkew_Z`** | $\frac{ratio\_skew - \mu_w}{\sigma_w}$ | **포지션 집중도 Z-score** | ratio_skew의 롤링 표준화 값. 극단적 값(±2 이상)은 비정상적 포지션 편향을 의미 |
| **`CrowdingPressure`** | $\tanh(3.0 \times ratio\_skew\_z)$ | **군중 압력 지수** | Z-score를 tanh로 압축(-1~+1)하여 과도한 편향 시 반전 신호로 활용 |
| **`OI_Z`** | $\frac{OI_t - \mu_w}{\sigma_w}$ | **미결제약정 Z-score** | OI의 표준화된 수준. 급등(+2 이상) 시 신규 자금 유입, 급감(-2 이하) 시 청산 압력 |
| **`PriceOIRegime`** | $sign(price\_change) \times sign(oi\_change)$ | **가격-OI 체제 판별** | +1: 상승+OI증가(건전한 상승) 또는 하락+OI감소(건전한 하락), -1: 불일치(반전 주의) |
| **`OI_XSkew`** | $oi\_z \times ratio\_skew\_z$ | **OI-포지션 상호작용** | OI 급증과 포지션 쏠림이 동시 발생 시 극단적 값 → 추세 가속 또는 반전 임박 신호 |

**핵심 로직:**
1. **`x_mt_oi_diff_rate`**: OI 변화율은 윈도우별로 계산되지만 중간 변수로만 사용되며 최종 출력에는 포함되지 않습니다.
2. **Rolling Z-score 안정화**: 
   - 윈도우가 5 미만일 때는 강제로 최소 5개 구간을 사용하여 과도한 노이즈 방지
   - 윈도우가 5 이상일 때는 해당 윈도우 전체를 사용하여 정확한 통계 계산
3. **다중 시간대 분석**: 동일한 피처를 여러 윈도우로 생성함으로써 모델이 단기/중기/장기 패턴을 동시에 학습 가능


### 3. 타겟 변수 (`y_generator.py`) - Output Features
모델은 절대적인 수익률이 아닌, **시장 평균 대비 초과 수익(Alpha)**을 예측합니다.

- **방식**: Forward Difference Neutralized
- **수식**: $y_{t, w} = (Close_{t+w} - Close_{t}) - \text{Mean}_{market}(Close_{t+w} - Close_{t})$
- **예측 구간**: 1분, 5분, 15분, 30분, 60분 뒤의 초과 수익.
- **목표**: 향후 $w$분 동안 시장 평균보다 더 많이 오를(혹은 덜 떨어질) 종목을 식별.

---

## 🧠 모델 (Model)

프로젝트에서는 다양한 딥러닝 및 머신러닝 모델을 구현하여 암호화폐 가격 예측을 수행합니다.

### 1. CNN
- **아키텍처**: 1D Convolutional Neural Network
  - Conv1D (32 filters) → MaxPool → Conv1D (64 filters) → MaxPool → Global Average Pooling → Fully Connected
- **입력 형태**: `(Batch, Time, Features)` - 시계열 데이터를 시간 차원으로 처리
- **특징**: 
  - 단일 종목의 시계열 패턴 학습에 특화
  - 경량 모델로 빠른 학습 및 추론 가능
  - 입력 정규화를 모델 내부에서 수행

### 2. CryptoMamba
- **아키텍처**: Mamba-based State Space Model
  - Temporal Mamba Layers (시간 차원 처리) + Spatial Mamba Layers (종목 간 관계 학습)
- **입력 형태**: `(Batch, Time, Nodes, Features)` - 다중 종목의 시공간 데이터
- **특징**:
  - Mamba SSM을 활용한 효율적인 장기 의존성 학습
  - 종목 간 상관관계를 공간 차원에서 모델링
  - 다양한 Loss Function 지원 (MSE, Hybrid, Adaptive, Directional, IC Loss)

### 3. SpatioTemporalTransformer
- **아키텍처**: 1D-CNN + Transformer Encoder
  - Temporal Encoder (1D-CNN) → Spatial Encoder (Transformer) → Prediction Head
- **입력 형태**: `(Batch, Time, Nodes, Features)` - 다중 종목의 시공간 데이터
- **특징**:
  - CNN으로 시간적 패턴 추출 후 Transformer로 종목 간 관계 학습
  - Regression 및 Classification 모드 지원
  - Auxiliary Loss를 통한 다중 태스크 학습

### 4. LightGBM (LGBM)
- **아키텍처**: Gradient Boosting Decision Tree
- **입력 형태**: Tabular Data (Flattened Features)
- **특징**:
  - 전통적인 머신러닝 기법으로 빠른 학습 및 해석 가능성
  - Feature Importance 분석 가능
  - Grid Search를 통한 하이퍼파라미터 최적화 지원

---

## 🚀 사용 방법 (Usage)

### 데이터 준비

#### 1. 유니버스 생성
```bash
python universe_builder.py --start_date 2025-02-01 --end_date 2025-03-01 --top_n 50
```

#### 2. Feature 데이터(X) 생성
```bash
python x_generator.py --date 2025-03-01 --top 50
```

#### 3. Label 데이터(y) 생성
```bash
python y_generator.py --date 2025-03-01 --top 50
```

### 모델 학습

#### CNN 모델
```bash
# 기본 학습 모드
python model/CNN_train.py

# Rolling Window 학습 (5개 윈도우로 반복 학습)
$env:MODE="rolling"; python model/CNN_train.py

# Hyperparameter Search
$env:MODE="search"; python model/CNN_train.py
```

**설정 파일**: `model/CNN_train.py`의 `CONFIG` 딕셔너리에서 날짜 범위, 배치 크기, 학습률 등을 조정할 수 있습니다.

#### CryptoMamba 모델
```bash
python model/CryptoMamba_train.py
```

**설정 파일**: `model/CryptoMamba_train.py`의 `CONFIG`에서 Mamba 레이어 수, hidden dimension, loss function 타입 등을 설정할 수 있습니다.

#### SpatioTemporalTransformer 모델
```bash
python model/SpatioTemporal_train.py
```

**설정 파일**: `model/SpatioTemporal_train.py`의 `CONFIG`에서 Transformer 레이어 수, attention heads, 모드(regression/classification) 등을 설정할 수 있습니다.

#### LightGBM 모델
```bash
# 학습 + 평가
python -m machine_learning.lgbm \
  --mode train_eval \
  --train_start 2025-02-01 --train_end 2025-04-30 \
  --valid_start 2025-05-01 --valid_end 2025-05-14 \
  --test_start  2025-05-15 --test_end  2025-05-28 \
  --y_name y_60m --topn 30 \
  --max_rows 600000 \
  --num_boost_round 4000 \
  --early_stopping_rounds 200

# 추론만 수행
python machine_learning/lgbm.py \
  --mode infer \
  --infer_start 2024-03-02 --infer_end 2024-03-05 \
  --y_name y_60m --topn 30 \
  --model_path /path/to/model.pkl \
  --meta_path  /path/to/meta.json \
  --save_preds
```

#### LGBM Grid Search
```bash
python machine_learning/lgbm_grid.py
```

---

## 📁 디렉토리 구조 (Directory Structure)

```
.
├── data/
│   ├── xy/                        # 통합 Feature + Label 데이터 (날짜별 .h5 파일)
│   └── datasets/                  # 전처리된 학습용 데이터셋
│       ├── cnn/                   # CNN 모델용 데이터셋
│       ├── ml/                    # 머신러닝 모델용 데이터셋
│       └── spatiotfm/             # SpatioTemporal 모델용 데이터셋
│
├── model/                         # 딥러닝 모델 코드
│   ├── CNN.py                     # CNN 모델 정의
│   ├── CNN_train.py               # CNN 학습 스크립트
│   ├── CNN_dataloader.py          # CNN 데이터 로더
│   ├── CryptoMamba.py             # CryptoMamba 모델 정의
│   ├── CryptoMamba_train.py       # CryptoMamba 학습 스크립트
│   ├── CryptoMamba_dataloader.py # CryptoMamba 데이터 로더
│   ├── SpatioTemporalTransformer.py  # SpatioTemporal 모델 정의
│   ├── SpatioTemporal_train.py    # SpatioTemporal 학습 스크립트
│   └── SpatioTemporal_dataloader.py # SpatioTemporal 데이터 로더
│    
│
├── machine_learning/              # 머신러닝 모델 코드
│   ├── lgbm.py                    # LightGBM 학습/추론 스크립트
│   ├── lgbm_grid.py               # LightGBM Grid Search
│   ├── datasets.py                # 데이터셋 빌더
│   └── linear_model.py            # 선형 모델
│
├── models/                        # 학습된 모델 체크포인트 (.pt, .pkl)
│
├── results/                       # 실험 결과
│   ├── cnn/                       # CNN 실험 결과
│   ├── lgbm/                      # LGBM 실험 결과
│   └── sttfm/                     # SpatioTemporal 실험 결과
│
├── feature_list/                  # Feature 리스트 JSON 파일
│   └── y_60m/                     # 60분 예측용 feature 리스트
│
├── universe_builder.py            # 종목 선정 스크립트
├── preprocessor.py                # 데이터 정제 및 검증 모듈
├── x_generator.py                 # Feature 엔지니어링 ($X$ 생성)
├── y_generator.py                 # 타겟 레이블링 ($y$ 생성)
└── README.md
```










