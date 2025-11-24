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

### 3. 타겟 변수 (`y_generator.py`) - Output Features
모델은 절대적인 수익률이 아닌, **시장 평균 대비 초과 수익(Alpha)**을 예측합니다.

- **방식**: Forward Difference Neutralized
- **수식**: $y_{t, w} = (Close_{t+w} - Close_{t}) - \text{Mean}_{market}(Close_{t+w} - Close_{t})$
- **예측 구간**: 1분, 5분, 15분, 30분, 60분 뒤의 초과 수익.
- **목표**: 향후 $w$분 동안 시장 평균보다 더 많이 오를(혹은 덜 떨어질) 종목을 식별.

---

## 🧠 모델 (Model)

- **아키텍처**: 
- **입력 형태**: 
- **특징**:
    

---

## 🚀 사용 방법 (Usage)

### 1. 유니버스 생성
```bash
python universe_builder.py --start_date 2025-02-01 --end_date 2025-03-01 --top_n 50
```
### 2. Feature 데이터(X) 생성
```bash
python x_generator.py --date 2025-03-01 --top 50
```
### 3. Label 데이터(y) 생성
```bash
python y_generator.py --date 2025-03-01 --top 50
```
---
## 📁 디렉토리 구조 (Directory Structure)

```
.
├── data/
│   ├── 1m_raw_data/       # 원본 OHLCV 데이터 (.h5)
│   ├── x/                 # 생성된 Feature 데이터 ($X$)
│   └── y/                 # 생성된 Label 데이터 ($y$)
├── universe_builder.py    # 종목 선정 스크립트
├── preprocessor.py        # 데이터 정제 및 검증 모듈
├── x_generator.py         # Feature 엔지니어링 ($X$ 생성)
├── y_generator.py         # 타겟 레이블링 ($y$ 생성)
└── README.md
```










