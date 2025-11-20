import os
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import argparse
from preprocessor import preprocess_data

# data name: x_{window}_{type}_{neut} ex) x_30m_O2C_neut, x_30m_O2C, etc.

WINDOW_LIST = [1, 5, 15, 30, 60, 120, 240, 480, 720, 1440]
TYPE_LIST = ["O2C", "O2H", "O2L", "H2C", "L2C", 
             "RVol", "EffRatio", "OI_P_Corr", "Force", "PctB" # 새로 만든 지표
             ]


def _compute_ohlc_window_features(g: pd.DataFrame, window: int) -> pd.DataFrame:
  """
  심볼 하나에 대해 과거 window 분 동안의 OHLC 관계식 계산.

  • window개 1분봉 기준
    O2C = (C_last / O_first - 1) * 10000
    O2H = (H_max  / O_first - 1) * 10000
    O2L = (L_min  / O_first - 1) * 10000
    H2C = (C_last / H_max  - 1) * 10000
    L2C = (C_last / L_min  - 1) * 10000
  """
  """
  추가 지표 : 
  1. RVol (Realized Volatility): 로그 수익률의 표준편차 (리스크 측정)
  2. EffRatio (Efficiency Ratio): 추세의 순도/직선성 (노이즈 필터링)
  3. OI_P_Corr (OI-Price Correlation): 가격과 미결제약정의 상관관계 (추세 진성 여부)
  4. Force (Force Index): 거래량을 가중한 가격 모멘텀 (매수/매도 세력의 힘)
  5. PctB (Bollinger %B): 볼린저 밴드 내 상대적 위치 (Stationarity 확보)
  """

  g = g.sort_values("start_time_ms")
  roll = g[["open", "high", "low", "close"]].rolling(window=window, min_periods=window)

  o_first = roll["open"].apply(lambda x: x.iloc[0])
  c_last = roll["close"].apply(lambda x: x.iloc[-1])
  h_max = roll["high"].max()
  l_min = roll["low"].min()

  # bp(=basis point) 단위 변환 = ×10000
  o2c = (c_last / o_first - 1.0) * 10000
  o2h = (h_max / o_first - 1.0) * 10000
  o2l = (l_min / o_first - 1.0) * 10000
  h2c = (c_last / h_max - 1.0) * 10000
  l2c = (c_last / l_min - 1.0) * 10000

  # (1) RVol (Realized Volatility)
  log_ret = np.log(g["close"] / g["close"].shift(1)).fillna(0)
  rvol = log_ret.rolling(window=window, min_periods=window).std() * 10000

  # (2) EffRatio (Efficiency Ratio)
  direction = (g["close"] - g["close"].shift(window)).abs()
  volatility = g["close"].diff().abs().rolling(window=window, min_periods=window).sum()
  eff_ratio = direction / volatility.replace(0, 1.0) # 0으로 나누기 방지

  # (3) OI_P_Corr (OI-Price Correlation)
  if "mt_oi" in g.columns:
      oi_p_corr = g["close"].rolling(window=window, min_periods=window).corr(g["mt_oi"]).fillna(0)
  else:
      oi_p_corr = pd.Series(0.0, index=g.index)

  # (4) Force (Force Index)
  price_diff = g["close"].diff().fillna(0)
  volume = g["quote_volume"] # 거래량
  
  raw_force = (price_diff * volume).rolling(window=window, min_periods=window).mean()
  vol_mean = volume.rolling(window=window, min_periods=window).mean().replace(0, 1.0)

  force_idx = raw_force / vol_mean

  # (5) PctB (Bollinger %B)
  mavg = roll["close"].mean() # 종가 기준 이동평균
  mstd = roll["close"].std() # 종가 기준 이동표준편차
  upper = mavg + 2 * mstd
  lower = mavg - 2 * mstd
  
  # 밴드 폭이 0일 경우(가격 변동 없음) 0.5(중간)로 처리
  denom = (upper - lower).replace(0, 1.0) 
  pct_b = (g["close"] - lower) / denom
  pct_b = pct_b.where((upper - lower) != 0, 0.5)

  out = pd.DataFrame(
    {
      f"{window}m_O2C": o2c,
      f"{window}m_O2H": o2h,
      f"{window}m_O2L": o2l,
      f"{window}m_H2C": h2c,
      f"{window}m_L2C": l2c,
      f"{window}m_RVol": rvol,
      f"{window}m_EffRatio": eff_ratio,
      f"{window}m_OI_P_Corr": oi_p_corr,
      f"{window}m_Force": force_idx,
      f"{window}m_PctB": pct_b,
    },
    index=g.index,
  )

  return out


def x_generator(
  date_str: str,
  universe: list[str],
  data_dir: str = "data/1m_raw_data",
  add_neutralized: bool = True,
) -> pd.DataFrame:
  """
  date_str의 X를 만들기 위해
  (date_str - 1일, date_str) 두 날짜의 1분 데이터를 읽어 window feature 계산.
  최종 반환은 date_str 날짜의 행만 포함.
  """

  target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
  prev_date_str = (target_date - timedelta(days=1)).strftime("%Y-%m-%d")

  prev_path = os.path.join(data_dir, f"{prev_date_str}.h5")
  cur_path = os.path.join(data_dir, f"{date_str}.h5")

  paths = []
  if os.path.exists(prev_path):
    paths.append(prev_path)
  if os.path.exists(cur_path):
    paths.append(cur_path)

  if not paths:
    raise FileNotFoundError(f"no h5 files for {prev_date_str} or {date_str}")

  # 하루 전 + 당일 데이터 concat
  df_list = [pd.read_hdf(p) for p in paths]
  df = pd.concat(df_list, ignore_index=True)
  print(df.head())
  # universe 필터 (타깃 날짜 기준 topN)
  df = df[df["symbol"].isin(universe)].copy()
  print(df.head())
  df = preprocess_data(df, num_dates=2)
  print(df.head())
  df.sort_values(["symbol", "start_time_ms"], inplace=True)
  
  
  feat_list = []
  by_sym = df.groupby("symbol", group_keys=False)

  for window in WINDOW_LIST:
    feat_w = by_sym.apply(_compute_ohlc_window_features, window=window, include_groups=False)
    feat_list.append(feat_w)

  feat_df = pd.concat(feat_list, axis=1)
  feat_df = feat_df.add_prefix("x_")
  full = pd.concat([df, feat_df], axis=1)
  
  if add_neutralized:
    factor_cols = [
      c for c in full.columns
      if any(c.endswith(t) for t in TYPE_LIST)
      and not c.endswith("_neut")
    ]

    # 시각별 cross-sectional neutralization
    for col in factor_cols:
      group_mean = full.groupby("start_time_ms")[col].transform("mean")
      group_std = full.groupby("start_time_ms")[col].transform(lambda x: x.std(ddof=0))
      group_std = group_std.replace(0, 1.0)
      full[f"{col}_neut"] = (full[col] - group_mean) / group_std

  # >>> 여기서부터가 핵심: 타깃 날짜만 남기기 (start_time_ms 기준) <<<
  dt = pd.to_datetime(full["start_time_ms"], unit="ms", utc=True)
  full = full[dt.dt.date == target_date].copy()

  return full


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--date", type=str, required=True, help="YYYY-MM-DD")
  parser.add_argument(
    "--top",
    type=int,
    required=True,
    help="universe size (e.g. 30 -> use top30_universe.json)",
  )
  parser.add_argument("--data_dir", type=str, default="data/1m_raw_data")
  args = parser.parse_args()

  universe_file = f"top{args.top}_universe.json"
  if not os.path.exists(universe_file):
    raise FileNotFoundError(universe_file)

  with open(universe_file, "r") as f:
    universe = json.load(f)

  x_df = x_generator(args.date, universe[args.date], data_dir=args.data_dir)
  out_path = os.path.join("data/x", f"{args.date}_x_top{args.top}.h5")
  x_df.to_hdf(out_path, key="x", mode="w")
  
  print(f"Saved x to {out_path}")
# python x_generator.py --date 2025-10-01 --top 30 --data_dir data/1m_raw_data


#TODO
# add more features + do it with multiprocessing by dates
