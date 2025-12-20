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

  g = g.sort_values("start_time_ms")
  roll = g[["open", "high", "low", "close"]].rolling(window=window, min_periods=window)

  o_first = roll["open"].apply(lambda x: x.iloc[0])
  c_last = roll["close"].apply(lambda x: x.iloc[-1])
  h_max = roll["high"].max()
  l_min = roll["low"].min()

  o2c = (c_last / o_first - 1.0) * 10000
  o2h = (h_max / o_first - 1.0) * 10000
  o2l = (l_min / o_first - 1.0) * 10000
  h2c = (c_last / h_max - 1.0) * 10000
  l2c = (c_last / l_min - 1.0) * 10000

  log_ret = np.log(g["close"] / g["close"].shift(1)).fillna(0)
  rvol = log_ret.rolling(window=window, min_periods=window).std() * 10000

  direction = (g["close"] - g["close"].shift(window)).abs()
  volatility = g["close"].diff().abs().rolling(window=window, min_periods=window).sum()
  eff_ratio = direction / volatility.replace(0, 1.0)

  if "mt_oi" in g.columns:
      oi_p_corr = g["close"].rolling(window=window, min_periods=window).corr(g["mt_oi"]).fillna(0)
  else:
      oi_p_corr = pd.Series(0.0, index=g.index)

  price_diff = g["close"].diff().fillna(0)
  volume = g["quote_volume"]
  
  raw_force = (price_diff * volume).rolling(window=window, min_periods=window).mean()
  vol_mean = volume.rolling(window=window, min_periods=window).mean().replace(0, 1.0)

  force_idx = raw_force / vol_mean

  mavg = roll["close"].mean()
  mstd = roll["close"].std()
  upper = mavg + 2 * mstd
  lower = mavg - 2 * mstd
  
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

  df_list = [pd.read_hdf(p) for p in paths]
  df = pd.concat(df_list, ignore_index=True)
  df = df[df["symbol"].isin(universe)].copy()
  df = preprocess_data(df, num_dates=2)
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

    for col in factor_cols:
      group_mean = full.groupby("start_time_ms")[col].transform("mean")
      group_std = full.groupby("start_time_ms")[col].transform(lambda x: x.std(ddof=0))
      group_std = group_std.replace(0, 1.0)
      full[f"{col}_neut"] = (full[col] - group_mean) / group_std

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

# 시각화 
'''
import matplotlib.pyplot as plt
# ==========================================
# 1. 설정 (Settings)
# ==========================================
symbol = "BTCUSDT"        # 시각화할 심볼
date_str = "2025-03-01"   # 날짜
top_n = 50
x_path = f"data/x/{date_str}_x_top{top_n}.h5"  # 파일 경로

# ==========================================
# 2. 데이터 로드
# ==========================================
if os.path.exists(x_path):
    # X 데이터 로드
    df_x = pd.read_hdf(x_path, key="x")
    
    # 특정 심볼 필터링 & 시간순 정렬
    df_plot = df_x[df_x["symbol"] == symbol].copy()
    df_plot = df_plot.sort_values("start_time_ms")
    
    # 시간 축 변환 (ms -> datetime)
    df_plot["datetime"] = pd.to_datetime(df_plot["start_time_ms"], unit="ms")

    # ==========================================
    # 3. 시각화 (Plotting)
    # ==========================================
    # 60분 윈도우 컬럼명 정의
    features = [
        "close",             # 종가
        "x_60m_RVol",        # 실현 변동성
        "x_60m_EffRatio",    # 효율성 지수
        "x_60m_OI_P_Corr",   # OI-가격 상관계수
        "x_60m_Force",       # 포스 인덱스
        "x_60m_PctB"         # 볼린저 %B
    ]
    
    titles = [
        f"Close Price ({symbol})", 
        "Realized Volatility (Risk)", 
        "Efficiency Ratio (Trend Quality)", 
        "OI-Price Correlation (Trend Strength)", 
        "Force Index (Volume Pressure)", 
        "Bollinger %B (Relative Position)"
    ]
    
    colors = ["#333333", "blue", "green", "purple", "orange", "red"]

    # 캔버스 설정 (6행 1열)
    fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(12, 20), sharex=True)

    for i, ax in enumerate(axes):
        col_name = features[i]
        
        if col_name in df_plot.columns:
            ax.plot(df_plot["datetime"], df_plot[col_name], color=colors[i], linewidth=1.5)
            ax.set_title(titles[i], fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend([col_name], loc='upper right')
            
            # 상관계수나 Force처럼 0이 기준인 지표는 기준선 추가
            if "Corr" in col_name or "Force" in col_name:
                ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
                
            # %B는 0과 1 기준선 추가
            if "PctB" in col_name:
                ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
                ax.axhline(1, color='gray', linestyle=':', alpha=0.5)
        else:
            ax.text(0.5, 0.5, f"{col_name} Not Found", ha='center', va='center')

    plt.xlabel("Time (UTC)")
    plt.tight_layout()
    plt.show()
    
else:
    print(f"파일을 찾을 수 없습니다: {x_path}")
'''

#TODO
# add more features + do it with multiprocessing by dates
