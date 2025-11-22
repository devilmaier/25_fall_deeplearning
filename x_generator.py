import os
import json
from datetime import datetime, timedelta
import argparse

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
import warnings

from preprocessor import preprocess_data

# --- warnings 무시 ---
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PerformanceWarning)

# data name: x_{window}_{type}_{neut} ex) x_30m_O2C_neut, x_30m_O2C, etc.

WINDOW_LIST = [1, 5, 15, 30, 60, 120, 240, 480, 720, 1440]
TYPE_LIST = [
  "O2C", "O2H", "O2L", "H2C", "L2C",
  "H2L_Vol", "OI_Chg", "AvgTrade", "WhaleGap", "NetTaker", "C2VWAP", "Premium",
  # added classical/technical factors
  "RVol", "EffRatio", "OI_P_Corr", "Force", "PctB",
]


def _compute_ohlc_window_features(g: pd.DataFrame, window: int) -> pd.DataFrame:
  """
  심볼 하나에 대해 과거 window 분 동안의 OHLC + 파생 지표 계산.

  • window개 1분봉 기준
    O2C = (C_last / O_first - 1) * 10000
    O2H = (H_max  / O_first - 1) * 10000
    O2L = (L_min  / O_first - 1) * 10000
    H2C = (C_last / H_max  - 1) * 10000
    L2C = (C_last / L_min  - 1) * 10000
    H2L_Vol = (H_max / L_min - 1) * 10000
  """

  g = g.sort_values("start_time_ms")
  cols = g.columns.tolist()

  target_cols = [
    c
    for c in [
      "open",
      "high",
      "low",
      "close",
      "quote_volume",
      "num_trades",
      "taker_buy_quote",
      "mt_oi",
      "mt_top_ls_ratio",
      "mt_ls_ratio_cnt",
      "pm_close",
    ]
    if c in cols
  ]
  roll = g[target_cols].rolling(window=window, min_periods=window)

  # --- OHLC 기반 기본 팩터 ---
  o_first = roll["open"].apply(lambda x: x.iloc[0])
  c_last = roll["close"].apply(lambda x: x.iloc[-1])
  h_max = roll["high"].max()
  l_min = roll["low"].min()

  o2c = (c_last / o_first - 1.0) * 10000
  o2h = (h_max / o_first - 1.0) * 10000
  o2l = (l_min / o_first - 1.0) * 10000
  h2c = (c_last / h_max - 1.0) * 10000
  l2c = (c_last / l_min - 1.0) * 10000
  h2l_vol = (h_max / l_min - 1.0) * 10000

  # --- OI_Chg ---
  if "mt_oi" in cols:
    oi_first = roll["mt_oi"].apply(lambda x: x.iloc[0])
    oi_last = roll["mt_oi"].apply(lambda x: x.iloc[-1])
    oi_chg = (oi_last / oi_first - 1.0).fillna(0.0) * 10000
  else:
    oi_chg = pd.Series(np.nan, index=g.index)

  # --- AvgTrade ---
  if "quote_volume" in cols and "num_trades" in cols:
    avg_trade = roll["quote_volume"].sum() / roll["num_trades"].sum()
  else:
    avg_trade = pd.Series(np.nan, index=g.index)

  # --- WhaleGap ---
  if "mt_top_ls_ratio" in cols and "mt_ls_ratio_cnt" in cols:
    whale_gap = roll["mt_top_ls_ratio"].mean() - roll["mt_ls_ratio_cnt"].mean()
  else:
    whale_gap = pd.Series(np.nan, index=g.index)

  # --- NetTaker ---
  if "taker_buy_quote" in cols and "quote_volume" in cols:
    sum_buy = roll["taker_buy_quote"].sum()
    sum_total = roll["quote_volume"].sum()
    sum_sell = sum_total - sum_buy
    net_taker = (sum_buy - sum_sell) / sum_total
  else:
    net_taker = pd.Series(np.nan, index=g.index)

  # --- C2VWAP ---
  if "quote_volume" in cols:
    tp = (g["high"] + g["low"] + g["close"]) / 3.0
    pv = tp * g["quote_volume"]
    sum_pv = pv.rolling(window=window, min_periods=window).sum()
    sum_vol = roll["quote_volume"].sum()
    vwap = sum_pv / sum_vol
    c2vwap = (c_last / vwap - 1.0) * 10000
  else:
    c2vwap = pd.Series(np.nan, index=g.index)

  # --- Premium ---
  if "pm_close" in cols:
    premium = roll["pm_close"].mean()
  else:
    premium = pd.Series(np.nan, index=g.index)

  # ============================
  #   새로 추가된 테크니컬 팩터
  # ============================

  # (1) RVol (Realized Volatility): 로그 수익률의 표준편차
  if "close" in cols:
    log_ret = np.log(g["close"] / g["close"].shift(1)).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    rvol = log_ret.rolling(window=window, min_periods=window).std(ddof=0) * 10000
  else:
    rvol = pd.Series(np.nan, index=g.index)

  # (2) EffRatio (Efficiency Ratio): 추세의 직선성
  if "close" in cols:
    direction = (g["close"] - g["close"].shift(window)).abs()
    volatility = g["close"].diff().abs().rolling(window=window, min_periods=window).sum()
    eff_ratio = direction / volatility.replace(0, 1.0)
  else:
    eff_ratio = pd.Series(np.nan, index=g.index)

  # (3) OI_P_Corr (OI-Price Correlation)
  if "mt_oi" in cols and "close" in cols:
    oi_p_corr = g["close"].rolling(window=window, min_periods=window).corr(g["mt_oi"]).fillna(0.0)
  else:
    oi_p_corr = pd.Series(0.0, index=g.index)

  # (4) Force (Force Index): 거래량을 가중한 가격 모멘텀
  if "quote_volume" in cols and "close" in cols:
    price_diff = g["close"].diff().fillna(0.0)
    volume = g["quote_volume"]
    raw_force = (price_diff * volume).rolling(window=window, min_periods=window).mean()
    vol_mean = volume.rolling(window=window, min_periods=window).mean().replace(0, 1.0)
    force_idx = raw_force / vol_mean
  else:
    force_idx = pd.Series(np.nan, index=g.index)

  # (5) PctB (Bollinger %B): 볼린저 밴드 내 상대 위치
  if "close" in cols:
    mavg = g["close"].rolling(window=window, min_periods=window).mean()
    mstd = g["close"].rolling(window=window, min_periods=window).std(ddof=0)
    upper = mavg + 2 * mstd
    lower = mavg - 2 * mstd
    denom = (upper - lower).replace(0, 1.0)
    pct_b = (g["close"] - lower) / denom
    pct_b = pct_b.where((upper - lower) != 0, 0.5)
  else:
    pct_b = pd.Series(0.5, index=g.index)

  # ============================
  #   크로스 섹션 / 레짐 관련 팩터
  # ============================

  by_sym_full = g.groupby("symbol", group_keys=False)

  # Close diff rate per window
  g[f"{window}x_close_diff_rate"] = by_sym_full["close"].pct_change(window).fillna(0.0)

  # OI diff rate per window (없으면 0) — intermediate only
  if "mt_oi" in cols:
    g["x_mt_oi_diff_rate"] = by_sym_full["mt_oi"].pct_change(window).fillna(0.0)
  else:
    g["x_mt_oi_diff_rate"] = 0.0

  # Ratio skew (상위 - 전체 의 포지션 집중도), window 붙인 버전
  if "mt_top_ls_ratio" in cols and "mt_ls_ratio" in cols:
    g[f"{window}x_ratio_skew"] = g["mt_top_ls_ratio"] - g["mt_ls_ratio"]
  else:
    g[f"{window}x_ratio_skew"] = 0.0

  def _rolling_zscore(s: pd.Series, w: int, minp=None) -> pd.Series:
    """
    의도:
      - window가 충분히 크면(window >= 5) 그 window 전체를 써서 z-score
      - window가 너무 작으면(window < 5) 최소 5개 구간으로 강제해서
        너무 noisy한 z-score를 피한다.
    """
    if w < 5:
      window_eff = 5
      minp_eff = 5 if minp is None else min(minp, window_eff)
    else:
      window_eff = w
      if minp is None:
        minp_eff = w
      else:
        minp_eff = min(minp, window_eff)

    r = s.rolling(window=window_eff, min_periods=minp_eff)
    return (s - r.mean()) / (r.std(ddof=0) + 1e-9)

  # Z-scores of ratio skew (window prefix 버전 사용)
  g[f"{window}x_ratio_skew_z"] = by_sym_full[f"{window}x_ratio_skew"].transform(
    lambda s: _rolling_zscore(s, window)
  )

  # Crowding Pressure
  g[f"{window}x_crowding_pressure"] = np.tanh(3.0 * g[f"{window}x_ratio_skew_z"])

  # OI z-score
  if "mt_oi" in cols:
    g[f"{window}x_oi_z"] = by_sym_full["mt_oi"].transform(
      lambda s: _rolling_zscore(s, window)
    )
  else:
    g[f"{window}x_oi_z"] = 0.0

  # Regime & interaction
  g[f"{window}x_price_oi_regime"] = (
    np.sign(g[f"{window}x_close_diff_rate"]) * np.sign(g["x_mt_oi_diff_rate"])
  )
  g[f"{window}x_oi_x_skew"] = g[f"{window}x_oi_z"] * g[f"{window}x_ratio_skew_z"]

  # ============================
  #   최종 DataFrame 구성
  # ============================

  out_main = pd.DataFrame(
    {
      f"{window}m_O2C": o2c,
      f"{window}m_O2H": o2h,
      f"{window}m_O2L": o2l,
      f"{window}m_H2C": h2c,
      f"{window}m_L2C": l2c,
      f"{window}m_H2L_Vol": h2l_vol,
      f"{window}m_OI_Chg": oi_chg,
      f"{window}m_AvgTrade": avg_trade,
      f"{window}m_WhaleGap": whale_gap,
      f"{window}m_NetTaker": net_taker,
      f"{window}m_C2VWAP": c2vwap,
      f"{window}m_Premium": premium,
      f"{window}m_RVol": rvol,
      f"{window}m_EffRatio": eff_ratio,
      f"{window}m_OI_P_Corr": oi_p_corr,
      f"{window}m_Force": force_idx,
      f"{window}m_PctB": pct_b,
    },
    index=g.index,
  )

  # x-쪽 feature들
  base_extra_cols = [f"{window}x_close_diff_rate"]
  mt_extra_cols = [
    f"{window}x_ratio_skew",
    f"{window}x_ratio_skew_z",
    f"{window}x_crowding_pressure",
    f"{window}x_oi_z",
    f"{window}x_price_oi_regime",
    f"{window}x_oi_x_skew",
  ]

  if window == 1:
    # 1분짜리 mt_ 기반 window feature는 feature set에서 제외
    extra_cols = base_extra_cols
  else:
    extra_cols = base_extra_cols + mt_extra_cols

  extra_cols = [c for c in extra_cols if c in g.columns]
  out_extra = g[extra_cols]

  out = pd.concat([out_main, out_extra], axis=1)

  # 1분짜리에서 완전히 드랍할 mt_ 기반 feature들
  if window == 1:
    drop_cols = [
      f"{window}m_OI_Chg",
      f"{window}m_WhaleGap",
      f"{window}m_OI_P_Corr",
      f"{window}x_ratio_skew",
      f"{window}x_ratio_skew_z",
      f"{window}x_crowding_pressure",
      f"{window}x_oi_z",
      f"{window}x_price_oi_regime",
      f"{window}x_oi_x_skew",
    ]
    drop_cols = [c for c in drop_cols if c in out.columns]
    if drop_cols:
      out = out.drop(columns=drop_cols)

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
    raise FileNotFoundError(f"No data for {prev_date_str} or {date_str} in {data_dir}")

  df_list = []
  for p in paths:
    # 무조건 /data 사용
    with pd.HDFStore(p, mode="r") as store:
      if "/data" not in store.keys():
        raise KeyError(f"No '/data' key in {p}, keys={store.keys()}")
    raw = pd.read_hdf(p, key="data")
    mt = None  # metrics 없다고 가정 (있으면 여기서 읽어서 넘기면 됨)

    # universe 먼저 필터
    raw = raw[raw["symbol"].isin(universe)].copy()

    # preprocess_data는 method를 명시해야 함
    merged = preprocess_data(raw, method=mt)
    df_list.append(merged)

  df = pd.concat(df_list, ignore_index=True)

  # universe 필터 한 번 더 (혹시 모를 잔여 심볼 제거)
  df = df[df["symbol"].isin(universe)].copy()

  # 심볼별로 window feature 계산
  feat_list = []
  for window in WINDOW_LIST:
    feat = (
      df.groupby("symbol", group_keys=False)
        .apply(_compute_ohlc_window_features, window=window)
    )
    feat_list.append(feat)

  feat_df = pd.concat(feat_list, axis=1)
  feat_df = feat_df.add_prefix("x_")
  full = pd.concat([df, feat_df], axis=1)

  if add_neutralized:
    factor_cols = [
      c
      for c in full.columns
      if any(c.endswith(t) for t in TYPE_LIST) and not c.endswith("_neut")
    ]

    neut_dict = {}
    grouped = full.groupby("start_time_ms")
    for col in factor_cols:
      group_mean = grouped[col].transform("mean")
      group_std = grouped[col].transform(lambda x: x.std(ddof=0)).replace(0, 1.0)
      neut_dict[f"{col}_neut"] = (full[col] - group_mean) / group_std

    neut_df = pd.DataFrame(neut_dict, index=full.index)
    full = pd.concat([full, neut_df], axis=1)

  # 타깃 날짜만 남기기 (start_time_ms 기준)
  dt = pd.to_datetime(full["start_time_ms"], unit="ms", utc=True)
  full = full[dt.dt.date == target_date].copy()

  return full


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--date", type=str, required=True, help="target date (YYYY-MM-DD)")
  parser.add_argument(
    "--top",
    type=int,
    default=30,
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
  os.makedirs(os.path.dirname(out_path), exist_ok=True)
  x_df.to_hdf(out_path, key="x", mode="w")

  print(f"Saved x to {out_path}")
# python x_generator.py --date 2025-10-01 --top 30 --data_dir data/1m_raw_data