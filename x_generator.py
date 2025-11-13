import os
import json
from datetime import datetime, timedelta
from this import d
import pandas as pd
import argparse
from preprocessor import preprocess_data

# data name: {window}_{type}_{neut} ex) 30m_O2C_neut, 30m_O2C, etc.

WINDOW_LIST = [1, 5, 15, 30, 60, 120, 240, 480, 720, 1440]
TYPE_LIST = ["O2C", "O2H", "O2L", "H2C", "L2C"]


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

  # bp(=basis point) 단위 변환 = ×10000
  o2c = (c_last / o_first - 1.0) * 10000
  o2h = (h_max / o_first - 1.0) * 10000
  o2l = (l_min / o_first - 1.0) * 10000
  h2c = (c_last / h_max - 1.0) * 10000
  l2c = (c_last / l_min - 1.0) * 10000

  out = pd.DataFrame(
    {
      f"{window}m_O2C": o2c,
      f"{window}m_O2H": o2h,
      f"{window}m_O2L": o2l,
      f"{window}m_H2C": h2c,
      f"{window}m_L2C": l2c,
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
  df = preprocess_data(df[1440:], num_dates=2)
  print(df.head())
  df.sort_values(["symbol", "start_time_ms"], inplace=True)
  
  
  feat_list = []
  by_sym = df.groupby("symbol", group_keys=False)

  for window in WINDOW_LIST:
    feat_w = by_sym.apply(_compute_ohlc_window_features, window=window, include_groups=False)
    feat_list.append(feat_w)

  feat_df = pd.concat(feat_list, axis=1)
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
