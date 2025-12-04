import os
import json
from datetime import datetime, timedelta
import argparse
import pandas as pd

from preprocessor import preprocess_data


# Forward difference label generator
# WINDOW_LIST defines horizons (in minutes)
# For each symbol & base minute t, compute diff = close(t + window) - close(t)
# Then cross-sectional neutralize: diff - mean(diff across universe at base t)
# Output rows for target date base times only.

WINDOW_LIST = [1, 5, 15, 30, 60, 240]


def _compute_symbol_forward_diff(g: pd.DataFrame, window: int, price_col: str) -> pd.Series:
  """Compute forward difference close(t+window) - close(t) for a single symbol.
  Assumes g sorted by start_time_ms and equally spaced 1m bars.
  Uses shift(-window) to access future close; returns NaN if future bar missing.
  """
  g = g.sort_values("start_time_ms")
  future = g[price_col].shift(-window)
  return (future - g[price_col]) / g[price_col]


def y_generator(
  date_str: str,
  universe: list[str],
  data_dir: str = "data/1m_raw_data",
  price_col: str = "close",
  window_list: list[int] = None,
) -> pd.DataFrame:
  """Generate forward diff neutralized labels for `date_str`.

  For each base minute t (in target_date) and each window w in window_list:
    raw_diff_w(t, symbol) = close_{t+w} - close_t
    y_w(t, symbol) = raw_diff_w(t, symbol) - mean_symbol[ raw_diff_w(t, :) ]

  Data requirements: we load current day + next day to have forward horizons.
  Rows near day end lacking future data produce NaN and are dropped per window.

  Returns DataFrame with columns:
    symbol, start_time_ms, {w}_y for each window.
  """

  if window_list is None:
    window_list = WINDOW_LIST

  target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
  next_date_str = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")

  cur_path = os.path.join(data_dir, f"{date_str}.h5")
  next_path = os.path.join(data_dir, f"{next_date_str}.h5")

  paths = []
  if os.path.exists(cur_path):
    paths.append(cur_path)
  if os.path.exists(next_path):
    paths.append(next_path)
  if not paths:
    raise FileNotFoundError(f"no h5 files for {date_str} or {next_date_str}")

  df_list = [pd.read_hdf(p) for p in paths]
  df = pd.concat(df_list, ignore_index=True)

  df = df[df["symbol"].isin(universe)].copy()

  # Use number of loaded days for validation
  df = preprocess_data(df, num_dates=len(paths))
  df.sort_values(["symbol", "start_time_ms"], inplace=True)

  by_sym = df.groupby("symbol", group_keys=False)

  diff_cols = []
  for w in window_list:
    col = by_sym.apply(lambda g: _compute_symbol_forward_diff(g, w, price_col), include_groups=False)
    col.name = f"{w}_diff"
    diff_cols.append(col)

  diff_df = pd.concat(diff_cols, axis=1)

  # Limit to base times within target_date only
  base = pd.concat([df[["symbol", "start_time_ms"]], diff_df], axis=1)
  dt = pd.to_datetime(base["start_time_ms"], unit="ms", utc=True)
  base = base[dt.dt.date == target_date].copy()

  # Cross-sectional mean subtraction per timestamp for each window
  for w in window_list:
    raw_col = f"{w}_diff"
    y_col = f"y_{w}m"
    cs_mean = base.groupby("start_time_ms")[raw_col].transform("mean")
    base[y_col] = base[raw_col] - cs_mean
    base.drop(columns=[raw_col], inplace=True)

  return base


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--date", type=str, required=True, help="YYYY-MM-DD (base date)")
  parser.add_argument("--top", type=int, required=True, help="universe size (e.g. 30 -> use top30_universe.json)")
  parser.add_argument("--data_dir", type=str, default="data/1m_raw_data")
  parser.add_argument("--price_col", type=str, default="close", help="price column for forward diff (default: close)")
  parser.add_argument("--windows", type=str, default="1,5,15,30,60,240", help="comma-separated forward horizons in minutes")
  args = parser.parse_args()

  universe_file = f"top{args.top}_universe.json"
  if not os.path.exists(universe_file):
    raise FileNotFoundError(universe_file)

  with open(universe_file, "r") as f:
    universe_map = json.load(f)

  if args.date in universe_map:
    universe = universe_map[args.date]
  else:
    # Fallback: derive universe from available data if JSON lacks the date
    cur_h5 = os.path.join(args.data_dir, f"{args.date}.h5")
    if not os.path.exists(cur_h5):
      raise KeyError(f"{args.date} not found in {universe_file} and no data file {cur_h5} to infer universe")
    df_day = pd.read_hdf(cur_h5)
    if "symbol" not in df_day.columns:
      raise KeyError(f"Cannot infer universe from {cur_h5}: 'symbol' column missing")
    # Use all symbols present; optionally cap to top size if more than requested
    symbols = df_day["symbol"].dropna().astype(str).unique().tolist()
    if len(symbols) > args.top:
      # Keep first N in sorted order for determinism
      symbols = sorted(symbols)[:args.top]
    universe = symbols
  windows = [int(x) for x in args.windows.split(",") if x.strip()]

  y = y_generator(args.date, universe, data_dir=args.data_dir, price_col=args.price_col, window_list=windows)

  out_dir = os.path.join("data", "y")
  os.makedirs(out_dir, exist_ok=True)
  out_path = os.path.join(out_dir, f"{args.date}_y_top{args.top}.h5")
  y.to_hdf(out_path, key="y", mode="w")
  print(f"Saved y to {out_path} ({len(y)} rows)")

# python y_generator.py --date 2025-03-01 --top 30 --data_dir 1m_raw_data
# python y_generator.py --date 2025-03-01 --top 30 --price_col close