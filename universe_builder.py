import os
import json
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import argparse


def _load_banned_symbols(path: str = "banned_symbols.json") -> dict:
  if not os.path.exists(path):
    return {}
  with open(path, "r") as f:
    data = json.load(f)
  for k, v in data.items():
    if not isinstance(v, list):
      data[k] = list(v)
  return data


def select_top_symbols_by_usd_volume(
  end_date_str: str,
  buffer_days: int = 14,
  top_n: int = 50,
  data_dir: str = "data/1m_raw_data",
  banned_json_path: str = "banned_symbols.json",
) -> list[str]:

  end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
  start_date = end_date - timedelta(days=buffer_days - 1)

  total_volume = defaultdict(float)

    
  cur = start_date
  while cur <= end_date:
    date_str = cur.strftime("%Y-%m-%d")
    path = os.path.join(data_dir, f"{date_str}.h5")

    if not os.path.exists(path):
      print(f"[WARN] Missing file: {path}")
      cur += timedelta(days=1)
      continue

    df = pd.read_hdf(path)

    if "symbol" not in df.columns or "quote_volume" not in df.columns:
      raise ValueError(f"{path} 에 symbol / quote_volume 컬럼이 없음")

    daily_sum = df.groupby("symbol")["quote_volume"].sum()

    for sym, vol in daily_sum.items():
      total_volume[sym] += float(vol)

    cur += timedelta(days=1)

  if not total_volume:
    print("[WARN] No data loaded in this period.")
    return []



  banned_map = _load_banned_symbols(banned_json_path)
  banned_symbols = set()

  if end_date_str in banned_map:
    banned_symbols.update(banned_map[end_date_str])

  prev_date_str = (end_date - timedelta(days=1)).strftime("%Y-%m-%d")
  if prev_date_str in banned_map:
    banned_symbols.update(banned_map[prev_date_str])

  if banned_symbols:
    print("[INFO] Banned (today + yesterday):")
    for s in sorted(banned_symbols):
      print("  -", s)

  sorted_symbols = sorted(total_volume.items(), key=lambda x: x[1], reverse=True)
  filtered = [sym for sym, _ in sorted_symbols if sym not in banned_symbols]

  top_symbols = filtered[:top_n]

  universe_json = f"top{top_n}_universe.json"

  if os.path.exists(universe_json):
    if os.path.getsize(universe_json) > 0:
      with open(universe_json, "r") as f:
        universe_data = json.load(f)
    else:
      universe_data = {}
  else:
    universe_data = {}

  universe_data[end_date_str] = top_symbols
  universe_data = {k: universe_data[k] for k in sorted(universe_data.keys())}

  with open(universe_json, "w") as f:
    json.dump(universe_data, f, indent=2, separators=(",", ": "))

  print(f"[INFO] Saved universe → {universe_json}")

  return top_symbols


def _iter_dates(start_date_str: str, end_date_str: str):
  start = datetime.strptime(start_date_str, "%Y-%m-%d").date()
  end = datetime.strptime(end_date_str, "%Y-%m-%d").date()
  cur = start
  while cur <= end:
    yield cur.strftime("%Y-%m-%d")
    cur += timedelta(days=1)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--start_date", type=str, required=True)
  parser.add_argument("--end_date", type=str, required=True)
  parser.add_argument("--buffer_days", type=int, default=14)
  parser.add_argument("--top_n", type=int, default=50)
  parser.add_argument("--data_dir", type=str, default="data/1m_raw_data")
  parser.add_argument("--banned_json_path", type=str, default="banned_symbols.json")
  return parser.parse_args()


def main():
  args = parse_args()

  print(f"[INFO] Universe dump from {args.start_date} to {args.end_date}")

  for date_str in _iter_dates(args.start_date, args.end_date):
    print(f"[INFO] Processing {date_str}")

    lst = select_top_symbols_by_usd_volume(
      end_date_str=date_str,
      buffer_days=args.buffer_days,
      top_n=args.top_n,
      data_dir=args.data_dir,
      banned_json_path=args.banned_json_path,
    )

    print(f"[INFO] Universe for {date_str}: {len(lst)} symbols")


if __name__ == "__main__":
  main()
    
# python3 universe_builder.py --start_date 2025-02-01 --end_date 2025-03-01 --buffer_days 14 --top_n 50