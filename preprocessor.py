# This is the functions to preprocess the raw data
# - Check all symbols are valid (each symbol has 1440 rows)
# - Check there are no NaNs in required columns
# - Fill NaN values with min of adjacent rows or forward fill

import pandas as pd
import json
import os
from collections import Counter


def symbol_checker(df: pd.DataFrame, num_dates=1) -> bool:
  """
  Check all symbols have exactly 1440 rows each.
  Uses Counter on df["symbol"].
  """
  if "symbol" not in df.columns:
    raise ValueError("DataFrame에 'symbol' 컬럼이 없습니다.")

  counts = Counter(df["symbol"])
  invalid = {sym: cnt for sym, cnt in counts.items() if cnt != 1440 * num_dates}

  if invalid:
    print("[WARN] These symbols have unexpected row counts:")
    for sym, cnt in invalid.items():
      print(f"  {sym}: {cnt} rows")
    return False

  return True


def ban_symbols_with_nan(date_str: str, threshold: int = 30):
  df = pd.read_hdf(f"data/1m_raw_data/{date_str}.h5")

  if "symbol" not in df.columns:
    raise ValueError("'symbol' column is not found in the DataFrame.")
  if not symbol_checker(df):
    raise ValueError("Symbol checker failed.")
  
  banned_symbols = []

  # 1) NaN row 개수 기준 ban
  for sym, g in df.groupby("symbol"):
    nan_rows = g.isna().any(axis=1).sum()
    if nan_rows >= threshold:
      banned_symbols.append(sym)

  # 2) 이름이 "USD" 로 시작하는 심볼 전부 ban
  for sym in df["symbol"].unique():
    if sym.startswith("USD") and sym not in banned_symbols:
      banned_symbols.append(sym)

  json_path = "banned_symbols.json"

  if os.path.exists(json_path):
    with open(json_path, "r") as f:
      data = json.load(f)
  else:
    data = {}

  data[date_str] = banned_symbols
  data = {k: data[k] for k in sorted(data.keys())}
  
  with open(json_path, "w") as f:
    json.dump(data, f, indent=2)

  print(f"[INFO] Banned symbols on {date_str}: {banned_symbols}")
  print(f"[INFO] Saved to {json_path}")

  return banned_symbols


def check_all_columns_valid(df: pd.DataFrame) -> bool:
  """
  symbol 단위로 NaN 존재 여부를 체크.
  어떤 symbol이라도 NaN이 하나라도 있으면 해당 symbol 출력 후 False 반환.
  """
  if "symbol" not in df.columns:
    raise ValueError("'symbol' column is not found in the DataFrame.")

  bad_symbols = []

  # symbol 단위로 그룹핑해서 NaN 존재 여부 체크
  for sym, g in df.groupby("symbol"):
    # g.isna().any().any() → 전체 DataFrame에 NaN 있으면 True
    if g.isna().any().any():
      bad_symbols.append(sym)
  print(f"bad_symbols: {bad_symbols}")
  if bad_symbols:
    print("[WARN] These symbols have NaN values:")
    for sym in bad_symbols:
      print(f"  {sym}")
    return False
  return True
  


def fill_nan_values(df: pd.DataFrame, method: str = "forward") -> pd.DataFrame:
  df = df.copy()
  numeric_cols = df.select_dtypes(include="number").columns

  if method == "forward":
    df[numeric_cols] = (
      df.groupby("symbol")[numeric_cols]
      .transform(lambda x: x.ffill())
    )
    df[numeric_cols] = df[numeric_cols].groupby(df["symbol"]).transform(lambda x: x.bfill())
    return df

  else:
    raise ValueError(f"Invalid method: {method}")


def preprocess_data(df: pd.DataFrame, method: str = "forward", num_dates=1):
  if not symbol_checker(df, num_dates):
    raise ValueError("Symbol checker failed.")
  df = fill_nan_values(df, method)
  if not check_all_columns_valid(df):
    raise ValueError(f"All columns are not valid,")
  return df
