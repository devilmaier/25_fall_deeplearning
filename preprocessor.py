# This is the functions to preprocess the raw data
# - Check all symbols are valid (each symbol has 1440 rows)
# - Check there are no NaNs in required columns
# - Fill NaN values with min of adjacent rows or forward fill

import pandas as pd
import json
import os
from collections import Counter


def symbol_checker(df: pd.DataFrame) -> bool:
  """
  Check all symbols have exactly 1440 rows each.
  Uses Counter on df["symbol"].
  """
  if "symbol" not in df.columns:
    raise ValueError("DataFrame에 'symbol' 컬럼이 없습니다.")

  counts = Counter(df["symbol"])
  invalid = {sym: cnt for sym, cnt in counts.items() if cnt != 1440}

  if invalid:
    print("[WARN] These symbols have unexpected row counts:")
    for sym, cnt in invalid.items():
      print(f"  {sym}: {cnt} rows")
    return False

  return True


def ban_symbols_with_nan(date_str: str, threshold: int = 30):
  """
  하루(df)에서 symbol 별 NaN row 개수를 세고,
  threshold 이상이면 ban list에 넣는다.

  banned_symbols.json 파일에 다음 구조로 저장:
    {
      "2025-10-01": ["BTCUSDT", "ETHUSDT"],
      "2025-10-02": ["FOOUSDT"]
    }
  """
  df = pd.read_hdf(f"data/1m_raw_data/{date_str}.h5")
  if "symbol" not in df.columns:
    raise ValueError("'symbol' column is not found in the DataFrame.")
  if not symbol_checker(df):
    raise ValueError("Symbol checker failed.")
  
  banned_symbols = []

  for sym, g in df.groupby("symbol"):
    nan_rows = g.isna().any(axis=1).sum()
    if nan_rows >= threshold:
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
    df[numeric_cols] = df[numeric_cols].ffill()
    return df
  else:
    raise ValueError(f"Invalid method: {method}")


def preprocess_data(df: pd.DataFrame, method: str = "forward"):
  if not symbol_checker(df):
    raise ValueError("Symbol checker failed.")
  df = fill_nan_values(df, method)
  if not check_all_columns_valid(df):
    raise ValueError("All columns are not valid.")
  return df
