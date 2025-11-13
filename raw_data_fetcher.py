#!/usr/bin/env python3
import argparse
import io
import zipfile
from datetime import datetime, timedelta, timezone
from multiprocessing import Pool, cpu_count
from functools import partial

import pandas as pd
import requests
from tqdm import tqdm


# ---------------------------------------------------------
# 1. CLI args
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect Binance USDT-M PERPETUAL 1m klines + premium index klines "
            "+ 5m metrics (backfilled to 1m) for a given UTC date range, "
            "merge, and save {date}.h5 for each date."
        )
    )

    parser.add_argument(
        "--start_date",
        type=str,
        required=False,
        help="UTC start date (YYYY-MM-DD). Default = yesterday UTC."
    )
    parser.add_argument(
        "--end_date",
        type=str,
        required=False,
        help="UTC end date (YYYY-MM-DD). Default = same as start-date."
    )
    parser.add_argument(
        "--workers",
        type=int,
        required=False,
        help="Parallel worker processes. Default = cpu_count()."
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # 날짜 기본값 처리 (기존 로직 스타일 유지)
    # ---------------------------------------------------------
    utc_now = datetime.now(timezone.utc)
    default_day = (utc_now - timedelta(days=1)).strftime("%Y-%m-%d")
    if args.start_date is None and args.end_date is None:
        
        args.start_date = default_day
        args.end_date = default_day

    # 둘 중 하나만 넣으면 오류
    elif args.end_date is None:
        args.end_date = default_day
    elif args.start_date is None:
        raise ValueError(f"{args.start_date} and {args.end_date} must be provided.")

    return args


# ---------------------------------------------------------
# 2. symbol list (USDT-M perp only)
# ---------------------------------------------------------
def get_all_symbols_usdtm():
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; raw_data_fetcher/1.0)"
    }

    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
    except requests.HTTPError as e:
        print(f"[ERROR] exchangeInfo HTTPError: {e}")
        return []
    except requests.RequestException as e:
        print(f"[ERROR] exchangeInfo RequestException: {e}")
        return []

    data = r.json()

    symbols = []
    for s in data.get("symbols", []):
        if (
            s.get("status") == "TRADING"
            and s.get("contractType") == "PERPETUAL"
            and s.get("marginAsset") == "USDT"
        ):
            symbols.append(s["symbol"])

    return symbols


# ---------------------------------------------------------
# 3. zip url builder / downloader
# ---------------------------------------------------------
def build_zip_url(symbol: str, date_str: str, interval: str, market_type: str):
    if market_type == "klines":
        return (
            "https://data.binance.vision/"
            f"data/futures/um/daily/klines/{symbol}/1m/"
            f"{symbol}-1m-{date_str}.zip"
        )
    elif market_type == "premiumIndexKlines":
        return (
            "https://data.binance.vision/"
            f"data/futures/um/daily/premiumIndexKlines/{symbol}/1m/"
            f"{symbol}-1m-{date_str}.zip"
        )
    elif market_type == "metrics":
        return (
            "https://data.binance.vision/"
            f"data/futures/um/daily/metrics/{symbol}/"
            f"{symbol}-metrics-{date_str}.zip"
        )
    else:
        raise ValueError(
            "market_type must be 'klines', 'premiumIndexKlines', or 'metrics'"
        )


def download_zip_if_exists(symbol: str, date_str: str, market_type: str):
    url = build_zip_url(symbol, date_str, "1m", market_type)
    if market_type == "metrics":
      url = build_zip_url(symbol, date_str, "5m", market_type)
    try:
        r = requests.get(url, timeout=15)
    except Exception as e:
        print(f"[WARN] {symbol} {date_str} download error ({market_type}): {e}")
        return None

    if r.status_code != 200:
        return None

    return r.content


# ---------------------------------------------------------
# 4. csv loader
# ---------------------------------------------------------
def load_csv_from_zip_bytes(zip_bytes: bytes) -> pd.DataFrame:
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            csv_name = None
            for name in zf.namelist():
                if name.endswith(".csv"):
                    csv_name = name
                    break
            if csv_name is None:
                return pd.DataFrame()

            raw_bytes = zf.read(csv_name)
    except Exception as e:
        print(f"[WARN] load_csv_from_zip_bytes unzip failed: {e}")
        return pd.DataFrame()

    # 1) header 있는 버전 시도
    try:
        df_try_header = pd.read_csv(
            io.StringIO(raw_bytes.decode("utf-8")),
            sep=None,
            engine="python",
            header=0,
        )
        first_colname = str(df_try_header.columns[0]).lower()
        if ("open_time" in first_colname) or ("opentime" in first_colname) or ("create_time" in first_colname):
            return df_try_header
    except Exception:
        pass

    # 2) header 없는 버전
    try:
        df_no_header = pd.read_csv(
            io.StringIO(raw_bytes.decode("utf-8")),
            sep=None,
            engine="python",
            header=None,
        )
        return df_no_header
    except Exception as e:
        print(f"[WARN] load_csv_from_zip_bytes parse failed: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------
# 5.  common columns normalization
# ---------------------------------------------------------
def normalize_columns(df_raw: pd.DataFrame) -> pd.DataFrame:
    cols = list(df_raw.columns)
    first_col_lower = str(cols[0]).lower()

    # snake_case 헤더 버전
    if (
        "open_time" in first_col_lower
        or "opentime" in first_col_lower
        or "open time" in first_col_lower
    ):
        rename_map = {
            cols[0]:  "openTime",
            cols[1]:  "open",
            cols[2]:  "high",
            cols[3]:  "low",
            cols[4]:  "close",
            cols[5]:  "volume_base",
            cols[6]:  "closeTime",
            cols[7]:  "quote_volume",
            cols[8]:  "num_trades",
            cols[9]:  "taker_buy_base",
            cols[10]: "taker_buy_quote",
            cols[11]: "ignore",
        }
        df_norm = df_raw.rename(columns=rename_map)
        return df_norm[
            [
                "openTime",
                "open",
                "high",
                "low",
                "close",
                "volume_base",
                "closeTime",
                "quote_volume",
                "num_trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ]
        ].copy()

    # no-header numeric 버전
    if len(cols) == 12 and all(str(c).isdigit() for c in cols):
        df_norm = df_raw.copy()
        df_norm.columns = [
            "openTime",
            "open",
            "high",
            "low",
            "close",
            "volume_base",
            "closeTime",
            "quote_volume",
            "num_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ]
        return df_norm

    print("[WARN] normalize_columns: unexpected columns, first few:", cols[:5], "...")
    return pd.DataFrame()


# ---------------------------------------------------------
# 6. kline processing
# ---------------------------------------------------------
def process_main_kline_df(df_raw: pd.DataFrame, symbol: str, date_str: str) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame()

    df_norm = normalize_columns(df_raw)
    if df_norm.empty:
        return pd.DataFrame()

    n = len(df_norm)
    if n == 1441:
        df_norm = df_norm.iloc[:-1].copy()
        n = len(df_norm)

    if n != 1440:
        print(f"[SKIP] {symbol} {date_str}: unexpected row count {n}")
        return pd.DataFrame()

    df_norm["start_time_ms"] = pd.to_numeric(
        df_norm["openTime"], errors="coerce", downcast="integer"
    )
    df_norm["start_time_utc"] = pd.to_datetime(
        df_norm["openTime"], unit="ms", utc=True
    ).dt.strftime("%Y-%m-%d %H:%M:%S")

    float_cols = [
        "open", "high", "low", "close",
        "volume_base", "quote_volume",
        "taker_buy_base", "taker_buy_quote",
    ]
    for col in float_cols:
        df_norm[col] = pd.to_numeric(df_norm[col], errors="coerce")

    df_norm["num_trades"] = pd.to_numeric(
        df_norm["num_trades"], errors="coerce", downcast="integer"
    )

    out = df_norm[[
        "start_time_ms",
        "start_time_utc",
        "open",
        "high",
        "low",
        "close",
        "volume_base",
        "quote_volume",
        "num_trades",
        "taker_buy_base",
        "taker_buy_quote",
    ]].copy()

    out.insert(0, "symbol", symbol)

    return out


# ---------------------------------------------------------
# 7. premium index
# ---------------------------------------------------------
def process_premium_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame()

    df_norm = normalize_columns(df_raw)
    if df_norm.empty:
        return pd.DataFrame()

    df_norm["start_time_ms"] = pd.to_numeric(
        df_norm["openTime"], errors="coerce", downcast="integer"
    )

    out = df_norm[[
        "start_time_ms",
        "open",
        "high",
        "low",
        "close",
    ]].copy()

    out.rename(
        columns={
            "open": "pm_open",
            "high": "pm_high",
            "low": "pm_low",
            "close": "pm_close",
        },
        inplace=True,
    )

    return out


# ---------------------------------------------------------
# 8. 5m -> 1m processing
# ---------------------------------------------------------

def _normalize_metrics_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    metrics csv columns (실제 확인한 포맷):

      create_time                              <-- "2025-11-01 00:05:00" (UTC 시각 문자열)
      symbol                                   <-- "1000000BOBUSDT"
      sum_open_interest                        <-- (안 씀)
      sum_open_interest_value                  <-- 우리가 oi 로 쓸 것
      count_toptrader_long_short_ratio         <-- 우리가 top_ls_ratio_cnt 로 쓸 것
      sum_toptrader_long_short_ratio           <-- 우리가 top_ls_ratio 로 쓸 것
      count_long_short_ratio                   <-- 우리가 ls_ratio_cnt 로 쓸 것
      sum_taker_long_short_vol_ratio           <-- 우리가 ls_vol_ratio 로 쓸 것

    출력 컬럼:
      start_time_ms            (int, epoch ms)
      oi
      top_ls_ratio_cnt
      top_ls_ratio
      ls_ratio_cnt
      ls_vol_ratio
    """

    if df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()

    if all(str(c).isdigit() for c in df.columns):
        # 최소 8열 필요
        if len(df.columns) < 8:
            print("[WARN] _normalize_metrics_df: numeric cols but not enough columns")
            return pd.DataFrame()

        df = df.iloc[:, :8].copy()
        df.columns = [
            "create_time",
            "symbol",
            "sum_open_interest",
            "sum_open_interest_value",
            "count_toptrader_long_short_ratio",
            "sum_toptrader_long_short_ratio",
            "count_long_short_ratio",
            "sum_taker_long_short_vol_ratio",
        ]

    if "create_time" not in df.columns:
        print("[WARN] _normalize_metrics_df: create_time not found in columns")
        return pd.DataFrame()


    ts = pd.to_datetime(df["create_time"], utc=True, errors="coerce")
    df["start_time_ms"] = (ts.astype("int64") // 1_000_000).astype("int64")

    keep_map = {
        "sum_open_interest_value": "oi",
        "count_toptrader_long_short_ratio": "top_ls_ratio_cnt",
        "sum_toptrader_long_short_ratio": "top_ls_ratio",
        "count_long_short_ratio": "ls_ratio_cnt",
        "sum_taker_long_short_vol_ratio": "ls_vol_ratio",
    }

    missing_cols = [c for c in keep_map if c not in df.columns]
    if missing_cols:
        print("[WARN] _normalize_metrics_df: missing cols", missing_cols)

    sub_cols = ["start_time_ms"] + [c for c in keep_map if c in df.columns]
    sub = df[sub_cols].copy()
    sub = sub.rename(columns=keep_map)

    for c in sub.columns:
        if c != "start_time_ms":
            sub[c] = pd.to_numeric(sub[c], errors="coerce")


    return sub



def prepare_metrics_with_prev_day(metrics_today_df: pd.DataFrame,
                                  metrics_prev_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_today_df.empty and metrics_prev_df.empty:
        return pd.DataFrame()

    prev_tail = pd.DataFrame()
    if not metrics_prev_df.empty:
        prev_tail = metrics_prev_df.tail(1).copy()

    merged_metrics = pd.concat(
        [prev_tail, metrics_today_df],
        ignore_index=True
    )

    merged_metrics = (
        merged_metrics
        .sort_values("start_time_ms")
        .drop_duplicates(subset=["start_time_ms"], keep="last")
        .reset_index(drop=True)
    )

    return merged_metrics


def build_metrics_1m_for_merge(df_main_1m: pd.DataFrame,
                               metrics_today_df_raw: pd.DataFrame,
                               metrics_prev_df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_main_1m.empty:
        return pd.DataFrame()
    mt_today = _normalize_metrics_df(metrics_today_df_raw) if not metrics_today_df_raw.empty else pd.DataFrame()
    mt_prev  = _normalize_metrics_df(metrics_prev_df_raw)  if not metrics_prev_df_raw.empty  else pd.DataFrame()
    
    if mt_today.empty and mt_prev.empty:
        return pd.DataFrame()

    mt_full = prepare_metrics_with_prev_day(mt_today, mt_prev)
    if mt_full.empty:
        return pd.DataFrame()

    keep_cols = [c for c in mt_full.columns if c not in ("start_time_ms",)]
    rename_map = {c: f"mt_{c}" for c in keep_cols}
    mt_prefixed = mt_full.rename(columns=rename_map)

    #merge
    mt_prefixed = mt_prefixed.sort_values("start_time_ms").reset_index(drop=True)
    df_main_sorted = df_main_1m.sort_values("start_time_ms").reset_index(drop=True)

    merged = pd.merge_asof(
        df_main_sorted,
        mt_prefixed,
        on="start_time_ms",
        direction="backward",
    )

    return merged


# ---------------------------------------------------------
# 9. each symbol fetching
# ---------------------------------------------------------
def fetch_symbol_for_date(symbol: str, date_str: str) -> pd.DataFrame:
    # 1) main klines
    zip_main = download_zip_if_exists(symbol, date_str, market_type="klines")
    if zip_main is None:
        return pd.DataFrame()

    raw_main = load_csv_from_zip_bytes(zip_main)
    if raw_main is None or raw_main.empty:
        return pd.DataFrame()

    df_main = process_main_kline_df(raw_main, symbol, date_str)
    if df_main.empty:
        return pd.DataFrame()

    # 2) premium index
    zip_prem = download_zip_if_exists(symbol, date_str, market_type="premiumIndexKlines")
    if zip_prem is not None:
        raw_prem = load_csv_from_zip_bytes(zip_prem)
        if raw_prem is not None and not raw_prem.empty:
            df_prem = process_premium_df(raw_prem)
        else:
            df_prem = pd.DataFrame()
    else:
        df_prem = pd.DataFrame()

    if not df_prem.empty:
        df_tmp = pd.merge(
            df_main,
            df_prem,
            on="start_time_ms",
            how="left",
        )
    else:
        df_tmp = df_main.copy()
        df_tmp["pm_open"] = pd.NA
        df_tmp["pm_high"] = pd.NA
        df_tmp["pm_low"] = pd.NA
        df_tmp["pm_close"] = pd.NA

    # 3) metrics (5m → 1m backward fill)
    zip_metrics_today = download_zip_if_exists(symbol, date_str, market_type="metrics")
    raw_metrics_today = (
        load_csv_from_zip_bytes(zip_metrics_today)
        if zip_metrics_today is not None
        else pd.DataFrame()
    )

    prev_date_dt = datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
    prev_date_str = prev_date_dt.strftime("%Y-%m-%d")

    zip_metrics_prev = download_zip_if_exists(symbol, prev_date_str, market_type="metrics")
    raw_metrics_prev = (
        load_csv_from_zip_bytes(zip_metrics_prev)
        if zip_metrics_prev is not None
        else pd.DataFrame()
    )

    df_with_metrics = build_metrics_1m_for_merge(
        df_main_1m=df_tmp,
        metrics_today_df_raw=raw_metrics_today,
        metrics_prev_df_raw=raw_metrics_prev,
    )

    if df_with_metrics.empty:
        df_with_metrics = df_tmp.copy()

    return df_with_metrics


# ---------------------------------------------------------
# 10. Pool worker
# ---------------------------------------------------------
def _worker_for_pool(symbol: str, date_str: str) -> pd.DataFrame:
    try:
        return fetch_symbol_for_date(symbol, date_str)
    except Exception as e:
        print(f"[WARN] {symbol} {date_str} failed: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------
# 11. parallel fetch all symbols
# ---------------------------------------------------------
def collect_day_all_symbols_from_binance_vision(date_str: str, max_workers=None) -> pd.DataFrame:
    symbols = get_all_symbols_usdtm()
    if not symbols:
        print("[WARN] empty symbol list (maybe 418 / blocked).")
        return pd.DataFrame()

    if max_workers is None:
        max_workers = cpu_count()

    worker_fn = partial(_worker_for_pool, date_str=date_str)

    dfs = []
    with Pool(processes=max_workers) as pool:
        for df_sym in tqdm(
            pool.imap_unordered(worker_fn, symbols),
            total=len(symbols),
            desc=f"Collect {date_str} UTC (binance.vision)"
        ):
            if df_sym is not None and not df_sym.empty:
                dfs.append(df_sym)

    if not dfs:
        return pd.DataFrame()

    big_df = pd.concat(dfs, ignore_index=True)

    big_df = big_df.sort_values(
        by=["start_time_ms", "symbol"]
    ).reset_index(drop=True)

    return big_df


# ---------------------------------------------------------
# 12. save h5
# ---------------------------------------------------------
def save_h5(df: pd.DataFrame, date_str: str):
    root_path = "/Users/minchul/Desktop/서울대/딥기/crypto_pred/data/1m_raw_data"
    filename = f"{root_path}/{date_str}.h5"
    df.to_hdf(filename, key="data", mode="w", format="table")
    print(f"[OK] {date_str}: saved {len(df)} rows to {filename}")


# ---------------------------------------------------------
# 13. iter_dates
# ---------------------------------------------------------
def _iter_dates(start_date_str: str, end_date_str: str):
    """
    'YYYY-MM-DD' 문자열 두 개를 받아서
    start_date ~ end_date (둘 다 포함) 날짜 문자열을 순회.
    """
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d").date()

    if end_dt < start_dt:
        raise ValueError(f"end_date({end_date_str}) 가 start_date({start_date_str}) 보다 앞입니다.")

    cur = start_dt
    one_day = timedelta(days=1)
    while cur <= end_dt:
        yield cur.strftime("%Y-%m-%d")
        cur += one_day


# ---------------------------------------------------------
# 14. main
# ---------------------------------------------------------
def main():
  args = parse_args()
  start_date_str = args.start_date
  end_date_str = args.end_date
  max_workers = args.workers if args.workers is not None else None

  print(f"[INFO] Dumping from {start_date_str} to {end_date_str} (inclusive)")

  for date_str in _iter_dates(start_date_str, end_date_str):
    print(f"[INFO] Fetching {date_str} ...")

    df_day = collect_day_all_symbols_from_binance_vision(
      date_str=date_str,
      max_workers=max_workers,
    )

    if df_day.empty:
      print(f"[WARN] {date_str}: no symbols available / no data fetched")
    else:
      save_h5(df_day, date_str)
      print(f"[INFO] Saved HDF5 for {date_str}")


if __name__ == "__main__":
    main()

#usage: python raw_data_fetcher.py --start-date 2025-10-01 --end-date 2025-10-07 --workers 16