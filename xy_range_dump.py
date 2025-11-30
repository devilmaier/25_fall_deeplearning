#!/usr/bin/env python
import os
import json
import argparse
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

# 같은 디렉토리에 있는 x_generator.py, y_generator.py 에서 함수 import
from x_generator import x_generator as make_x
from y_generator import y_generator as make_y


def generate_date_list(start_date_str: str, end_date_str: str):
    start = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    if end < start:
        raise ValueError("end_date must be >= start_date")
    dates = []
    cur = start
    while cur <= end:
        dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)
    return dates


def load_universe_map(top: int, universe_file: str | None = None) -> dict:
    """
    top30_universe.json 같은 파일을 한 번만 읽어서 dict 로 올려둔다.
    { "2025-03-01": ["BTCUSDT", ...], ... }
    """
    if universe_file is None:
        universe_file = f"top{top}_universe.json"
    if not os.path.exists(universe_file):
        raise FileNotFoundError(universe_file)

    with open(universe_file, "r") as f:
        universe_map = json.load(f)

    if not isinstance(universe_map, dict):
        raise ValueError(f"{universe_file} must be a dict mapping date_str -> list[symbol]")

    return universe_map


def _merge_x_y_for_date(
    date_str: str,
    top: int,
    data_dir: str,
    universe_map: dict,
    price_col: str,
    windows: list[int],
    out_dir: str,
    how: str = "inner",
) -> tuple[str, int]:
    """
    한 날짜에 대해:
      1) X 생성 (in-memory)
      2) Y 생성 (in-memory)
      3) symbol + start_time_ms 기준으로 merge
      4) data/dataset/{date}_xy_top{top}.h5 로 저장

    return: (date_str, return_code) 0 이면 성공, 그 외는 실패
    """
    try:
        if date_str not in universe_map:
            raise KeyError(f"{date_str} not in universe_map")

        universe = universe_map[date_str]

        # --- X 생성 ---
        x_df = make_x(
            date_str=date_str,
            universe=universe,
            data_dir=data_dir,
            add_neutralized=True,
        )

        # --- Y 생성 ---
        y_df = make_y(
            date_str=date_str,
            universe=universe,
            data_dir=data_dir,
            price_col=price_col,
            window_list=windows,
        )

        # y_generator 는 open_time_ms 로 rename 했으므로 다시 start_time_ms 로 맞춰준다.
        if "start_time_ms" not in y_df.columns:
            if "open_time_ms" not in y_df.columns:
                raise KeyError(
                    f"Y for {date_str} is missing both 'start_time_ms' and 'open_time_ms'"
                )
            y_df = y_df.copy()
            y_df["start_time_ms"] = y_df["open_time_ms"]

        # --- merge ---
        merged = pd.merge(
            x_df,
            y_df,
            on=["symbol", "start_time_ms"],
            how=how,
            suffixes=("", "_ydup"),
        )

        # open_time_ms 는 정보상 start_time_ms 와 동일하므로 있으면 제거
        if "open_time_ms" in merged.columns:
            merged = merged.drop(columns=["open_time_ms"])

        # 혹시 suffix 붙은 중복 컬럼이 있다면 정리
        dup_cols = [c for c in merged.columns if c.endswith("_ydup")]
        if dup_cols:
            merged = merged.drop(columns=dup_cols)

        merged = merged.sort_values(["start_time_ms", "symbol"]).reset_index(drop=True)

        # --- 저장 ---
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{date_str}_xy_top{top}.h5")
        merged.to_hdf(out_path, key="xy", mode="w")

        print(f"[OK] {date_str}: saved {len(merged)} rows to {out_path}")
        return date_str, 0

    except Exception as e:
        print(f"[ERROR] {date_str}: {e}")
        return date_str, 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate merged X+Y datasets for a date range in parallel."
    )
    parser.add_argument("--start_date", type=str, required=True, help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--end_date", type=str, required=True, help="YYYY-MM-DD (inclusive)")
    parser.add_argument("--top", type=int, required=True, help="universe size, e.g. 30")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/1m_raw_data",
        help="raw 1m data directory (h5 files)",
    )
    parser.add_argument(
        "--universe_file",
        type=str,
        default="",
        help="universe json (default: top{top}_universe.json)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/dataset",
        help="output directory for merged XY datasets",
    )
    parser.add_argument(
        "--price_col",
        type=str,
        default="close",
        help="price column for forward diff labels (default: close)",
    )
    parser.add_argument(
        "--windows",
        type=str,
        default="1,5,15,30,60,240",
        help="comma-separated forward horizons in minutes for Y (e.g. 1,5,15...)",
    )
    parser.add_argument(
        "--how",
        type=str,
        default="inner",
        choices=["inner", "left", "right"],
        help="merge type (default: inner)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="number of parallel processes (default: 4)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="print which dates would be processed, without actually running",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dates = generate_date_list(args.start_date, args.end_date)
    print(f"Total {len(dates)} days from {args.start_date} to {args.end_date}")

    universe_file = args.universe_file if args.universe_file else None
    universe_map = load_universe_map(args.top, universe_file=universe_file)

    windows = [int(x) for x in args.windows.split(",") if x.strip()]

    if args.dry_run:
        print("Dry-run mode: will process the following dates:")
        for d in dates:
            print("  ", d)
        return

    failed = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                _merge_x_y_for_date,
                d,
                args.top,
                args.data_dir,
                universe_map,
                args.price_col,
                windows,
                args.out_dir,
                args.how,
            ): d
            for d in dates
        }

        for fut in as_completed(futures):
            date_str = futures[fut]
            try:
                _, rc = fut.result()
                if rc != 0:
                    failed.append(date_str)
            except Exception as e:
                print(f"[EXCEPTION] {date_str}: {e}")
                failed.append(date_str)

    if failed:
        print("\n=== Failed dates ===")
        for d in failed:
            print(d)
    else:
        print("\nAll dates finished successfully.")


if __name__ == "__main__":
    main()
"""
python xy_range_dump.py \
  --start_date 2025-03-01 \
  --end_date 2025-10-01 \
  --top 30 \
  --data_dir data/1m_raw_data \
  --out_dir data/dataset \
  --max_workers 16
  """