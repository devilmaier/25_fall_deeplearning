#!/usr/bin/env python

import argparse
from datetime import datetime, timedelta
from preprocessor import ban_symbols_with_nan
from tqdm import tqdm


def daterange(start_date, end_date):
    """Yield YYYY-MM-DD strings from start_date to end_date inclusive."""
    cur = start_date
    while cur <= end_date:
        yield cur.strftime("%Y-%m-%d")
        cur += timedelta(days=1)


def main():
    parser = argparse.ArgumentParser(
        description="Run ban_symbols_with_nan for a date range and update banned_symbols.json"
    )

    parser.add_argument(
        "--start_date",
        required=True,
        help="시작 날짜 (예: 2025-03-01)",
    )
    parser.add_argument(
        "--end_date",
        required=True,
        help="끝 날짜 (예: 2025-03-10)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=30,
        help="NaN row 개수 기준 (기본값: 30)",
    )

    args = parser.parse_args()

    # 문자열 → datetime 변환
    s = datetime.strptime(args.start_date, "%Y-%m-%d")
    e = datetime.strptime(args.end_date, "%Y-%m-%d")

    # 날짜 목록 만들기
    date_list = list(daterange(s, e))

    print(f"[INFO] Total {len(date_list)} days to process.")

    # tqdm progress bar
    for date_str in tqdm(date_list, desc="Processing dates"):
        try:
            ban_symbols_with_nan(date_str, threshold=args.threshold)
        except FileNotFoundError:
            tqdm.write(f"[WARN] {date_str} 파일 없음 → 스킵")
        except Exception as ex:
            tqdm.write(f"[ERROR] {date_str} 처리 중 오류 발생: {ex}")


if __name__ == "__main__":
    main()
"""
python symbol_ban.py \
    --start_date 2024-01-01 \
    --end_date 2025-11-30 \
    --threshold 30
"""