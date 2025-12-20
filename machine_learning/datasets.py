# machine_learning/datasets.py

import os
import json
import ast
from datetime import datetime, timedelta
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

# =========================
# 프로젝트 루트 기준 경로들
# =========================
BASE_DIR = '/Users/minchul/Desktop/서울대/딥기/crypto_pred'
DATA_DIR_DEFAULT = os.path.join(BASE_DIR, "data")
FEATURE_ROOT_DEFAULT = os.path.join(BASE_DIR, "feature_list")
EXPORT_DIR_DEFAULT = os.path.join(DATA_DIR_DEFAULT, "datasets", "ml")
BAN_PATH_DEFAULT = os.path.join(BASE_DIR, "global_ban_dates.json")


def _iter_dates(start_date: str, end_date: str) -> List[str]:
    """
    start_date, end_date: "YYYY-MM-DD" 문자열 (inclusive)
    global_ban_dates.json 에 포함된 날짜들은 제거한다.
    - missing_dates: 리스트
    - nan_dates: dict 의 key (날짜 문자열) 들
    """
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    if end < start:
        raise ValueError(f"end_date({end_date}) must be >= start_date({start_date})")

    cur = start
    dates: List[str] = []
    while cur <= end:
        dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)

    ban_path = BAN_PATH_DEFAULT

    if os.path.exists(ban_path):
        try:
            with open(ban_path, "r") as f:
                obj = json.load(f)

            missing_dates = set(obj.get("missing_dates", []))
            nan_dates = set(obj.get("nan_dates", {}).keys())
            banned = missing_dates | nan_dates

            before_n = len(dates)
            dates = [d for d in dates if d not in banned]
            removed_n = before_n - len(dates)

            if removed_n > 0:
                print(
                    f"[INFO] _iter_dates: removed {removed_n} banned dates "
                    f"from global_ban_dates.json"
                )
        except Exception as e:
            print(f"[WARN] Failed to apply global_ban_dates.json: {e}")

    return dates


def _load_feature_list(
    y_name: str,
    topn: int,
    feature_root: str = FEATURE_ROOT_DEFAULT,
) -> List[str]:
    """
    feature_list/{y_name}/top{topn}_example_features_166.json 을 읽어서
    feature 이름 리스트를 리턴한다.

    JSON 포맷:
      1) 권장: ["x_...", "x_...", ...]   # 진짜 JSON 리스트
      2) 현재처럼: "['x_...', 'x_...', ...]"  # 문자열 안에 파이썬 리스트가 들어간 형태도 지원
    """
    json_path = os.path.join(
        feature_root, y_name, f"top{topn}_example_features_166.json"
    )
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Feature list JSON not found: {json_path}")

    with open(json_path, "r") as f:
        features = json.load(f)

    # case 1: 이미 올바른 리스트
    if isinstance(features, list):
        pass

    # case 2: 문자열 안에 "['x_1m', 'x_5m', ...]" 꼴로 들어가 있음
    elif isinstance(features, str):
        try:
            parsed = ast.literal_eval(features)
        except Exception as e:
            raise ValueError(
                f"Feature JSON is a string but cannot be parsed as list: {json_path}, err={e}"
            )
        if not isinstance(parsed, list):
            raise ValueError(
                f"Parsed feature string is not a list: {json_path}, got type={type(parsed)}"
            )
        features = parsed

    else:
        raise ValueError(
            f"Feature JSON must be a list or string-encoded list: {json_path}, got type={type(features)}"
        )

    return [str(c) for c in features]


def build_xy_dataset(
    start_date: str,
    end_date: str,
    y_name: str,
    topn: int,
    export: bool = True,
    max_rows: Optional[int] = None,
    data_dir: str = DATA_DIR_DEFAULT,
    feature_root: str = FEATURE_ROOT_DEFAULT,
    export_dir: str = EXPORT_DIR_DEFAULT,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    LGBM / Linear Regression 용 X, y 데이터셋을 만드는 함수.

    Parameters
    ----------
    start_date : str
        "YYYY-MM-DD" 형식의 시작 날짜 (inclusive)
    end_date : str
        "YYYY-MM-DD" 형식의 끝 날짜 (inclusive)
    y_name : str
        타겟 컬럼 이름 (예: "y_60m", "y_30m", ...)
    topn : int
        universe 크기 (예: 30 -> *_xy_top30.h5)
    export : bool, default True
        True면 export_dir에 HDF5 파일로 저장
    max_rows : Optional[int], default None
        전체 row 중에서 max_rows개 만큼만 샘플링 (None 이면 전체 사용)
    data_dir : str
        원본 xy HDF5 파일이 있는 최상위 디렉토리 (기본: 프로젝트루트/data)
    feature_root : str
        feature json이 있는 최상위 디렉토리 (기본: 프로젝트루트/feature_list)
    export_dir : str
        export=True일 때 결과를 저장할 디렉토리 (기본: 프로젝트루트/data/datasets)
    random_state : int
        max_rows 샘플링 시에 사용할 random seed

    Returns
    -------
    X : pd.DataFrame
        모델 입력 feature 데이터프레임
    y : pd.Series
        타겟 시리즈
    """
    feature_list = _load_feature_list(y_name=y_name, topn=topn, feature_root=feature_root)

    all_rows = []
    dates = _iter_dates(start_date, end_date)

    for d in dates:
        file_name = f"{d}_xy_top{topn}.h5"
        file_path = os.path.join(data_dir, "xy", file_name)

        if not os.path.exists(file_path):
            print(f"[WARN] File not found, skip: {file_path}")
            continue

        try:
            df = pd.read_hdf(file_path)
        except Exception as e:
            print(f"[WARN] Failed to read {file_path}: {e}")
            continue

        needed_cols = [y_name] + feature_list
        existing_cols = [c for c in needed_cols if c in df.columns]

        if y_name not in existing_cols:
            print(
                f"[WARN] y_name '{y_name}' not found in {file_path}, skip this file."
            )
            continue

        missing_feats = [c for c in feature_list if c not in df.columns]
        if missing_feats:
            print(
                f"[INFO] {file_path}: {len(missing_feats)} features missing "
                f"(e.g. {missing_feats[:3]} ...), they will be ignored."
            )

        sub = df[existing_cols].copy()

        sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
        if sub.empty:
            print(f"[WARN] After dropna, no rows left in {file_path}")
            continue

        sub["__date__"] = d
        all_rows.append(sub)

        print(f"[INFO] Loaded {len(sub):6d} rows from {file_path}")

    if not all_rows:
        raise ValueError("No data collected. Check date range / files / y_name.")

    full_df = pd.concat(all_rows, axis=0, ignore_index=True)
    print(f"[INFO] Total rows before sampling: {len(full_df)}")

    if (max_rows is not None) and (len(full_df) > max_rows):
        full_df = full_df.sample(n=max_rows, random_state=random_state)
        full_df = full_df.sort_index()
        print(f"[INFO] Sampled down to {len(full_df)} rows with max_rows={max_rows}")

    final_feature_cols = [c for c in feature_list if c in full_df.columns]
    X = full_df[final_feature_cols].copy()
    y = full_df[y_name].copy()

    print(f"[INFO] Final X shape: {X.shape}, y shape: {y.shape}")

    if export:
        os.makedirs(export_dir, exist_ok=True)
        out_fname = f"{y_name}_top{topn}_{start_date}_{end_date}_n{len(full_df)}.h5"
        out_path = os.path.join(export_dir, out_fname)

        export_df = full_df[[y_name] + final_feature_cols + ["__date__"]].copy()
        export_df.to_hdf(out_path, key="df", mode="w")

        meta = {
            "y_name": y_name,
            "topn": topn,
            "start_date": start_date,
            "end_date": end_date,
            "n_rows": int(len(export_df)),
            "n_features": int(len(final_feature_cols)),
            "feature_list_path": os.path.join(
                feature_root, y_name, f"top{topn}_example_features_166.json"
            ),
            "feature_cols": final_feature_cols,
        }
        meta_path = out_path.replace(".h5", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[INFO] Saved dataset to: {out_path}")
        print(f"[INFO] Saved meta to:     {meta_path}")

    return X, y

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build XY dataset for LGBM / Linear Regression"
    )
    parser.add_argument("--start_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--y_name", required=True, help="target column, e.g. y_60m")
    parser.add_argument("--topn", type=int, required=True, help="universe size (e.g. 30)")
    parser.add_argument("--max_rows", type=int, default=None, help="cap total rows (optional)")
    parser.add_argument(
    )
    parser.add_argument(
        "--no_export",
        action="store_true",
        help="set this flag if you do NOT want to export to HDF5",
    )
    args = parser.parse_args()

    build_xy_dataset(
        start_date=args.start_date,
        end_date=args.end_date,
        y_name=args.y_name,
        topn=args.topn,
        export=not args.no_export,
        max_rows=args.max_rows,
    )
    
"""
command:
python machine_learning/datasets.py \
  --start_date 2024-02-01 \
  --end_date 2024-02-10 \
  --y_name y_60m \
  --topn 30

python machine_learning/datasets.py \
    --start_date 2024-02-01 \
    --end_date 2024-02-10 \
    --y_name y_60m \
    --topn 30 \
    --no_export

python machine_learning/datasets.py \
    --start_date 2024-02-01 \
    --end_date 2024-03-01 \
    --y_name y_60m \
    --topn 30 \
    --max_rows 50000

"""