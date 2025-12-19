# machine_learning/lgbm_grid.py
"""
Grid search wrapper for LightGBM using train_and_evaluate() from lgbm.py

Usage
-----
python -m machine_learning.lgbm_grid
"""

import itertools
import json
import os
from datetime import datetime

from machine_learning.lgbm import TrainConfig, train_and_evaluate
from machine_learning.datasets import DATA_DIR_DEFAULT, FEATURE_ROOT_DEFAULT, EXPORT_DIR_DEFAULT


# =========================
# 1. 고정 실험 설정
# =========================
Y_NAME = "y_60m"
TOPN = 30

TRAIN_RANGE = ("2024-11-01", "2025-04-30")
VALID_RANGE = ("2025-05-01", "2025-05-31")
TEST_RANGE  = ("2025-06-01", "2025-06-14")

MAX_ROWS = 600_000
SAMPLE_BY_DAY = True
RANDOM_STATE = 42


# =========================
# 2. Grid 정의 (여기만 만지면 됨)
# =========================
GRID = {
    "learning_rate": [0.01, 0.03],
    "num_leaves": [31, 63],
    "min_data_in_leaf": [20, 200],
    "feature_fraction": [0.2, 0.4],
    "bagging_fraction": [0.2, 0.4],
}


# =========================
# 3. Grid runner
# =========================
def run_grid():
    keys = list(GRID.keys())
    values = list(GRID.values())

    runs = list(itertools.product(*values))
    print(f"[INFO] Total grid runs: {len(runs)}")

    results = []

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_dir = os.path.join(EXPORT_DIR_DEFAULT, "lgbm_grid", f"{ts}_{TRAIN_RANGE[0]}")
    os.makedirs(summary_dir, exist_ok=True)

    for i, combo in enumerate(runs, 1):
        params = dict(zip(keys, combo))
        print("\n" + "=" * 80)
        print(f"[GRID {i}/{len(runs)}] params = {params}")

        cfg = TrainConfig(
            y_name=Y_NAME,
            topn=TOPN,
            random_state=RANDOM_STATE,
            data_dir=DATA_DIR_DEFAULT,
            feature_root=FEATURE_ROOT_DEFAULT,
            max_rows=MAX_ROWS,
            sample_by_day=SAMPLE_BY_DAY,
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            min_data_in_leaf=params["min_data_in_leaf"],
            feature_fraction=params["feature_fraction"],
            bagging_fraction=params["bagging_fraction"],
            out_dir=os.path.join(summary_dir, "models"),
        )

        out = train_and_evaluate(
            cfg=cfg,
            train_range=TRAIN_RANGE,
            valid_range=VALID_RANGE,
            test_range=TEST_RANGE,
            save_preds=False,   # grid에선 preds 저장 안 함
        )

        results.append({
            "params": params,
            "train_ic": out["train_ic"],
            "valid_ic": out["valid_ic"],
            "test_ic":  out["test_ic"],
            "run_dir": out["run_dir"],
        })

        # 중간 저장 (crash 대비)
        with open(os.path.join(summary_dir, "grid_results_partial.json"), "w") as f:
            json.dump(results, f, indent=2)

    # 최종 저장
    result_path = os.path.join(summary_dir, "grid_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n[INFO] Grid search finished.")
    print("Results saved to:", result_path)


if __name__ == "__main__":
    run_grid()
