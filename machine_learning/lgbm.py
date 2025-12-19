# machine_learning/lgbm.py
"""
End-to-end LightGBM training + inference + evaluation (IC) using datasets.py

Usage examples
--------------
# Train + eval on a date range (dataset will be built on the fly)
python -m machine_learning.lgbm \
  --mode train_eval \
  --train_start 2025-02-01 --train_end 2025-04-30 \
  --valid_start 2025-05-01 --valid_end 2025-05-14 \
  --test_start  2025-05-15 --test_end  2025-05-28 \
  --y_name y_60m --topn 30 \
  --max_rows 600000 \
  --sample_by_day \
  --num_boost_round 4000 \
  --early_stopping_rounds 200

# Inference only on some dates (requires trained model + feature meta)
python machine_learning/lgbm.py \
  --mode infer \
  --infer_start 2024-03-02 --infer_end 2024-03-05 \
  --y_name y_60m --topn 30 \
  --model_path /path/to/model.pkl \
  --meta_path  /path/to/meta.json \
  --save_preds

Notes
-----
- This file expects:
  - Your existing: machine_learning/datasets.py (build_xy_dataset)
  - HDF5 per-day files: data/xy/{YYYY-MM-DD}_xy_top{topn}.h5
  - feature list json: feature_list/{y_name}/top{topn}_example_features_166.json

- IC is computed as:
  - "ic_pearson": Pearson corr between y and prediction
  - "ic_spearman": Spearman rank corr between y and prediction
  - "daily_ic_*": compute IC per __date__ then average (more robust)
"""

import os
import json
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

import numpy as np
import pandas as pd


# LightGBM import
try:
    import lightgbm as lgb
except Exception as e:
    raise ImportError(
        "lightgbm is not installed. Try: pip install lightgbm"
    ) from e

# Our dataset builder
from machine_learning.datasets import build_xy_dataset, DATA_DIR_DEFAULT, EXPORT_DIR_DEFAULT, FEATURE_ROOT_DEFAULT

# -------------------------
# Utils: metrics (IC)
# -------------------------
def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0:
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    # Spearman = Pearson of ranks
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0:
        return float("nan")
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    return _pearson_corr(ra, rb)


def compute_ic(
    df: pd.DataFrame,
    y_col: str,
    pred_col: str,
    date_col: str = "__date__",
) -> Dict[str, float]:
    """
    Returns overall IC + daily IC mean/std.
    """
    out: Dict[str, float] = {}

    y = df[y_col].to_numpy()
    p = df[pred_col].to_numpy()

    out["ic_pearson"] = _pearson_corr(y, p)
    out["ic_spearman"] = _spearman_corr(y, p)

    if date_col in df.columns:
        daily = []
        for d, g in df.groupby(date_col):
            yy = g[y_col].to_numpy()
            pp = g[pred_col].to_numpy()
            daily.append((_pearson_corr(yy, pp), _spearman_corr(yy, pp)))
        daily = np.asarray(daily, dtype=float)
        if daily.size > 0:
            out["daily_ic_pearson_mean"] = float(np.nanmean(daily[:, 0]))
            out["daily_ic_pearson_std"] = float(np.nanstd(daily[:, 0]))
            out["daily_ic_spearman_mean"] = float(np.nanmean(daily[:, 1]))
            out["daily_ic_spearman_std"] = float(np.nanstd(daily[:, 1]))
            out["n_days"] = float(len(daily))
        else:
            out["daily_ic_pearson_mean"] = float("nan")
            out["daily_ic_pearson_std"] = float("nan")
            out["daily_ic_spearman_mean"] = float("nan")
            out["daily_ic_spearman_std"] = float("nan")
            out["n_days"] = 0.0

    return out


# -------------------------
# Sampling helper
# -------------------------
def sample_by_day(
    full_df: pd.DataFrame,
    max_rows: int,
    date_col: str = "__date__",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    More stable than global .sample(): cap total rows while keeping days balanced.
    """
    if max_rows is None or len(full_df) <= max_rows:
        return full_df

    if date_col not in full_df.columns:
        # fallback
        return full_df.sample(n=max_rows, random_state=random_state).sort_index()

    rng = np.random.RandomState(random_state)
    days = sorted(full_df[date_col].unique().tolist())
    if len(days) == 0:
        return full_df.sample(n=max_rows, random_state=random_state).sort_index()

    per_day = max(1, max_rows // len(days))
    chunks = []
    for d in days:
        g = full_df[full_df[date_col] == d]
        if len(g) <= per_day:
            chunks.append(g)
        else:
            idx = rng.choice(g.index.values, size=per_day, replace=False)
            chunks.append(g.loc[idx])
    out = pd.concat(chunks, axis=0).sort_index()
    # If still over max_rows due to rounding, trim
    if len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=random_state).sort_index()
    return out


# -------------------------
# Data loading (X, y + meta df)
# -------------------------
def build_xy_with_meta(
    start_date: str,
    end_date: str,
    y_name: str,
    topn: int,
    max_rows: Optional[int],
    data_dir: str,
    feature_root: str,
    random_state: int,
    do_sample_by_day: bool,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Uses build_xy_dataset (which returns X,y) but also reconstructs meta df by re-loading
    exported HDF if export=True, or by directly reading per-day files again.

    To keep things simple & reliable: we build an internal dataframe in one pass here
    by calling build_xy_dataset(export=False) and then re-reading per-day to get __date__.
    But your datasets.py already attaches __date__ internally; we can just replicate its logic.

    Easiest approach:
      - call build_xy_dataset(export=False) to get X,y (already cleaned)
      - separately call build_xy_dataset(export=True) in training mode to save meta if desired
    Here we instead implement a small extra pass to attach __date__ by re-calling build_xy_dataset
    with export=True to create an HDF and read it back (single source of truth).

    We’ll do:
      - export=False for speed? No, we want __date__. So we do export=True into a temp path if needed.
    """
    # Build with export to get y + X + __date__ in one DF saved, then read it.
    # We'll save into export_dir under a deterministic file name and reuse if exists.
    os.makedirs(EXPORT_DIR_DEFAULT, exist_ok=True)

    # create dataset file (or reuse if already exists)
    X, y = build_xy_dataset(
        start_date=start_date,
        end_date=end_date,
        y_name=y_name,
        topn=topn,
        export=True,              # make the HDF + meta json
        max_rows=None,            # we will do sampling ourselves (optionally day-balanced)
        data_dir=data_dir,
        feature_root=feature_root,
        export_dir=EXPORT_DIR_DEFAULT,
        random_state=random_state,
    )

    # Find latest exported file that matches pattern (same params).
    # datasets.py uses: f"{y_name}_top{topn}_{start}_{end}_n{len(full_df)}.h5"
    # We don't know n in advance, so we locate by prefix.
    prefix = f"{y_name}_top{topn}_{start_date}_{end_date}_n"
    candidates = [
        fn for fn in os.listdir(EXPORT_DIR_DEFAULT)
        if fn.startswith(prefix) and fn.endswith(".h5")
    ]
    if not candidates:
        raise RuntimeError("Exported dataset file not found after build_xy_dataset(export=True).")

    # pick most recent by mtime
    candidates = sorted(
        candidates,
        key=lambda fn: os.path.getmtime(os.path.join(EXPORT_DIR_DEFAULT, fn)),
        reverse=True,
    )
    ds_path = os.path.join(EXPORT_DIR_DEFAULT, candidates[0])
    df = pd.read_hdf(ds_path, key="df")

    # Optional sampling
    if max_rows is not None and len(df) > max_rows:
        if do_sample_by_day:
            df = sample_by_day(df, max_rows=max_rows, random_state=random_state)
        else:
            df = df.sample(n=max_rows, random_state=random_state).sort_index()

    # Separate back to X,y with exact columns present
    feature_cols = [c for c in df.columns if c not in (y_name, "__date__")]
    X2 = df[feature_cols].copy()
    y2 = df[y_name].copy()
    return X2, y2, df


# -------------------------
# Model IO
# -------------------------
def save_model(model: lgb.Booster, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save_model(path)


def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_model(path: str) -> lgb.Booster:
    if not os.path.exists(path):
        raise FileNotFoundError(f"model_path not found: {path}")
    return lgb.Booster(model_file=path)


def load_meta(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"meta_path not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


# -------------------------
# Train / Eval
# -------------------------
@dataclass
class TrainConfig:
    y_name: str
    topn: int
    random_state: int = 42

    # data
    data_dir: str = DATA_DIR_DEFAULT
    feature_root: str = None  # default handled below
    max_rows: Optional[int] = None
    sample_by_day: bool = False

    # lgb params
    objective: str = "regression"
    metric: str = "rmse"
    learning_rate: float = 0.02
    num_leaves: int = 255
    min_data_in_leaf: int = 200
    feature_fraction: float = 0.2
    bagging_fraction: float = 0.2
    bagging_freq: int = 1
    lambda_l1: float = 0.0
    lambda_l2: float = 0.0
    max_depth: int = -1

    # training control
    num_boost_round: int = 4000
    early_stopping_rounds: int = 200

    # output
    out_dir: str = os.path.join(EXPORT_DIR_DEFAULT, "lgbm_models")


def train_and_evaluate(
    cfg: TrainConfig,
    train_range: Tuple[str, str],
    valid_range: Tuple[str, str],
    test_range: Tuple[str, str],
    save_preds: bool = True,
) -> Dict[str, Any]:
    if cfg.feature_root is None:
        cfg.feature_root = FEATURE_ROOT_DEFAULT
    else:
        cfg.feature_root = cfg.feature_root

    t0 = time.time()

    # Build datasets (with __date__)
    Xtr, ytr, dftr = build_xy_with_meta(
        start_date=train_range[0], end_date=train_range[1],
        y_name=cfg.y_name, topn=cfg.topn,
        max_rows=cfg.max_rows,
        data_dir=cfg.data_dir,
        feature_root=cfg.feature_root,
        random_state=cfg.random_state,
        do_sample_by_day=cfg.sample_by_day,
    )
    Xva, yva, dfva = build_xy_with_meta(
        start_date=valid_range[0], end_date=valid_range[1],
        y_name=cfg.y_name, topn=cfg.topn,
        max_rows=None,  # usually keep full valid
        data_dir=cfg.data_dir,
        feature_root=cfg.feature_root,
        random_state=cfg.random_state,
        do_sample_by_day=False,
    )
    Xte, yte, dfte = build_xy_with_meta(
        start_date=test_range[0], end_date=test_range[1],
        y_name=cfg.y_name, topn=cfg.topn,
        max_rows=None,
        data_dir=cfg.data_dir,
        feature_root=cfg.feature_root,
        random_state=cfg.random_state,
        do_sample_by_day=False,
    )

    # Align columns across splits (robustness: some missing features in some days)
    common_cols = sorted(list(set(Xtr.columns) & set(Xva.columns) & set(Xte.columns)))
    if len(common_cols) == 0:
        raise ValueError("No common features across train/valid/test after alignment.")
    Xtr = Xtr[common_cols]
    Xva = Xva[common_cols]
    Xte = Xte[common_cols]

    # LightGBM datasets
    dtrain = lgb.Dataset(Xtr, label=ytr, free_raw_data=False)
    dvalid = lgb.Dataset(Xva, label=yva, reference=dtrain, free_raw_data=False)

    params = dict(
        objective=cfg.objective,
        metric=cfg.metric,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        min_data_in_leaf=cfg.min_data_in_leaf,
        feature_fraction=cfg.feature_fraction,
        bagging_fraction=cfg.bagging_fraction,
        bagging_freq=cfg.bagging_freq,
        lambda_l1=cfg.lambda_l1,
        lambda_l2=cfg.lambda_l2,
        max_depth=cfg.max_depth,
        seed=cfg.random_state,
        feature_pre_filter=False,
        verbosity=-1,
    )

    # Train
    booster = lgb.train(
        params=params,
        train_set=dtrain,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        num_boost_round=cfg.num_boost_round,
        callbacks=[
            lgb.early_stopping(cfg.early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    best_iter = booster.best_iteration or cfg.num_boost_round

    # Predict
    dftr_pred = dftr.copy()
    dfva_pred = dfva.copy()
    dfte_pred = dfte.copy()

    dftr_pred["pred"] = booster.predict(Xtr, num_iteration=best_iter)
    dfva_pred["pred"] = booster.predict(Xva, num_iteration=best_iter)
    dfte_pred["pred"] = booster.predict(Xte, num_iteration=best_iter)

    # IC metrics
    train_ic = compute_ic(dftr_pred, y_col=cfg.y_name, pred_col="pred")
    valid_ic = compute_ic(dfva_pred, y_col=cfg.y_name, pred_col="pred")
    test_ic  = compute_ic(dfte_pred, y_col=cfg.y_name, pred_col="pred")

    # Save artifacts
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"lgbm_{cfg.y_name}_top{cfg.topn}_{train_range[0]}_{train_range[1]}__{ts}"
    run_dir = os.path.join(cfg.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    model_path = os.path.join(run_dir, "model.txt")
    save_model(booster, model_path)

    meta = {
        "run_name": run_name,
        "y_name": cfg.y_name,
        "topn": cfg.topn,
        "feature_cols": common_cols,
        "train_range": list(train_range),
        "valid_range": list(valid_range),
        "test_range": list(test_range),
        "best_iteration": int(best_iter),
        "lgb_params": params,
        "train_ic": train_ic,
        "valid_ic": valid_ic,
        "test_ic": test_ic,
        "n_train": int(len(dftr_pred)),
        "n_valid": int(len(dfva_pred)),
        "n_test": int(len(dfte_pred)),
        "elapsed_sec": float(time.time() - t0),
    }
    meta_path = os.path.join(run_dir, "meta.json")
    save_json(meta, meta_path)

    # Feature importance
    imp = pd.DataFrame({
        "feature": common_cols,
        "importance_gain": booster.feature_importance(importance_type="gain"),
        "importance_split": booster.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)
    imp_path = os.path.join(run_dir, "feature_importance.csv")
    imp.to_csv(imp_path, index=False)

    # Save predictions
    if save_preds:
        dftr_pred.to_parquet(os.path.join(run_dir, "preds_train.parquet"), index=False)
        dfva_pred.to_parquet(os.path.join(run_dir, "preds_valid.parquet"), index=False)
        dfte_pred.to_parquet(os.path.join(run_dir, "preds_test.parquet"), index=False)

    print("\n========== IC Summary ==========")
    print("[TRAIN]", train_ic)
    print("[VALID]", valid_ic)
    print("[TEST ]",  test_ic)
    print("Artifacts saved to:", run_dir)

    return {
        "run_dir": run_dir,
        "model_path": model_path,
        "meta_path": meta_path,
        "train_ic": train_ic,
        "valid_ic": valid_ic,
        "test_ic": test_ic,
        "best_iteration": best_iter,
    }


# -------------------------
# Inference only
# -------------------------
def infer(
    model_path: str,
    meta_path: str,
    infer_range: Tuple[str, str],
    y_name: str,
    topn: int,
    data_dir: str,
    feature_root: str,
    max_rows: Optional[int],
    random_state: int,
    sample_by_day_flag: bool,
    save_preds: bool,
    out_dir: str,
) -> Dict[str, Any]:
    booster = load_model(model_path)
    meta = load_meta(meta_path)

    # Build dataset with __date__
    X, y, df = build_xy_with_meta(
        start_date=infer_range[0], end_date=infer_range[1],
        y_name=y_name, topn=topn,
        max_rows=max_rows,
        data_dir=data_dir,
        feature_root=feature_root,
        random_state=random_state,
        do_sample_by_day=sample_by_day_flag,
    )

    # Use the exact feature cols from meta (safe)
    feat_cols = meta.get("feature_cols", None)
    if feat_cols is None:
        raise ValueError("meta.json missing 'feature_cols'")
    # Some columns might be missing (rare); intersect
    use_cols = [c for c in feat_cols if c in X.columns]
    if len(use_cols) == 0:
        raise ValueError("No usable features found in inference data after intersect with meta feature_cols.")
    X = X[use_cols]

    df_out = df.copy()
    df_out["pred"] = booster.predict(X, num_iteration=meta.get("best_iteration", None))

    # If y exists, compute IC
    metrics = {}
    if y_name in df_out.columns:
        metrics = compute_ic(df_out, y_col=y_name, pred_col="pred")

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"infer_{y_name}_top{topn}_{infer_range[0]}_{infer_range[1]}__{ts}"
    run_dir = os.path.join(out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    if save_preds:
        out_path = os.path.join(run_dir, "preds.parquet")
        df_out.to_parquet(out_path, index=False)
        save_json(
            {
                "model_path": model_path,
                "meta_path": meta_path,
                "infer_range": list(infer_range),
                "n_rows": int(len(df_out)),
                "metrics": metrics,
                "saved_preds": out_path,
            },
            os.path.join(run_dir, "infer_meta.json"),
        )
        print("Saved inference outputs to:", run_dir)

    if metrics:
        print("\n========== Inference IC ==========")
        print(metrics)

    return {"run_dir": run_dir, "metrics": metrics, "n_rows": int(len(df_out))}


# -------------------------
# CLI
# -------------------------
def main():
    import argparse

    p = argparse.ArgumentParser(description="LightGBM train/eval/infer using datasets.py")
    p.add_argument("--mode", choices=["train_eval", "infer"], required=True)

    # shared
    p.add_argument("--y_name", required=True)
    p.add_argument("--topn", type=int, required=True)
    p.add_argument("--data_dir", default=DATA_DIR_DEFAULT)
    p.add_argument("--feature_root", default=FEATURE_ROOT_DEFAULT)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--max_rows", type=int, default=None)
    p.add_argument("--sample_by_day", action="store_true")

    # train ranges
    p.add_argument("--train_start", default=None)
    p.add_argument("--train_end", default=None)
    p.add_argument("--valid_start", default=None)
    p.add_argument("--valid_end", default=None)
    p.add_argument("--test_start", default=None)
    p.add_argument("--test_end", default=None)

    # lgb params
    p.add_argument("--learning_rate", type=float, default=0.03)
    p.add_argument("--num_leaves", type=int, default=255)
    p.add_argument("--min_data_in_leaf", type=int, default=200)
    p.add_argument("--feature_fraction", type=float, default=0.8)
    p.add_argument("--bagging_fraction", type=float, default=0.8)
    p.add_argument("--bagging_freq", type=int, default=1)
    p.add_argument("--lambda_l1", type=float, default=0.0)
    p.add_argument("--lambda_l2", type=float, default=0.0)
    p.add_argument("--max_depth", type=int, default=-1)
    p.add_argument("--num_boost_round", type=int, default=4000)
    p.add_argument("--early_stopping_rounds", type=int, default=200)

    # outputs
    p.add_argument("--out_dir", default=os.path.join(EXPORT_DIR_DEFAULT, "lgbm_models"))
    p.add_argument("--save_preds", action="store_true")

    # inference args
    p.add_argument("--model_path", default=None)
    p.add_argument("--meta_path", default=None)
    p.add_argument("--infer_start", default=None)
    p.add_argument("--infer_end", default=None)

    args = p.parse_args()

    if args.mode == "train_eval":
        required = ["train_start", "train_end", "valid_start", "valid_end", "test_start", "test_end"]
        missing = [k for k in required if getattr(args, k) is None]
        if missing:
            raise ValueError(f"Missing args for train_eval: {missing}")

        cfg = TrainConfig(
            y_name=args.y_name,
            topn=args.topn,
            random_state=args.random_state,
            data_dir=args.data_dir,
            feature_root=args.feature_root,
            max_rows=args.max_rows,
            sample_by_day=args.sample_by_day,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            min_data_in_leaf=args.min_data_in_leaf,
            feature_fraction=args.feature_fraction,
            bagging_fraction=args.bagging_fraction,
            bagging_freq=args.bagging_freq,
            lambda_l1=args.lambda_l1,
            lambda_l2=args.lambda_l2,
            max_depth=args.max_depth,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping_rounds,
            out_dir=args.out_dir,
        )

        train_and_evaluate(
            cfg=cfg,
            train_range=(args.train_start, args.train_end),
            valid_range=(args.valid_start, args.valid_end),
            test_range=(args.test_start, args.test_end),
            save_preds=args.save_preds,
        )

    else:  # infer
        required = ["model_path", "meta_path", "infer_start", "infer_end"]
        missing = [k for k in required if getattr(args, k) is None]
        if missing:
            raise ValueError(f"Missing args for infer: {missing}")

        infer(
            model_path=args.model_path,
            meta_path=args.meta_path,
            infer_range=(args.infer_start, args.infer_end),
            y_name=args.y_name,
            topn=args.topn,
            data_dir=args.data_dir,
            feature_root=args.feature_root,
            max_rows=args.max_rows,
            random_state=args.random_state,
            sample_by_day_flag=args.sample_by_day,
            save_preds=args.save_preds,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
