import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class SpatioTemporalDataset(Dataset):
    """
    PyTorch Dataset for SpatioTemporalTransformer model.
    Creates graph-structured data where each sample contains multiple symbols (nodes)
    at the same time window.
    
    Output shape: (Batch, Time, Nodes, Features)
    """
    def __init__(self, df, feature_cols, target_col, seq_len=60, num_nodes=30, mode='train'):
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.target_col = target_col
        self.feature_cols = feature_cols
        
        init_len = len(df)
        df = df.dropna(subset=[target_col])
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)
        
        print(f"[{mode.upper()}] Cleaned NaNs: {init_len} -> {len(df)} rows (Dropped {init_len - len(df)})")

        # Sort by time and symbol
        df = df.sort_values(['start_time_ms', 'symbol'])
        
        # Get unique timestamps
        self.timestamps = df['start_time_ms'].unique()
        self.timestamps.sort()
        
        # Group data by timestamp
        self.time_groups = {}
        print(f"[INFO] Building {mode} dataset (Graph Mode)...")
        for ts in tqdm(self.timestamps, desc=f"Building {mode}"):
            ts_data = df[df['start_time_ms'] == ts]
            symbols = ts_data['symbol'].unique()
            
            if len(symbols) >= num_nodes:
                symbols = sorted(symbols)[:num_nodes]
                
                features_dict = {}
                targets_dict = {}
                
                for sym in symbols:
                    sym_data = ts_data[ts_data['symbol'] == sym].iloc[0]
                    features_dict[sym] = sym_data[feature_cols].values.astype(np.float32)
                    targets_dict[sym] = sym_data[target_col].astype(np.float32)
                
                self.time_groups[ts] = {
                    'symbols': symbols,
                    'features': features_dict,
                    'targets': targets_dict
                }
        
        # Create valid sample indices (sequences of timestamps)
        self.sample_indices = []
        
        for i in range(len(self.timestamps) - seq_len + 1):
            time_window = self.timestamps[i:i + seq_len]
            
            if all(ts in self.time_groups for ts in time_window):
                symbol_sets = [set(self.time_groups[ts]['symbols']) for ts in time_window]
                common_symbols = set.intersection(*symbol_sets)
                
                if len(common_symbols) >= num_nodes:
                    common_symbols = sorted(list(common_symbols))[:num_nodes]
                    self.sample_indices.append((i, common_symbols))
        
        print(f"[{mode.upper()}] Created {len(self.sample_indices)} valid graph sequences")

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        start_idx, symbols = self.sample_indices[idx]
        time_window = self.timestamps[start_idx:start_idx + self.seq_len]
        
        num_features = len(self.feature_cols)
        x = np.zeros((self.seq_len, self.num_nodes, num_features), dtype=np.float32)
        y = np.zeros(self.num_nodes, dtype=np.float32)
        
        for t_idx, ts in enumerate(time_window):
            group = self.time_groups[ts]
            for n_idx, sym in enumerate(symbols):
                x[t_idx, n_idx, :] = group['features'][sym]
                if t_idx == self.seq_len - 1:
                    y[n_idx] = group['targets'][sym]
        
        return torch.tensor(x), torch.tensor(y)



def get_spatiotemporal_loaders(
    data_dir, 
    start_date, 
    end_date, 
    top_n=30, 
    feature_list=None, 
    target_col='y_60m', 
    seq_len=60, 
    num_nodes=30,
    batch_size=32, 
    num_workers=4,
    ban_list_path=None,
    export_path=None    # <-- ⭐ 추가됨
):
    """
    Function to load data for SpatioTemporalTransformer model and return DataLoaders.
    """

    # =====================================================================
    # ⭐ 0. CACHE 기능: 이미 export된 데이터가 있으면 바로 load
    # =====================================================================
    if export_path is not None:
        train_fp = os.path.join(export_path, "train_df.h5")
        val_fp   = os.path.join(export_path, "val_df.h5")
        test_fp  = os.path.join(export_path, "test_df.h5")
        meta_fp  = os.path.join(export_path, "meta.json")

        if all(os.path.exists(f) for f in [train_fp, val_fp, test_fp, meta_fp]):
            print(f"[CACHE] Loading cached dataset from {export_path}")

            train_df = pd.read_hdf(train_fp)
            val_df   = pd.read_hdf(val_fp)
            test_df  = pd.read_hdf(test_fp)

            with open(meta_fp, "r") as f:
                meta = json.load(f)

            features = meta["feature_list"]
            seq_len  = meta["seq_len"]
            num_nodes = meta["num_nodes"]

            train_ds = SpatioTemporalDataset(train_df, features, target_col, seq_len, num_nodes, 'train')
            val_ds   = SpatioTemporalDataset(val_df,   features, target_col, seq_len, num_nodes, 'val')
            test_ds  = SpatioTemporalDataset(test_df,  features, target_col, seq_len, num_nodes, 'test')

            return (
                DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers),
                DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers),
                DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers),
                len(features)
            )

        else:
            print("[CACHE] No valid cache found. Building dataset...")

    # =====================================================================
    # 1. Load Ban List
    # =====================================================================
    missing_dates = []
    nan_dates_map = {}
    
    if ban_list_path and os.path.exists(ban_list_path):
        print(f"[ST LOADER] Loading ban list from {ban_list_path}")
        with open(ban_list_path, 'r') as f:
            ban = json.load(f)
            missing_dates = ban.get("missing_dates", [])
            nan_dates_map = ban.get("nan_dates", {})

    # =====================================================================
    # 2. Load Feature List
    # =====================================================================
    if isinstance(feature_list, str) and feature_list.endswith('.json'):
        with open(feature_list, 'r') as f:
            ft = json.load(f)
            if isinstance(ft, str):
                import ast
                ft = ast.literal_eval(ft)
        features = ft
    elif isinstance(feature_list, list):
        features = feature_list
    else:
        features = None

    # =====================================================================
    # 3. Load .h5 data files
    # =====================================================================
    all_dates = pd.date_range(start_date, end_date).strftime("%Y-%m-%d").tolist()
    dates = [d for d in all_dates if d not in missing_dates]

    print(f"[ST LOADER] Loading {len(dates)} files...")
    df_list = []
    
    for d in dates:
        fp = os.path.join(data_dir, f"{d}_xy_top{top_n}.h5")
        if os.path.exists(fp):
            df = pd.read_hdf(fp)

            if d in nan_dates_map:
                bad_feats = [c for c in nan_dates_map[d] if c in df.columns]
                df.loc[:, bad_feats] = 0

            df_list.append(df)

    if not df_list:
        raise FileNotFoundError("No valid data files loaded.")

    full_df = pd.concat(df_list, ignore_index=True)

    # Auto-detect features
    if features is None:
        features = [c for c in full_df.columns if c.startswith("x_")]
    else:
        features = [f for f in features if f in full_df.columns]

    print(f"[ST LOADER] Features={len(features)}, Target={target_col}, Seq={seq_len}, Nodes={num_nodes}")

    # =====================================================================
    # 4. Time-based split
    # =====================================================================
    times = full_df["start_time_ms"].unique()
    times.sort()

    n = len(times)
    train_t = times[: int(n * 0.8)]
    val_t   = times[int(n * 0.8) : int(n * 0.9)]
    test_t  = times[int(n * 0.9):]

    train_df = full_df[full_df["start_time_ms"].isin(train_t)]
    val_df   = full_df[full_df["start_time_ms"].isin(val_t)]
    test_df  = full_df[full_df["start_time_ms"].isin(test_t)]

    # =====================================================================
    # 5. EXPORT 기능: full, train, val, test, meta 저장
    # =====================================================================
    if export_path is not None:
        os.makedirs(export_path, exist_ok=True)

        full_df.to_hdf(os.path.join(export_path, "full_df.h5"), key="df", mode="w")
        train_df.to_hdf(os.path.join(export_path, "train_df.h5"), key="df", mode="w")
        val_df.to_hdf(os.path.join(export_path, "val_df.h5"), key="df", mode="w")
        test_df.to_hdf(os.path.join(export_path, "test_df.h5"), key="df", mode="w")

        meta = {
            "feature_list": features,
            "target_col": target_col,
            "seq_len": seq_len,
            "num_nodes": num_nodes,
            "start_date": start_date,
            "end_date": end_date,
            "top_n": top_n
        }

        with open(os.path.join(export_path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[EXPORT] Saved dataset to: {export_path}")

    # =====================================================================
    # 6. Dataset + Loader
    # =====================================================================
    train_ds = SpatioTemporalDataset(train_df, features, target_col, seq_len, num_nodes, 'train')
    val_ds   = SpatioTemporalDataset(val_df,   features, target_col, seq_len, num_nodes, 'val')
    test_ds  = SpatioTemporalDataset(test_df,  features, target_col, seq_len, num_nodes, 'test')

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers),
        len(features)
    )
