import os
import json
import pandas as pd
import numpy as np
import torch
import gc
import ast
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

class CryptoDataset(Dataset):
    """Memory-optimized dataset for time series sequences."""
    def __init__(self, df, feature_cols, target_col, seq_len=60, mode='train'):
        self.seq_len = seq_len
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.mode = mode
        
        if len(df) == 0:
            print(f"[WARNING] Empty dataset initialized for {mode}")
            self.data_feat = np.array([], dtype=np.float32)
            self.data_target = np.array([], dtype=np.float32)
            self.valid_indices = np.array([], dtype=np.int32)
            return

        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} not found")

        self.data_feat = df[feature_cols].values.astype(np.float32)
        self.data_target = df[target_col].values.astype(np.float32)
        
        if 'symbol' in df.columns:
            symbols = df['symbol'].values
        else:
            symbols = np.zeros(len(df))

        self.valid_indices = []
        
        if seq_len > 1:
            change_points = np.where(symbols[:-1] != symbols[1:])[0] + 1
            boundaries = np.concatenate(([0], change_points, [len(df)]))
            
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i+1]
                block_len = end - start
                
                if block_len <= seq_len:
                    continue
                
                valid_starts = np.arange(start, end - seq_len + 1)
                self.valid_indices.extend(valid_starts)
                
            self.valid_indices = np.array(self.valid_indices, dtype=np.int32)
        else:
            self.valid_indices = np.arange(len(df), dtype=np.int32)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_row = self.valid_indices[idx]
        
        if self.seq_len > 1:
            x = self.data_feat[start_row : start_row + self.seq_len]
            y = self.data_target[start_row + self.seq_len - 1]
            return torch.tensor(x), torch.tensor(y)
        else:
            x = self.data_feat[start_row]
            y = self.data_target[start_row]
            return torch.tensor(x), torch.tensor(y)


def get_loaders(
    data_dir, 
    start_date, 
    end_date, 
    top_n=30, 
    feature_list=None, 
    target_col='y_60m', 
    seq_len=60, 
    batch_size=32, 
    num_workers=4,
    ban_list_path=None,
    export_path=None,      
    num_samples=None,
    train_start_date=None, train_end_date=None,
    val_start_date=None, val_end_date=None,
    test_start_date=None, test_end_date=None
):
    """Memory-optimized data loader with incremental processing and outlier removal."""

    if export_path is not None:
        os.makedirs(export_path, exist_ok=True)
        train_fp = os.path.join(export_path, "train_df.h5")
        val_fp   = os.path.join(export_path, "val_df.h5")
        test_fp  = os.path.join(export_path, "test_df.h5")
        meta_fp  = os.path.join(export_path, "meta.json")

        if all(os.path.exists(x) for x in [train_fp, meta_fp]):
            print(f"[CACHE] Using cached dataset from {export_path}")
            
            try:
                with open(meta_fp, "r") as f:
                    meta = json.load(f)
                features = meta["feature_list"]
                
                train_df = pd.read_hdf(train_fp)
                val_df   = pd.read_hdf(val_fp) if os.path.exists(val_fp) else pd.DataFrame()
                test_df  = pd.read_hdf(test_fp) if os.path.exists(test_fp) else pd.DataFrame()

                train_ds = CryptoDataset(train_df, features, target_col, seq_len, 'train')
                val_ds   = CryptoDataset(val_df,   features, target_col, seq_len, 'val')
                test_ds  = CryptoDataset(test_df,  features, target_col, seq_len, 'test')
                
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
                test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
                
                return train_loader, val_loader, test_loader, len(features)
            except Exception as e:
                print(f"[CACHE ERROR] Failed to load cache: {e}")
                print("[CACHE] Rebuilding dataset...")

    missing_dates = []
    nan_dates_map = {}
    
    if ban_list_path and os.path.exists(ban_list_path):
        with open(ban_list_path, 'r') as f:
            ban_data = json.load(f)
            missing_dates = ban_data.get("missing_dates", [])
            nan_dates_map = ban_data.get("nan_dates", {})

    all_dates = pd.date_range(start_date, end_date).strftime("%Y-%m-%d").tolist()
    dates = [d for d in all_dates if d not in missing_dates]

    def _filter_dates(date_list, start, end):
        if start is None and end is None: return []
        return [d for d in date_list if (start is None or d >= start) and (end is None or d <= end)]

    manual_split = any([train_start_date, train_end_date, val_start_date, val_end_date, test_start_date, test_end_date])

    if manual_split:
        train_dates = _filter_dates(dates, train_start_date, train_end_date)
        val_dates   = _filter_dates(dates, val_start_date,   val_end_date)
        test_dates  = _filter_dates(dates, test_start_date,  test_end_date)
    else:
        n = len(dates)
        train_dates = dates[:int(n*0.8)]
        val_dates   = dates[int(n*0.8):int(n*0.9)]
        test_dates  = dates[int(n*0.9):]

    if isinstance(feature_list, str) and feature_list.endswith('.json'):
        with open(feature_list, 'r') as f:
            features = json.load(f)
            if isinstance(features, str):
                features = ast.literal_eval(features)
    elif isinstance(feature_list, list):
        features = feature_list
    else:
        features = None

    scaler = StandardScaler()
    print("[LOADER] Pass 1: Fitting Scaler incrementally (Train dates)...")
    
    if features is None:
        first_file = os.path.join(data_dir, f"{train_dates[0]}_xy_top{top_n}.h5")
        if os.path.exists(first_file):
            temp_df = pd.read_hdf(first_file)
            features = [c for c in temp_df.columns if c.startswith('x_')]
            del temp_df
        else:
            raise ValueError("First file not found for feature detection.")

    for d in tqdm(train_dates, desc="Fitting Scaler"):
        file_path = os.path.join(data_dir, f"{d}_xy_top{top_n}.h5")
        if not os.path.exists(file_path): continue
        
        try:
            df = pd.read_hdf(file_path)
            
            if d in nan_dates_map:
                bad = [c for c in nan_dates_map[d] if c in df.columns]
                if bad: df.loc[:, bad] = 0
            
            f_cols = df.select_dtypes(include=['float64']).columns
            df[f_cols] = df[f_cols].astype(np.float32)
            
            df = df.dropna(subset=[target_col]).fillna(0)
            df = df[df[target_col].abs() <= 1.0]

            if len(df) > 0:
                scaler.partial_fit(df[features])
                
        except Exception as e:
            print(f"Skipping {d}: {e}")
        
        del df
        gc.collect()

    print("[LOADER] Pass 2: Transforming & Saving incrementally...")

    def process_and_save(date_list, filename, is_train=False):
        out_path = os.path.join(export_path, filename)
        if os.path.exists(out_path): os.remove(out_path)
        
        saved_count = 0
        
        samples_per_file = None
        if is_train and num_samples is not None:
            samples_per_file = max(1, int(num_samples / len(date_list)))

        for d in tqdm(date_list, desc=f"Saving {filename}"):
            file_path = os.path.join(data_dir, f"{d}_xy_top{top_n}.h5")
            if not os.path.exists(file_path): continue
            
            try:
                df = pd.read_hdf(file_path)
                
                if d in nan_dates_map:
                    bad = [c for c in nan_dates_map[d] if c in df.columns]
                    if bad: df.loc[:, bad] = 0
                
                f_cols = df.select_dtypes(include=['float64']).columns
                df[f_cols] = df[f_cols].astype(np.float32)
                
                df = df.dropna(subset=[target_col]).fillna(0)
                df = df[df[target_col].abs() <= 1.0]
                
                if is_train and samples_per_file and len(df) > samples_per_file:
                    df = df.sample(n=samples_per_file)
                
                if len(df) == 0: continue
                
                df[features] = scaler.transform(df[features]).astype(np.float32)
                
                int_cols = df.select_dtypes(include=['int16', 'int32', 'int64']).columns
                if len(int_cols) > 0:
                    df[int_cols] = df[int_cols].astype(np.int32)
                
                df = df.sort_values(['symbol', 'start_time_ms'])
                
                df.to_hdf(
                    out_path, key='df', mode='a', format='table', 
                    append=True, complib='blosc', complevel=5,
                    min_itemsize={'symbol': 20}
                )
                saved_count += len(df)
                
            except Exception as e:
                print(f"Error {d}: {e}")
            
            del df
            gc.collect()
            
        print(f"Saved {saved_count} rows to {filename}")

    process_and_save(train_dates, "train_df.h5", is_train=True)
    process_and_save(val_dates,   "val_df.h5",   is_train=False)
    process_and_save(test_dates,  "test_df.h5",  is_train=False)

    meta = {
        "feature_list": features,
        "target_col": target_col,
        "seq_len": seq_len,
        "start_date": start_date,
        "end_date": end_date,
        "top_n": top_n,
        "scaled": True
    }
    with open(os.path.join(export_path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("[LOADER] Reloading processed datasets into memory...")
    
    train_df = pd.read_hdf(os.path.join(export_path, "train_df.h5"))
    val_df   = pd.read_hdf(os.path.join(export_path, "val_df.h5")) if os.path.exists(os.path.join(export_path, "val_df.h5")) else pd.DataFrame()
    test_df  = pd.read_hdf(os.path.join(export_path, "test_df.h5")) if os.path.exists(os.path.join(export_path, "test_df.h5")) else pd.DataFrame()

    print("[LOADER] Global sorting by Symbol & Time...")
    if len(train_df) > 0:
        train_df = train_df.sort_values(['symbol', 'start_time_ms']).reset_index(drop=True)
    if len(val_df) > 0:
        val_df = val_df.sort_values(['symbol', 'start_time_ms']).reset_index(drop=True)
    if len(test_df) > 0:
        test_df = test_df.sort_values(['symbol', 'start_time_ms']).reset_index(drop=True)

    # Create and save full_df
    if export_path is not None:
        df_list = []
        if len(train_df) > 0:
            df_list.append(train_df)
        if len(val_df) > 0:
            df_list.append(val_df)
        if len(test_df) > 0:
            df_list.append(test_df)
        
        if df_list:
            full_df = pd.concat(df_list, ignore_index=True)
            full_df = full_df.sort_values(['symbol', 'start_time_ms']).reset_index(drop=True)
            full_df_path = os.path.join(export_path, "full_df.h5")
            full_df.to_hdf(full_df_path, key='df', mode='w', format='table', 
                          complib='blosc', complevel=5, min_itemsize={'symbol': 20})
            print(f"[LOADER] Saved full_df to: {full_df_path} ({len(full_df)} rows)")

    train_ds = CryptoDataset(train_df, features, target_col, seq_len, 'train')
    val_ds   = CryptoDataset(val_df,   features, target_col, seq_len, 'val')
    test_ds  = CryptoDataset(test_df,  features, target_col, seq_len, 'test')
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, len(features)