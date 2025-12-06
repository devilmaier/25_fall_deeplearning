import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class CryptoDataset(Dataset):
    """
    PyTorch Dataset for cryptocurrency data.
    Supports both time-series (seq_len > 1) and flat (seq_len = 1) data.
    """
    def __init__(self, df, feature_cols, target_col, seq_len=60, mode='train'):
        self.seq_len = seq_len
        self.target_col = target_col
        self.feature_cols = feature_cols
        
        init_len = len(df)
        df = df.dropna(subset=[target_col])
        
        df = df.fillna(0)
        
        df = df.replace([np.inf, -np.inf], 0)
        
        print(f"[{mode.upper()}] Cleaned NaNs: {init_len} -> {len(df)} rows (Dropped {init_len - len(df)})")

        # Group data by symbol to prevent mixing sequences between different coins
        self.grouped_data = []
        self.grouped_targets = []
        
        # --- Option 1: Time-Series Data (for LSTM, CNN) ---
        if seq_len > 1:
            # print(f"[INFO] Building {mode} dataset (Sequence Mode)...") # 너무 시끄러우면 주석 처리
            for sym, group in tqdm(df.groupby('symbol'), desc=f"Building {mode}"):
                group = group.sort_values('start_time_ms')
                
                feats = group[feature_cols].values.astype(np.float32)
                targets = group[target_col].values.astype(np.float32)
                
                if len(feats) <= seq_len:
                    continue
                
                self.grouped_data.append(feats)
                self.grouped_targets.append(targets)
                
            # Pre-calculate indices for all valid windows
            self.sample_indices = []
            for g_idx, g_data in enumerate(self.grouped_data):
                num_windows = len(g_data) - seq_len
                for start_row in range(num_windows):
                    self.sample_indices.append((g_idx, start_row))
                    
        # --- Option 2: Flat Data (for MLP, LightGBM) ---
        else:
            self.data = df[feature_cols].values.astype(np.float32)
            self.targets = df[target_col].values.astype(np.float32)

    def __len__(self):
        if self.seq_len > 1:
            return len(self.sample_indices)
        return len(self.data)

    def __getitem__(self, idx):
        if self.seq_len > 1:
            # Retrieve sequence window
            group_idx, start_row = self.sample_indices[idx]
            data_block = self.grouped_data[group_idx]
            target_block = self.grouped_targets[group_idx]
            
            # Input: (seq_len, features)
            x = data_block[start_row : start_row + self.seq_len]
            # Target: Scalar value at the last step of the window
            y = target_block[start_row + self.seq_len - 1]
            
            return torch.tensor(x), torch.tensor(y)
        else:
            # Retrieve single row
            return torch.tensor(self.data[idx]), torch.tensor(self.targets[idx])


def get_loaders(
    data_dir, 
    start_date, 
    end_date, 
    top_n=30, 
    feature_list=None, 
    target_col='y_60m', 
    seq_len=60, 
    batch_size=32, 
    num_workers=4
):
    """
    Standard function to load data, split into train/val/test, and return DataLoaders.
    """
    # 1. Load Feature List
    if isinstance(feature_list, str) and feature_list.endswith('.json'):
        with open(feature_list, 'r') as f:
            features = json.load(f)
            # Handle case where JSON contains a string representation of a list
            if isinstance(features, str):
                import ast
                features = ast.literal_eval(features)
    elif isinstance(feature_list, list):
        features = feature_list
    else:
        features = None # Auto-detection

    # 2. Load and Merge Data Files
    dates = pd.date_range(start_date, end_date).strftime("%Y-%m-%d").tolist()
    file_paths = [os.path.join(data_dir, f"{d}_xy_top{top_n}.h5") for d in dates]
    file_paths = [p for p in file_paths if os.path.exists(p)]
    
    if not file_paths:
        raise FileNotFoundError("No data files found for the given date range.")
    
    print(f"[LOADER] Loading {len(file_paths)} files...")
    df_list = [pd.read_hdf(p, mode='r') for p in file_paths]
    full_df = pd.concat(df_list, ignore_index=True)
    
    # Auto-detect features if not provided (looks for '_neut' suffix)
    if features is None:
        features = [c for c in full_df.columns if c.startswith('x_') and c.endswith('_neut')]
    else:
        # Filter features to only include those that exist in the dataframe
        available_cols = set(full_df.columns)
        original_features = features.copy() if isinstance(features, list) else list(features)
        original_count = len(original_features)
        features = [f for f in original_features if f in available_cols]
        
        if len(features) < original_count:
            missing = [f for f in original_features if f not in available_cols]
            print(f"[WARNING] {original_count - len(features)} features not found in data")
            if len(missing) <= 10:
                print(f"[WARNING] Missing features: {missing}")
            else:
                print(f"[WARNING] Missing features (first 10): {missing[:10]}...")
            print(f"[LOADER] Using {len(features)} available features out of {original_count} requested")
        
        if len(features) == 0:
            raise ValueError("No valid features found in the dataframe. Please check your feature list.")
    
    print(f"[LOADER] Features: {len(features)}, Target: {target_col}, Seq_Len: {seq_len}")

    # 3. Time-based Split (Train 80% / Val 10% / Test 10%)
    dates_sorted = full_df['start_time_ms'].unique()
    dates_sorted.sort()
    
    n_dates = len(dates_sorted)
    train_idx = int(n_dates * 0.8)
    val_idx = int(n_dates * 0.9)
    
    train_times = dates_sorted[:train_idx]
    val_times = dates_sorted[train_idx:val_idx]
    test_times = dates_sorted[val_idx:]
    
    train_df = full_df[full_df['start_time_ms'].isin(train_times)]
    val_df = full_df[full_df['start_time_ms'].isin(val_times)]
    test_df = full_df[full_df['start_time_ms'].isin(test_times)]

    # 4. Create Datasets & Loaders
    train_ds = CryptoDataset(train_df, features, target_col, seq_len, 'train')
    val_ds = CryptoDataset(val_df, features, target_col, seq_len, 'val')
    test_ds = CryptoDataset(test_df, features, target_col, seq_len, 'test')
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, len(features)