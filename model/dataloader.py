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
        
        # 1. Drop rows where target (y) is NaN (e.g., end of dataset)
        init_len = len(df)
        df = df.dropna(subset=[target_col])
        
        # 2. Fill NaN features with 0 to prevent model instability
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)
        
        print(f"[{mode.upper()}] Cleaned NaNs: {init_len} -> {len(df)} rows (Dropped {init_len - len(df)})")

        self.grouped_data = []
        self.grouped_targets = []
        
        # --- Option 1: Time-Series Data (for LSTM, CNN) ---
        if seq_len > 1:
            # Group by symbol to prevent mixing sequences between different coins
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
    num_workers=4,
    ban_list_path=None
):
    """
    Standard function to load data, apply ban list logic, split, and return DataLoaders.
    """
    # 1. Load Ban List (Missing dates & NaN masking)
    missing_dates = []
    nan_dates_map = {}
    
    if ban_list_path and os.path.exists(ban_list_path):
        print(f"[LOADER] Loading ban list from {ban_list_path}")
        with open(ban_list_path, 'r') as f:
            ban_data = json.load(f)
            missing_dates = ban_data.get("missing_dates", [])
            nan_dates_map = ban_data.get("nan_dates", {})

    # 2. Load Feature List
    if isinstance(feature_list, str) and feature_list.endswith('.json'):
        with open(feature_list, 'r') as f:
            features = json.load(f)
            if isinstance(features, str):
                import ast
                features = ast.literal_eval(features)
    elif isinstance(feature_list, list):
        features = feature_list
    else:
        features = None # Auto-detection

    # 3. Generate Date Range & Filter Missing Dates
    all_dates = pd.date_range(start_date, end_date).strftime("%Y-%m-%d").tolist()
    dates = [d for d in all_dates if d not in missing_dates]
    
    if len(dates) < len(all_dates):
        print(f"[LOADER] Skipped {len(all_dates) - len(dates)} dates defined in ban list.")

    # 4. Load Files & Apply Masking
    print(f"[LOADER] Loading {len(dates)} files...")
    df_list = []
    
    for d in dates:
        file_path = os.path.join(data_dir, f"{d}_xy_top{top_n}.h5")
        if os.path.exists(file_path):
            daily_df = pd.read_hdf(file_path, mode='r')
            
            # Apply Masking: Set specific features to 0 for specific dates
            if d in nan_dates_map:
                bad_features = nan_dates_map[d]
                # Filter features that exist in the dataframe
                valid_bad_features = [c for c in bad_features if c in daily_df.columns]
                
                if valid_bad_features:
                    daily_df.loc[:, valid_bad_features] = 0
            
            df_list.append(daily_df)
    
    if not df_list:
        raise FileNotFoundError("No valid data files loaded. Check date range or ban list.")

    full_df = pd.concat(df_list, ignore_index=True)
    
    # 5. Feature Selection
    if features is None:
        features = [c for c in full_df.columns if c.startswith('x_') and c.endswith('_neut')]
    else:
        available_cols = set(full_df.columns)
        original_features = features.copy() if isinstance(features, list) else list(features)
        original_count = len(original_features)
        features = [f for f in original_features if f in available_cols]
        
        if len(features) < original_count:
            missing = [f for f in original_features if f not in available_cols]
            print(f"[WARNING] {original_count - len(features)} features not found in data.")
        
        if len(features) == 0:
            raise ValueError("No valid features found in the dataframe.")
    
    print(f"[LOADER] Features: {len(features)}, Target: {target_col}, Seq_Len: {seq_len}")

    # 6. Time-based Split (Train 80% / Val 10% / Test 10%)
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

    # 7. Create Datasets & Loaders
    train_ds = CryptoDataset(train_df, features, target_col, seq_len, 'train')
    val_ds = CryptoDataset(val_df, features, target_col, seq_len, 'val')
    test_ds = CryptoDataset(test_df, features, target_col, seq_len, 'test')
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, len(features)