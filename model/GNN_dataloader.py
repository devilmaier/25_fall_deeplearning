import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class CryptoGraphDataset(Dataset):
    """
    PyTorch Dataset for GNN-LSTM model.
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
            
            # Get symbols at this timestamp
            symbols = ts_data['symbol'].unique()
            
            # Only keep timestamps where we have enough symbols
            if len(symbols) >= num_nodes:
                # Take first num_nodes symbols (sorted alphabetically for consistency)
                symbols = sorted(symbols)[:num_nodes]
                
                # Extract features and targets for each symbol
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
        
        # Create valid sample indices (sequences of consecutive timestamps)
        self.sample_indices = []
        
        for i in range(len(self.timestamps) - seq_len + 1):
            # Check if we have consecutive timestamps with enough data
            time_window = self.timestamps[i:i + seq_len]
            
            # Check if all timestamps in window have data
            if all(ts in self.time_groups for ts in time_window):
                # Check if symbols are consistent across the window
                # (Use intersection of symbols to ensure all nodes have data across time)
                symbol_sets = [set(self.time_groups[ts]['symbols']) for ts in time_window]
                common_symbols = set.intersection(*symbol_sets)
                
                if len(common_symbols) >= num_nodes:
                    # Use first num_nodes symbols (sorted for consistency)
                    common_symbols = sorted(list(common_symbols))[:num_nodes]
                    self.sample_indices.append((i, common_symbols))
        
        print(f"[{mode.upper()}] Created {len(self.sample_indices)} valid graph sequences")

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        start_idx, symbols = self.sample_indices[idx]
        time_window = self.timestamps[start_idx:start_idx + self.seq_len]
        
        # Initialize tensors
        # x shape: (Time, Nodes, Features)
        # y shape: (Nodes,) - targets for the last timestamp
        num_features = len(self.feature_cols)
        x = np.zeros((self.seq_len, self.num_nodes, num_features), dtype=np.float32)
        y = np.zeros(self.num_nodes, dtype=np.float32)
        
        # Fill in the data
        for t_idx, ts in enumerate(time_window):
            group = self.time_groups[ts]
            for n_idx, sym in enumerate(symbols):
                x[t_idx, n_idx, :] = group['features'][sym]
                
                # For the last timestamp, store the target
                if t_idx == self.seq_len - 1:
                    y[n_idx] = group['targets'][sym]
        
        return torch.tensor(x), torch.tensor(y)


def get_gnn_loaders(
    data_dir, 
    start_date, 
    end_date, 
    top_n=30, 
    feature_list=None, 
    target_col='y_60m', 
    seq_len=60, 
    num_nodes=30,
    batch_size=32, 
    num_workers=4
):
    """
    Function to load data for GNN-LSTM model and return DataLoaders.
    
    Args:
        data_dir: Directory containing .h5 files
        start_date: Start date for data loading
        end_date: End date for data loading
        top_n: Top N coins (should match num_nodes)
        feature_list: Path to JSON file or list of feature names
        target_col: Target column name
        seq_len: Sequence length (time steps)
        num_nodes: Number of nodes (symbols) per graph
        batch_size: Batch size
        num_workers: Number of worker threads
    
    Returns:
        train_loader, val_loader, test_loader, feature_dim
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
    
    print(f"[GNN LOADER] Loading {len(file_paths)} files...")
    df_list = [pd.read_hdf(p, mode='r') for p in file_paths]
    full_df = pd.concat(df_list, ignore_index=True)
    
    # Auto-detect features if not provided
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
            print(f"[GNN LOADER] Using {len(features)} available features out of {original_count} requested")
        
        if len(features) == 0:
            raise ValueError("No valid features found in the dataframe. Please check your feature list.")
    
    print(f"[GNN LOADER] Features: {len(features)}, Target: {target_col}, Seq_Len: {seq_len}, Num_Nodes: {num_nodes}")

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
    train_ds = CryptoGraphDataset(train_df, features, target_col, seq_len, num_nodes, 'train')
    val_ds = CryptoGraphDataset(val_df, features, target_col, seq_len, num_nodes, 'val')
    test_ds = CryptoGraphDataset(test_df, features, target_col, seq_len, num_nodes, 'test')
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, len(features)
