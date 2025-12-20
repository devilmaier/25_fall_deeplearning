import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import itertools
from pathlib import Path
from tqdm import tqdm
import pandas as pd

import sys
sys.path.append(str(Path(__file__).parent / 'model'))

from model.SpatioTemporal_dataloader import get_spatiotemporal_loaders
from model.SpatioTemporalTransformer import SpatioTemporalTransformer

PROJECT_ROOT = Path(__file__).parent

PARAM_GRID = {
    'seq_len': [30, 60],
    'hidden_dim': [64, 128],
    'num_transformer_layers': [2,4],
    'num_heads': [2,4,8],
    'lr': [0.001, 0.005, 0.01],
}

BASE_CONFIG = {
    'mode': 'regression',
    'data_dir': str(PROJECT_ROOT / 'data' / 'xy'),
    'start_date': '2024-10-01',
    'end_date': '2024-11-29',
    'vali_date': '2024-11-01',
    'test_date': '2024-11-15',
    'top_n': 30,
    'num_nodes': 30,
    'output_dim': 1,
    'dropout': 0.2,
    'batch_size': 96,
    'epochs': 3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'feature_list': str(PROJECT_ROOT / 'feature_list' / 'y_60m' / 'top30_example_features_166.json'),
    'ban_list_path': str(PROJECT_ROOT / 'global_ban_dates.json'),
    'export_path': str(PROJECT_ROOT / 'data' / 'datasets' / 'spatiotfm'),
    'cnn_loss_weight': 0.2,
    'transformer_loss_weight': 0.3,
    'train_num': 1000000,
    'val_num': 200000,
    'test_num': 100000,
}

def compute_stats(loader):
    print("[INFO] Computing input statistics...")
    sum_x = 0
    sum_sq_x = 0
    count = 0
    for x, _ in tqdm(loader, desc="Scanning Data", leave=False):
        x_flat = x.view(-1, x.size(-1))
        sum_x += x_flat.sum(dim=0)
        sum_sq_x += (x_flat ** 2).sum(dim=0)
        count += x_flat.size(0)
    mean = sum_x / count
    var = (sum_sq_x / count) - (mean ** 2)
    std = torch.sqrt(torch.clamp(var, min=1e-9))
    return mean.view(1, 1, 1, -1), std.view(1, 1, 1, -1)

def train_evaluate(config, run_id):
    print(f"\n[Run {run_id}] Starting training with: {config}")
    
    train_loader, val_loader, test_loader, feature_dim = get_spatiotemporal_loaders(
        data_dir=BASE_CONFIG['data_dir'],
        start_date=BASE_CONFIG['start_date'],
        end_date=BASE_CONFIG['end_date'],
        vali_date=BASE_CONFIG['vali_date'],
        test_date=BASE_CONFIG['test_date'],
        top_n=BASE_CONFIG['top_n'],
        num_nodes=BASE_CONFIG['num_nodes'],
        feature_list=BASE_CONFIG['feature_list'],
        seq_len=config['seq_len'],
        batch_size=BASE_CONFIG['batch_size'],
        ban_list_path=BASE_CONFIG['ban_list_path'],
        export_path=BASE_CONFIG['export_path'],
        dataset_workers=8,
        train_num=BASE_CONFIG['train_num'],
        val_num=BASE_CONFIG['val_num'],
        test_num=BASE_CONFIG['test_num']
    )
    
    input_dim = feature_dim
    input_mean, input_std = compute_stats(train_loader)
    input_mean = input_mean.to(BASE_CONFIG['device'])
    input_std = input_std.to(BASE_CONFIG['device'])

    model = SpatioTemporalTransformer(
        num_nodes=BASE_CONFIG['num_nodes'],
        input_dim=input_dim,
        hidden_dim=config['hidden_dim'],
        output_dim=BASE_CONFIG['output_dim'],
        dropout=BASE_CONFIG['dropout'],
        num_transformer_layers=config['num_transformer_layers'],
        num_heads=config['num_heads'],
        mean=input_mean,
        std=input_std
    ).to(BASE_CONFIG['device'])

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    best_ic = -1.0
    
    for epoch in range(BASE_CONFIG['epochs']):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Run {run_id} Epoch {epoch+1}", leave=False):
            x, y = x.to(BASE_CONFIG['device']), y.to(BASE_CONFIG['device'])
            optimizer.zero_grad()
            final_pred, transformer_pred, cnn_pred = model(x)
            
            final_pred = final_pred.squeeze(-1)
            transformer_pred = transformer_pred.squeeze(-1)
            cnn_pred = cnn_pred.squeeze(-1)
            
            loss = criterion(final_pred, y) + \
                   BASE_CONFIG['transformer_loss_weight'] * criterion(transformer_pred, y) + \
                   BASE_CONFIG['cnn_loss_weight'] * criterion(cnn_pred, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        preds_list = []
        targets_list = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(BASE_CONFIG['device']), y.to(BASE_CONFIG['device'])
                final_pred, _, _ = model(x)
                final_pred = final_pred.squeeze(-1)
                
                preds_list.append(final_pred.cpu().numpy())
                targets_list.append(y.cpu().numpy())
        
        preds_arr = np.concatenate(preds_list).flatten()
        targets_arr = np.concatenate(targets_list).flatten()
        
        current_ic = 0.0
        if len(preds_arr) > 1:
            current_ic = np.corrcoef(preds_arr, targets_arr)[0, 1]
            if np.isnan(current_ic):
                current_ic = 0.0

        if current_ic > best_ic:
            best_ic = current_ic

    return best_ic

def main():
    keys, values = zip(*PARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = []
    print(f"[INFO] Total combinations to test: {len(combinations)}")
    
    for i, config in enumerate(combinations):
        try:
            best_ic = train_evaluate(config, i+1)
            result = {**config, 'best_ic': best_ic}
            results.append(result)
            print(f"[Result] Run {i+1}: {result}")
        except Exception as e:
            print(f"[Error] Run {i+1} failed: {e}")
            continue

    # Save results
    if not results:
        print("[Error] No results to save. All runs failed.")
        return

    df = pd.DataFrame(results)
    if 'best_ic' in df.columns:
        df = df.sort_values('best_ic', ascending=False)
    else:
        print("[Warn] 'best_ic' column not found in results.")
    
    df.to_csv('grid_search_results.csv', index=False)
    print("\n[INFO] Grid Search Completed. Top 5 configurations:")
    print(df.head())

if __name__ == "__main__":
    main()
