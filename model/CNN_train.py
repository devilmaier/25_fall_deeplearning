import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
import random
import gc
from pathlib import Path
from tqdm import tqdm
import pandas as pd

from CNN_dataloader import get_loaders
from CNN import TimeSeries1DCNN

PROJECT_ROOT = Path(__file__).parent.parent

START_DATE = '2024-08-01' 
END_DATE   = '2025-03-14'
DATASET_FOLDER_NAME = f"cnn_{START_DATE}_to_{END_DATE}"

CONFIG = {
    'data_dir': str(PROJECT_ROOT / 'data' / 'xy'),
    'export_path': str(PROJECT_ROOT / 'data' / 'datasets' / 'cnn' / DATASET_FOLDER_NAME),
    'start_date': START_DATE,
    'end_date':   END_DATE,
    'train_start_date': '2024-08-01',
    'train_end_date':   '2025-01-31',
    'val_start_date':   '2025-02-01',
    'val_end_date':     '2025-02-28',
    'test_start_date':  '2025-03-01',
    'test_end_date':    '2025-03-14',
    'top_n': 30,
    'seq_len': 60,      
    'input_dim': 480,   
    'output_dim': 1,    
    'batch_size': 512,
    'epochs': 1,
    'lr': 0.0003,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'feature_list': str(PROJECT_ROOT / 'feature_list' / 'y_60m' / 'top30_example_features_166.json'),
    'ban_list_path': str(PROJECT_ROOT / 'global_ban_dates.json'),
    'save_path': str(PROJECT_ROOT / 'best_model.pt'),
    'num_samples': 600000,
}

def compute_stats(loader):
    """Compute mean and std for input normalization."""
    print("[INFO] Computing input statistics (Mean, Std)...")
    sum_x = 0
    sum_sq_x = 0
    count = 0

    for x, _ in tqdm(loader, desc="Scanning Data"):
        x_flat = x.view(-1, x.size(-1))
        sum_x += x_flat.sum(dim=0)
        sum_sq_x += (x_flat ** 2).sum(dim=0)
        count += x_flat.size(0)

    mean = sum_x / count
    var = (sum_sq_x / count) - (mean ** 2)
    std = torch.sqrt(torch.clamp(var, min=1e-9)) 
    
    return mean.view(1, 1, -1), std.view(1, 1, -1)


def train():
    print(f"[INFO] Device: {CONFIG['device']}")
    print(f"[INFO] Dataset Path: {CONFIG['export_path']}")
    
    try:
        train_loader, val_loader, test_loader, feature_dim = get_loaders(
            data_dir=CONFIG['data_dir'],
            start_date=CONFIG['start_date'],
            end_date=CONFIG['end_date'],
            top_n=CONFIG['top_n'],
            feature_list=CONFIG['feature_list'],
            seq_len=CONFIG['seq_len'],
            batch_size=CONFIG['batch_size'],
            ban_list_path=CONFIG['ban_list_path'],
            num_samples=CONFIG['num_samples'],
            export_path=CONFIG['export_path'],
            train_start_date=CONFIG.get('train_start_date'),
            train_end_date=CONFIG.get('train_end_date'),
            val_start_date=CONFIG.get('val_start_date'),
            val_end_date=CONFIG.get('val_end_date'),
            test_start_date=CONFIG.get('test_start_date'),
            test_end_date=CONFIG.get('test_end_date'),
        )
    except Exception as e:
        print(f"[ERROR] DataLoader failed: {e}")
        return {"best_val_loss": float('inf'), "test_loss": float('inf'), "test_corr": 0}

    CONFIG['input_dim'] = feature_dim
    print(f"[INFO] Input feature dim: {feature_dim}")

    input_mean, input_std = compute_stats(train_loader)
    input_mean = input_mean.to(CONFIG['device'])
    input_std = input_std.to(CONFIG['device'])

    model = TimeSeries1DCNN(
        input_dim=CONFIG['input_dim'], 
        output_dim=CONFIG['output_dim'],
        mean=input_mean,
        std=input_std
    ).to(CONFIG['device'])
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    best_val_loss = float('inf')
    last_val_loss = None
    model_saved = False

    print("[INFO] Start Training...")
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        
        if len(val_loader) > 0:
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
                    pred = model(x)
                    loss = criterion(pred, y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            last_val_loss = avg_val_loss
        else:
            avg_val_loss = float('inf')
            last_val_loss = None
            print(f"[WARNING] Validation set is empty. Skipping validation.")
        
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] "
              f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(CONFIG['save_path']), exist_ok=True)
            torch.save(model.state_dict(), CONFIG['save_path'])
            model_saved = True

    print("\n[INFO] Evaluating on Test Set...")
    if model_saved and os.path.exists(CONFIG['save_path']):
        model.load_state_dict(torch.load(CONFIG['save_path']))
    
    model.eval()
    test_loss = 0.0
    preds = []
    targets = []
    
    if len(test_loader) > 0:
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
                pred = model(x)
                loss = criterion(pred, y)
                test_loss += loss.item()
                
                preds.extend(pred.cpu().numpy())
                targets.extend(y.cpu().numpy())

        avg_test_loss = test_loss / len(test_loader)
        print(f"Final Test MSE: {avg_test_loss:.6f}")
        
        test_corr = 0.0
        if len(preds) > 1:
            test_corr = float(np.corrcoef(np.array(preds).flatten(), np.array(targets).flatten())[0, 1])
            print(f"Prediction Correlation (IC): {test_corr:.4f}")
    else:
        print("[WARNING] Test set is empty. Skipping evaluation.")
        avg_test_loss = float('inf')
        test_corr = 0.0

    return {
        "best_val_loss": float(best_val_loss),
        "final_val_loss": float(last_val_loss) if last_val_loss else None,
        "test_loss": float(avg_test_loss),
        "test_corr": test_corr,
    }


def train_with_overrides(hparam_overrides=None):
    """Run training with temporary configuration overrides."""
    global CONFIG
    base_config = CONFIG.copy()
    if hparam_overrides:
        base_config.update(hparam_overrides)
    CONFIG = base_config

    try:
        metrics = train()
    finally:
        CONFIG = base_config
        gc.collect()
        torch.cuda.empty_cache()

    return metrics


def hyperparam_search(
    n_random_trials: int = 10,
    result_path: Path | None = None,
):
    """Randomized grid search with dynamic caching."""
    if result_path is None:
        result_path = PROJECT_ROOT / "results" / "cnn_hparam_search.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)

    param_grid = {
        "lr": [1e-4, 3e-4, 1e-3],
        "batch_size": [256, 512, 1024],
        "epochs": [5, 10],
        "num_samples": [300_000, 600_000], 
    }

    all_combinations = []
    for lr in param_grid["lr"]:
        for batch_size in param_grid["batch_size"]:
            for epochs in param_grid["epochs"]:
                for num_samples in param_grid["num_samples"]:
                    all_combinations.append({
                        "lr": lr,
                        "batch_size": batch_size,
                        "epochs": epochs,
                        "num_samples": num_samples,
                    })

    random.seed(42)
    random.shuffle(all_combinations)
    selected_combos = all_combinations[: min(n_random_trials, len(all_combinations))]

    if result_path.exists():
        try:
            with open(result_path, "r") as f:
                results = json.load(f)
        except:
            results = []
    else:
        results = []

    print(f"[HSEARCH] Running {len(selected_combos)} hyperparameter trials...")

    for run_idx, hparams in enumerate(selected_combos, start=1):
        print(f"\n[HSEARCH] Trial {run_idx}/{len(selected_combos)}: {hparams}")

        samp_suffix = "full" if hparams["num_samples"] is None else f"samp{hparams['num_samples']}"
        seq_suffix = f"seq{CONFIG['seq_len']}" 
        tuning_folder = f"cnn_tuning_{CONFIG['start_date']}_{CONFIG['end_date']}_{samp_suffix}_{seq_suffix}"
        dynamic_export_path = str(PROJECT_ROOT / 'data' / 'datasets' / 'cnn' / tuning_folder)

        overrides = {
            "lr": hparams["lr"],
            "batch_size": hparams["batch_size"],
            "epochs": hparams["epochs"],
            "num_samples": hparams["num_samples"],
            "export_path": dynamic_export_path, 
            "save_path": str(PROJECT_ROOT / "models" / f"best_model_trial_{run_idx}.pt") 
        }
        
        metrics = train_with_overrides(overrides)

        run_record = {
            "run_index": len(results) + 1,
            "hyperparams": hparams,
            "export_path_used": dynamic_export_path,
            "metrics": metrics,
        }
        results.append(run_record)

        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"[HSEARCH] Trial {run_idx} done. IC: {metrics['test_corr']:.4f}")


def rolling_window_train(base_start_date='2024-10-01', num_windows=5, window_months=1):
    """
    Train model with rolling window dates.
    Each window shifts by 1 month from the previous one.
    
    Args:
        base_start_date: Starting date for first window (format: 'YYYY-MM-DD')
        num_windows: Number of windows to train (default: 5)
        window_months: Number of months for each window (default: 1)
    """
    print(f"[ROLLING] Starting rolling window training")
    print(f"[ROLLING] Base start date: {base_start_date}")
    print(f"[ROLLING] Number of windows: {num_windows}")
    print(f"[ROLLING] Window size: {window_months} month(s)")
    print("="*80)
    
    results = []
    start_date = pd.to_datetime(base_start_date)
    
    for window_idx in range(num_windows):
        window_start = start_date + pd.DateOffset(months=window_idx)
        
        window_start_str = window_start.strftime('%Y-%m-%d')
        
        print(f"\n{'='*80}")
        print(f"[ROLLING] Window {window_idx + 1}/{num_windows}")
        print(f"[ROLLING] Window start: {window_start_str}")
        print(f"{'='*80}\n")
        
        # Train: 6 months, Val: 1 month, Test: 2 weeks
        train_start = window_start
        train_end = train_start + pd.DateOffset(months=6)
        val_start = train_end
        val_end = val_start + pd.DateOffset(months=1)
        test_start = val_end
        test_end = test_start + pd.Timedelta(days=14)
        
        window_end_str = test_end.strftime('%Y-%m-%d')
        dataset_folder = f"cnn_{window_start_str}_to_{window_end_str}"
        
        print(f"[ROLLING] Data split:")
        print(f"  Train: {train_start.strftime('%Y-%m-%d')} ~ {train_end.strftime('%Y-%m-%d')} (6 months)")
        print(f"  Val:   {val_start.strftime('%Y-%m-%d')} ~ {val_end.strftime('%Y-%m-%d')} (1 month)")
        print(f"  Test:  {test_start.strftime('%Y-%m-%d')} ~ {test_end.strftime('%Y-%m-%d')} (2 weeks)")
        print()
        
        overrides = {
            'start_date': window_start_str,
            'end_date': window_end_str,
            'export_path': str(PROJECT_ROOT / 'data' / 'datasets' / 'cnn' / dataset_folder),
            'save_path': str(PROJECT_ROOT / 'models' / f'best_model_window_{window_idx + 1}.pt'),
            'train_start_date': train_start.strftime('%Y-%m-%d'),
            'train_end_date': train_end.strftime('%Y-%m-%d'),
            'val_start_date': val_start.strftime('%Y-%m-%d'),
            'val_end_date': val_end.strftime('%Y-%m-%d'),
            'test_start_date': test_start.strftime('%Y-%m-%d'),
            'test_end_date': test_end.strftime('%Y-%m-%d'),
        }
        
        try:
            metrics = train_with_overrides(overrides)
            
            export_path = overrides['export_path']
            meta_path = os.path.join(export_path, 'meta.json')
            
            meta_info = {}
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta_info = json.load(f)
            else:
                meta_info = {
                    'feature_list': CONFIG.get('feature_list', ''),
                    'target_col': 'y_60m',
                    'seq_len': CONFIG['seq_len'],
                    'start_date': window_start_str,
                    'end_date': window_end_str,
                    'top_n': CONFIG['top_n'],
                    'scaled': True,
                    'train_start_date': overrides['train_start_date'],
                    'train_end_date': overrides['train_end_date'],
                    'val_start_date': overrides['val_start_date'],
                    'val_end_date': overrides['val_end_date'],
                    'test_start_date': overrides['test_start_date'],
                    'test_end_date': overrides['test_end_date'],
                }
                os.makedirs(export_path, exist_ok=True)
                with open(meta_path, 'w') as f:
                    json.dump(meta_info, f, indent=2)
                print(f"[ROLLING] Meta info saved to: {meta_path}")
            
            metrics['window_idx'] = window_idx + 1
            metrics['start_date'] = window_start_str
            metrics['end_date'] = window_end_str
            metrics['export_path'] = export_path
            metrics['meta'] = meta_info
            results.append(metrics)
            
            print(f"\n[ROLLING] Window {window_idx + 1} completed:")
            print(f"  Test MSE: {metrics['test_loss']:.6f}")
            print(f"  Test IC: {metrics['test_corr']:.4f}")
            
        except Exception as e:
            print(f"[ERROR] Window {window_idx + 1} failed: {e}")
            results.append({
                'window_idx': window_idx + 1,
                'start_date': window_start_str,
                'end_date': window_end_str,
                'export_path': overrides.get('export_path', ''),
                'error': str(e)
            })
        
        gc.collect()
        torch.cuda.empty_cache()
    
    print(f"\n{'='*80}")
    print(f"[ROLLING] All windows completed!")
    print(f"{'='*80}\n")
    
    result_path = PROJECT_ROOT / "results" / "rolling_window_results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"[ROLLING] Results saved to: {result_path}")
    
    print("\n[ROLLING] Summary:")
    for i, r in enumerate(results, 1):
        if 'error' not in r:
            print(f"  Window {i} ({r['start_date']} ~ {r['end_date']}): "
                  f"IC={r['test_corr']:.4f}, MSE={r['test_loss']:.6f}")
        else:
            print(f"  Window {i} ({r['start_date']} ~ {r['end_date']}): ERROR - {r['error']}")


if __name__ == "__main__":
    mode = os.environ.get("MODE", "train")
    if mode == "search":
        hyperparam_search()
    elif mode == "rolling":
        rolling_window_train()
    else:
        train()