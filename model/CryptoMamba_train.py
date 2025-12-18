import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import custom modules
from CryptoMamba_dataloader import get_mamba_loaders
from CryptoMamba import CryptoMamba
from loss_functions import get_loss_function, HybridFinancialLoss, AdaptiveHybridLoss

# Get project root directory (parent of model directory)
PROJECT_ROOT = Path(__file__).parent.parent

# ==========================================
# Configuration
# ==========================================
CONFIG = {
    'data_dir': str(PROJECT_ROOT / 'data' / 'xy'),
    'start_date': '2024-03-01',
    'end_date': '2024-03-30',
    'top_n': 30,
    'num_nodes': 30,          # Number of nodes in graph
    'seq_len': 240,           # Time window size
    'input_dim': 166,         # Will be updated automatically based on features
    'hidden_dim': 128,        # Mamba hidden dimension
    'output_dim': 1,    
    'num_mamba_layers': 2,    # Number of Mamba layers
    'd_state': 16,            # SSM state expansion factor
    'd_conv': 4,              # Local convolution width
    'expand': 2,              # Block expansion factor
    'dropout': 0.2,
    'batch_size': 64,
    'epochs': 10,
    'lr': 0.001,
    'num_workers': 24,        # Number of workers for data loading (0 = single process)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'feature_list': str(PROJECT_ROOT / 'feature_list' / 'y_60m' / 'top30_example_features_166.json'),     # Set to None to auto-detect '_neut' features
    'ban_list_path': str(PROJECT_ROOT / 'global_ban_dates.json'),
    'save_path': str(PROJECT_ROOT / 'best_cryptomamba_model.pt'),
    'export_path': str(PROJECT_ROOT / 'data' / 'datasets' / 'mamba'),
    'skip_normalization': False,  # Set to True if data is already normalized
    
    # ==========================================
    # Loss Function Configuration
    # ==========================================
    'loss_type': 'hybrid',  # Options: 'mse', 'hybrid', 'adaptive', 'directional', 'ic'
    
    # Hybrid Loss weights (used if loss_type='hybrid')
    'mse_weight': 0.2,      # Weight for MSE component
    'dir_weight': 0.1,      # Weight for directional component (방향 중시!)
    'ic_weight': 0.4,       # Weight for IC component
    
    # Adaptive Loss settings (used if loss_type='adaptive')
    'initial_weights': (0.8, 0.1, 0.1),  # (mse, dir, ic) for early epochs
    'final_weights': (0.2, 0.4, 0.4),    # (mse, dir, ic) for late epochs
    
    # Temperature for smooth sign function
    'temperature': 0.2,
}


def compute_stats(loader, num_workers=0):
    """
    Scans the training set to compute Mean and Std for each feature.
    For Mamba: input shape is (Batch, Time, Nodes, Features)
    Returns tensors of shape (1, 1, 1, Feature_Dim).
    
    Args:
        loader: DataLoader (can have num_workers=0)
        num_workers: Number of workers for parallel data loading
    
    Note: This function will recreate the DataLoader with specified num_workers
    """
    print(f"[INFO] Computing input statistics (Mean, Std) for CryptoMamba...")
    if num_workers > 0:
        print(f"[INFO] Using {num_workers} workers for parallel data loading")
    
    # Recreate loader with specified num_workers for faster loading
    if num_workers > 0:
        fast_loader = DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            shuffle=False,  # Don't shuffle for statistics
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False,
            persistent_workers=True if num_workers > 0 else False
        )
    else:
        fast_loader = loader
    
    sum_x = 0
    sum_sq_x = 0
    count = 0

    # Iterate over the entire training set
    for x, _ in tqdm(fast_loader, desc="Scanning Data"):
        # x shape: (Batch, Time, Nodes, Features)
        
        # Flatten all dimensions except Features
        # Shape: (Batch * Time * Nodes, Features)
        x_flat = x.view(-1, x.size(-1))
        
        sum_x += x_flat.sum(dim=0)
        sum_sq_x += (x_flat ** 2).sum(dim=0)
        count += x_flat.size(0)

    # Calculate Mean
    mean = sum_x / count
    
    # Calculate Std: sqrt(E[X^2] - (E[X])^2)
    var = (sum_sq_x / count) - (mean ** 2)
    std = torch.sqrt(torch.clamp(var, min=1e-9))
    
    # Reshape to (1, 1, 1, Features) for broadcasting
    return mean.view(1, 1, 1, -1), std.view(1, 1, 1, -1)


def compute_stats_fast(dataset, batch_size=32, num_workers=24):
    """
    Fast version of compute_stats using optimized DataLoader settings.
    
    Args:
        dataset: Dataset object
        batch_size: Batch size for loading (larger = faster but more memory)
        num_workers: Number of parallel workers (default: 24)
    
    Returns:
        mean, std: Statistics tensors
    """
    print(f"[FAST] Computing statistics with {num_workers} workers, batch_size={batch_size}")
    
    # Create optimized DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True,
        prefetch_factor=2  # Prefetch 2 batches per worker
    )
    
    sum_x = 0
    sum_sq_x = 0
    count = 0

    for x, _ in tqdm(loader, desc="Computing Stats (Fast)"):
        x_flat = x.view(-1, x.size(-1))
        sum_x += x_flat.sum(dim=0)
        sum_sq_x += (x_flat ** 2).sum(dim=0)
        count += x_flat.size(0)

    mean = sum_x / count
    var = (sum_sq_x / count) - (mean ** 2)
    std = torch.sqrt(torch.clamp(var, min=1e-9))
    
    return mean.view(1, 1, 1, -1), std.view(1, 1, 1, -1)


def load_from_cache(export_path, batch_size, num_workers=4, device='cpu', compute_normalization=True):
    """
    Directly load DataLoaders from cached files without rebuilding dataset.
    This is faster when you already have processed and exported data.
    
    Args:
        export_path: Path to cached dataset directory
        batch_size: Batch size for DataLoader
        num_workers: Number of workers for DataLoader
        device: Device to load statistics to (for normalization)
        compute_normalization: If False, skip normalization computation (use zeros/ones)
    
    Returns:
        train_loader, val_loader, test_loader, feature_dim, input_mean, input_std
    """
    print(f"[CACHE] Loading dataset directly from {export_path}")
    
    # Check if all required files exist
    required_files = {
        'train': os.path.join(export_path, 'train_df.h5'),
        'val': os.path.join(export_path, 'val_df.h5'),
        'test': os.path.join(export_path, 'test_df.h5'),
        'meta': os.path.join(export_path, 'meta.json')
    }
    
    for name, filepath in required_files.items():
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"[ERROR] Required file not found: {filepath}")
    
    # Load metadata
    with open(required_files['meta'], 'r') as f:
        meta = json.load(f)
    
    features = meta['feature_list']
    target_col = meta.get('target_col', 'y_60m')
    seq_len = meta['seq_len']
    num_nodes = meta['num_nodes']
    
    print(f"[CACHE] Metadata loaded:")
    print(f"  - Features: {len(features)}")
    print(f"  - Target: {target_col}")
    print(f"  - Sequence length: {seq_len}")
    print(f"  - Number of nodes: {num_nodes}")
    
    # Load dataframes
    print(f"[CACHE] Loading dataframes...")
    train_df = pd.read_hdf(required_files['train'])
    val_df = pd.read_hdf(required_files['val'])
    test_df = pd.read_hdf(required_files['test'])
    
    print(f"[CACHE] Loaded {len(train_df)} train rows, {len(val_df)} val rows, {len(test_df)} test rows")
    
    # Create datasets
    from CryptoMamba_dataloader import CryptoMambaDataset
    
    train_ds = CryptoMambaDataset(train_df, features, target_col, seq_len, num_nodes, 'train')
    val_ds = CryptoMambaDataset(val_df, features, target_col, seq_len, num_nodes, 'val')
    test_ds = CryptoMambaDataset(test_df, features, target_col, seq_len, num_nodes, 'test')
    
    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"[CACHE] DataLoaders created:")
    print(f"  - Train samples: {len(train_ds)}")
    print(f"  - Val samples: {len(val_ds)}")
    print(f"  - Test samples: {len(test_ds)}")
    
    # Compute or skip normalization statistics
    if compute_normalization:
        print(f"[CACHE] Computing normalization statistics with {num_workers} workers...")
        # Use fast version with dataset directly for maximum speed
        input_mean, input_std = compute_stats_fast(
            train_ds, 
            batch_size=batch_size * 2,  # Use larger batch for stats computation
            num_workers=num_workers
        )
        input_mean = input_mean.to(device)
        input_std = input_std.to(device)
    else:
        print(f"[CACHE] Skipping normalization (data already normalized)")
        # Return identity normalization (mean=0, std=1)
        input_mean = torch.zeros(1, 1, 1, len(features)).to(device)
        input_std = torch.ones(1, 1, 1, len(features)).to(device)
    
    return train_loader, val_loader, test_loader, len(features), input_mean, input_std


def train():
    print(f"[INFO] Device: {CONFIG['device']}")
    print(f"[INFO] Model: CryptoMamba (Mamba-based Time Series Model)")
    
    # 1. Check if we should use cached data directly
    use_direct_cache = False
    if CONFIG['export_path'] is not None:
        cache_files = [
            os.path.join(CONFIG['export_path'], 'train_df.h5'),
            os.path.join(CONFIG['export_path'], 'val_df.h5'),
            os.path.join(CONFIG['export_path'], 'test_df.h5'),
            os.path.join(CONFIG['export_path'], 'meta.json')
        ]
        
        if all(os.path.exists(f) for f in cache_files):
            print(f"\n{'='*70}")
            print(f"[CACHE] Found existing cached dataset at:")
            print(f"        {CONFIG['export_path']}")
            print(f"[CACHE] Using cached data directly (skip data preprocessing)")
            print(f"{'='*70}\n")
            use_direct_cache = True
    
    # 2. Load data (either from cache or rebuild)
    if use_direct_cache:
        # Direct cache loading - fastest method
        train_loader, val_loader, test_loader, feature_dim, input_mean, input_std = load_from_cache(
            export_path=CONFIG['export_path'],
            batch_size=CONFIG['batch_size'],
            num_workers=CONFIG['num_workers'],  # Use configured num_workers
            device=CONFIG['device'],
            compute_normalization=not CONFIG['skip_normalization']  # Skip if data already normalized
        )
    else:
        # Normal loading with get_mamba_loaders (will create cache if export_path is set)
        print(f"[INFO] Loading data using get_mamba_loaders...")
        train_loader, val_loader, test_loader, feature_dim = get_mamba_loaders(
            data_dir=CONFIG['data_dir'],
            start_date=CONFIG['start_date'],
            end_date=CONFIG['end_date'],
            top_n=CONFIG['top_n'],
            num_nodes=CONFIG['num_nodes'],
            feature_list=CONFIG['feature_list'],
            seq_len=CONFIG['seq_len'],
            batch_size=CONFIG['batch_size'],
            ban_list_path=CONFIG['ban_list_path'],
            export_path=CONFIG['export_path'],
        )
        
        # Compute statistics for normalization (or skip if already normalized)
        if CONFIG['skip_normalization']:
            print(f"[INFO] Skipping normalization (data already normalized)")
            input_mean = torch.zeros(1, 1, 1, feature_dim).to(CONFIG['device'])
            input_std = torch.ones(1, 1, 1, feature_dim).to(CONFIG['device'])
        else:
            print(f"[INFO] Computing normalization statistics with {CONFIG['num_workers']} workers...")
            # Use fast version with dataset directly
            input_mean, input_std = compute_stats_fast(
                train_loader.dataset,
                batch_size=CONFIG['batch_size'] * 2,
                num_workers=CONFIG['num_workers']
            )
            input_mean = input_mean.to(CONFIG['device'])
            input_std = input_std.to(CONFIG['device'])
    
    CONFIG['input_dim'] = feature_dim
    print(f"\n[INFO] Input feature dim: {feature_dim}")

    # 3. Initialize Model with Stats
    model = CryptoMamba(
        num_nodes=CONFIG['num_nodes'],
        input_dim=CONFIG['input_dim'], 
        hidden_dim=CONFIG['hidden_dim'],
        output_dim=CONFIG['output_dim'],
        dropout=CONFIG['dropout'],
        num_mamba_layers=CONFIG['num_mamba_layers'],
        d_state=CONFIG['d_state'],
        d_conv=CONFIG['d_conv'],
        expand=CONFIG['expand'],
        mean=input_mean,
        std=input_std
    ).to(CONFIG['device'])
    
    print(f"[INFO] Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 4. Initialize Loss Function
    print(f"\n[INFO] Loss Function: {CONFIG['loss_type']}")
    
    if CONFIG['loss_type'] == 'mse':
        criterion = nn.MSELoss()
        print(f"  - Using MSE Loss")
    
    elif CONFIG['loss_type'] == 'hybrid':
        criterion = HybridFinancialLoss(
            mse_weight=CONFIG['mse_weight'],
            dir_weight=CONFIG['dir_weight'],
            ic_weight=CONFIG['ic_weight'],
            temperature=CONFIG['temperature']
        )
        print(f"  - Hybrid Loss Weights:")
        print(f"    MSE:        {CONFIG['mse_weight']:.2f}")
        print(f"    Directional: {CONFIG['dir_weight']:.2f}")
        print(f"    IC:         {CONFIG['ic_weight']:.2f}")
    
    elif CONFIG['loss_type'] == 'adaptive':
        criterion = AdaptiveHybridLoss(
            total_epochs=CONFIG['epochs'],
            initial_weights=CONFIG['initial_weights'],
            final_weights=CONFIG['final_weights'],
            temperature=CONFIG['temperature']
        )
        print(f"  - Adaptive Hybrid Loss:")
        print(f"    Initial weights (MSE, Dir, IC): {CONFIG['initial_weights']}")
        print(f"    Final weights (MSE, Dir, IC):   {CONFIG['final_weights']}")
    
    else:
        # Use factory function for other loss types
        criterion = get_loss_function(
            loss_type=CONFIG['loss_type'],
            temperature=CONFIG.get('temperature', 1.0)
        )
        print(f"  - Using {CONFIG['loss_type']} loss")
    
    # Track if using custom loss (returns dict)
    use_custom_loss = CONFIG['loss_type'] in ['hybrid', 'adaptive', 'directional', 'ic']
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    model_saved = False

    # 5. Training Loop
    print("\n[INFO] Start Training...")
    for epoch in range(CONFIG['epochs']):
        # Update adaptive loss weights if using adaptive loss
        if CONFIG['loss_type'] == 'adaptive':
            criterion.set_epoch(epoch)
            weights = (
                criterion.loss_fn.mse_weight,
                criterion.loss_fn.dir_weight,
                criterion.loss_fn.ic_weight
            )
            print(f"\n[Epoch {epoch+1}] Adaptive weights: MSE={weights[0]:.2f}, Dir={weights[1]:.2f}, IC={weights[2]:.2f}")
        
        # --- Train Step ---
        model.train()
        train_loss = 0.0
        train_loss_components = {'mse': 0.0, 'directional': 0.0, 'ic_loss': 0.0, 'ic_value': 0.0}
        
        for batch_idx, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]")):
            x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
            
            # Debug: Check for NaN/Inf in input data
            if torch.isnan(x).any():
                print(f"[WARNING] NaN detected in input data at batch {batch_idx}")
                continue
            if torch.isinf(x).any():
                print(f"[WARNING] Inf detected in input data at batch {batch_idx}")
                continue
            if torch.isnan(y).any():
                print(f"[WARNING] NaN detected in target data at batch {batch_idx}")
                continue
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(x)  # (Batch, Nodes)
            
            # Calculate loss
            if use_custom_loss:
                # Custom loss returns (loss, loss_dict)
                loss, loss_dict = criterion(predictions, y)
                
                # Accumulate loss components for logging
                for key in loss_dict:
                    if key in train_loss_components:
                        train_loss_components[key] += loss_dict[key]
            else:
                # Standard MSE loss
                loss = criterion(predictions, y)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"[WARNING] NaN loss at batch {batch_idx}, skipping...")
                continue
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Log loss components if using custom loss
        if use_custom_loss:
            num_batches = len(train_loader)
            print(f"\n  Train Loss Components:")
            if 'mse' in train_loss_components and train_loss_components['mse'] > 0:
                print(f"    MSE:         {train_loss_components['mse'] / num_batches:.6f}")
            if 'directional' in train_loss_components and train_loss_components['directional'] > 0:
                print(f"    Directional: {train_loss_components['directional'] / num_batches:.6f}")
            if 'ic_loss' in train_loss_components and train_loss_components['ic_loss'] > 0:
                print(f"    IC Loss:     {train_loss_components['ic_loss'] / num_batches:.6f}")
            if 'ic_value' in train_loss_components:
                print(f"    IC Value:    {train_loss_components['ic_value'] / num_batches:.6f}")

        # --- Validation Step ---
        model.eval()
        val_loss = 0.0
        val_loss_components = {'mse': 0.0, 'directional': 0.0, 'ic_loss': 0.0, 'ic_value': 0.0}
        
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Val]"):
                x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
                
                predictions = model(x)
                
                if use_custom_loss:
                    loss, loss_dict = criterion(predictions, y)
                    # Accumulate validation loss components
                    for key in loss_dict:
                        if key in val_loss_components:
                            val_loss_components[key] += loss_dict[key]
                else:
                    loss = criterion(predictions, y)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Step scheduler
        scheduler.step(avg_val_loss)
        
        print(f"\nEpoch [{epoch+1}/{CONFIG['epochs']}]")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss:   {avg_val_loss:.6f}")
        
        # Log validation components if using custom loss
        if use_custom_loss:
            num_val_batches = len(val_loader)
            print(f"  Val Loss Components:")
            if 'mse' in val_loss_components and val_loss_components['mse'] > 0:
                print(f"    MSE:         {val_loss_components['mse'] / num_val_batches:.6f}")
            if 'directional' in val_loss_components and val_loss_components['directional'] > 0:
                print(f"    Directional: {val_loss_components['directional'] / num_val_batches:.6f}")
            if 'ic_loss' in val_loss_components and val_loss_components['ic_loss'] > 0:
                print(f"    IC Loss:     {val_loss_components['ic_loss'] / num_val_batches:.6f}")
            if 'ic_value' in val_loss_components:
                print(f"    IC Value:    {val_loss_components['ic_value'] / num_val_batches:.6f}")

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'config': CONFIG,
                'input_mean': input_mean,
                'input_std': input_std
            }, CONFIG['save_path'])
            model_saved = True
            print(f"  -> Model saved (Val loss improved)")

    # 5. Final Evaluation (Test Set)
    print("\n[INFO] Evaluating on Test Set...")
    
    # Load best model if it was saved
    if model_saved and os.path.exists(CONFIG['save_path']):
        checkpoint = torch.load(CONFIG['save_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] Loaded best model from epoch {checkpoint['epoch']+1}")
    else:
        print("[INFO] No model checkpoint found, using current model state")
    
    model.eval()
    
    test_loss = 0.0
    predictions_list = []
    targets_list = []
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
            
            predictions = model(x)
            
            if use_custom_loss:
                loss, _ = criterion(predictions, y)
            else:
                loss = criterion(predictions, y)
            
            test_loss += loss.item()
            
            # Flatten predictions and targets for correlation calculation
            predictions_list.extend(predictions.cpu().numpy().flatten())
            targets_list.extend(y.cpu().numpy().flatten())

    avg_test_loss = test_loss / len(test_loader)
    
    print(f"\n{'='*70}")
    print(f"Final Test Results:")
    print(f"  Test MSE: {avg_test_loss:.6f}")
    
    # Calculate Information Coefficient (IC)
    if len(predictions_list) > 1:
        ic = np.corrcoef(predictions_list, targets_list)[0, 1]
        print(f"  Test IC (Correlation): {ic:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    train()
