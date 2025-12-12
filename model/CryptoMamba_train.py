import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# Import custom modules
from CryptoMamba_dataloader import get_mamba_loaders
from CryptoMamba import CryptoMamba

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
    'input_dim': 258,         # Will be updated automatically based on features
    'hidden_dim': 128,        # Mamba hidden dimension
    'output_dim': 1,    
    'num_mamba_layers': 4,    # Number of Mamba layers
    'd_state': 16,            # SSM state expansion factor
    'd_conv': 4,              # Local convolution width
    'expand': 2,              # Block expansion factor
    'dropout': 0.1,
    'batch_size': 96,
    'epochs': 10,
    'lr': 0.0001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'feature_list': None, #str(PROJECT_ROOT / 'feature_list' / 'y_60m' / 'top30_example_features_98.json'),     # Set to None to auto-detect '_neut' features
    'ban_list_path': str(PROJECT_ROOT / 'global_ban_dates.json'),
    'save_path': str(PROJECT_ROOT / 'best_cryptomamba_model.pt'),
    'export_path': str(PROJECT_ROOT / 'data' / 'datasets' / 'mamba'),
}


def compute_stats(loader):
    """
    Scans the training set to compute Mean and Std for each feature.
    For Mamba: input shape is (Batch, Time, Nodes, Features)
    Returns tensors of shape (1, 1, 1, Feature_Dim).
    """
    print("[INFO] Computing input statistics (Mean, Std) for CryptoMamba...")
    sum_x = 0
    sum_sq_x = 0
    count = 0

    # Iterate over the entire training set
    for x, _ in tqdm(loader, desc="Scanning Data"):
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


def train():
    print(f"[INFO] Device: {CONFIG['device']}")
    print(f"[INFO] Model: CryptoMamba (Mamba-based Time Series Model)")
    
    # 1. Prepare Data Loaders
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
    CONFIG['input_dim'] = feature_dim
    print(f"[INFO] Input feature dim: {feature_dim}")

    # 2. Compute Statistics for Normalization
    input_mean, input_std = compute_stats(train_loader)
    
    # Move stats to GPU
    input_mean = input_mean.to(CONFIG['device'])
    input_std = input_std.to(CONFIG['device'])

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
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    model_saved = False

    # 4. Training Loop
    print("[INFO] Start Training...")
    for epoch in range(CONFIG['epochs']):
        # --- Train Step ---
        model.train()
        train_loss = 0.0
        
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

        # --- Validation Step ---
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Val]"):
                x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
                
                predictions = model(x)
                loss = criterion(predictions, y)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Step scheduler
        scheduler.step(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}]")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss:   {avg_val_loss:.6f}")

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
