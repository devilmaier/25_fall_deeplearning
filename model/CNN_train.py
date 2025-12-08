#author : snuzeus
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# Import custom modules
from dataloader import get_loaders
from CNN import TimeSeries1DCNN

# Get project root directory (parent of model directory)
PROJECT_ROOT = Path(__file__).parent.parent

# ==========================================
# Configuration
# ==========================================
CONFIG = {
    'data_dir': str(PROJECT_ROOT / 'data' / 'xy'),
    'start_date': '2025-03-01',
    'end_date': '2025-03-31',
    'top_n': 30,
    'seq_len': 60,      # Window size
    'input_dim': 480,   # Will be updated automatically
    'output_dim': 1,    
    'batch_size': 512,
    'epochs': 10,
    'lr': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'feature_list': str(PROJECT_ROOT / 'feature_list' / 'y_60m' / 'top30_example_features_44.json'),
    'ban_list_path': str(PROJECT_ROOT / 'global_ban_dates.json'), # [NEW] Path to ban list
    'save_path': str(PROJECT_ROOT / 'best_model.pt'),
    'export_path': str(PROJECT_ROOT / 'data' / 'datasets' / 'cnn'),
}

def compute_stats(loader):
    """
    Scans the training set to compute Mean and Std for each feature.
    Returns tensors of shape (1, 1, Feature_Dim).
    """
    print("[INFO] Computing input statistics (Mean, Std)...")
    sum_x = 0
    sum_sq_x = 0
    count = 0

    # Iterate over the entire training set
    for x, _ in tqdm(loader, desc="Scanning Data"):
        # x shape: (Batch, Seq_Len, Feature)
        
        # Flatten Batch and Time dimensions to calculate stats per Feature
        # Shape: (Batch * Seq_Len, Feature)
        x_flat = x.view(-1, x.size(-1))
        
        sum_x += x_flat.sum(dim=0)
        sum_sq_x += (x_flat ** 2).sum(dim=0)
        count += x_flat.size(0)

    # Calculate Mean
    mean = sum_x / count
    
    # Calculate Std: sqrt(E[X^2] - (E[X])^2)
    var = (sum_sq_x / count) - (mean ** 2)
    std = torch.sqrt(torch.clamp(var, min=1e-9)) # Clamp to avoid sqrt of negative
    
    # Reshape to (1, 1, Feature) for broadcasting in model
    return mean.view(1, 1, -1), std.view(1, 1, -1)


def train():
    print(f"[INFO] Device: {CONFIG['device']}")
    
    # 1. Prepare Data Loaders
    train_loader, val_loader, test_loader, feature_dim = get_loaders(
        data_dir=CONFIG['data_dir'],
        start_date=CONFIG['start_date'],
        end_date=CONFIG['end_date'],
        top_n=CONFIG['top_n'],
        feature_list=CONFIG['feature_list'],
        seq_len=CONFIG['seq_len'],
        batch_size=CONFIG['batch_size'],
        ban_list_path=CONFIG['ban_list_path'],
        export_path=CONFIG['export_path'],
        # [NEW] Pass ban list path
    )
    CONFIG['input_dim'] = feature_dim
    print(f"[INFO] Input feature dim: {feature_dim}")

    # 2. Compute Statistics for Normalization
    input_mean, input_std = compute_stats(train_loader)
    
    # Move stats to GPU
    input_mean = input_mean.to(CONFIG['device'])
    input_std = input_std.to(CONFIG['device'])

    # 3. Initialize Model with Stats
    model = TimeSeries1DCNN(
        input_dim=CONFIG['input_dim'], 
        output_dim=CONFIG['output_dim'],
        mean=input_mean,
        std=input_std
    ).to(CONFIG['device'])
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    best_val_loss = float('inf')
    model_saved = False  # Track if model was saved

    # 4. Training Loop
    print("[INFO] Start Training...")
    for epoch in range(CONFIG['epochs']):
        # --- Train Step ---
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

        # --- Validation Step ---
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] "
              f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), CONFIG['save_path'])
            model_saved = True
            print(f"  -> Model saved (Val loss improved)")

    # 5. Final Evaluation (Test Set)
    print("\n[INFO] Evaluating on Test Set...")
    
    # Load best model if it was saved, otherwise use current model
    if model_saved and os.path.exists(CONFIG['save_path']):
        model.load_state_dict(torch.load(CONFIG['save_path']))
        print("[INFO] Loaded best model from checkpoint")
    else:
        print("[INFO] No model checkpoint found, using current model state")
    model.eval()
    
    test_loss = 0.0
    preds = []
    targets = []
    
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
    
    # Calculate Information Coefficient (IC)
    if len(preds) > 1:
        corr = np.corrcoef(preds, targets)[0, 1]
        print(f"Prediction Correlation (IC): {corr:.4f}")

if __name__ == "__main__":
    train()