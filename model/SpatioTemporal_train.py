import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# Import custom modules
from SpatioTemporal_dataloader import get_spatiotemporal_loaders
from SpatioTemporalTransformer import SpatioTemporalTransformer

# Get project root directory (parent of model directory)
PROJECT_ROOT = Path(__file__).parent.parent

# ==========================================
# Configuration
# ==========================================
CONFIG = {
    'data_dir': str(PROJECT_ROOT / 'data'),
    'start_date': '2024-02-01',
    'end_date': '2024-02-15',
    'top_n': 30,
    'num_nodes': 30,    # Number of nodes in graph
    'seq_len': 9,       # Time window size
    'input_dim': 480,   # Will be updated automatically based on features
    'hidden_dim': 64,   # CNN and Transformer hidden dimension
    'output_dim': 1,    
    'num_transformer_layers': 2,  # Number of Transformer encoder layers
    'num_heads': 4,     # Number of attention heads
    'dropout': 0.2,
    'batch_size': 16,
    'epochs': 10,
    'lr': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'feature_list': str(PROJECT_ROOT / 'feature_list' / 'y_60m' / 'top30_example_features_44.json'),
    'ban_list_path': str(PROJECT_ROOT / 'global_ban_dates.json'), # [NEW] Path to ban list
    'save_path': str(PROJECT_ROOT / 'best_spatiotemporal_model.pt'),
    'export_path': str(PROJECT_ROOT / 'data' / 'datasets' / 'spatiotfm'),
    'cnn_loss_weight': 0.2,         # Weight for CNN auxiliary loss
    'transformer_loss_weight': 0.3  # Weight for Transformer auxiliary loss
}

def compute_stats(loader):
    """
    Scans the training set to compute Mean and Std for each feature.
    For GNN: input shape is (Batch, Time, Nodes, Features)
    Returns tensors of shape (1, 1, 1, Feature_Dim).
    """
    print("[INFO] Computing input statistics (Mean, Std) for SpatioTemporalTransformer...")
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
    print(f"[INFO] Model: SpatioTemporalTransformer (1D-CNN -> Transformer with Residual)")
    
    # 1. Prepare Data Loaders
    train_loader, val_loader, test_loader, feature_dim = get_spatiotemporal_loaders(
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
    model = SpatioTemporalTransformer(
        num_nodes=CONFIG['num_nodes'],
        input_dim=CONFIG['input_dim'], 
        hidden_dim=CONFIG['hidden_dim'],
        output_dim=CONFIG['output_dim'],
        dropout=CONFIG['dropout'],
        num_transformer_layers=CONFIG['num_transformer_layers'],
        num_heads=CONFIG['num_heads'],
        mean=input_mean,
        std=input_std
    ).to(CONFIG['device'])
    
    print(f"[INFO] Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    model_saved = False

    # 4. Training Loop
    print("[INFO] Start Training...")
    for epoch in range(CONFIG['epochs']):
        # --- Train Step ---
        model.train()
        train_loss = 0.0
        train_final_loss = 0.0
        train_transformer_loss = 0.0
        train_cnn_loss = 0.0
        
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]"):
            x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
            
            optimizer.zero_grad()
            
            # Forward pass - model returns (final_pred, transformer_pred, cnn_pred)
            # All have shape (Batch, Nodes)
            final_pred, transformer_pred, cnn_pred = model(x)
            
            # y shape: (Batch, Nodes)
            # Calculate losses
            final_loss = criterion(final_pred, y)
            transformer_loss = criterion(transformer_pred, y)
            cnn_loss = criterion(cnn_pred, y)
            
            # Combined loss: final prediction + auxiliary losses
            loss = final_loss + \
                   CONFIG['transformer_loss_weight'] * transformer_loss + \
                   CONFIG['cnn_loss_weight'] * cnn_loss
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_final_loss += final_loss.item()
            train_transformer_loss += transformer_loss.item()
            train_cnn_loss += cnn_loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_final_loss = train_final_loss / len(train_loader)
        avg_train_transformer_loss = train_transformer_loss / len(train_loader)
        avg_train_cnn_loss = train_cnn_loss / len(train_loader)

        # --- Validation Step ---
        model.eval()
        val_loss = 0.0
        val_final_loss = 0.0
        val_transformer_loss = 0.0
        val_cnn_loss = 0.0
        
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Val]"):
                x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
                
                final_pred, transformer_pred, cnn_pred = model(x)
                
                final_loss = criterion(final_pred, y)
                transformer_loss = criterion(transformer_pred, y)
                cnn_loss = criterion(cnn_pred, y)
                
                loss = final_loss + \
                       CONFIG['transformer_loss_weight'] * transformer_loss + \
                       CONFIG['cnn_loss_weight'] * cnn_loss
                
                val_loss += loss.item()
                val_final_loss += final_loss.item()
                val_transformer_loss += transformer_loss.item()
                val_cnn_loss += cnn_loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_final_loss = val_final_loss / len(val_loader)
        avg_val_transformer_loss = val_transformer_loss / len(val_loader)
        avg_val_cnn_loss = val_cnn_loss / len(val_loader)
        
        # Step scheduler
        scheduler.step(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}]")
        print(f"  Train - Total: {avg_train_loss:.6f} | Final: {avg_train_final_loss:.6f} | "
              f"Transformer: {avg_train_transformer_loss:.6f} | CNN: {avg_train_cnn_loss:.6f}")
        print(f"  Val   - Total: {avg_val_loss:.6f} | Final: {avg_val_final_loss:.6f} | "
              f"Transformer: {avg_val_transformer_loss:.6f} | CNN: {avg_val_cnn_loss:.6f}")

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
    test_final_loss = 0.0
    test_transformer_loss = 0.0
    test_cnn_loss = 0.0
    final_preds = []
    transformer_preds = []
    cnn_preds = []
    targets = []
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
            
            final_pred, transformer_pred, cnn_pred = model(x)
            
            final_loss = criterion(final_pred, y)
            transformer_loss = criterion(transformer_pred, y)
            cnn_loss = criterion(cnn_pred, y)
            
            loss = final_loss + \
                   CONFIG['transformer_loss_weight'] * transformer_loss + \
                   CONFIG['cnn_loss_weight'] * cnn_loss
            
            test_loss += loss.item()
            test_final_loss += final_loss.item()
            test_transformer_loss += transformer_loss.item()
            test_cnn_loss += cnn_loss.item()
            
            # Flatten predictions and targets for correlation calculation
            final_preds.extend(final_pred.cpu().numpy().flatten())
            transformer_preds.extend(transformer_pred.cpu().numpy().flatten())
            cnn_preds.extend(cnn_pred.cpu().numpy().flatten())
            targets.extend(y.cpu().numpy().flatten())

    avg_test_loss = test_loss / len(test_loader)
    avg_test_final_loss = test_final_loss / len(test_loader)
    avg_test_transformer_loss = test_transformer_loss / len(test_loader)
    avg_test_cnn_loss = test_cnn_loss / len(test_loader)
    
    print(f"\n{'='*70}")
    print(f"Final Test Results:")
    print(f"  Total MSE: {avg_test_loss:.6f}")
    print(f"  Final (Residual) MSE: {avg_test_final_loss:.6f}")
    print(f"  Transformer MSE: {avg_test_transformer_loss:.6f}")
    print(f"  CNN MSE: {avg_test_cnn_loss:.6f}")
    
    # Calculate Information Coefficient (IC)
    if len(final_preds) > 1:
        final_corr = np.corrcoef(final_preds, targets)[0, 1]
        transformer_corr = np.corrcoef(transformer_preds, targets)[0, 1]
        cnn_corr = np.corrcoef(cnn_preds, targets)[0, 1]
        print(f"  Final (Residual) IC: {final_corr:.4f}")
        print(f"  Transformer IC: {transformer_corr:.4f}")
        print(f"  CNN IC: {cnn_corr:.4f}")
    print(f"{'='*70}")

if __name__ == "__main__":
    train()
