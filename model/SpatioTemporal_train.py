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
    'mode': 'regression',  # 'regression' or 'classification'
    'data_dir': str(PROJECT_ROOT / 'data' / 'xy'),
    'start_date': '2025-02-01',
    'end_date': '2025-09-14',
    'vali_date': '2025-08-01',
    'test_date': '2025-09-01',
    'top_n': 30,
    'num_nodes': 30,    # Number of nodes in graph
    'seq_len': 60,       # Time window size
    'input_dim': 480,   # Will be updated automatically based on features
    'hidden_dim': 64,   # CNN and Transformer hidden dimension
    'output_dim': 2,    # 2 for classification (binary), 1 for regression
    'num_transformer_layers': 2,  # Number of Transformer encoder layers
    'num_heads': 4,     # Number of attention heads
    'dropout': 0.2,
    'batch_size': 96,
    'epochs': 5,
    'lr': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'feature_list': str(PROJECT_ROOT / 'feature_list' / 'y_60m' / 'top30_example_features_166.json'),
    'ban_list_path': str(PROJECT_ROOT / 'global_ban_dates.json'), # [NEW] Path to ban list
    'save_path': str(PROJECT_ROOT / 'best_spatiotemporal_model.pt'),
    'export_path': str(PROJECT_ROOT / 'data' / 'datasets' / 'spatiotfm'),
    'cnn_loss_weight': 0.2,         # Weight for CNN auxiliary loss
    'transformer_loss_weight': 0.3, # Weight for Transformer auxiliary loss
    'negate_prediction': False      # Option to negate predictions for IC calculation
}

def compute_stats(loader):
    print("[INFO] Computing input statistics (Mean, Std) for SpatioTemporalTransformer...")
    sum_x = 0
    sum_sq_x = 0
    count = 0

    for x, _ in tqdm(loader, desc="Scanning Data"):
        # Flatten all dimensions except Features
        x_flat = x.view(-1, x.size(-1))
        
        sum_x += x_flat.sum(dim=0)
        sum_sq_x += (x_flat ** 2).sum(dim=0)
        count += x_flat.size(0)

    mean = sum_x / count
    var = (sum_sq_x / count) - (mean ** 2)
    std = torch.sqrt(torch.clamp(var, min=1e-9))
    
    return mean.view(1, 1, 1, -1), std.view(1, 1, 1, -1)


def train():
    print(f"[INFO] Device: {CONFIG['device']}")
    print(f"[INFO] Mode: {CONFIG['mode'].upper()}")
    print(f"[INFO] Model: SpatioTemporalTransformer (1D-CNN -> Transformer with Residual)")
    print(f"[INFO] Loading data from {CONFIG['start_date']} to {CONFIG['end_date']}")
    
    # Adjust output_dim based on mode
    if CONFIG['mode'] == 'classification':
        CONFIG['output_dim'] = 2  # Binary classification
    else:  # regression
        CONFIG['output_dim'] = 1
    
    # 1. Prepare Data Loaders (with parallel dataset construction)
    train_loader, val_loader, test_loader, feature_dim = get_spatiotemporal_loaders(
        data_dir=CONFIG['data_dir'],
        start_date=CONFIG['start_date'],
        end_date=CONFIG['end_date'],
        vali_date=CONFIG['vali_date'],
        test_date=CONFIG['test_date'],
        top_n=CONFIG['top_n'],
        num_nodes=CONFIG['num_nodes'],
        feature_list=CONFIG['feature_list'],
        seq_len=CONFIG['seq_len'],
        batch_size=CONFIG['batch_size'],
        ban_list_path=CONFIG['ban_list_path'],
        export_path=CONFIG['export_path'],
        dataset_workers=24
    )
    CONFIG['input_dim'] = feature_dim
    print(f"[INFO] Input feature dim: {feature_dim}")

    # Compute Statistics for Normalization
    input_mean, input_std = compute_stats(train_loader)
    
    input_mean = input_mean.to(CONFIG['device'])
    input_std = input_std.to(CONFIG['device'])

    # Initialize Model
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
    
    # Select loss function
    if CONFIG['mode'] == 'classification':
        criterion = nn.CrossEntropyLoss()
        print(f"[INFO] Using CrossEntropyLoss for classification")
    else:  # regression
        criterion = nn.MSELoss()
        print(f"[INFO] Using MSELoss for regression")
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    model_saved = False

    # Training Loop
    print("[INFO] Start Training...")
    for epoch in range(CONFIG['epochs']):
        # Train Step
        model.train()
        train_loss = 0.0
        train_final_loss = 0.0
        train_transformer_loss = 0.0
        train_cnn_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]"):
            x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
            
            optimizer.zero_grad()
            
            # Forward pass
            final_pred, transformer_pred, cnn_pred = model(x)
            
            if CONFIG['mode'] == 'classification':
                # Convert y to binary labels
                y_labels = (y > 0).long()
                
                # Reshape for loss calculation
                batch_size, num_nodes, num_classes = final_pred.shape
                final_pred_flat = final_pred.view(-1, num_classes)
                transformer_pred_flat = transformer_pred.view(-1, num_classes)
                cnn_pred_flat = cnn_pred.view(-1, num_classes)
                y_labels_flat = y_labels.view(-1)
                
                # Calculate losses
                final_loss = criterion(final_pred_flat, y_labels_flat)
                transformer_loss = criterion(transformer_pred_flat, y_labels_flat)
                cnn_loss = criterion(cnn_pred_flat, y_labels_flat)
                
                # Calculate accuracy
                pred_labels = torch.argmax(final_pred, dim=2)
                train_correct += (pred_labels == y_labels).sum().item()
                train_total += y_labels.numel()
            else:  # regression
                # Squeeze output for regression
                final_pred = final_pred.squeeze(-1)
                transformer_pred = transformer_pred.squeeze(-1)
                cnn_pred = cnn_pred.squeeze(-1)
                
                # Calculate losses
                final_loss = criterion(final_pred, y)
                transformer_loss = criterion(transformer_pred, y)
                cnn_loss = criterion(cnn_pred, y)
            
            # Combined loss
            loss = final_loss + \
                   CONFIG['transformer_loss_weight'] * transformer_loss + \
                   CONFIG['cnn_loss_weight'] * cnn_loss
            
            loss.backward()
            
            # Gradient clipping
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
        train_accuracy = 100.0 * train_correct / train_total if train_total > 0 else 0.0

        # Validation Step
        model.eval()
        val_loss = 0.0
        val_final_loss = 0.0
        val_transformer_loss = 0.0
        val_cnn_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Val]"):
                x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
                
                final_pred, transformer_pred, cnn_pred = model(x)
                
                if CONFIG['mode'] == 'classification':
                    # Convert y to binary labels
                    y_labels = (y > 0).long()
                    
                    # Reshape for loss calculation
                    batch_size, num_nodes, num_classes = final_pred.shape
                    final_pred_flat = final_pred.view(-1, num_classes)
                    transformer_pred_flat = transformer_pred.view(-1, num_classes)
                    cnn_pred_flat = cnn_pred.view(-1, num_classes)
                    y_labels_flat = y_labels.view(-1)
                    
                    final_loss = criterion(final_pred_flat, y_labels_flat)
                    transformer_loss = criterion(transformer_pred_flat, y_labels_flat)
                    cnn_loss = criterion(cnn_pred_flat, y_labels_flat)
                    
                    # Calculate accuracy
                    pred_labels = torch.argmax(final_pred, dim=2)
                    val_correct += (pred_labels == y_labels).sum().item()
                    val_total += y_labels.numel()
                else:  # regression
                    # Squeeze output for regression
                    final_pred = final_pred.squeeze(-1)
                    transformer_pred = transformer_pred.squeeze(-1)
                    cnn_pred = cnn_pred.squeeze(-1)
                    
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
        val_accuracy = 100.0 * val_correct / val_total if val_total > 0 else 0.0
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}]")
        if CONFIG['mode'] == 'classification':
            print(f"  Train - Loss: {avg_train_loss:.6f} | Accuracy: {train_accuracy:.2f}%")
            print(f"  Train - Final: {avg_train_final_loss:.6f} | Transformer: {avg_train_transformer_loss:.6f} | CNN: {avg_train_cnn_loss:.6f}")
            print(f"  Val   - Loss: {avg_val_loss:.6f} | Accuracy: {val_accuracy:.2f}%")
            print(f"  Val   - Final: {avg_val_final_loss:.6f} | Transformer: {avg_val_transformer_loss:.6f} | CNN: {avg_val_cnn_loss:.6f}")
        else:  # regression
            print(f"  Train - MSE: {avg_train_loss:.6f}")
            print(f"  Train - Final: {avg_train_final_loss:.6f} | Transformer: {avg_train_transformer_loss:.6f} | CNN: {avg_train_cnn_loss:.6f}")
            print(f"  Val   - MSE: {avg_val_loss:.6f}")
            print(f"  Val   - Final: {avg_val_final_loss:.6f} | Transformer: {avg_val_transformer_loss:.6f} | CNN: {avg_val_cnn_loss:.6f}")

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
            if CONFIG['mode'] == 'classification':
                print(f"  -> Model saved (Val loss improved: {avg_val_loss:.6f}, Val accuracy: {val_accuracy:.2f}%)")
            else:
                print(f"  -> Model saved (Val loss improved: {avg_val_loss:.6f})")

    # Final Evaluation
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
    test_correct = 0
    test_total = 0
    final_preds = []
    targets = []
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
            
            final_pred, transformer_pred, cnn_pred = model(x)
            
            if CONFIG['mode'] == 'classification':
                # Convert y to binary labels
                y_labels = (y > 0).long()
                
                # Reshape for loss calculation
                batch_size, num_nodes, num_classes = final_pred.shape
                final_pred_flat = final_pred.view(-1, num_classes)
                transformer_pred_flat = transformer_pred.view(-1, num_classes)
                cnn_pred_flat = cnn_pred.view(-1, num_classes)
                y_labels_flat = y_labels.view(-1)
                
                final_loss = criterion(final_pred_flat, y_labels_flat)
                transformer_loss = criterion(transformer_pred_flat, y_labels_flat)
                cnn_loss = criterion(cnn_pred_flat, y_labels_flat)
                
                # Calculate accuracy
                pred_labels = torch.argmax(final_pred, dim=2)
                test_correct += (pred_labels == y_labels).sum().item()
                test_total += y_labels.numel()
                
                # Collect predictions and true labels for analysis
                final_preds.extend(pred_labels.cpu().numpy().flatten())
                targets.extend(y_labels.cpu().numpy().flatten())
            else:  # regression
                # Squeeze output for regression
                final_pred = final_pred.squeeze(-1)
                transformer_pred = transformer_pred.squeeze(-1)
                cnn_pred = cnn_pred.squeeze(-1)
                
                final_loss = criterion(final_pred, y)
                transformer_loss = criterion(transformer_pred, y)
                cnn_loss = criterion(cnn_pred, y)
                
                # Collect predictions and targets for correlation
                final_preds.extend(final_pred.cpu().numpy().flatten())
                targets.extend(y.cpu().numpy().flatten())
            
            loss = final_loss + \
                   CONFIG['transformer_loss_weight'] * transformer_loss + \
                   CONFIG['cnn_loss_weight'] * cnn_loss
            
            test_loss += loss.item()
            test_final_loss += final_loss.item()
            test_transformer_loss += transformer_loss.item()
            test_cnn_loss += cnn_loss.item()

    avg_test_loss = test_loss / len(test_loader)
    avg_test_final_loss = test_final_loss / len(test_loader)
    avg_test_transformer_loss = test_transformer_loss / len(test_loader)
    avg_test_cnn_loss = test_cnn_loss / len(test_loader)
    test_accuracy = 100.0 * test_correct / test_total if test_total > 0 else 0.0
    
    print(f"\n{'='*70}")
    print(f"Final Test Results:")
    
    if CONFIG['mode'] == 'classification':
        print(f"  Test Loss: {avg_test_loss:.6f}")
        print(f"  Test Accuracy: {test_accuracy:.2f}%")
        print(f"  Final Loss: {avg_test_final_loss:.6f}")
        print(f"  Transformer Loss: {avg_test_transformer_loss:.6f}")
        print(f"  CNN Loss: {avg_test_cnn_loss:.6f}")
        
        # Calculate classification metrics
        if len(final_preds) > 1:
            final_preds = np.array(final_preds)
            targets = np.array(targets)
            
            # Calculate accuracy
            accuracy = 100.0 * (final_preds == targets).sum() / len(targets)
            
            # Calculate precision, recall
            tp = ((final_preds == 1) & (targets == 1)).sum()
            fp = ((final_preds == 1) & (targets == 0)).sum()
            fn = ((final_preds == 0) & (targets == 1)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"  Accuracy (verified): {accuracy:.2f}%")
            print(f"  Precision (positive class): {precision:.4f}")
            print(f"  Recall (positive class): {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
    else:  # regression
        print(f"  Test MSE: {avg_test_loss:.6f}")
        print(f"  Final MSE: {avg_test_final_loss:.6f}")
        print(f"  Transformer MSE: {avg_test_transformer_loss:.6f}")
        print(f"  CNN MSE: {avg_test_cnn_loss:.6f}")
        
        # Calculate IC
        if len(final_preds) > 1:
            final_preds = np.array(final_preds)
            
            if CONFIG.get('negate_prediction', False):
                print("[INFO] Negating predictions for IC calculation (Contrarian Mode)...")
                final_preds = -final_preds
            
            targets = np.array(targets)
            ic = np.corrcoef(final_preds, targets)[0, 1]
            print(f"  Information Coefficient (IC): {ic:.4f}")
            return ic
    
    print(f"{'='*70}")
    return 0.0

if __name__ == "__main__":
    train()
