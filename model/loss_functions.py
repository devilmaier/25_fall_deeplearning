import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridFinancialLoss(nn.Module):
    """
    Hybrid Loss Function for Financial Time Series Prediction
    
    Combines three objectives:
    1. MSE: Basic regression accuracy
    2. Directional: Direction prediction accuracy (up/down)
    3. IC: Information Coefficient (correlation-based ranking)
    
    Args:
        mse_weight: Weight for MSE loss (default: 0.3)
        dir_weight: Weight for directional loss (default: 0.4)
        ic_weight: Weight for IC loss (default: 0.3)
        temperature: Temperature for smooth sign function (default: 1.0)
    """
    def __init__(self, mse_weight=0.3, dir_weight=0.4, ic_weight=0.3, temperature=1.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.dir_weight = dir_weight
        self.ic_weight = ic_weight
        self.temperature = temperature
        
        # Normalize weights
        total = mse_weight + dir_weight + ic_weight
        self.mse_weight /= total
        self.dir_weight /= total
        self.ic_weight /= total
    
    def forward(self, predictions, targets):
        """
        Compute hybrid loss
        
        Args:
            predictions: (Batch, Nodes) or (Batch*Nodes,)
            targets: (Batch, Nodes) or (Batch*Nodes,)
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual losses
        """
        # Flatten if needed
        if predictions.dim() > 1:
            predictions = predictions.view(-1)
            targets = targets.view(-1)
        
        # 1. MSE Loss
        mse_loss = torch.mean((predictions - targets) ** 2)
        
        # 2. Directional Loss (smooth version)
        # Use tanh for smooth, differentiable sign function
        pred_sign = torch.tanh(predictions / self.temperature)
        target_sign = torch.tanh(targets / self.temperature)
        
        # Agreement: +1 if same direction, -1 if opposite
        agreement = pred_sign * target_sign
        
        # FIXED: Make directional loss positive by negating
        # We want to MINIMIZE disagreement = MAXIMIZE agreement
        # Loss should be positive and decrease as agreement increases
        dir_loss = 1.0 - torch.mean(agreement)  # Range: [0, 2], lower is better
        
        # 3. IC Loss (maximize correlation)
        # Pearson correlation
        pred_mean = torch.mean(predictions)
        target_mean = torch.mean(targets)
        
        pred_centered = predictions - pred_mean
        target_centered = targets - target_mean
        
        numerator = torch.sum(pred_centered * target_centered)
        denominator = torch.sqrt(
            torch.sum(pred_centered ** 2) * torch.sum(target_centered ** 2) + 1e-8
        )
        
        ic = numerator / denominator
        
        # FIXED: Make IC loss positive
        # IC ranges from -1 to +1
        # We want to MAXIMIZE IC, so loss should DECREASE as IC increases
        ic_loss = 1.0 - ic  # Range: [0, 2], lower is better
        
        # Combined loss (all positive components)
        total_loss = (
            self.mse_weight * mse_loss +
            self.dir_weight * dir_loss +
            self.ic_weight * ic_loss
        )
        
        # Return loss and components for logging
        loss_dict = {
            'total': total_loss.item(),
            'mse': mse_loss.item(),
            'directional': dir_loss.item(),
            'ic_loss': ic_loss.item(),
            'ic_value': ic.item()  # Actual IC value (not loss)
        }
        
        return total_loss, loss_dict


class AdaptiveHybridLoss(nn.Module):
    """
    Adaptive Hybrid Loss that changes weights during training
    
    Strategy:
    - Early epochs: Focus on MSE (stable learning)
    - Mid epochs: Balanced
    - Late epochs: Focus on Direction + IC (trading performance)
    
    Args:
        total_epochs: Total number of training epochs
        initial_weights: (mse, dir, ic) for early training
        final_weights: (mse, dir, ic) for late training
    """
    def __init__(self, total_epochs=20, 
                 initial_weights=(0.6, 0.2, 0.2),
                 final_weights=(0.2, 0.4, 0.4),
                 temperature=1.0):
        super().__init__()
        self.total_epochs = total_epochs
        self.initial_weights = initial_weights
        self.final_weights = final_weights
        self.temperature = temperature
        
        # Current epoch (to be set externally)
        self.current_epoch = 0
        
        # Initialize with initial weights
        self.loss_fn = HybridFinancialLoss(
            mse_weight=initial_weights[0],
            dir_weight=initial_weights[1],
            ic_weight=initial_weights[2],
            temperature=temperature
        )
    
    def set_epoch(self, epoch):
        """Update weights based on current epoch"""
        self.current_epoch = epoch
        
        # Linear interpolation between initial and final weights
        progress = min(epoch / self.total_epochs, 1.0)
        
        mse_w = self.initial_weights[0] + progress * (self.final_weights[0] - self.initial_weights[0])
        dir_w = self.initial_weights[1] + progress * (self.final_weights[1] - self.initial_weights[1])
        ic_w = self.initial_weights[2] + progress * (self.final_weights[2] - self.initial_weights[2])
        
        # Update loss function weights
        self.loss_fn = HybridFinancialLoss(
            mse_weight=mse_w,
            dir_weight=dir_w,
            ic_weight=ic_w,
            temperature=self.temperature
        )
    
    def forward(self, predictions, targets):
        return self.loss_fn(predictions, targets)


class DirectionalLoss(nn.Module):
    """
    Simple directional loss (can be used standalone)
    
    Penalizes incorrect direction predictions
    Loss is positive and decreases as directional agreement increases
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, predictions, targets):
        if predictions.dim() > 1:
            predictions = predictions.view(-1)
            targets = targets.view(-1)
        
        # Smooth sign
        pred_sign = torch.tanh(predictions / self.temperature)
        target_sign = torch.tanh(targets / self.temperature)
        
        # Agreement: +1 if same direction, -1 if opposite
        agreement = pred_sign * target_sign
        
        # FIXED: Positive loss (1 - agreement)
        # Range: [0, 2], lower is better
        loss = 1.0 - torch.mean(agreement)
        
        # Calculate directional accuracy for logging
        with torch.no_grad():
            pred_dir = (predictions > 0).float()
            target_dir = (targets > 0).float()
            dir_acc = (pred_dir == target_dir).float().mean()
        
        return loss, {'directional_loss': loss.item(), 'dir_accuracy': dir_acc.item()}


class ICLoss(nn.Module):
    """
    Information Coefficient Loss (maximize correlation)
    
    Loss is positive and decreases as IC increases
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions, targets):
        if predictions.dim() > 1:
            predictions = predictions.view(-1)
            targets = targets.view(-1)
        
        # Center
        pred_centered = predictions - torch.mean(predictions)
        target_centered = targets - torch.mean(targets)
        
        # Correlation
        numerator = torch.sum(pred_centered * target_centered)
        denominator = torch.sqrt(
            torch.sum(pred_centered ** 2) * torch.sum(target_centered ** 2) + 1e-8
        )
        
        ic = numerator / denominator
        
        # FIXED: Positive loss (1 - IC)
        # IC ranges from -1 to +1
        # Loss ranges from 0 to 2, lower is better
        loss = 1.0 - ic
        
        return loss, {'ic_loss': loss.item(), 'ic_value': ic.item()}


class RankingLoss(nn.Module):
    """
    Pairwise ranking loss
    
    Ensures that higher predicted values correspond to higher actual returns
    Good for Long-Short strategies
    
    Warning: O(N²) complexity, use with smaller batches
    """
    def __init__(self, margin=0.01):
        super().__init__()
        self.margin = margin
    
    def forward(self, predictions, targets):
        if predictions.dim() > 1:
            predictions = predictions.view(-1)
            targets = targets.view(-1)
        
        n = predictions.size(0)
        loss = 0.0
        count = 0
        
        # Sample pairs if too many
        max_pairs = 1000
        if n * (n - 1) // 2 > max_pairs:
            # Random sampling
            indices = torch.randperm(n)[:int(torch.sqrt(torch.tensor(max_pairs * 2)).item())]
        else:
            indices = torch.arange(n)
        
        for i in indices:
            for j in indices:
                if i >= j:
                    continue
                
                # If target_i > target_j (by margin)
                if targets[i] > targets[j] + self.margin:
                    # pred_i should be > pred_j
                    diff = predictions[j] - predictions[i]
                    loss += F.relu(diff + self.margin)
                    count += 1
                
                # If target_j > target_i (by margin)
                elif targets[j] > targets[i] + self.margin:
                    # pred_j should be > pred_i
                    diff = predictions[i] - predictions[j]
                    loss += F.relu(diff + self.margin)
                    count += 1
        
        if count > 0:
            loss = loss / count
        
        return loss, {'ranking_loss': loss.item()}


# Utility function to get loss function by name
def get_loss_function(loss_type='hybrid', **kwargs):
    """
    Factory function to get loss function
    
    Args:
        loss_type: 'mse', 'hybrid', 'adaptive', 'directional', 'ic', 'ranking'
        **kwargs: Additional arguments for loss function
    
    Returns:
        loss_fn: Loss function
    """
    if loss_type == 'mse':
        return nn.MSELoss()
    
    elif loss_type == 'hybrid':
        return HybridFinancialLoss(
            mse_weight=kwargs.get('mse_weight', 0.3),
            dir_weight=kwargs.get('dir_weight', 0.4),
            ic_weight=kwargs.get('ic_weight', 0.3),
            temperature=kwargs.get('temperature', 1.0)
        )
    
    elif loss_type == 'adaptive':
        return AdaptiveHybridLoss(
            total_epochs=kwargs.get('total_epochs', 20),
            initial_weights=kwargs.get('initial_weights', (0.6, 0.2, 0.2)),
            final_weights=kwargs.get('final_weights', (0.2, 0.4, 0.4)),
            temperature=kwargs.get('temperature', 1.0)
        )
    
    elif loss_type == 'directional':
        return DirectionalLoss(temperature=kwargs.get('temperature', 1.0))
    
    elif loss_type == 'ic':
        return ICLoss()
    
    elif loss_type == 'ranking':
        return RankingLoss(margin=kwargs.get('margin', 0.01))
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Example usage
    print("Testing Loss Functions")
    print("="*50)
    
    # Generate sample data
    batch_size = 32
    num_nodes = 30
    predictions = torch.randn(batch_size, num_nodes) * 0.01
    targets = torch.randn(batch_size, num_nodes) * 0.01
    
    # Test different loss functions
    losses = {
        'MSE': nn.MSELoss(),
        'Hybrid': HybridFinancialLoss(),
        'Directional': DirectionalLoss(),
        'IC': ICLoss()
    }
    
    for name, loss_fn in losses.items():
        if isinstance(loss_fn, nn.MSELoss):
            loss = loss_fn(predictions, targets)
            print(f"{name:15s}: {loss.item():.6f}")
        else:
            loss, loss_dict = loss_fn(predictions, targets)
            print(f"{name:15s}: {loss_dict}")
    
    print("\n" + "="*50)
    print("Adaptive Loss Test (3 epochs)")
    print("="*50)
    
    adaptive_loss = AdaptiveHybridLoss(total_epochs=10)
    
    for epoch in [0, 5, 10]:
        adaptive_loss.set_epoch(epoch)
        loss, loss_dict = adaptive_loss(predictions, targets)
        
        weights = (
            adaptive_loss.loss_fn.mse_weight,
            adaptive_loss.loss_fn.dir_weight,
            adaptive_loss.loss_fn.ic_weight
        )
        
        print(f"\nEpoch {epoch}:")
        print(f"  Weights (MSE, Dir, IC): ({weights[0]:.2f}, {weights[1]:.2f}, {weights[2]:.2f})")
        print(f"  Loss: {loss_dict}")