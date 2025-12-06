import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeSeries1DCNN(nn.Module):
    def __init__(self, input_dim, output_dim=1, mean=None, std=None):
        super(TimeSeries1DCNN, self).__init__()
        
        # -------------------------------------------------------
        # 1. In-model Normalization
        # Stores Mean and Std as part of the model state.
        # These are not trained (buffers), but saved with the model.
        # -------------------------------------------------------
        if mean is not None and std is not None:
            self.register_buffer('input_mean', mean)
            self.register_buffer('input_std', std)
        else:
            # Default: No normalization (Mean=0, Std=1)
            self.register_buffer('input_mean', torch.zeros(1, 1, input_dim))
            self.register_buffer('input_std', torch.ones(1, 1, input_dim))

        # -------------------------------------------------------
        # 2. Convolutional Layers (Feature Extraction)
        # -------------------------------------------------------
        # Block 1: Input (Batch, Input_Dim, Seq_Len) -> Hidden 32
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2) # Downsample time by 2
        
        # Block 2: Hidden 32 -> Hidden 64
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2) # Downsample time by 2 again
        
        # -------------------------------------------------------
        # 3. Global Aggregation & Output
        # -------------------------------------------------------
        # Collapses remaining time dimension into a single vector
        self.gap = nn.AdaptiveAvgPool1d(1) 
        
        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        # Input x shape: (Batch, Seq_Len, Features)
        
        # [Step 1] Apply Normalization
        # Formula: (x - mean) / (std + epsilon)
        x = (x - self.input_mean) / (self.input_std + 1e-9)

        # [Step 2] Permute for Conv1d
        # Change shape to (Batch, Features, Seq_Len)
        x = x.permute(0, 2, 1) 
        
        # [Step 3] Convolution Blocks
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # [Step 4] Global Pooling & Prediction
        # Shape: (Batch, 64, 1) -> (Batch, 64)
        x = self.gap(x).squeeze(-1)
        
        out = self.fc(x)
        
        return out.squeeze()

