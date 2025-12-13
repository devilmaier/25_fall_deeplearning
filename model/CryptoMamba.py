import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import causal_conv1d_cuda
    from mamba_ssm.modules.mamba_simple import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    print("[WARNING] mamba_ssm not installed. Please install it to use CryptoMamba model.")
    MAMBA_AVAILABLE = False


class RMSNorm(nn.Module):
    """RMS Normalization used in Mamba architecture"""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight


class CryptoMamba(nn.Module):
    def __init__(self, 
                 num_nodes=30,
                 input_dim=258,
                 hidden_dim=128,
                 output_dim=1,
                 dropout=0.1,
                 num_mamba_layers=4,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 mean=None,
                 std=None):
        """
        CryptoMamba Model for cryptocurrency prediction with cross-coin correlation learning
        
        Architecture:
        1. Per-coin feature projection
        2. Temporal Mamba layers (process time dimension)
        3. Spatial Mamba layers (process coin dimension)
        4. Prediction head
        
        Args:
            num_nodes: Number of coins/symbols (default: 30)
            input_dim: Input feature dimension (default: 258)
            hidden_dim: Model hidden dimension (default: 128)
            output_dim: Output dimension (default: 1)
            dropout: Dropout rate (default: 0.1)
            num_mamba_layers: Number of Mamba layers (default: 4)
            d_state: SSM state expansion factor (default: 16)
            d_conv: Local convolution width (default: 4)
            expand: Block expansion factor (default: 2)
            mean: Input normalization mean (optional)
            std: Input normalization std (optional)
        """
        super().__init__()
        
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba_ssm package is required. Install it with: pip install mamba-ssm")
        
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # -------------------------------------------------------
        # 0. In-model Normalization
        # -------------------------------------------------------
        if mean is not None and std is not None:
            self.register_buffer('input_mean', mean)
            self.register_buffer('input_std', std)
        else:
            # Default: No normalization (Mean=0, Std=1)
            # Shape: (1, 1, 1, input_dim) for broadcasting to (Batch, Time, Nodes, Features)
            self.register_buffer('input_mean', torch.zeros(1, 1, 1, input_dim))
            self.register_buffer('input_std', torch.ones(1, 1, 1, input_dim))
        
        # -------------------------------------------------------
        # 1. Feature Projection (per-coin)
        # Project each coin's features independently
        # -------------------------------------------------------
        self.feature_projection = nn.Linear(input_dim, hidden_dim)
        
        # -------------------------------------------------------
        # 2. Temporal Mamba Layers
        # Process time dimension for each coin
        # Learn temporal patterns: price movements over time
        # -------------------------------------------------------
        self.temporal_mamba_layers = nn.ModuleList([
            Mamba(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            ) for _ in range(num_mamba_layers // 2)
        ])
        self.temporal_norms = nn.ModuleList([
            RMSNorm(hidden_dim) for _ in range(num_mamba_layers // 2)
        ])
        
        # -------------------------------------------------------
        # 3. Spatial Mamba Layers
        # Process coin dimension across time
        # Learn cross-coin correlations: BTC affects ETH, etc.
        # -------------------------------------------------------
        self.spatial_mamba_layers = nn.ModuleList([
            Mamba(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            ) for _ in range(num_mamba_layers // 2)
        ])
        self.spatial_norms = nn.ModuleList([
            RMSNorm(hidden_dim) for _ in range(num_mamba_layers // 2)
        ])
        
        # -------------------------------------------------------
        # 4. Final temporal aggregation
        # Aggregate information across time for final prediction
        # -------------------------------------------------------
        self.temporal_pooling = nn.AdaptiveAvgPool1d(1)
        
        # -------------------------------------------------------
        # 5. Prediction Head
        # Predict future returns for each coin
        # -------------------------------------------------------
        self.final_norm = RMSNorm(hidden_dim)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize model parameters"""
        # Initialize projection
        nn.init.xavier_uniform_(self.feature_projection.weight)
        if self.feature_projection.bias is not None:
            nn.init.zeros_(self.feature_projection.bias)
        
        # Initialize prediction head
        for layer in self.prediction_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Forward pass with cross-coin correlation learning
        
        Args:
            x: Input tensor of shape (Batch, Time, Nodes, Features)
        
        Returns:
            predictions: Output tensor of shape (Batch, Nodes)
        """
        # x shape: (Batch, Time, Nodes, Features)
        B, L, N, F = x.size()
        
        # --- Step 0: Apply Normalization ---
        x = (x - self.input_mean) / (self.input_std + 1e-9)
        
        # --- Step 1: Feature Projection ---
        # Process each coin's features independently
        # (B, L, N, F) -> (B, L, N, H)
        x = self.feature_projection(x)
        
        # --- Step 2: Temporal Processing ---
        # Process time dimension for each coin
        # Learn: "How does each coin's price change over time?"
        
        # Reshape to (B*N, L, H) to process each coin's time series
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, N, L, H)
        x = x.view(B * N, L, self.hidden_dim)   # (B*N, L, H)
        
        # Apply temporal Mamba layers
        for mamba_layer, norm in zip(self.temporal_mamba_layers, self.temporal_norms):
            x_norm = norm(x)
            x = x + mamba_layer(x_norm)  # Residual connection
        
        # Reshape back to (B, N, L, H)
        x = x.view(B, N, L, self.hidden_dim)
        
        # --- Step 3: Spatial Processing ---
        # Process coin dimension across time
        # Learn: "How do different coins correlate with each other?"
        
        # For each time step, process relationships between coins
        # Reshape to (B*L, N, H) to process coin relationships at each time
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, L, N, H)
        x = x.view(B * L, N, self.hidden_dim)   # (B*L, N, H)
        
        # Apply spatial Mamba layers
        for mamba_layer, norm in zip(self.spatial_mamba_layers, self.spatial_norms):
            x_norm = norm(x)
            x = x + mamba_layer(x_norm)  # Residual connection
        
        # Reshape back to (B, L, N, H)
        x = x.view(B, L, N, self.hidden_dim)
        
        # --- Step 4: Temporal Aggregation ---
        # Aggregate temporal information for final prediction
        # (B, L, N, H) -> (B, N, H)
        
        # Permute to (B, N, H, L) for pooling
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, N, H, L)
        x = x.view(B * N, self.hidden_dim, L)   # (B*N, H, L)
        
        # Global average pooling over time
        x = self.temporal_pooling(x).squeeze(-1)  # (B*N, H)
        
        # Reshape to (B, N, H)
        x = x.view(B, N, self.hidden_dim)
        
        # --- Step 5: Prediction ---
        # Apply final normalization and prediction head
        x = self.final_norm(x)  # (B, N, H)
        
        # Flatten for prediction head
        x = x.view(B * N, self.hidden_dim)  # (B*N, H)
        predictions = self.prediction_head(x)  # (B*N, 1)
        
        # Reshape to (B, N)
        predictions = predictions.view(B, N)
        
        return predictions
