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
    """RMS Normalization."""
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
        CryptoMamba Model for cryptocurrency prediction.
        """
        super().__init__()
        
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba_ssm package is required. Install it with: pip install mamba-ssm")
        
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Normalization
        if mean is not None and std is not None:
            self.register_buffer('input_mean', mean)
            self.register_buffer('input_std', std)
        else:
            self.register_buffer('input_mean', torch.zeros(1, 1, 1, input_dim))
            self.register_buffer('input_std', torch.ones(1, 1, 1, input_dim))
        
        # Feature Projection
        self.feature_projection = nn.Linear(input_dim, hidden_dim)
        
        # Temporal Mamba Layers
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
        
        # Spatial Mamba Layers
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
        
        # Temporal aggregation
        self.temporal_pooling = nn.AdaptiveAvgPool1d(1)
        
        # Prediction Head
        self.final_norm = RMSNorm(hidden_dim)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.reset_parameters()

    def reset_parameters(self):
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
        # x shape: (Batch, Time, Nodes, Features)
        B, L, N, F = x.size()
        
        # Normalization
        x = (x - self.input_mean) / (self.input_std + 1e-9)
        
        # Feature Projection
        x = self.feature_projection(x)
        
        # Temporal Processing
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, N, L, H)
        x = x.view(B * N, L, self.hidden_dim)   # (B*N, L, H)
        
        # Apply temporal Mamba layers
        for mamba_layer, norm in zip(self.temporal_mamba_layers, self.temporal_norms):
            x_norm = norm(x)
            x = x + mamba_layer(x_norm)  # Residual connection
        
        x = x.view(B, N, L, self.hidden_dim)
        
        # Spatial Processing
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, L, N, H)
        x = x.view(B * L, N, self.hidden_dim)   # (B*L, N, H)
        
        # Apply spatial Mamba layers
        for mamba_layer, norm in zip(self.spatial_mamba_layers, self.spatial_norms):
            x_norm = norm(x)
            x = x + mamba_layer(x_norm)  # Residual connection
        
        x = x.view(B, L, N, self.hidden_dim)
        
        # Temporal Aggregation
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, N, H, L)
        x = x.view(B * N, self.hidden_dim, L)   # (B*N, H, L)
        
        x = self.temporal_pooling(x).squeeze(-1)  # (B*N, H)
        
        x = x.view(B, N, self.hidden_dim)
        
        # Prediction
        x = self.final_norm(x)  # (B, N, H)
        
        x = x.view(B * N, self.hidden_dim)  # (B*N, H)
        predictions = self.prediction_head(x)  # (B*N, 1)
        
        predictions = predictions.view(B, N)
        
        return predictions
