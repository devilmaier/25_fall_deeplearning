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
        CryptoMamba Model for cryptocurrency prediction
        
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
        # 1. Feature Projection (input_dim -> hidden_dim)
        # -------------------------------------------------------
        self.encoder = nn.Linear(input_dim, hidden_dim)
        
        # -------------------------------------------------------
        # 2. Mamba Backbone
        # Stack Mamba blocks to learn temporal features
        # -------------------------------------------------------
        self.layers = nn.ModuleList([
            Mamba(
                d_model=hidden_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            ) for _ in range(num_mamba_layers)
        ])
        
        # Layer Normalization between blocks
        self.norms = nn.ModuleList([RMSNorm(hidden_dim) for _ in range(num_mamba_layers)])
        
        # -------------------------------------------------------
        # 3. Prediction Head
        # Use the last time step state to predict returns
        # -------------------------------------------------------
        self.final_norm = RMSNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize model parameters"""
        # Initialize encoder
        nn.init.xavier_uniform_(self.encoder.weight)
        if self.encoder.bias is not None:
            nn.init.zeros_(self.encoder.bias)
        
        # Initialize prediction head
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (Batch, Time, Nodes, Features)
        
        Returns:
            predictions: Output tensor of shape (Batch, Nodes)
        """
        # x shape: (Batch, Time, Nodes, Features)
        B, L, N, F = x.size()
        
        # --- Step 0: Apply Normalization ---
        x = (x - self.input_mean) / (self.input_std + 1e-9)
        
        # --- Step 1: Reshape for Mamba ---
        # Mamba expects (Batch, Seq_Len, Dim) input
        # Treat each coin as an independent sample
        # (Batch * Nodes, Seq_Len, Features)
        x = x.view(B * N, L, F)
        
        # --- Step 2: Feature Projection ---
        x = self.encoder(x)  # (B*N, L, hidden_dim)
        
        # --- Step 3: Mamba Layers (with Residual Connection) ---
        for layer, norm in zip(self.layers, self.norms):
            # Pre-Norm & Residual
            x_norm = norm(x)
            x = x + layer(x_norm)
        
        # --- Step 4: Prediction ---
        # Use only the last time step (t) information
        x = self.final_norm(x)
        last_state = x[:, -1, :]  # (B*N, hidden_dim)
        
        out = self.head(last_state)  # (B*N, 1)
        
        # --- Step 5: Reshape back to (Batch, Nodes) ---
        predictions = out.view(B, N)
        
        return predictions
