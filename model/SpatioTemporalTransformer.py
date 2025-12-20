import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatioTemporalTransformer(nn.Module):
    def __init__(self, 
                 num_nodes=30, 
                 input_dim=480, 
                 hidden_dim=64, 
                 output_dim=1, 
                 dropout=0.2,
                 num_transformer_layers=2,
                 num_heads=4,
                 mean=None,
                 std=None):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Normalization
        if mean is not None and std is not None:
            self.register_buffer('input_mean', mean)
            self.register_buffer('input_std', std)
        else:
            self.register_buffer('input_mean', torch.zeros(1, 1, 1, input_dim))
            self.register_buffer('input_std', torch.ones(1, 1, 1, input_dim))
        
        # Temporal Encoder (1D-CNN)
        self.conv1 = nn.Conv1d(
            in_channels=input_dim, 
            out_channels=hidden_dim, 
            kernel_size=3, 
            padding=1
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(
            in_channels=hidden_dim, 
            out_channels=hidden_dim, 
            kernel_size=3, 
            padding=1
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu2 = nn.ReLU()
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Prediction Heads
        self.transformer_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Auxiliary CNN prediction head
        self.cnn_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize CNN
        for layer in [self.conv1, self.conv2]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        
        # Initialize Prediction Heads
        for head in [self.transformer_head, self.cnn_head]:
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, x):
        # x shape: (Batch, Time, Nodes, Features)
        N, T, V, C = x.size()
        
        # Normalization
        x = (x - self.input_mean) / (self.input_std + 1e-9)
        
        # Temporal Encoding (1D-CNN)
        x = x.permute(0, 2, 3, 1).contiguous()  # (N, V, C, T)
        x = x.view(N * V, C, T)  # (N*V, C, T)
        
        # Apply CNN layers
        x = self.relu1(self.bn1(self.conv1(x)))  # (N*V, hidden_dim, T)
        x = self.relu2(self.bn2(self.conv2(x)))  # (N*V, hidden_dim, T)
        
        cnn_embedding = self.gap(x).squeeze(-1)  # (N*V, hidden_dim)
        
        cnn_embedding = cnn_embedding.view(N, V, self.hidden_dim)
        
        # Transformer Encoding
        transformer_out = self.transformer_encoder(cnn_embedding)  # (N, V, hidden_dim)
        
        # Residual connection + Layer Norm
        transformer_embedding = self.layer_norm(transformer_out + cnn_embedding)
        
        # Prediction with Residual
        transformer_out_flat = transformer_embedding.reshape(N * V, -1)
        transformer_pred = self.transformer_head(transformer_out_flat)  # (N*V, output_dim)
        transformer_pred = transformer_pred.reshape(N, V, self.output_dim)
        
        # CNN Prediction
        cnn_out_flat = cnn_embedding.reshape(N * V, -1)
        cnn_pred = self.cnn_head(cnn_out_flat)  # (N*V, output_dim)
        cnn_pred = cnn_pred.reshape(N, V, self.output_dim)
        
        # Residual Prediction
        final_pred = transformer_pred + 0.3 * cnn_pred
        
        return final_pred, transformer_pred, cnn_pred
