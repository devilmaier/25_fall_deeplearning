import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNLSTM(nn.Module):
    def __init__(self, 
                 num_nodes=30, 
                 input_dim=480, 
                 hidden_dim=64, 
                 output_dim=1, 
                 dropout=0.2,
                 num_layers=2,
                 gnn_type='transformer'): # gnn_type: 'transformer' or 'gat'
        super().__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # 1. Temporal Encoder (LSTM)
        # 각 symbol의 시계열 데이터를 독립적으로 인코딩 
        # 모든 symbol에 대해 동일한 LSTM - weight sharing 
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 2. Graphical Encoder
        # 각 노드(symbol) 간의 관계를 모델링

        # Transformer's Encoder Layer (Dense attention) 
        if gnn_type == 'transformer':
            self.gnn_layer = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
        # Graph Attention Network (GAT)
        elif gnn_type == 'gat':
            pass

        # 3. Prediction Heads
        # Head for GNN output 
        self.gnn_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )
        
        # Auxiliary Head for the LSTM output 
        self.lstm_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # x shape: (Batch, Time, Nodes, Features)
        N, T, V, C = x.size()
        
        # --- Step 1: Temporal Encoding (LSTM) ---
        # (N*V, T, C): 각 노드의 시계열 데이터를 독립적으로 처리하기 위해 reshape
        x_flat = x.permute(0, 2, 1, 3).contiguous().view(N * V, T, C)
        
        # Run LSTM
        # out: (N*V, T, Hidden), (h_n, c_n)
        _, (h_n, _) = self.lstm(x_flat)
        
        # Take the last hidden state of the last layer
        # h_n shape: (Num_Layers, N*V, Hidden) -> 마지막 layer: (N*V, Hidden)
        lstm_embedding = h_n[-1]
        
        # Reshape back to (Batch, Nodes, Hidden)
        lstm_embedding = lstm_embedding.view(N, V, self.hidden_dim)
        

        # --- Step 2: Graphical Encoding ---
        # Self-Attention across the "Nodes" dimension
        # Query=Key=Value = lstm_embedding
        # attn_output: (Batch, Nodes, Hidden)
        gnn_embedding, _ = self.gnn_layer(lstm_embedding, lstm_embedding, lstm_embedding)
        
        # Residual connection + Norm 
        gnn_embedding = gnn_embedding + lstm_embedding
        # normalization needed?
        # gnn_embedding = F.layer_norm(gnn_embedding, gnn_embedding.size()[1:])

        
        # --- Step 3: Prediction ---
        # 1. GNN Prediction (Main)
        # Flatten to (N*V, Hidden)
        gnn_out_flat = gnn_embedding.view(N * V, -1)
        gnn_pred = self.gnn_head(gnn_out_flat)
        gnn_pred = gnn_pred.view(N, V) # (Batch, Nodes)
        
        # 2. LSTM Prediction (Auxiliary)
        lstm_out_flat = lstm_embedding.view(N * V, -1)
        lstm_pred = self.lstm_head(lstm_out_flat)
        lstm_pred = lstm_pred.view(N, V) # (Batch, Nodes)
        
        return gnn_pred, lstm_pred