import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import glob
import os
import sys

from mamba_dataloader import CryptoMambaDataset

import causal_conv1d_cuda
from mamba_ssm.modules.mamba_simple import Mamba

BATCH_SIZE = 16        # GPU 메모리에 맞춰 조절 (16, 32, 64 ...)
LEARNING_RATE = 1e-4   # Mamba/Transformer 계열은 보통 낮게 시작
EPOCHS = 10
SEQ_LEN = 240
N_COINS = 30
N_FEATURES = 258
DATA_DIR = "./data"    # h5 파일들이 들어있는 폴더 경로

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class RMSNorm(nn.Module):
    """Mamba에서 주로 사용하는 RMS Normalization"""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight

class CryptoMamba(nn.Module):
    def __init__(self, 
                 d_input=258,     # 입력 Feature 수
                 d_model=128,     # 모델 내부 Hidden Dimension (조절 가능)
                 n_layers=4,      # Mamba 레이어 쌓는 횟수
                 n_coins=30,      # 코인 개수
                 dropout=0.1):
        super().__init__()
        
        self.d_input = d_input
        self.d_model = d_model
        self.n_coins = n_coins

        # 1. Feature Projection (258 -> 128)
        # 입력 차원을 모델 차원으로 변환
        self.encoder = nn.Linear(d_input, d_model)
        
        # 2. Mamba Backbone
        # Mamba 블록들을 쌓아 시계열 특성 학습
        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model, # Model dimension d_model
                d_state=16,      # SSM state expansion factor
                d_conv=4,        # Local convolution width
                expand=2,        # Block expansion factor
            ) for _ in range(n_layers)
        ])
        
        # 레이어 사이의 정규화 (Normalization)
        self.norms = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        
        # 3. Final Prediction Head
        # 시퀀스의 마지막 상태를 이용해 수익률 예측
        self.final_norm = RMSNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1) # 최종 출력: 1개의 스칼라 (예측 수익률)
        )

    def forward(self, x):
        """
        Input x: (Batch, Seq_Len, Coins, Features) -> (B, 240, 30, 258)
        """
        B, L, N, F = x.shape
        
        # [Step 1] Reshape for Mamba
        # Mamba는 (Batch, Seq_Len, Dim) 입력을 받습니다.
        # 여기서 각 코인을 독립적인 샘플로 취급하여 배치 차원과 합칩니다.
        # (Batch * Coins, Seq_Len, Features)
        x = x.view(B * N, L, F)
        
        # [Step 2] Feature Projection
        x = self.encoder(x) # (B*N, L, d_model)
        
        # [Step 3] Mamba Layers (Residual Connection 적용)
        for layer, norm in zip(self.layers, self.norms):
            # Pre-Norm & Residual
            x_norm = norm(x)
            x = x + layer(x_norm)
            
        # [Step 4] Prediction
        # 시퀀스의 가장 마지막 시점(t)의 정보만 가져옵니다.
        x = self.final_norm(x)
        last_state = x[:, -1, :] # (B*N, d_model)
        
        out = self.head(last_state) # (B*N, 1)
        
        # [Step 5] Reshape back to (Batch, Coins)
        return out.view(B, N)

def get_dataloaders():
    # 1. 파일 리스트 가져오기 및 정렬 (날짜 순서 보장 필수)
    file_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.h5")))
    
    if len(file_paths) == 0:
        raise ValueError(f"{DATA_DIR} 폴더에 .h5 파일이 없습니다.")
        
    print(f"Total files found: {len(file_paths)}")

    # 2. 날짜 기준으로 분할 (Time Series Split)
    # 예: 31개 파일 -> Train(24), Val(3), Test(4)
    n_total = len(file_paths)
    n_train = int(n_total * 0.8) # 약 24일
    n_val = int(n_total * 0.1)   # 약 3일
    
    train_files = file_paths[:n_train]
    val_files = file_paths[n_train : n_train + n_val]
    test_files = file_paths[n_train + n_val :]
    
    print(f"Split: Train({len(train_files)}) / Val({len(val_files)}) / Test({len(test_files)})")

    # 3. Dataset & DataLoader 생성
    # Train은 셔플을 해도 됨 (윈도우 단위로 잘려있으므로 순서 상관 없음, 오히려 학습에 도움)
    train_dataset = CryptoMambaDataset(train_files, seq_len=SEQ_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    # Val/Test는 순서대로 평가하는 것이 일반적
    val_dataset = CryptoMambaDataset(val_files, seq_len=SEQ_LEN)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader

def train_model():
    train_loader, val_loader = get_dataloaders()
    
    # 모델 초기화 (실제 CryptoMamba 모델로 교체 필요)
    model = CryptoMamba().to(device)
    
    # Loss & Optimizer
    criterion = nn.MSELoss() # 회귀 문제이므로 MSE 사용
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')

    print("\nStarting Training...")
    
    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # data: (B, 240, 30, 258), target: (B, 30)
            data, target = data.to(device), target.to(device)

            # --- [디버깅 코드 추가] ---
            if torch.isnan(data).any():
                print(f"🚨 [Error] 입력 데이터(X)에 NaN이 포함되어 있습니다! (Batch {batch_idx})")
                break
            if torch.isinf(data).any():
                print(f"🚨 [Error] 입력 데이터(X)에 무한대(Inf)가 포함되어 있습니다! (Batch {batch_idx})")
                break
            if torch.isnan(target).any():
                print(f"🚨 [Error] 타겟 데이터(Y)에 NaN이 포함되어 있습니다! (Batch {batch_idx})")
                break
            # ------------------------
            
            optimizer.zero_grad()
            
            # Forward
            output = model(data) # Output: (B, 30)
            
            # Loss 계산
            loss = criterion(output, target)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Step [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.6f}")

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"==> Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        
        # Best Model 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_cryptomamba_model.pth")
            print("    (Best model saved)")

if __name__ == "__main__":
    train_model()
