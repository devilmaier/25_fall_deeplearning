import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from functools import lru_cache

class CryptoMambaDataset(Dataset):
    def __init__(self, file_paths, seq_len=240, n_coins=30, steps_per_file=1440):
        self.file_paths = file_paths
        self.seq_len = seq_len
        self.n_coins = n_coins 
        self.steps_per_file = steps_per_file
        self.total_steps = len(file_paths) * steps_per_file
        
        # 메타데이터 저장용 변수 (첫 파일 로드 시 채워짐)
        self.feature_names = None
        self.coin_symbols = None
        
        if self.total_steps < seq_len:
            raise ValueError(f"데이터셋 길이가 너무 짧습니다.")
        self.valid_samples = self.total_steps - (seq_len - 1)

    def __len__(self):
        return self.valid_samples

    @lru_cache(maxsize=4)
    def _load_file_data(self, file_idx):
        file_path = self.file_paths[file_idx]
        try:
            df = pd.read_hdf(file_path, mode='r')
        except Exception as e:
            raise IOError(f"파일 로드 오류: {file_path}. Error: {e}")

        # 1. Feature 이름 추출 (원본 컬럼명 유지)
        if self.feature_names is None:
            self.feature_names = [c for c in df.columns if '_neut' in c]
            
        feature_cols = self.feature_names # 저장된 순서 보장
        target_col = 'y_60m'

        # 2. Symbol 정보 추출 (검증용)
        # 가정: 데이터에 'symbol' 컬럼이 있거나, 인덱스에 포함되어 있다고 가정.
        # 여기서는 'symbol'이라는 컬럼이 있다고 가정합니다.
        if self.coin_symbols is None:
            if 'symbol' in df.columns:
                # 시간 순 정렬이므로 첫 30개 행이 30개 코인 리스트라고 가정
                self.coin_symbols = df['symbol'].iloc[:self.n_coins].values.tolist()
            else:
                # symbol 컬럼이 없으면 인덱스 등에서 확인 필요 (여기서는 임시 처리)
                print("Warning: 'symbol' column not found. Using generic IDs.")
                self.coin_symbols = [f'Coin_{i}' for i in range(self.n_coins)]

        # 3. 데이터 변환 (Reshape)
        X_flat = df[feature_cols].values
        Y_flat = df[target_col].values
        
        # (Total_Rows, Features) -> (Time, Coins, Features)
        try:
            X_panel = X_flat.reshape(self.steps_per_file, self.n_coins, -1)
            Y_panel = Y_flat.reshape(self.steps_per_file, self.n_coins)
        except ValueError:
            raise ValueError(f"Reshape 실패: {file_path}. 행 개수 확인 필요.")

        return {'X': X_panel, 'Y': Y_panel}

    def __getitem__(self, idx):
        # 기존 로직과 동일
        global_end_idx = idx + (self.seq_len - 1)
        global_start_idx = global_end_idx - self.seq_len + 1
        
        start_file_idx = global_start_idx // self.steps_per_file
        start_local_idx = global_start_idx % self.steps_per_file
        end_file_idx = global_end_idx // self.steps_per_file
        end_local_idx = global_end_idx % self.steps_per_file
        
        if start_file_idx == end_file_idx:
            data = self._load_file_data(start_file_idx)
            X_seq = data['X'][start_local_idx : end_local_idx + 1]
            Y_target = data['Y'][end_local_idx]
        else:
            data_prev = self._load_file_data(start_file_idx)
            X_prev = data_prev['X'][start_local_idx : ]
            data_curr = self._load_file_data(end_file_idx)
            X_curr = data_curr['X'][ : end_local_idx + 1]
            X_seq = np.concatenate([X_prev, X_curr], axis=0)
            Y_target = data_curr['Y'][end_local_idx]

        return torch.FloatTensor(X_seq), torch.FloatTensor(Y_target)
