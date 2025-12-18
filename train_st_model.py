# train_st_loader.py

import torch
from model.SpatioTemporal_dataloader import get_spatiotemporal_loaders


def main():
    # ⚠️ 처음에는 메모리 안전하게 보수적으로 세팅
    train_loader, val_loader, test_loader, feat_dim = get_spatiotemporal_loaders(
        data_dir="data/xy",
        start_date="2025-04-01",  # 일단 반년 정도로 줄여서 테스트 (원하면 다시 늘리면 됨)
        end_date="2025-08-15",
        top_n=30,
        feature_list="feature_list/y_60m/top30_example_features_166.json",
        ban_list_path="global_ban_dates.json",

        # h5 export/caching: 처음에는 None으로 해보고, 나중에 안정되면 켜도 됨
        export_path="data/datasets/spatiotfm",

        # ❗ DataLoader worker: 0으로 (subprocess 안 만들게)
        num_workers=0,

        # ❗ Dataset 내부 병렬: 1로 (Pool 안 쓰게, 메모리/프로세스 안정)
        dataset_workers=1,

        batch_size=64,

        # 시퀀스 개수 제한 (처음엔 확 줄여서 메모리 확인)
        train_num=10000,
        val_num=3000,
        test_num=3000,

        # 날짜 기준 split
        vali_date="2025-07-01",
        test_date="2025-08-01",

        seed=42,
    )

    print("=== Dataset info ===")
    print("feat_dim:", feat_dim)
    print("len(train):", len(train_loader.dataset))
    print("len(val):  ", len(val_loader.dataset))
    print("len(test): ", len(test_loader.dataset))

    # 한 배치만 꺼내서 shape 확인
    xb, yb = next(iter(train_loader))
    print("xb.shape:", xb.shape)
    print("yb.shape:", yb.shape)


if __name__ == "__main__":
    main()
