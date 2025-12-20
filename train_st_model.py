# train_st_loader.py

import torch
from model.SpatioTemporal_dataloader import get_spatiotemporal_loaders


def main():
    train_loader, val_loader, test_loader, feat_dim = get_spatiotemporal_loaders(
        data_dir="data/xy",
        start_date="2025-04-01",
        end_date="2025-08-15",
        top_n=30,
        feature_list="feature_list/y_60m/top30_example_features_166.json",
        ban_list_path="global_ban_dates.json",
        export_path="data/datasets/spatiotfm",
        num_workers=0,
        dataset_workers=1,
        batch_size=64,
        train_num=10000,
        val_num=3000,
        test_num=3000,
        vali_date="2025-07-01",
        test_date="2025-08-01",
        seed=42,
    )

    print("=== Dataset info ===")
    print("feat_dim:", feat_dim)
    print("len(train):", len(train_loader.dataset))
    print("len(val):  ", len(val_loader.dataset))
    print("len(test): ", len(test_loader.dataset))

    xb, yb = next(iter(train_loader))
    print("xb.shape:", xb.shape)
    print("yb.shape:", yb.shape)


if __name__ == "__main__":
    main()
