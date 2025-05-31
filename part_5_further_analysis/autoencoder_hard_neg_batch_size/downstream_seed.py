import os
import sys
import torch
import random
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from downstream_model_lstm_no_decoder.downstream_task_main import downstream_task as downstream_task_lstm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    # --- Data Loading and Preprocessing ---
    BATCH_SIZE = 128 // 3
    data = torch.load("/vol/bitbucket/nb324/ERA5_64x32_daily_850.pt")
    n_samples = data.shape[0]
    n_train = int(n_samples * 0.6)
    n_valid = int(n_samples * 0.2)

    train_data = data[:n_train]
    valid_data = data[n_train : n_train + n_valid]
    test_data = data[n_train + n_valid :]

    mean = train_data.mean(dim=(0, 2, 3), keepdim=True)
    std = train_data.std(dim=(0, 2, 3), keepdim=True)
    train_data = (train_data - mean) / std
    valid_data = (valid_data - mean) / std
    test_data = (test_data - mean) / std

    print(f"Train: {train_data.shape}, Valid: {valid_data.shape}, Test: {test_data.shape}")

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_decoder = torch.load("det_autoencoder.pth", weights_only=False)

    # --- Downstream Tasks ---
    print("Starting Downstream Task")
    downstream_configs = [
        {"context_window": 5, "stride": 5, "save": "downstream_model_no_decoder_weight_decay_s_5_cw_5_2.pth"},
        {"context_window": 5, "stride": 10, "save": "downstream_model_no_decoder_weight_decay_s_10_cw_5_2.pth"},

    ]
    seeds = [0, 42, 123]
    for seed in seeds:
        set_seed(seed)
        for cfg in downstream_configs:
            downstream_task_lstm(
                num_epochs=100,
                data=test_data,
                encoder_model=model_decoder.encoder,
                latent_dim=1000,
                context_window=cfg["context_window"],
                target_length=1,
                stride=cfg["stride"],
                model_save_path=f"{seed}_" + cfg["save"],
                weight_decay=1e-5,
            )


if __name__ == "__main__":
    main()
