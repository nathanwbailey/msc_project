import os
import sys

import torch
from dataset import WeatherBenchDataset
from model import AutoEncoder
from torch.utils.data import DataLoader
from torchsummary import summary
from train import train_autoencoder

# Add parent directory to sys.path for downstream imports
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from downstream_model_lstm_no_decoder.downstream_task_main import \
    downstream_task as downstream_task_lstm
from latent_diffusion_model_conditional_attn.latent_model_main import \
    downstream_task as downstream_task_latent_diffusion_conditional_attn


def main():
    # Hyperparameters
    BATCH_SIZE = 128 // 3
    NUM_EPOCHS = 180
    LEARNING_RATE = 1e-3
    MODEL_SAVE_PATH = "det_autoencoder.pth"
    DATA_PATH = "/vol/bitbucket/nb324/ERA5_64x32_daily_850.pt"

    # Load and split data
    data = torch.load(DATA_PATH)
    n_samples = data.shape[0]
    n_train = int(n_samples * 0.6)
    n_valid = int(n_samples * 0.2)
    train_data = data[:n_train]
    valid_data = data[n_train : n_train + n_valid]
    test_data = data[n_train + n_valid :]

    # Normalization
    mean = train_data.mean(dim=(0, 2, 3), keepdim=True)
    std = train_data.std(dim=(0, 2, 3), keepdim=True)
    train_data = (train_data - mean) / std
    valid_data = (valid_data - mean) / std
    test_data = (test_data - mean) / std

    print(f"Train shape: {train_data.shape}")
    print(f"Valid shape: {valid_data.shape}")
    print(f"Test shape: {test_data.shape}")

    # Datasets and loaders
    train_dataset = WeatherBenchDataset(
        data=train_data, mask_prob_low=0.5, mask_prob_high=0.9
    )
    valid_dataset = WeatherBenchDataset(
        data=valid_data, mask_prob_low=0.5, mask_prob_high=0.9
    )

    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=3,
        multiprocessing_context="forkserver",
    )

    trainloader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    testloader = DataLoader(valid_dataset, shuffle=False, **loader_kwargs)

    # Model setup
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    C, H, W = next(iter(trainloader))[1].shape[2:]
    print(f"Shape: {(C, H, W)}")

    model = AutoEncoder(C).to(DEVICE)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, threshold=0.0001
    )

    # Train autoencoder
    # train_autoencoder(
    #     model,
    #     NUM_EPOCHS,
    #     trainloader,
    #     testloader,
    #     optimizer,
    #     scheduler,
    #     DEVICE,
    #     loss_fn,
    #     model_save_path=MODEL_SAVE_PATH,
    #     add_l1=True,
    #     l1_lambda=1e-6,
    # )
    model = torch.load(MODEL_SAVE_PATH, weights_only=False)

    print("Starting Downstream Task")
    downstream_configs = [
        {
            "context_window": 30,
            "stride": 1,
            "save": "downstream_model_no_decoder_weight_decay.pth",
        },
    ]
    # for cfg in downstream_configs:
    #     downstream_task_lstm(
    #         num_epochs=100,
    #         data=test_data,
    #         encoder_model=model,
    #         latent_dim=1000,
    #         context_window=cfg["context_window"],
    #         target_length=1,
    #         stride=cfg["stride"],
    #         model_save_path=cfg["save"],
    #         weight_decay=1e-5,
    #     )

    print("Starting Latent Downstream Task")
    downstream_task_latent_diffusion_conditional_attn(
        num_epochs=300,
        data=test_data,
        model_encoder=model,
        model_decoder=model.decoder,
    )


if __name__ == "__main__":
    main()
