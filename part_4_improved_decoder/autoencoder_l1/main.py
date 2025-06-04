import os
import sys

import torch
from dataset import WeatherBenchDataset
from model import AutoEncoder
from torch.utils.data import DataLoader
from torchsummary import summary
from train import train_autoencoder

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from downstream_model_lstm_no_decoder.downstream_task_main import \
    downstream_task as downstream_task_lstm
from latent_diffusion_model_conditional_attn.latent_model_main import \
    downstream_task as downstream_task_latent_diffusion_conditional_attn


def main():
    BATCH_SIZE = 128
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

    print(train_data.shape)
    print(valid_data.shape)
    print(test_data.shape)

    train_dataset = WeatherBenchDataset(
        data=train_data, mask_prob_low=0.5, mask_prob_high=0.9
    )
    valid_dataset = WeatherBenchDataset(
        data=valid_data, mask_prob_low=0.5, mask_prob_high=0.9
    )
    trainloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=3,
        multiprocessing_context="forkserver",
    )

    testloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=3,
        multiprocessing_context="forkserver",
    )
    num_epochs = 180
    learning_rate = 1e-3
    latent_dim = 128

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    DEVICE = torch.device(DEVICE)

    C, H, W = next(iter(trainloader))[1].shape[1:]
    print(f"Shape: {C, H, W}")
    model = AutoEncoder(C, latent_dim)
    model = model.to(DEVICE)
    # summary(model, (C, H, W), depth=10)
    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-6
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, threshold=0.0001
    )

    # train_autoencoder(model, num_epochs, trainloader, testloader, optimizer, scheduler, DEVICE, loss_fn, model_save_path="det_autoencoder.pth", add_l1=True, l1_lambda=1e-6)
    model = torch.load("det_autoencoder.pth", weights_only=False)

    # print('Starting Downstream Task')
    downstream_task_lstm(
        num_epochs=100,
        data=test_data,
        encoder_model=model.encoder,
        latent_dim=1000,
        context_window=5,
        target_length=1,
        stride=5,
        model_save_path="downstream_model_no_decoder_weight_decay_s_5_cw_5.pth",
        weight_decay=1e-5,
    )

    downstream_task_lstm(
        num_epochs=100,
        data=test_data,
        encoder_model=model.encoder,
        latent_dim=1000,
        context_window=5,
        target_length=1,
        stride=10,
        model_save_path="downstream_model_no_decoder_weight_decay_s_10_cw_5.pth",
        weight_decay=1e-5,
    )

    downstream_task_lstm(
        num_epochs=100,
        data=test_data,
        encoder_model=model.encoder,
        latent_dim=1000,
        context_window=3,
        target_length=1,
        stride=5,
        model_save_path="downstream_model_no_decoder_weight_decay_cw_3.pth",
        weight_decay=1e-5,
    )

    downstream_task_lstm(
        num_epochs=100,
        data=test_data,
        encoder_model=model.encoder,
        latent_dim=1000,
        context_window=1,
        target_length=1,
        stride=10,
        model_save_path="downstream_model_no_decoder_weight_decay_cw_1.pth",
        weight_decay=1e-5,
    )

    print("Starting Latent Downstream Task")
    # downstream_task_latent_diffusion(num_epochs=100, data=test_data, model=model)
    # downstream_task_latent_diffusion_conditional_attn(num_epochs=300, data=test_data, model_encoder=model.encoder, model_decoder=model.decoder)


if __name__ == "__main__":
    main()
