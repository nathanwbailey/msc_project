import os
import sys

import torch
from dataset import WeatherBenchDataset
from model_decoder import SIMCLR, SIMCLRDecoder
from pytorch_metric_learning.losses import NTXentLoss, SelfSupervisedLoss
from torch.utils.data import DataLoader
from train_decoder import train_decoder, train_encoder_decoder

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from downstream_model_lstm_no_decoder.downstream_task_main import \
    downstream_task as downstream_task_lstm


def main():
    # Hyperparameters and paths
    BATCH_SIZE_OUTER = 128 // 3
    BATCH_SIZE_INNER = 32 // 3
    DATA_PATH = "/vol/bitbucket/nb324/ERA5_64x32_daily_850.pt"
    SIMCLR_PATH = "simclr.pth"
    SIMCLR_DECODER_PATH = "simclr_decoder.pth"
    SIMCLR_DECODER_FREEZE_PATH = "simclr_decoder_freeze.pth"
    NUM_EPOCHS = 250
    LEARNING_RATE = 1e-4
    LEARNING_RATE_DECODER = 1e-3
    LATENT_DIM = 128

    # Load and normalize data
    data = torch.load(DATA_PATH)
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

    print(
        f"Train shape: {train_data.shape}, Valid shape: {valid_data.shape}, Test shape: {test_data.shape}"
    )

    # Datasets and loaders
    train_dataset = WeatherBenchDataset(
        data=train_data, mask_prob_low=0.5, mask_prob_high=0.9
    )
    valid_dataset = WeatherBenchDataset(
        data=valid_data, mask_prob_low=0.5, mask_prob_high=0.9
    )

    loader_kwargs = dict(
        pin_memory=True,
        num_workers=6,
        persistent_workers=True,
        prefetch_factor=5,
        multiprocessing_context="forkserver",
    )

    trainloader_outer = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_OUTER,
        shuffle=True,
        **loader_kwargs,
    )
    validloader_outer = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE_OUTER,
        shuffle=False,
        **loader_kwargs,
    )
    trainloader_inner = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_INNER,
        shuffle=True,
        **loader_kwargs,
    )
    validloader_inner = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE_INNER,
        shuffle=False,
        **loader_kwargs,
    )

    # Losses
    loss_fn_contrastive = SelfSupervisedLoss(NTXentLoss(temperature=0.3))
    loss_fn_reconstruct = torch.nn.MSELoss()
    cycle_loss = torch.nn.MSELoss()

    # Device and model setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    C, H, W = next(iter(trainloader_outer))[0].shape[2:]
    print(f"Data shape: (C={C}, H={H}, W={W})")

    # SIMCLR pretraining (load or train)
    model = SIMCLR(in_channels=C, latent_dim=LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, threshold=0.0001
    )
    # Uncomment to train from scratch:
    # train_model(model, 100, trainloader_outer, validloader_outer, optimizer, scheduler, device, loss_fn_contrastive, cycle_loss, model_save_path=SIMCLR_PATH)
    model = torch.load(SIMCLR_PATH, weights_only=False)

    # Decoder fine-tuning
    model_decoder = SIMCLRDecoder(in_channels=C, model=model).to(device)
    optimizer_mae = torch.optim.Adam(
        model_decoder.parameters(), lr=LEARNING_RATE_DECODER, weight_decay=0
    )
    optimizer_simclr = torch.optim.Adam(
        model_decoder.model.parameters(), lr=LEARNING_RATE, weight_decay=0
    )
    print("Fine Tuning Both Encoder and Decoder")

    # train_encoder_decoder(
    #     model=model_decoder,
    #     num_epochs=NUM_EPOCHS,
    #     trainloader_outer=trainloader_outer,
    #     testloader_outer=validloader_outer,
    #     trainloader_inner=trainloader_inner,
    #     testloader_inner=validloader_inner,
    #     optimizer_simclr=optimizer_simclr,
    #     optimizer_mae=optimizer_mae,
    #     device=device,
    #     loss_fn_contrastive=loss_fn_contrastive,
    #     loss_fn_reconstruct=loss_fn_reconstruct,
    #     cycle_loss=cycle_loss,
    #     model_save_path=SIMCLR_DECODER_PATH,
    #     add_l1=True,
    #     l1_lambda=1e-6,
    #     add_l2=True,
    #     l2_lambda=1e-6
    # )

    model_dict = torch.load("simclr_decoder.pth")
    model = SIMCLR(in_channels=5, latent_dim=128).to(device)
    model_decoder = SIMCLRDecoder(in_channels=5, model=model).to(device)
    model_decoder.load_state_dict(model_dict["model_state_dict"])

    # Downstream task (LSTM)
    print("Starting Downstream Task")
    downstream_task_lstm(
        num_epochs=100,
        data=test_data,
        encoder_model=model_decoder.model.encoder,
        latent_dim=1000,
        context_window=30,
        target_length=1,
        stride=1,
        model_save_path="downstream_model_no_decoder_weight_decay.pth",
        weight_decay=1e-5,
    )

    # Freeze encoder, train decoder only
    for param in model_decoder.model.parameters():
        param.requires_grad = False
    model_decoder.model.eval()

    optimizer = torch.optim.Adam(
        model_decoder.decoder.parameters(), lr=LEARNING_RATE, weight_decay=0
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, threshold=0.0001
    )

    print("Training Decoder Only (Encoder Frozen)")
    train_decoder(
        model=model_decoder,
        num_epochs=200,
        trainloader=trainloader_inner,
        testloader=validloader_inner,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        loss_fn_reconstruct=loss_fn_reconstruct,
        model_save_path=SIMCLR_DECODER_FREEZE_PATH,
    )


if __name__ == "__main__":
    main()
