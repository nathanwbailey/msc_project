import gc
import os
import sys

import torch
from dataset import WeatherBenchDataset
from model_decoder import SIMCLR, SIMCLRDecoder
from pytorch_metric_learning.losses import NTXentLoss, SelfSupervisedLoss
from torch.utils.data import DataLoader
from train import train_model
from train_decoder import train_decoder, train_encoder_decoder

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from downstream_model_lstm_no_decoder.downstream_task_main import \
    downstream_task as downstream_task_lstm
from latent_classification_model.latent_model_main import \
    downstream_task as downstream_task_latent_classification
from latent_diffusion_model_conditional_attn.latent_model_main import \
    downstream_task as downstream_task_latent_diffusion_conditional_attn


def main():
    # --- Data Loading and Preprocessing ---
    BATCH_SIZE = 256 // 3
    data = torch.load("/vol/bitbucket/nb324/ERA5_64x32_daily_850.pt")
    labels = torch.load("/vol/bitbucket/nb324/ERA5_64x32_daily_850_labels.pt")
    n_samples = data.shape[0]
    n_train = int(n_samples * 0.6)
    n_valid = int(n_samples * 0.2)
    train_data = data[:n_train]
    train_labels = labels[:n_train]
    valid_data = data[n_train : n_train + n_valid]
    valid_labels = labels[n_train : n_train + n_valid]
    test_data = data[n_train + n_valid :]
    test_labels = labels[n_train + n_valid :]

    mean = train_data.mean(dim=(0, 2, 3), keepdim=True)
    std = train_data.std(dim=(0, 2, 3), keepdim=True)
    train_data = (train_data - mean) / std
    valid_data = (valid_data - mean) / std
    test_data = (test_data - mean) / std

    print(
        f"Train: {train_data.shape}, Valid: {valid_data.shape}, Test: {test_data.shape}"
    )

    # --- Dataset and DataLoader ---
    train_dataset = WeatherBenchDataset(
        data=train_data, mask_prob_low=0.5, mask_prob_high=0.9
    )
    valid_dataset = WeatherBenchDataset(
        data=valid_data, mask_prob_low=0.5, mask_prob_high=0.9
    )

    loader_args = dict(
        batch_size=BATCH_SIZE,
        pin_memory=True,
        num_workers=6,
        persistent_workers=True,
        prefetch_factor=5,
        multiprocessing_context="forkserver",
    )
    trainloader = DataLoader(train_dataset, shuffle=True, **loader_args)
    validloader = DataLoader(valid_dataset, shuffle=False, **loader_args)

    # --- Loss Functions and Training Setup ---
    loss_fn_contrastive = SelfSupervisedLoss(NTXentLoss(temperature=0.3))
    loss_fn_reconstruct = torch.nn.MSELoss()
    cycle_loss = torch.nn.MSELoss()
    num_epochs = 250
    learning_rate = 1e-4
    learning_rate_decoder = 1e-4
    latent_dim = 128

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    C, H, W = next(iter(trainloader))[0].shape[2:]  # Adjusted for dict output
    print(f"Input shape: (C={C}, H={H}, W={W})")

    # --- Model Initialization ---
    model = SIMCLR(in_channels=C, latent_dim=latent_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, threshold=0.0001
    )

    # --- Pretrain Encoder (SimCLR) ---
    # train_model(
    #     model, 100, trainloader, validloader, optimizer, scheduler, DEVICE,
    #     loss_fn_contrastive, cycle_loss, model_save_path="simclr.pth"
    # )
    model = torch.load("simclr.pth", weights_only=False)

    # --- Fine-tune Encoder + Decoder ---
    model_decoder = SIMCLRDecoder(in_channels=C, model=model).to(DEVICE)
    optimizer = torch.optim.Adam(
        model_decoder.parameters(), lr=learning_rate_decoder
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, threshold=0.0001
    )

    print("Fine Tuning Both")
    # train_encoder_decoder(
    #     model=model_decoder,
    #     num_epochs=num_epochs,
    #     trainloader=trainloader,
    #     testloader=validloader,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    #     device=DEVICE,
    #     loss_fn_contrastive=loss_fn_contrastive,
    #     loss_fn_reconstruct=loss_fn_reconstruct,
    #     cycle_loss=cycle_loss,
    #     model_save_path="simclr_decoder.pth",
    # )
    model_decoder = torch.load("simclr_decoder.pth", weights_only=False)

    # --- Downstream Tasks ---
    # print("Starting Downstream Task")
    # downstream_configs = [
    #     {"context_window": 30, "stride": 1, "save": "downstream_model_no_decoder_weight_decay.pth"},
    #     {"context_window": 5, "stride": 1, "save": "downstream_model_no_decoder_weight_decay_cw_5.pth"},
    #     {"context_window": 5, "stride": 5, "save": "downstream_model_no_decoder_weight_decay_s_5_cw_5.pth"},
    #     {"context_window": 5, "stride": 10, "save": "downstream_model_no_decoder_weight_decay_s_10_cw_5.pth"},
    #     {"context_window": 3, "stride": 1, "save": "downstream_model_no_decoder_weight_decay_cw_3.pth"},
    #     {"context_window": 1, "stride": 1, "save": "downstream_model_no_decoder_weight_decay_cw_1.pth"},
    # ]
    # for cfg in downstream_configs:
    #     torch.cuda.empty_cache()
    #     gc.collect()
    #     downstream_task_lstm(
    #         num_epochs=100,
    #         data=test_data,
    #         encoder_model=model_decoder.model.encoder,
    #         latent_dim=1000,
    #         context_window=cfg["context_window"],
    #         target_length=1,
    #         stride=cfg["stride"],
    #         model_save_path=cfg["save"],
    #         weight_decay=1e-5,
    #     )

    # --- Freeze Encoder, Train Decoder Only ---
    # for param in model_decoder.model.parameters():
    #     param.requires_grad = False
    # model_decoder.model.eval()

    # optimizer = torch.optim.Adam(model_decoder.decoder.parameters(), lr=learning_rate_decoder)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, threshold=0.0001)
    # torch.cuda.empty_cache()
    # gc.collect()
    print("Training Decoder")
    # train_decoder(
    #     model=model_decoder,
    #     num_epochs=200,
    #     trainloader=trainloader,
    #     testloader=validloader,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    #     device=DEVICE,
    #     loss_fn_reconstruct=loss_fn_reconstruct,
    #     model_save_path="simclr_decoder_freeze.pth",
    # )
    model_decoder = torch.load(
        "simclr_decoder_freeze.pth", weights_only=False
    )
    print("Starting Latent Downstream Task")
    downstream_task_latent_diffusion_conditional_attn(
        num_epochs=300,
        data=test_data,
        model_encoder=model_decoder.model.encoder,
        model_decoder=model_decoder.decoder,
    )
    # downstream_task_latent_classification(
    #     num_epochs=100,
    #     data=test_data,
    #     labels=test_labels,
    #     model_encoder=model_decoder.model.encoder,
    #     mask_prob_low=0.2,
    #     mask_prob_high=0.9,
    #     learning_rate=1e-3,
    # )


if __name__ == "__main__":
    main()
