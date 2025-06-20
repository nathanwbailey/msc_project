import os
import sys

import torch
from dataset import WeatherBenchDataset
from model_decoder import SIMCLR, SIMCLRDecoder
from pytorch_metric_learning.losses import NTXentLoss, SelfSupervisedLoss
from torch.utils.data import DataLoader
from torchsummary import summary
from train_decoder import train_decoder

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

    BATCH_SIZE = 128 // 3
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
        num_workers=6,
        persistent_workers=True,
        prefetch_factor=5,
        multiprocessing_context="forkserver",
    )

    validloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=6,
        persistent_workers=True,
        prefetch_factor=5,
        multiprocessing_context="forkserver",
    )

    loss_fn_contrastive = NTXentLoss(temperature=0.3)
    loss_fn_contrastive = SelfSupervisedLoss(loss_fn_contrastive)
    loss_fn_reconstruct = torch.nn.MSELoss()
    cycle_loss = torch.nn.MSELoss()
    num_epochs = 180
    learning_rate_decoder = 1e-3
    learning_rate_fine_tune = 1e-4
    latent_dim = 128

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    DEVICE = torch.device(DEVICE)
    C, H, W = next(iter(trainloader))[0].shape[2:]
    print(f"Shape: {C, H, W}")

    model = SIMCLR(in_channels=C, latent_dim=latent_dim)
    model = model.to(DEVICE)
    model_decoder = SIMCLRDecoder(in_channels=C, model=model)
    model_decoder = model_decoder.to(DEVICE)

    # Freeze the projector
    for param in model_decoder.model.projector.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        model_decoder.parameters(),
        lr=learning_rate_decoder,
        weight_decay=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, threshold=0.0001
    )

    print("Training Decoder")

    train_decoder(model=model_decoder, num_epochs=180, trainloader=trainloader, testloader=validloader, optimizer=optimizer, scheduler=scheduler, device=DEVICE, loss_fn_reconstruct=loss_fn_reconstruct, model_save_path='simclr_decoder.pth', add_l1=True, l1_lambda=1e-6)

    print('Starting Downstream Task')
    downstream_task_lstm(num_epochs=100, data=test_data, encoder_model=model_decoder.model.encoder, latent_dim=1000, context_window=30, target_length=1, stride=1, model_save_path='downstream_model_no_decoder_weight_decay.pth', weight_decay=1e-5)


    print("Starting Latent Downstream Task")
    downstream_task_latent_diffusion_conditional_attn(
        num_epochs=300,
        data=test_data,
        model_encoder=model_decoder.model.encoder,
        model_decoder=model_decoder.decoder,
    )
    downstream_task_latent_classification(
        num_epochs=100,
        data=test_data,
        labels=test_labels,
        model_encoder=model_decoder.model.encoder,
        mask_prob_low=0.2,
        mask_prob_high=0.9,
        learning_rate=1e-3,
    )


if __name__ == "__main__":
    main()
