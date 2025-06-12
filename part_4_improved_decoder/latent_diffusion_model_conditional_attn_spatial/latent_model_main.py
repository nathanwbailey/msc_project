import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from .dataset import WeatherBenchDataset
from .ddpm import DDPM
from .ddpm_sch import plot_ddpm_schedules
from .model import LatentNetwork
from .train import train_diffusion_model


def downstream_task(
    num_epochs,
    data,
    model_encoder,
    model_decoder,
    model_save_path="conditional_latent_model.pth",
    mask_prob_low=0.7,
    mask_prob_high=0.7,
    loss_fn=torch.nn.MSELoss,
    latent_dim=1000,
    learning_rate=1e-3,
):

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    DEVICE = torch.device(DEVICE)

    BATCH_SIZE = 64
    n_samples = data.shape[0]
    n_train = int(n_samples * 0.6)
    train_data = data[:n_train]
    valid_data = data[n_train:]
    mean = train_data.mean(dim=(0, 2, 3), keepdim=True)
    std = train_data.std(dim=(0, 2, 3), keepdim=True)
    train_data = (train_data - mean) / std
    valid_data = (valid_data - mean) / std

    train_dataset = WeatherBenchDataset(
        data=train_data,
        mask_prob_low=mask_prob_low,
        mask_prob_high=mask_prob_high,
    )
    valid_dataset = WeatherBenchDataset(
        data=valid_data,
        mask_prob_low=mask_prob_low,
        mask_prob_high=mask_prob_high,
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
        shuffle=True,
        pin_memory=True,
        num_workers=6,
        persistent_workers=True,
        prefetch_factor=5,
        multiprocessing_context="forkserver",
    )

    latent_model = LatentNetwork(
        latent_dim=latent_dim, time_emb_dim=256, emb_dim=1024
    ).to(DEVICE)

    plot_ddpm_schedules(beta1=0.0001, beta2=0.02, T=1000)

    ddpm = DDPM(
        eps_model=latent_model,
        betas=(0.0001, 0.02),
        n_T=1000,
        loss_fn=loss_fn,
        device=DEVICE,
    )

    optimizer = torch.optim.Adam(
        ddpm.eps_model.parameters(), lr=learning_rate
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, threshold=0.0001
    )
    for param in model_encoder.parameters():
        param.requires_grad = False
    model_encoder.eval()

    for param in model_decoder.parameters():
        param.requires_grad = False
    model_decoder.eval()

    mse_losses, train_losses = train_diffusion_model(
        ddpm=ddpm,
        num_epochs=num_epochs,
        device=DEVICE,
        encoder_model=model_encoder,
        decoder_model=model_decoder,
        trainloader=trainloader,
        validloader=validloader,
        latent_dim=latent_dim,
        model_save_path=model_save_path,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    epochs = range(1, len(mse_losses) + 1)

    plt.clf()
    plt.plot(epochs, mse_losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Generative MSE Loss vs Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("latent_model_mse_losses.png")

    plt.clf()
    plt.plot(epochs, train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Noise Prediction MSE Loss vs Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("latent_model_training_losses.png")
