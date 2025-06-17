import os
import sys

import torch
from dataset import WeatherBenchDataset
from loss import BarlowTwinsLoss
from model import BarlowTwins
from torch.utils.data import DataLoader
from torchsummary import summary
from train import train_model
from tsne import plot_tsne

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
import optuna
from eval_sim import eval_model
from optuna.samplers import GridSampler

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def objective(trial):

    learning_rate = trial.suggest_categorical("lr", [1e-5, 1e-4, 1e-3, 1e-2])
    latent_dim = trial.suggest_categorical("latent_dim", [128, 256, 512])
    lambd = trial.suggest_categorical("lambd", [0.0001, 0.001, 0.01, 0.1])
    scale_factor = 1
    decay = 0.9

    param_str = "_".join(f"{k}={v}" for k, v in trial.params.items())
    print(param_str)

    BATCH_SIZE = 128
    TRAIN_SPLIT = 0.8
    data = torch.load("/vol/bitbucket/nb324/era5_level0.pt")
    n_samples = data.shape[0]
    n_train = int(n_samples * TRAIN_SPLIT)
    train_data = data[:n_train]
    test_data = data[n_train:]

    data = train_data
    n_samples = data.shape[0]
    n_train = int(n_samples * TRAIN_SPLIT)
    train_data = data[:n_train]
    valid_data = data[n_train:]

    mean = train_data.mean(dim=(0, 2, 3), keepdim=True)
    std = train_data.std(dim=(0, 2, 3), keepdim=True)

    train_data = (train_data - mean) / std
    valid_data = (valid_data - mean) / std

    train_dataset = WeatherBenchDataset(data=train_data, decay=decay)
    valid_dataset = WeatherBenchDataset(data=valid_data, decay=decay)

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

    loss_fn = BarlowTwinsLoss(scale_factor=scale_factor, lambd=lambd)
    num_epochs = 100

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    DEVICE = torch.device(DEVICE)
    C, H, W = next(iter(trainloader))[0].shape[1:]

    model = BarlowTwins(in_channels=C, latent_dim=latent_dim)
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=0
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, threshold=0.0001
    )

    train_model(
        model,
        num_epochs,
        trainloader,
        testloader,
        optimizer,
        scheduler,
        DEVICE,
        loss_fn,
    )

    plot_tsne(
        train_data=train_data,
        valid_data=valid_data,
        filename=f"tsne_plots/{param_str}.png",
        decay=decay,
        model=model.encoder,
    )

    cos_sim, rand_cos_sim, mean_var = eval_model(
        model.encoder, testloader, DEVICE
    )

    print("Mean Cosine similarity:", cos_sim)
    print("Negative Mean Cosine similarity:", rand_cos_sim)
    print("Mean Variance of Embeddings", mean_var)

def main():
    param_grid = {
        "lr": [1e-3],
        "latent_dim": [128],
        "lambd": [0.001, 0.01, 0.1],
    }
    sampler = GridSampler(param_grid)
    study = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(objective)


if __name__ == "__main__":
    main()
