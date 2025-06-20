import os
import sys

import numpy as np
import optuna
import torch
from dataset import WeatherBenchDataset
from eval_sim import eval_model
from model import SCL
from optuna.samplers import GridSampler
from pytorch_metric_learning.losses import NTXentLoss, SelfSupervisedLoss
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
from train import train_model
from tsne import plot_tsne

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def objective(trial):
    learning_rate = trial.suggest_categorical("lr", [1e-5, 1e-4, 1e-3])
    latent_dim = trial.suggest_categorical("latent_dim", [128, 256])
    temperature = trial.suggest_categorical(
        "temperature", [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.3, 0.5]
    )
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
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=3,
        multiprocessing_context="forkserver",
    )

    loss_fn = NTXentLoss(temperature=temperature)
    loss_fn = SelfSupervisedLoss(loss_fn)
    num_epochs = 100

    C, H, W = next(iter(testloader))[0].shape[1:]
    print(f"Shape: {C, H, W}")
    model = SCL(in_channels=C, latent_dim=latent_dim)
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=0
    )
    train_model(
        model, num_epochs, trainloader, testloader, optimizer, DEVICE, loss_fn
    )
    model.projector_network = nn.Identity()
    cos_sim, rand_cos_sim, mean_var = eval_model(
        model.encoder, testloader, DEVICE
    )

    print("Mean Cosine similarity:", cos_sim)
    print("Negative Mean Cosine similarity:", rand_cos_sim)
    print("Mean Variance of Embeddings", mean_var)

    plot_tsne(
        train_data=train_data,
        valid_data=valid_data,
        filename=f"tsne_plots/{param_str}.png",
        decay=decay,
        model=model,
    )

def main():
    param_grid = {
        "lr": [1e-4],
        "latent_dim": [128],
        "temperature": [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.3],
    }

    sampler = GridSampler(param_grid)
    study = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(objective)


if __name__ == "__main__":
    main()
