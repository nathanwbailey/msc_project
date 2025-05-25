import os
import sys

import optuna
import torch
from dataset import WeatherBenchDataset
from eval_sim import eval_model
from model import SupConModel
from optuna.samplers import GridSampler
from pytorch_metric_learning.losses import SupConLoss
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
from train import train_model
from tsne import plot_tsne

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
# from downstream_task_transformer.downstream_task_transformer_main import downstream_task

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def objective(trial):
    learning_rate = trial.suggest_categorical("lr", [1e-5, 1e-4, 1e-3])
    latent_dim = trial.suggest_categorical("latent_dim", [128, 256])
    temperature = trial.suggest_categorical(
        "temperature", [0.01, 0.03, 0.05, 0.07, 0.08, 0.1, 0.3, 0.5]
    )
    decay = 0.9

    param_str = "_".join(f"{k}={v}" for k, v in trial.params.items())
    print(param_str)

    BATCH_SIZE = 128
    TRAIN_SPLIT = 0.8
    data = torch.load("/vol/bitbucket/nb324/era5_level0.pt")
    labels = torch.load("/vol/bitbucket/nb324/era5_level0_Y.pt")
    n_samples = data.shape[0]
    n_train = int(n_samples * TRAIN_SPLIT)
    train_data = data[:n_train]
    test_data = data[n_train:]
    train_labels = labels[:n_train]
    test_labels = labels[:n_train]

    data = train_data
    labels = train_labels
    n_samples = data.shape[0]
    n_train = int(n_samples * TRAIN_SPLIT)
    train_data = data[:n_train]
    valid_data = data[n_train:]
    train_labels = labels[:n_train]
    valid_labels = labels[:n_train]

    mean = train_data.mean(dim=(0, 2, 3), keepdim=True)
    std = train_data.std(dim=(0, 2, 3), keepdim=True)

    train_data = (train_data - mean) / std
    valid_data = (valid_data - mean) / std

    train_dataset = WeatherBenchDataset(
        data=train_data, labels=train_labels, decay=decay
    )
    valid_dataset = WeatherBenchDataset(
        data=valid_data, labels=valid_labels, decay=decay
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
    loss_fn = SupConLoss(temperature=temperature)
    num_epochs = 100

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    DEVICE = torch.device(DEVICE)

    C, H, W = next(iter(trainloader))[0].shape[1:]
    print(f"Shape: {C, H, W}")
    model = SupConModel(in_channels=C, latent_dim=latent_dim, dropout=0)

    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=20, threshold=0.0001)

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
        train_labels=train_labels,
        valid_labels=valid_labels,
        filename=f"tsne_plots/{param_str}.png",
        decay=decay,
        model=model,
    ),

    test_data = torch.load("/vol/bitbucket/nb324/CL_X_test_full.pt")

    print("Starting Downstream Task")
    # test_error = downstream_task(num_epochs=50, data=test_data, encoder_model=model, latent_dim=1000, context_window=30, target_length=1, stride=1)
    # return test_error


def main():
    # param_grid = {
    #     "lr": [1e-5, 1e-4, 1e-3],
    #     "latent_dim": [128, 256],
    #     "temperature": [0.01, 0.05, 0.08, 0.1, 0.3, 0.5],
    # }

    param_grid = {
        "lr": [1e-3],
        "latent_dim": [128],
        "temperature": [0.08],
    }
    sampler = GridSampler(param_grid)
    study = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(objective)


if __name__ == "__main__":
    main()
