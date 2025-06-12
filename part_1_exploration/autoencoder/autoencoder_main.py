import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
from train_autoencoder import train_autoencoder
from weatherbench_dataset import WeatherBenchDataset

from autoencoder import VAE


def main():
    TRAIN_SPLIT = 0.8
    data = torch.load("/vol/bitbucket/nb324/era5_level0.pt")
    n_samples = data.shape[0]
    n_train = int(n_samples * TRAIN_SPLIT)
    train_data = data[:n_train]
    valid_data = data[n_train:]

    mean = train_data.mean(dim=(0, 2, 3), keepdim=True)
    std = train_data.std(dim=(0, 2, 3), keepdim=True)

    train_data = (train_data - mean) / std
    valid_data = (valid_data - mean) / std

    train_dataset = WeatherBenchDataset(data=train_data)
    valid_dataset = WeatherBenchDataset(data=valid_data)
    trainloader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=3,
        multiprocessing_context="forkserver",
    )

    testloader = DataLoader(
        valid_dataset,
        batch_size=128,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=3,
        multiprocessing_context="forkserver",
    )
    num_epochs = 300
    learning_rate = 1e-3
    latent_dim = 128
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    DEVICE = torch.device(DEVICE)

    model = VAE(5, latent_dim)
    summary(model, (5, 64, 32), depth=10)
    model = model.to(DEVICE)
    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, threshold=0.0001
    )

    train_losses, test_losses = train_autoencoder(
        model,
        num_epochs,
        trainloader,
        testloader,
        loss_fn,
        optimizer,
        scheduler,
        DEVICE,
    )
    epochs = range(1, len(train_losses) + 1)

    plt.clf()
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, test_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_losses_autoencoder.png")


if __name__ == "__main__":
    main()
