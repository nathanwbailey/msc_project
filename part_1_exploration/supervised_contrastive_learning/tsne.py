import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from dataset import WeatherBenchDataset
from sklearn import decomposition
from torch.utils.data import DataLoader


def plot_tsne(
    train_data,
    valid_data,
    train_labels,
    valid_labels,
    model=torch.load(
        "sup_con_model.pth",
        weights_only=False,
        map_location=torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ),
    filename="./tsne.png",
    decay=0.9,
):

    train_dataset = WeatherBenchDataset(
        data=train_data, labels=train_labels, decay=decay
    )
    valid_dataset = WeatherBenchDataset(
        data=valid_data, labels=valid_labels, decay=decay
    )

    batch_size = 20

    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=3,
        multiprocessing_context="forkserver",
    )

    testloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=3,
        multiprocessing_context="forkserver",
    )

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    DEVICE = torch.device(DEVICE)

    test_batch = next(iter(testloader))

    X = test_batch[0].to(DEVICE)
    X_prime = test_batch[1].to(DEVICE)
    test_batch[2].to(DEVICE)

    with torch.no_grad():
        embeddings_x = model(X).cpu()
        embeddings_x_prime = model(X_prime).cpu()

    embeddings_x = embeddings_x.cpu().numpy()
    embeddings_x_prime = embeddings_x_prime.cpu().numpy()

    combined = np.vstack([embeddings_x, embeddings_x_prime])
    pca = decomposition.PCA(n_components=2)
    combined_proj = pca.fit_transform(combined)

    x_proj = combined_proj[: len(embeddings_x)]
    x_prime_proj = combined_proj[len(embeddings_x) :]

    df = pd.DataFrame(
        {
            "x": np.concatenate([x_proj[:, 0], x_prime_proj[:, 0]]),
            "y": np.concatenate([x_proj[:, 1], x_prime_proj[:, 1]]),
            "group": ["A"] * batch_size + ["B"] * batch_size,
            "point_id": list(range(batch_size)) * 2,
        }
    )

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df, x="x", y="y", hue="point_id", palette="tab20", s=60
    )
    plt.legend().remove()
    plt.title("PCA Embeddings (Matching Colors for Corresponding Points)")
    plt.tight_layout()
    plt.savefig(filename)


if __name__ == "__main__":
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
    plot_tsne(train_data, valid_data, train_labels, valid_labels)
