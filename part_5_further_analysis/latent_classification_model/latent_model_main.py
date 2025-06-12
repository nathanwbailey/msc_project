import torch
from torch.utils.data import DataLoader

from .dataset import WeatherBenchDataset
from .model import LatentClassificationModel
from .train import train_classification_model, test_classification_network


def downstream_task(
    num_epochs,
    data,
    labels,
    model_encoder,
    model_save_path="classification_latent_model.pth",
    mask_prob_low=0.7,
    mask_prob_high=0.7,
    loss_fn=torch.nn.CrossEntropyLoss,
    latent_dim=1000,
    learning_rate=1e-3,
):

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    DEVICE = torch.device(DEVICE)

    BATCH_SIZE = 64
    n_samples = data.shape[0]

    n_train = int(n_samples * 0.6)
    n_valid = int(n_samples * 0.2)

    num_labels = torch.unique(labels).numel()
    print(num_labels)

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

    train_dataset = WeatherBenchDataset(
        data=train_data,
        labels=train_labels,
        mask_prob_low=mask_prob_low,
        mask_prob_high=mask_prob_high,
    )
    valid_dataset = WeatherBenchDataset(
        data=valid_data,
        labels=valid_labels,
        mask_prob_low=mask_prob_low,
        mask_prob_high=mask_prob_high,
    )
    test_dataset = WeatherBenchDataset(
        data=test_data,
        labels=test_labels,
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
        shuffle=False,
        pin_memory=True,
        num_workers=6,
        persistent_workers=True,
        prefetch_factor=5,
        multiprocessing_context="forkserver",
    )
    testloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=6,
        persistent_workers=True,
        prefetch_factor=5,
        multiprocessing_context="forkserver",
    )

    latent_model = LatentClassificationModel(
        latent_dim=latent_dim, num_labels=num_labels
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        latent_model.parameters(), lr=learning_rate, weight_decay=1e-2
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, threshold=0.0001
    )
    for param in model_encoder.parameters():
        param.requires_grad = False
    model_encoder.eval()

    train_classification_model(
        model=latent_model,
        encoder_model=model_encoder,
        num_epochs=num_epochs,
        trainloader=trainloader,
        testloader=validloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn(),
        model_save_path=model_save_path,
        device=DEVICE,
    )
    test_classification_network(
        model=latent_model,
        encoder_model=model_encoder,
        device=DEVICE,
        loss_fn=loss_fn(),
        testloader=testloader
    )
