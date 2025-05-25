
from torch import nn
import torch
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
from dataset import WeatherBenchDataset
from model import SupConModel
from train import train_model
from pytorch_metric_learning.losses import SupConLoss
from eval_sim import eval_model
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from downstream_task_transformer.downstream_task_transformer_main import downstream_task

def main():
    # TRAIN_SPLIT = 0.7
    # data = torch.load('/vol/bitbucket/nb324/era5_level0.pt')
    # labels = torch.load('/vol/bitbucket/nb324/era5_level0_Y.pt')
    # print(len(labels))
    # print(len(data))
    # assert len(data) == len(labels)
    # n_samples = data.shape[0]
    # n_train = int(n_samples * TRAIN_SPLIT)
    # train_data = data[:n_train]
    # train_labels = labels[:n_train]
    # valid_data = data[n_train:]
    # valid_labels = labels[n_train:]

    # mean = train_data.mean(dim=(0, 2, 3), keepdim=True)
    # std = train_data.std(dim=(0, 2, 3), keepdim=True)

    # train_data = (train_data - mean) / std
    # valid_data = (valid_data - mean) / std
    BATCH_SIZE = 128
    TRAIN_SPLIT = 0.8
    data = torch.load('/vol/bitbucket/nb324/CL_X_train_full.pt')
    labels = torch.load('/vol/bitbucket/nb324/CL_Y_train.pt')
    n_samples = data.shape[0]
    n_train = int(n_samples * TRAIN_SPLIT)
    train_data = data[:n_train]
    valid_data = data[n_train:]
    train_labels = labels[:n_train]
    valid_labels = labels[n_train:]

    mean = train_data.mean(dim=(0, 2, 3), keepdim=True)
    std = train_data.std(dim=(0, 2, 3), keepdim=True)

    train_data = (train_data - mean) / std
    valid_data = (valid_data - mean) / std

    train_dataset = WeatherBenchDataset(data=train_data, labels=train_labels)
    valid_dataset = WeatherBenchDataset(data=valid_data, labels=valid_labels)
    trainloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=3,
        multiprocessing_context="forkserver"
    )

    testloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=3,
        multiprocessing_context="forkserver"
    )
    loss_fn = SupConLoss(temperature=0.1)
    num_epochs = 100
    learning_rate = 1e-4
    latent_dim = 100

    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    DEVICE = torch.device(DEVICE)

    C, H, W = next(iter(trainloader))[0].shape[1:]
    print(f'Shape: {C, H, W}')
    model = SupConModel(in_channels=C, latent_dim=latent_dim, dropout=0)

    summary(model, (C, H, W), depth=10)
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=20, threshold=0.0001)

    train_model(model, num_epochs, trainloader, testloader, optimizer, DEVICE, loss_fn)
    model.projector_network = nn.Identity()
    cos_sim, rand_cos_sim = eval_model(model.encoder, testloader, DEVICE)

    print("Mean cosine similarity:", cos_sim.mean().item())
    print("Random Mean cosine similarity:", rand_cos_sim.mean().item())

    test_data = torch.load('/vol/bitbucket/nb324/CL_X_test_full.pt')
    
    print('Starting Downstream Task')
    downstream_task(num_epochs=100, data=test_data, encoder_model=model, latent_dim=1000, context_window=30, target_length=1, stride=1)

if __name__ == '__main__':
    main()