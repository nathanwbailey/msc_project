
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
from dataset import WeatherBenchDataset
from model import SCL
from train import train_model
from pytorch_metric_learning.losses import NTXentLoss, SelfSupervisedLoss
from torch import nn
import sys
import os
from eval_sim import eval_model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from downstream_task_transformer.downstream_task_transformer_main import downstream_task


def main():

    BATCH_SIZE = 128
    TRAIN_SPLIT = 0.8
    data = torch.load('/vol/bitbucket/nb324/CL_X_train_full.pt')
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

    loss_fn = NTXentLoss(temperature=0.05)
    loss_fn = SelfSupervisedLoss(loss_fn)
    num_epochs = 100
    learning_rate = 3e-4
    latent_dim = 256
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    DEVICE = torch.device(DEVICE)

    C, H, W = next(iter(testloader))[0].shape[1:]
    print(f'Shape: {C, H, W}')
    model = SCL(in_channels=C, latent_dim=latent_dim)
    summary(model, (C, H, W), depth=10)
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    train_model(model, num_epochs, trainloader, testloader, optimizer, DEVICE, loss_fn, model_save_path="SCL_TEST.pth")

    model.projector_network = nn.Identity()
    cos_sim, rand_cos_sim = eval_model(model, testloader, DEVICE)

    print("Mean cosine similarity:", cos_sim.mean().item())
    print("Random Mean cosine similarity:", rand_cos_sim.mean().item())

    test_data = torch.load('/vol/bitbucket/nb324/CL_X_test_full.pt')
    print('Starting Downstream Task')
    downstream_task(num_epochs=100, data=test_data, encoder_model=model, latent_dim=1000, context_window=30, target_length=1, stride=1)

if __name__ == '__main__':
    main()
