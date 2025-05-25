
import torch
from torch.utils.data import DataLoader
from torchsummary import summary
from dataset import WeatherBenchDataset
from model import AutoEncoder
from train import train_autoencoder
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from downstream_model_lstm_no_decoder.downstream_task_main import downstream_task as downstream_task_lstm


def main():
    BATCH_SIZE = 128
    data = torch.load('/vol/bitbucket/nb324/ERA5_64x32_daily_850.pt')
    n_samples = data.shape[0]
    n_train = int(n_samples * 0.6)
    n_valid = int(n_samples * 0.2)
    train_data = data[:n_train]
    valid_data = data[n_train:n_train+n_valid]
    test_data = data[n_train+n_valid:]

    mean = train_data.mean(dim=(0, 2, 3), keepdim=True)
    std = train_data.std(dim=(0, 2, 3), keepdim=True)

    train_data = (train_data - mean) / std
    valid_data = (valid_data - mean) / std
    test_data = (test_data - mean) / std

    print(train_data.shape)
    print(valid_data.shape)
    print(test_data.shape)

    train_dataset = WeatherBenchDataset(data=train_data, mask_prob=0.7)
    valid_dataset = WeatherBenchDataset(data=valid_data, mask_prob=0.7)
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
    num_epochs = 180
    learning_rate = 1e-3
    latent_dim = 128

    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    DEVICE = torch.device(DEVICE)

        
    C, H, W = next(iter(trainloader))[1].shape[1:]
    print(f'Shape: {C, H, W}')
    model = AutoEncoder(C, latent_dim)
    model = model.to(DEVICE)
    # summary(model, (C, H, W), depth=10)
    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, threshold=0.0001)
    
    train_autoencoder(model, num_epochs, trainloader, testloader, optimizer, scheduler, DEVICE, loss_fn, model_save_path="det_autoencoder.pth")
    # model = torch.load("det_autoencoder.pth", weights_only=False)

    print('Starting Downstream Task')
    downstream_task_lstm(num_epochs=100, data=test_data, encoder_model=model.encoder, latent_dim=1000, context_window=30, target_length=1, stride=1, model_save_path='downstream_model_no_decoder.pth')

if __name__ == '__main__':
    main()