
import torch
from torch.utils.data import DataLoader
from torchsummary import summary
from train import train_model
from dataset import WeatherBenchDataset
from model_decoder import SIMCLRDecoder, SIMCLR
from train_decoder import train_encoder_decoder, train_decoder
from pytorch_metric_learning.losses import NTXentLoss, SelfSupervisedLoss

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from downstream_model_lstm_no_decoder.downstream_task_main import downstream_task as downstream_task_lstm



def main():

    BATCH_SIZE = 128//3
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

    train_dataset = WeatherBenchDataset(data=train_data, augment_sample_random_mask=0.7)
    valid_dataset = WeatherBenchDataset(data=valid_data, augment_sample_random_mask=0.7)

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

    validloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=3,
        multiprocessing_context="forkserver"
    )


    loss_fn_contrastive = NTXentLoss(temperature=0.3)
    loss_fn_reconstruct = torch.nn.MSELoss()
    num_epochs = 180
    learning_rate = 1e-4
    learning_rate_decoder = 1e-4
    latent_dim = 128

    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    DEVICE = torch.device(DEVICE)
    C, H, W = next(iter(trainloader))[0].shape[2:]
    print(f'Shape: {C, H, W}')

    model = SIMCLR(in_channels=C, latent_dim=latent_dim)
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, threshold=0.0001)

    train_model(model, 100, trainloader, validloader, optimizer, scheduler, DEVICE, loss_fn_contrastive, model_save_path='simclr_change2.pth')

    model_decoder = SIMCLRDecoder(in_channels=C, model=model)
    model_decoder = model_decoder.to(DEVICE)

    optimizer = torch.optim.Adam(model_decoder.parameters(), lr=learning_rate_decoder, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, threshold=0.0001)
    print('Fine Tuning Both')

    train_encoder_decoder(model=model_decoder, num_epochs=num_epochs, trainloader=trainloader, testloader=validloader, optimizer=optimizer, scheduler=scheduler, device=DEVICE, loss_fn_contrastive=loss_fn_contrastive, loss_fn_reconstruct=loss_fn_reconstruct, model_save_path='simclr_decoder_change.pth', alpha=0.1)

    model_decoder = torch.load('simclr_decoder_change.pth', weights_only=False)

    print('Starting Downstream Task')
    downstream_task_lstm(num_epochs=100, data=test_data, encoder_model=model_decoder.model.encoder, latent_dim=1000, context_window=30, target_length=1, stride=1, model_save_path='downstream_model_no_decoder_change2.pth')

    #downstream_task_lstm(num_epochs=100, data=test_data, encoder_model=model_decoder.model.encoder, latent_dim=1000, context_window=30, target_length=1, stride=1, model_save_path='downstream_model_no_decoder_variable_mask.pth', mask_prob_low=0.5, mask_prob_high=0.9)

    for param in model_decoder.model.parameters():
        param.requires_grad = False
    model_decoder.model.eval()

    optimizer = torch.optim.Adam(model_decoder.decoder.parameters(), lr=learning_rate_decoder, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, threshold=0.0001)

    print('Training Decoder')

    train_decoder(model=model_decoder, num_epochs=200, trainloader=trainloader, testloader=validloader, optimizer=optimizer, scheduler=scheduler, device=DEVICE, loss_fn_reconstruct=loss_fn_reconstruct, model_save_path='simclr_decoder_freeze_change2.pth')


if __name__ == '__main__':
    main()
