import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchsummary import summary

from .downstream_model_lstm import Seq2SeqModel
from .downstream_model_lstm_test import test_lstm_model
from .downstream_model_lstm_train import train_lstm_model
from .weatherbench_dataset_window import WeatherBenchDatasetWindow


def downstream_task(
    num_epochs,
    data,
    encoder_model,
    latent_dim,
    context_window,
    target_length,
    stride,
    model_save_path="downstream_model_no_decoder.pth",
):
    BATCH_SIZE = 64
    n_samples = data.shape[0]

    n_train = int(n_samples * 0.6)
    n_valid = int(n_samples * 0.2)

    train_data = data[:n_train]
    valid_data = data[n_train : n_train + n_valid]
    test_data = data[n_train + n_valid :]

    mean = train_data.mean(dim=(0, 2, 3), keepdim=True)
    std = train_data.std(dim=(0, 2, 3), keepdim=True)

    train_data = (train_data - mean) / std
    valid_data = (valid_data - mean) / std
    test_data = (test_data - mean) / std

    print(train_data.shape)
    print(valid_data.shape)
    print(test_data.shape)

    train_dataset = WeatherBenchDatasetWindow(
        data=train_data,
        context_length=context_window,
        target_length=target_length,
        stride=stride,
    )
    valid_dataset = WeatherBenchDatasetWindow(
        data=valid_data,
        context_length=context_window,
        target_length=target_length,
        stride=stride,
    )
    test_dataset = WeatherBenchDatasetWindow(
        data=test_data,
        context_length=context_window,
        target_length=target_length,
        stride=stride,
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

    validloader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=3,
        multiprocessing_context="forkserver",
    )
    testloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
        prefetch_factor=3,
        multiprocessing_context="forkserver",
    )
    learning_rate = 1e-2

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    DEVICE = torch.device(DEVICE)

    for param in encoder_model.parameters():
        param.requires_grad = False
    encoder_model.eval()

    C, H, W = next(iter(trainloader))[1].shape[2:]
    print(f"Shape: {C, H, W}")
    # summary(encoder_model, (C, H, W), depth=10)

    seq2seq_model = Seq2SeqModel(
        input_size=latent_dim,
        hidden_size=128,
        num_layers=1,
        output_len=target_length,
        bidirectional=False,
    )
    seq2seq_model = seq2seq_model.to(DEVICE)

    optimizer = torch.optim.Adam(seq2seq_model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, threshold=0.0001
    )
    loss_fn = torch.nn.MSELoss()

    train_plot, valid_plot = train_lstm_model(
        num_epochs,
        encoder_model,
        seq2seq_model,
        loss_fn,
        optimizer,
        scheduler,
        trainloader,
        validloader,
        DEVICE,
        model_save_path=model_save_path,
    )

    epochs = range(1, len(train_plot) + 1)

    plt.clf()
    plt.plot(epochs, train_plot, label="Training Loss")
    plt.plot(epochs, valid_plot, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_losses.png")
    test_error = test_lstm_model(
        encoder_model, seq2seq_model, loss_fn, validloader, DEVICE
    )
    test_error = test_lstm_model(
        encoder_model, seq2seq_model, loss_fn, testloader, DEVICE
    )
    return test_error, seq2seq_model
