import numpy as np
import torch
from torch import nn


def train_autoencoder(
    model,
    num_epochs,
    trainloader,
    testloader,
    optimizer,
    scheduler,
    device,
    loss_fn,
    alpha=0.5,
    model_save_path="det_autoencoder.pth",
):
    for epoch in range(num_epochs):
        train_loss = []
        model.train()
        for data in trainloader:
            optimizer.zero_grad()
            X = data[3].to(device)
            x_input = data[4].to(device)
            B, T, C, H, W = x_input.shape
            x_input = x_input.reshape(B * T, C, H, W)
            B, T, C, H, W = X.shape
            X = X.reshape(B * T, C, H, W)
            recon_data_X = model(x_input)
            loss_batch = loss_fn(recon_data_X, X)
            loss_batch.backward()
            optimizer.step()
            train_loss.append(loss_batch.item())

        valid_loss = []
        model.eval()
        with torch.no_grad():
            for data in testloader:
                X = data[3].to(device)
                x_input = data[4].to(device)
                B, T, C, H, W = x_input.shape
                x_input = x_input.reshape(B * T, C, H, W)
                B, T, C, H, W = X.shape
                X = X.reshape(B * T, C, H, W)
                recon_data_X = model(x_input)
                loss_batch = loss_fn(recon_data_X, X)
                valid_loss.append(loss_batch.item())

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch: {epoch}, Train Loss = {np.mean(train_loss):.2f}, Valid loss = {np.mean(valid_loss):.2f}, lr = {lr:.5f}"
        )
        torch.save(model, model_save_path)

        scheduler.step(np.mean(valid_loss))
