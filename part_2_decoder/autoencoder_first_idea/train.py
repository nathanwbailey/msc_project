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
        train_recon_1_loss = []
        train_recon_2_loss = []
        model.train()
        for data in trainloader:
            optimizer.zero_grad()
            X = data[0].to(device)
            X_prime = data[1].to(device)
            X_orig = data[2].to(device)
            X_orig_aug = data[3].to(device)
            recon_data_X = model(X)
            recon_data_X_prime = model(X_prime)
            recon_1 = loss_fn(recon_data_X, X_orig)
            recon_2 = loss_fn(recon_data_X_prime, X_orig)
            loss_batch = alpha * recon_1 + (1 - alpha) * recon_2
            loss_batch.backward()
            optimizer.step()
            train_recon_1_loss.append(recon_1.item())
            train_recon_2_loss.append(recon_2.item())
            train_loss.append(loss_batch.item())

        valid_loss = []
        valid_recon_1_loss = []
        valid_recon_2_loss = []
        model.eval()
        with torch.no_grad():
            for data in testloader:
                X = data[0].to(device)
                X_prime = data[1].to(device)
                X_orig = data[2].to(device)
                X_orig_aug = data[3].to(device)
                recon_data_X = model(X)
                recon_data_X_prime = model(X_prime)
                recon_1 = loss_fn(recon_data_X, X_orig)
                recon_2 = loss_fn(recon_data_X_prime, X_orig)
                loss_batch = alpha * recon_1 + (1 - alpha) * recon_2
                valid_recon_1_loss.append(recon_1.item())
                valid_recon_2_loss.append(recon_2.item())
                valid_loss.append(loss_batch.item())

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch: {epoch}, Train Loss = {np.mean(train_loss):.2f}, Train Recon Loss = {np.mean(train_recon_1_loss):.2f}, Train Masked Recon Loss = {np.mean(train_recon_2_loss):.2f}, Valid loss = {np.mean(valid_loss):.2f}, Valid Recon Loss = {np.mean(valid_recon_1_loss):.2f}, Valid Masked Recon Loss = {np.mean(valid_recon_2_loss):.2f}, lr = {lr:.5f}"
        )
        torch.save(model, model_save_path)

        scheduler.step(np.mean(valid_loss))
