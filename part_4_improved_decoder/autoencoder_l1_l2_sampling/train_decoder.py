import math

import numpy as np
import torch


def train_decoder(
    model,
    num_epochs,
    trainloader,
    testloader,
    optimizer,
    scheduler,
    device,
    loss_fn_reconstruct,
    model_save_path="simclr.pth",
    add_l1=False,
    l1_lambda=1e-5,
):

    for epoch in range(num_epochs):
        total_train_loss = []
        total_valid_loss = []
        model.decoder.train()
        model.model.eval()
        model.model.encoder.eval()
        for data in trainloader:
            optimizer.zero_grad()
            X = data[3].to(device)
            X_masked = data[4].to(device)
            B, T, C, H, W = X_masked.shape
            X_masked = X_masked.reshape(B * T, C, H, W)
            B, T, C, H, W = X.shape
            X = X.reshape(B * T, C, H, W)
            _, recon_masked = model(X_masked)
            loss_batch = torch.tensor(0.0, device=X.device)
            for c in range(C):
                loss_batch += loss_fn_reconstruct(
                    recon_masked[:, c, :, :], X[:, c, :, :]
                )

            if add_l1:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss_batch += l1_lambda * l1_norm

            loss_batch.backward()
            optimizer.step()
            with torch.no_grad():
                loss_batch = loss_fn_reconstruct(recon_masked, X)
            total_train_loss.append(loss_batch.item())

        model.eval()
        with torch.no_grad():
            for data in testloader:
                X = data[3].to(device)
                X_masked = data[4].to(device)
                B, T, C, H, W = X_masked.shape
                X_masked = X_masked.reshape(B * T, C, H, W)
                B, T, C, H, W = X.shape
                X = X.reshape(B * T, C, H, W)
                _, recon_masked = model(X_masked)
                loss_batch = loss_fn_reconstruct(recon_masked, X)
                total_valid_loss.append(loss_batch.item())

        torch.save(model, model_save_path)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch: {epoch}, Total Train Loss: {np.mean(total_train_loss):.2f}, Total Valid Loss: {np.mean(total_valid_loss):.2f}, Learning Rate: {lr}"
        )
        scheduler.step(np.mean(total_valid_loss))