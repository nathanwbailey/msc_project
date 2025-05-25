import math

import numpy as np
import torch


def train_encoder_decoder(
    model,
    num_epochs,
    trainloader,
    testloader,
    optimizer,
    scheduler,
    device,
    loss_fn_contrastive,
    loss_fn_reconstruct,
    model_save_path="barlow_twins.pth",
    alpha=0.5,
):
    alpha_start = 1.0
    alpha_end = 0.0
    k = 0.01
    for epoch in range(num_epochs):
        alpha = alpha_end + (alpha_start - alpha_end) * math.exp(-k * epoch)
        total_train_loss = []
        recon_train_loss = []
        contrastive_train_loss = []
        total_valid_loss = []
        recon_valid_loss = []
        contrastive_valid_loss = []
        model.train()
        for data in trainloader:
            optimizer.zero_grad()
            X_augment = data[0].to(device)
            X_prime_augment = data[1].to(device)
            X = data[2].to(device)

            z1, recon_X = model(X_augment)
            z2, recon_X_prime = model(X_prime_augment)
            loss_contrastive = loss_fn_contrastive(z1, z2)

            loss_recon_X = loss_fn_reconstruct(recon_X_prime, X)

            loss_batch = alpha * loss_contrastive + (1 - alpha) * (
                loss_recon_X
            )
            loss_batch.backward()
            optimizer.step()
            total_train_loss.append(loss_batch.item())
            recon_train_loss.append(loss_recon_X.item())
            contrastive_train_loss.append(loss_contrastive.item())

        model.model.train()
        model.decoder.eval()
        with torch.no_grad():
            for data in testloader:
                X_augment = data[0].to(device)
                X_prime_augment = data[1].to(device)
                X = data[2].to(device)

                z1, recon_X = model(X_augment)
                z2, recon_X_prime = model(X_prime_augment)
                loss_contrastive = loss_fn_contrastive(z1, z2)

                loss_recon_X = loss_fn_reconstruct(recon_X_prime, X)

                loss_batch = alpha * loss_contrastive + (1 - alpha) * (
                    loss_recon_X
                )
                total_valid_loss.append(loss_batch.item())
                recon_valid_loss.append(loss_recon_X.item())
                contrastive_valid_loss.append(loss_contrastive.item())

        torch.save(model, model_save_path)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch: {epoch}, Alpha: {alpha:.2f}, Total Train Loss: {np.mean(total_train_loss):.2f}, Contrastive Train Loss: {np.mean(contrastive_train_loss):.2f}, Recon Train Loss: {np.mean(recon_train_loss):.2f}, Total Valid Loss: {np.mean(total_valid_loss):.2f}, Contrastive Valid Loss: {np.mean(contrastive_valid_loss):.2f}, Recon Valid Loss: {np.mean(recon_valid_loss):.2f}, Learning Rate: {lr}"
        )
        # scheduler.step(np.mean(total_valid_loss))
