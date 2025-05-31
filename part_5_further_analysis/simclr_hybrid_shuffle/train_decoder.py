import math
import numpy as np
import torch
from torch.utils.data import DataLoader
import gc

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
):
    """
    Train only the decoder part of the model for reconstruction.
    """
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
            loss = torch.tensor(0.0, device=X.device)
            for c in range(C):
                loss += loss_fn_reconstruct(recon_masked[:, c, :, :], X[:, c, :, :])
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                loss_batch = loss_fn_reconstruct(recon_masked,X)
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
                loss = loss_fn_reconstruct(recon_masked, X)
                total_valid_loss.append(loss.item())

        torch.save(model, model_save_path)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch: {epoch}, "
            f"Train Loss: {np.mean(total_train_loss):.4f}, "
            f"Valid Loss: {np.mean(total_valid_loss):.4f}, "
            f"LR: {lr:.2e}"
        )
        scheduler.step(np.mean(total_valid_loss))


def train_encoder_decoder(
    model,
    num_epochs,
    train_dataset,
    dataloader_args, 
    testloader,
    optimizer,
    scheduler,
    device,
    loss_fn_contrastive,
    loss_fn_reconstruct,
    cycle_loss,
    model_save_path="barlow_twins.pth",
    add_l1=False,
    add_l2=False,
    l1_lambda=1e-6,
    l2_lambda=1e-6,
):
    """
    Train both encoder and decoder with contrastive and reconstruction losses.
    """
    alpha_start, alpha_end, k = 1.0, 0.0, 0.01
    trainloader = None
    for epoch in range(num_epochs):
        alpha = alpha_end + (alpha_start - alpha_end) * math.exp(-k * epoch)
        total_train_loss, recon_train_loss, recon_train_loss_2 = [], [], []
        contrastive_train_loss = []
        total_valid_loss, recon_valid_loss, recon_valid_loss_2 = [], [], []
        contrastive_valid_loss = []

        model.train()
        if trainloader is not None:
            del trainloader
            gc.collect()
            torch.cuda.empty_cache()
        if epoch % 2 == 0:
            trainloader = DataLoader(train_dataset, shuffle=False, **dataloader_args)
        else:
            trainloader = DataLoader(train_dataset, shuffle=True, **dataloader_args)
        for data in trainloader:
            optimizer.zero_grad()
            X_augment = data[0].to(device)
            X_prime_augment = data[1].to(device)
            X_prime_2 = data[2].to(device)
            X = data[3].to(device)
            X_masked = data[4].to(device)
            X_prime_recon = data[5].to(device)
            X_prime_2_recon = data[6].to(device)

            B, T, C, H, W = X_augment.shape
            X_augment = X_augment.reshape(B * T, C, H, W)
            X_prime_augment = X_prime_augment.reshape(B * T, C, H, W)
            X_prime_2 = X_prime_2.reshape(B * T, C, H, W)
            X_masked = X_masked.reshape(B * T, C, H, W)
            B, T, C, H, W = X.shape
            X = X.reshape(B * T, C, H, W)
            X_prime_recon = X_prime_recon.reshape(B * T, C, H, W)
            X_prime_2_recon = X_prime_2_recon.reshape(B * T, C, H, W)

            z1, _ = model(X_augment)
            z2, z2_recon = model(X_prime_augment)
            z3, z3_recon = model(X_prime_2)
            _, recon_masked = model(X_masked)

            loss_cycle = cycle_loss(z2 - 2 * z1 + z3, torch.zeros_like(z1))
            loss_contrastive_recon = torch.tensor(0.0, device=X.device)
            for c in range(C):
                loss_contrastive_recon += loss_fn_reconstruct(z2_recon[:, c, :, :], X_prime_recon[:, c, :, :]) + loss_fn_reconstruct(z3_recon[:, c, :, :], X_prime_2_recon[:, c, :, :])
            loss_contrastive = (
                loss_fn_contrastive(z1, z2)
                + loss_fn_contrastive(z1, z3)
                + loss_cycle
                + loss_contrastive_recon
            )
            loss_recon_X = torch.tensor(0.0, device=X.device)
            for c in range(C):
                loss_recon_X += loss_fn_reconstruct(recon_masked[:, c, :, :], X[:, c, :, :])

            if add_l1:
                l1_norm = sum(p.abs().sum() for p in model.decoder.parameters())
                loss_recon_X += l1_lambda * l1_norm
            if add_l2:
                l2_norm = sum((p**2).sum() for p in model.decoder.parameters())
                loss_recon_X += l2_lambda * l2_norm

            loss_batch = alpha * loss_contrastive + (1 - alpha) * loss_recon_X
            loss_batch.backward()
            optimizer.step()

            total_train_loss.append(loss_batch.item())
            recon_train_loss_2.append(loss_fn_reconstruct(recon_masked, X).item())
            recon_train_loss.append(loss_recon_X.item())
            contrastive_train_loss.append(loss_contrastive.item())

        model.eval()
        with torch.no_grad():
            for data in testloader:
                X_augment = data[0].to(device)
                X_prime_augment = data[1].to(device)
                X_prime_2 = data[2].to(device)
                X = data[3].to(device)
                X_masked = data[4].to(device)
                X_prime_recon = data[5].to(device)
                X_prime_2_recon = data[6].to(device)

                B, T, C, H, W = X_augment.shape
                X_augment = X_augment.reshape(B * T, C, H, W)
                X_prime_augment = X_prime_augment.reshape(B * T, C, H, W)
                X_prime_2 = X_prime_2.reshape(B * T, C, H, W)
                X_masked = X_masked.reshape(B * T, C, H, W)
                B, T, C, H, W = X.shape
                X = X.reshape(B * T, C, H, W)
                X_prime_recon = X_prime_recon.reshape(B * T, C, H, W)
                X_prime_2_recon = X_prime_2_recon.reshape(B * T, C, H, W)

                z1, _ = model(X_augment)
                z2, z2_recon = model(X_prime_augment)
                z3, z3_recon = model(X_prime_2)
                _, recon_masked = model(X_masked)

                loss_cycle = cycle_loss(z2 - 2 * z1 + z3, torch.zeros_like(z1))
                loss_contrastive_recon = torch.tensor(0.0, device=X.device)
                for c in range(C):
                    loss_contrastive_recon += loss_fn_reconstruct(z2_recon[:, c, :, :], X_prime_recon[:, c, :, :]) + loss_fn_reconstruct(z3_recon[:, c, :, :], X_prime_2_recon[:, c, :, :])
                loss_contrastive = (
                    loss_fn_contrastive(z1, z2)
                    + loss_fn_contrastive(z1, z3)
                    + loss_cycle
                    + loss_contrastive_recon
                )
                loss_recon_X = torch.tensor(0.0, device=X.device)
                for c in range(C):
                    loss_recon_X += loss_fn_reconstruct(recon_masked[:, c, :, :], X[:, c, :, :])
                loss_batch = alpha * loss_contrastive + (1 - alpha) * loss_recon_X

                total_valid_loss.append(loss_batch.item())
                recon_valid_loss.append(loss_recon_X.item())
                recon_valid_loss_2.append(loss_fn_reconstruct(recon_masked, X).item())
                contrastive_valid_loss.append(loss_contrastive.item())

        torch.save(model, model_save_path)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch: {epoch}, Alpha: {alpha:.2f}\n"
            f"  Train Loss:        {np.mean(total_train_loss):.4f}\n"
            f"    Contrastive:     {np.mean(contrastive_train_loss):.4f}\n"
            f"    Recon (all):     {np.mean(recon_train_loss):.4f}\n"
            f"    Recon (log):     {np.mean(recon_train_loss_2):.4f}\n"
            f"  Valid Loss:        {np.mean(total_valid_loss):.4f}\n"
            f"    Contrastive:     {np.mean(contrastive_valid_loss):.4f}\n"
            f"    Recon (all):     {np.mean(recon_valid_loss):.4f}\n"
            f"    Recon (log):     {np.mean(recon_valid_loss_2):.4f}\n"
            f"  LR: {lr:.2e}"
        )
        scheduler.step(np.mean(total_valid_loss))
