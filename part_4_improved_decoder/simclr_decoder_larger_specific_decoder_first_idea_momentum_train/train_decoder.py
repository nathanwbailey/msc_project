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
                loss_batch = torch.tensor(0.0, device=X.device)
                for c in range(C):
                    loss_batch += loss_fn_reconstruct(
                        recon_masked[:, c, :, :], X[:, c, :, :]
                    )
                loss_batch = loss_fn_reconstruct(recon_masked, X)
                total_valid_loss.append(loss_batch.item())

        torch.save(model, model_save_path)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch: {epoch}, Total Train Loss: {np.mean(total_train_loss):.2f}, Total Valid Loss: {np.mean(total_valid_loss):.2f}, Learning Rate: {lr}"
        )
        scheduler.step(np.mean(total_valid_loss))


@torch.no_grad()
def update_momentum_model(model, momentum_model, m=0.996):
    for param_q, param_k in zip(
        model.parameters(), momentum_model.parameters()
    ):
        param_k.data = param_k.data * m + param_q.data * (1 - m)


def train_encoder_decoder(
    model,
    momentum_model,
    num_epochs,
    trainloader,
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
    for epoch in range(num_epochs):
        total_train_loss = []
        recon_train_loss = []
        recon_train_loss_2 = []
        contrastive_train_loss = []
        total_valid_loss = []
        recon_valid_loss = []
        recon_valid_loss_2 = []
        contrastive_valid_loss = []
        model.train()
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
            Z_prime_2, z3_recon = model(X_prime_2)

            loss_cycle = cycle_loss(
                z2 - 2 * z1 + Z_prime_2, torch.zeros_like(z1)
            )

            loss_contrastive_recon = torch.tensor(0.0, device=X.device)
            for c in range(C):
                loss_contrastive_recon += loss_fn_reconstruct(
                    z2_recon[:, c, :, :], X_prime_recon[:, c, :, :]
                ) + loss_fn_reconstruct(
                    z3_recon[:, c, :, :], X_prime_2_recon[:, c, :, :]
                )

            loss_contrastive = (
                loss_fn_contrastive(z1, z2)
                + loss_fn_contrastive(z1, Z_prime_2)
                + loss_cycle
                + loss_contrastive_recon
            )

            with torch.no_grad():
                _, recon_masked = momentum_model(X_masked)

            loss_recon_X = torch.tensor(0.0, device=X.device)

            for c in range(C):
                loss_recon_X += loss_fn_reconstruct(
                    recon_masked[:, c, :, :], X[:, c, :, :]
                )

            loss_contrastive.backward()
            optimizer.step()
            update_momentum_model(model, momentum_model)
            loss_recon_X_log = loss_fn_reconstruct(recon_masked, X)
            recon_train_loss_2.append(loss_recon_X_log.item())
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
                Z_prime_2, z3_recon = model(X_prime_2)

                loss_cycle = cycle_loss(
                    z2 - 2 * z1 + Z_prime_2, torch.zeros_like(z1)
                )

                loss_contrastive_recon = torch.tensor(0.0, device=X.device)
                for c in range(C):
                    loss_contrastive_recon += loss_fn_reconstruct(
                        z2_recon[:, c, :, :], X_prime_recon[:, c, :, :]
                    ) + loss_fn_reconstruct(
                        z3_recon[:, c, :, :], X_prime_2_recon[:, c, :, :]
                    )

                loss_contrastive = (
                    loss_fn_contrastive(z1, z2)
                    + loss_fn_contrastive(z1, Z_prime_2)
                    + loss_cycle
                    + loss_contrastive_recon
                )

                with torch.no_grad():
                    _, recon_masked = momentum_model(X_masked)

                loss_recon_X = torch.tensor(0.0, device=X.device)
                for c in range(C):
                    loss_recon_X += loss_fn_reconstruct(
                        recon_masked[:, c, :, :], X[:, c, :, :]
                    )

                recon_valid_loss.append(loss_recon_X.item())
                loss_recon_X_log = loss_fn_reconstruct(recon_masked, X)
                recon_valid_loss_2.append(loss_recon_X_log.item())
                contrastive_valid_loss.append(loss_contrastive.item())

        torch.save(model, model_save_path)
        torch.save(momentum_model, "momentum_model.pth")
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch: {epoch}\n"
            f"  Contrastive Train Loss: {np.mean(contrastive_train_loss):.2f}\n"
            f"  Overall Recon Train Loss: {np.mean(recon_train_loss):.2f}\n"
            f"  Recon Train Loss:       {np.mean(recon_train_loss_2):.2f}\n"
            f"  Contrastive Valid Loss: {np.mean(contrastive_valid_loss):.2f}\n"
            f"  Overall Recon Valid Loss: {np.mean(recon_valid_loss):.2f}\n"
            f"  Recon Valid Loss:       {np.mean(recon_valid_loss_2):.2f}\n"
            f"Learning Rate: {lr}"
        )

        # scheduler.step(np.mean(total_valid_loss))
