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
    """
    Train only the decoder part of the model.
    """
    for epoch in range(num_epochs):
        total_train_loss = []
        total_valid_loss = []

        model.decoder.train()
        model.model.eval()
        model.model.encoder.eval()

        # --- Training ---
        for data in trainloader:
            optimizer.zero_grad()
            X = data["X_orig"].to(device)
            X_masked = data["X_masked"].to(device)

            B, T, C, H, W = X_masked.shape
            X_masked = X_masked.reshape(B * T, C, H, W)
            B, T, C, H, W = X.shape
            X = X.reshape(B * T, C, H, W)

            _, recon_masked, _ = model(X_masked)
            loss = torch.tensor(0.0, device=X.device)
            for c in range(C):
                loss += loss_fn_reconstruct(
                    recon_masked[:, c, :, :], X[:, c, :, :]
                )
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                loss_batch = loss_fn_reconstruct(recon_masked, X)
            total_train_loss.append(loss_batch.item())

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            for data in testloader:
                X = data["X_orig"].to(device)
                X_masked = data["X_masked"].to(device)

                B, T, C, H, W = X_masked.shape
                X_masked = X_masked.reshape(B * T, C, H, W)
                B, T, C, H, W = X.shape
                X = X.reshape(B * T, C, H, W)

                _, recon_masked, _ = model(X_masked)
                loss = loss_fn_reconstruct(recon_masked, X)
                total_valid_loss.append(loss.item())

        torch.save(model, model_save_path)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch: {epoch}, "
            f"Total Train Loss: {np.mean(total_train_loss):.2f}, "
            f"Total Valid Loss: {np.mean(total_valid_loss):.2f}, "
            f"Learning Rate: {lr}"
        )
        scheduler.step(np.mean(total_valid_loss))


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
    cycle_loss,
    model_save_path="barlow_twins.pth",
    add_l1=False,
    add_l2=False,
    l1_lambda=1e-6,
    l2_lambda=1e-6,
    alpha_cycle=1,
):
    """
    Train both encoder and decoder with contrastive, reconstruction, and cycle losses.
    """
    alpha_start = 1.0
    alpha_end = 0.0
    k = 0.01

    for epoch in range(num_epochs):
        alpha = alpha_end + (alpha_start - alpha_end) * math.exp(-k * epoch)
        (
            total_train_loss,
            recon_train_loss,
            recon_train_loss_2,
            contrastive_train_loss,
            cycle_train_loss,
        ) = ([], [], [], [], [])
        (
            total_valid_loss,
            recon_valid_loss,
            recon_valid_loss_2,
            contrastive_valid_loss,
            cycle_valid_loss,
        ) = ([], [], [], [], [])

        model.train()
        # --- Training ---
        for data in trainloader:
            optimizer.zero_grad()
            X_augment = data["x_pos_1"].to(device)
            X_prime_augment = data["x_pos_2"].to(device)
            X_prime_2 = data["x_pos_3"].to(device)
            X = data["X_orig"].to(device)
            X_masked = data["X_masked"].to(device)
            X_prime_recon = data["X_masked_delta"].to(device)
            X_prime_2_recon = data["X_masked_delta_2"].to(device)

            X_enc = data["X_enc"].to(device)
            X_delta = data["X_delta"].to(device)
            X_minus_delta = data["X_minus_delta"].to(device)

            # Reshape
            B, T, C, H, W = X_augment.shape
            X_augment = X_augment.reshape(B * T, C, H, W)
            X_prime_augment = X_prime_augment.reshape(B * T, C, H, W)
            X_prime_2 = X_prime_2.reshape(B * T, C, H, W)
            X_masked = X_masked.reshape(B * T, C, H, W)
            X_enc = X_enc.reshape(B * T, C, H, W)
            X_delta = X_delta.reshape(B * T, C, H, W)
            X_minus_delta = X_minus_delta.reshape(B * T, C, H, W)
            B, T, C, H, W = X.shape
            X = X.reshape(B * T, C, H, W)
            X_prime_recon = X_prime_recon.reshape(B * T, C, H, W)
            X_prime_2_recon = X_prime_2_recon.reshape(B * T, C, H, W)

            # Forward
            z1, _, z1_x = model(X_augment)
            z2, z2_recon, z2_x = model(X_prime_augment)
            z3, z3_recon, z3_x = model(X_prime_2)
            _, recon_masked, _ = model(X_masked)
            # Losses
            loss_cycle = cycle_loss(
                z2_x - 2 * z1_x + z3_x, torch.zeros_like(z1_x)
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
                + loss_fn_contrastive(z1, z3)
                + loss_contrastive_recon
                + alpha_cycle * loss_cycle
            )

            loss_recon_X = torch.tensor(0.0, device=X.device)
            for c in range(C):
                loss_recon_X += loss_fn_reconstruct(
                    recon_masked[:, c, :, :], X[:, c, :, :]
                )

            # Optional regularization
            if add_l1:
                l1_norm = sum(
                    p.abs().sum() for p in model.decoder.parameters()
                )
                loss_recon_X += l1_lambda * l1_norm
            if add_l2:
                l2_norm = sum(
                    (p**2).sum() for p in model.decoder.parameters()
                )
                loss_recon_X += l2_lambda * l2_norm

            loss_batch = alpha * loss_contrastive + (1 - alpha) * loss_recon_X
            loss_batch.backward()
            optimizer.step()

            total_train_loss.append(loss_batch.item())
            recon_train_loss.append(loss_recon_X.item())
            recon_train_loss_2.append(
                loss_fn_reconstruct(recon_masked, X).item()
            )
            contrastive_train_loss.append(loss_contrastive.item())
            cycle_train_loss.append(loss_cycle.item())

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            for data in testloader:
                X_augment = data["x_pos_1"].to(device)
                X_prime_augment = data["x_pos_2"].to(device)
                X_prime_2 = data["x_pos_3"].to(device)
                X = data["X_orig"].to(device)
                X_masked = data["X_masked"].to(device)
                X_prime_recon = data["X_masked_delta"].to(device)
                X_prime_2_recon = data["X_masked_delta_2"].to(device)

                X_enc = data["X_enc"].to(device)
                X_delta = data["X_delta"].to(device)
                X_minus_delta = data["X_minus_delta"].to(device)

                # Reshape
                B, T, C, H, W = X_augment.shape
                X_augment = X_augment.reshape(B * T, C, H, W)
                X_prime_augment = X_prime_augment.reshape(B * T, C, H, W)
                X_prime_2 = X_prime_2.reshape(B * T, C, H, W)
                X_masked = X_masked.reshape(B * T, C, H, W)
                X_enc = X_enc.reshape(B * T, C, H, W)
                X_delta = X_delta.reshape(B * T, C, H, W)
                X_minus_delta = X_minus_delta.reshape(B * T, C, H, W)
                B, T, C, H, W = X.shape
                X = X.reshape(B * T, C, H, W)
                X_prime_recon = X_prime_recon.reshape(B * T, C, H, W)
                X_prime_2_recon = X_prime_2_recon.reshape(B * T, C, H, W)

                # Forward
                z1, _, z1_x = model(X_augment)
                z2, z2_recon, z2_x = model(X_prime_augment)
                z3, z3_recon, z3_x = model(X_prime_2)
                _, recon_masked, _ = model(X_masked)

                # Losses
                loss_cycle = cycle_loss(
                    z2_x - 2 * z1_x + z3_x, torch.zeros_like(z1_x)
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
                    + loss_fn_contrastive(z1, z3)
                    + loss_contrastive_recon
                )

                loss_recon_X = torch.tensor(0.0, device=X.device)
                for c in range(C):
                    loss_recon_X += loss_fn_reconstruct(
                        recon_masked[:, c, :, :], X[:, c, :, :]
                    )

                # Optional regularization
                if add_l1:
                    l1_norm = sum(
                        p.abs().sum() for p in model.decoder.parameters()
                    )
                    loss_recon_X += l1_lambda * l1_norm
                if add_l2:
                    l2_norm = sum(
                        (p**2).sum() for p in model.decoder.parameters()
                    )
                    loss_recon_X += l2_lambda * l2_norm

                loss_batch = (
                    alpha * loss_contrastive + (1 - alpha) * loss_recon_X
                )
                total_valid_loss.append(loss_batch.item())
                recon_valid_loss.append(loss_recon_X.item())
                recon_valid_loss_2.append(
                    loss_fn_reconstruct(recon_masked, X).item()
                )
                contrastive_valid_loss.append(loss_contrastive.item())
                cycle_valid_loss.append(loss_cycle.item())

        torch.save(model, model_save_path)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch: {epoch}, Alpha: {alpha:.2f}\n"
            f"  Total Train Loss:        {np.mean(total_train_loss):.2f}\n"
            f"    Contrastive Train Loss: {np.mean(contrastive_train_loss):.2f}\n"
            f"    Overall Recon Train Loss: {np.mean(recon_train_loss):.2f}\n"
            f"    Recon Train Loss:       {np.mean(recon_train_loss_2):.2f}\n"
            f"    Cycle Train Loss:       {np.mean(cycle_train_loss):.2f}\n"
            f"  Total Valid Loss:        {np.mean(total_valid_loss):.2f}\n"
            f"    Contrastive Valid Loss: {np.mean(contrastive_valid_loss):.2f}\n"
            f"    Overall Recon Valid Loss: {np.mean(recon_valid_loss):.2f}\n"
            f"    Recon Valid Loss:       {np.mean(recon_valid_loss_2):.2f}\n"
            f"    Cycle Valid Loss:       {np.mean(cycle_valid_loss):.2f}\n"
            f"Learning Rate: {lr}"
        )
        scheduler.step(np.mean(total_valid_loss))
