import numpy as np
import torch


def train_model(
    model,
    num_epochs,
    trainloader,
    testloader,
    optimizer,
    scheduler,
    device,
    loss_fn,
    cycle_loss,
    model_save_path="simclr.pth",
    alpha_cycle=1,
):
    """
    Train a model with contrastive and cycle consistency losses.

    Args:
        model: PyTorch model.
        num_epochs (int): Number of epochs.
        trainloader: DataLoader for training data.
        testloader: DataLoader for validation data.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        device: Device to use.
        loss_fn: Contrastive loss function.
        cycle_loss: Cycle consistency loss function.
        model_save_path (str): Path to save the model.
        alpha_cycle (float): Weight for cycle loss.
        alpha_jerk (float): Weight for jerk loss.
    """
    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        con_train_loss_1, con_train_loss_2 = [], []
        train_loss, train_cycle_loss = [], []

        for batch in trainloader:
            optimizer.zero_grad()
            X = batch["x_pos_1"].to(device)
            X_prime = batch["x_pos_2"].to(device)
            X_prime_2 = batch["x_pos_3"].to(device)

            X_enc = batch["X_enc"].to(device)
            X_delta = batch["X_delta"].to(device)
            X_minus_delta = batch["X_minus_delta"].to(device)

            B, T, C, H, W = X.shape
            X = X.reshape(B * T, C, H, W)
            X_prime = X_prime.reshape(B * T, C, H, W)
            X_prime_2 = X_prime_2.reshape(B * T, C, H, W)
            X_enc = X_enc.reshape(B * T, C, H, W)
            X_delta = X_delta.reshape(B * T, C, H, W)
            X_minus_delta = X_minus_delta.reshape(B * T, C, H, W)

            Z, Z_X = model(X)
            Z_prime, Z_prime_X = model(X_prime)
            Z_prime_2, Z_prime_2_X = model(X_prime_2)

            loss_1 = loss_fn(Z, Z_prime)
            loss_2 = loss_fn(Z, Z_prime_2)
            loss_cycle = cycle_loss(
                Z_prime_X - 2 * Z_X + Z_prime_2_X, torch.zeros_like(Z_X)
            )
            loss_batch = loss_1 + loss_2 + alpha_cycle * loss_cycle

            loss_batch.backward()
            optimizer.step()

            con_train_loss_1.append(loss_1.item())
            con_train_loss_2.append(loss_2.item())
            train_loss.append(loss_batch.item())
            train_cycle_loss.append(loss_cycle.item())

        # --- Validation ---
        model.eval()
        con_valid_loss_1, con_valid_loss_2 = [], []
        valid_loss, valid_cycle_loss = [], []

        with torch.no_grad():
            for batch in testloader:
                X = batch["x_pos_1"].to(device)
                X_prime = batch["x_pos_2"].to(device)
                X_prime_2 = batch["x_pos_3"].to(device)

                X_enc = batch["X_enc"].to(device)
                X_delta = batch["X_delta"].to(device)
                X_minus_delta = batch["X_minus_delta"].to(device)

                B, T, C, H, W = X.shape
                X = X.reshape(B * T, C, H, W)
                X_prime = X_prime.reshape(B * T, C, H, W)
                X_prime_2 = X_prime_2.reshape(B * T, C, H, W)
                X_enc = X_enc.reshape(B * T, C, H, W)
                X_delta = X_delta.reshape(B * T, C, H, W)
                X_minus_delta = X_minus_delta.reshape(B * T, C, H, W)

                Z, Z_X = model(X)
                Z_prime, Z_prime_X = model(X_prime)
                Z_prime_2, Z_prime_2_X = model(X_prime_2)

                loss_1 = loss_fn(Z, Z_prime)
                loss_2 = loss_fn(Z, Z_prime_2)
                loss_cycle = cycle_loss(
                    Z_prime_X - 2 * Z_X + Z_prime_2_X, torch.zeros_like(Z_X)
                )

                loss_batch = loss_1 + loss_2 + alpha_cycle * loss_cycle

                con_valid_loss_1.append(loss_1.item())
                con_valid_loss_2.append(loss_2.item())
                valid_loss.append(loss_batch.item())
                valid_cycle_loss.append(loss_cycle.item())

        # --- Save Model and Print Stats ---
        torch.save(model, model_save_path)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch: {epoch}, "
            f"Con Train Loss 1: {np.mean(con_train_loss_1):.2f}, "
            f"Con Train Loss 2: {np.mean(con_train_loss_2):.2f}, "
            f"Train Cycle Loss: {np.mean(train_cycle_loss):.2f}, "
            f"Train Loss: {np.mean(train_loss):.2f}, "
            f"Con Valid Loss 1: {np.mean(con_valid_loss_1):.2f}, "
            f"Con Valid Loss 2: {np.mean(con_valid_loss_2):.2f}, "
            f"Valid Cycle Loss: {np.mean(valid_cycle_loss):.2f}, "
            f"Validation Loss: {np.mean(valid_loss):.2f}, "
            f"Learning Rate: {lr}"
        )
        scheduler.step(np.mean(valid_loss))
