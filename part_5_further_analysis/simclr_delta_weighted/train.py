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
    model_save_path="barlow_twins.pth",
):
    """
    Train a model with contrastive and cycle consistency losses.
    """
    for epoch in range(num_epochs):
        train_loss, train_cycle_loss = [], []
        con_train_loss_1, con_train_loss_2 = [], []
        valid_loss, valid_cycle_loss = [], []
        con_valid_loss_1, con_valid_loss_2 = [], []

        model.train()
        for data in trainloader:
            optimizer.zero_grad()
            X = data[0].to(device)
            X_prime = data[1].to(device)
            X_prime_2 = data[2].to(device)
            aug_idx = data[7].flatten().to(device)
            weights = 1 / torch.cat((aug_idx, aug_idx))

            B, T, C, H, W = X.shape
            X = X.reshape(B * T, C, H, W)
            X_prime = X_prime.reshape(B * T, C, H, W)
            X_prime_2 = X_prime_2.reshape(B * T, C, H, W)

            Z = model(X)
            Z_prime = model(X_prime)
            Z_prime_2 = model(X_prime_2)

            loss_1 = (weights * loss_fn(Z, Z_prime)["loss"]["losses"]).mean()
            loss_2 = (
                weights * loss_fn(Z, Z_prime_2)["loss"]["losses"]
            ).mean()
            cyc_loss = cycle_loss(
                Z_prime - 2 * Z + Z_prime_2, torch.zeros_like(Z)
            )
            loss_batch = loss_1 + loss_2 + cyc_loss

            loss_batch.backward()
            optimizer.step()

            con_train_loss_1.append(loss_1.item())
            con_train_loss_2.append(loss_2.item())
            train_cycle_loss.append(cyc_loss.item())
            train_loss.append(loss_batch.item())

        model.eval()
        with torch.no_grad():
            for data in testloader:
                X = data[0].to(device)
                X_prime = data[1].to(device)
                X_prime_2 = data[2].to(device)
                aug_idx = data[7].flatten().to(device)
                weights = 1 / torch.cat((aug_idx, aug_idx))
                B, T, C, H, W = X.shape
                X = X.reshape(B * T, C, H, W)
                X_prime = X_prime.reshape(B * T, C, H, W)
                X_prime_2 = X_prime_2.reshape(B * T, C, H, W)

                Z = model(X)
                Z_prime = model(X_prime)
                Z_prime_2 = model(X_prime_2)

                loss_1 = (
                    weights * loss_fn(Z, Z_prime)["loss"]["losses"]
                ).mean()
                loss_2 = (
                    weights * loss_fn(Z, Z_prime_2)["loss"]["losses"]
                ).mean()
                cyc_loss = cycle_loss(
                    Z_prime - 2 * Z + Z_prime_2, torch.zeros_like(Z)
                )
                loss_batch = loss_1 + loss_2 + cyc_loss

                con_valid_loss_1.append(loss_1.item())
                con_valid_loss_2.append(loss_2.item())
                valid_cycle_loss.append(cyc_loss.item())
                valid_loss.append(loss_batch.item())

        torch.save(model, model_save_path)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch: {epoch}\n"
            f"  Train Loss:        {np.mean(train_loss):.4f}\n"
            f"    Contrastive 1:   {np.mean(con_train_loss_1):.4f}\n"
            f"    Contrastive 2:   {np.mean(con_train_loss_2):.4f}\n"
            f"    Cycle:           {np.mean(train_cycle_loss):.4f}\n"
            f"  Valid Loss:        {np.mean(valid_loss):.4f}\n"
            f"    Contrastive 1:   {np.mean(con_valid_loss_1):.4f}\n"
            f"    Contrastive 2:   {np.mean(con_valid_loss_2):.4f}\n"
            f"    Cycle:           {np.mean(valid_cycle_loss):.4f}\n"
            f"  LR: {lr:.2e}"
        )
        scheduler.step(np.mean(valid_loss))
