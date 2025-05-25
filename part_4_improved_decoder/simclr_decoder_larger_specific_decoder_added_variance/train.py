import torch
import numpy as np


def variance_loss(z, gamma=1.0, eps=1e-4):
    std = torch.sqrt(z.var(dim=0) + eps)
    loss = torch.mean(torch.relu(gamma - std))
    return loss

def covariance_loss(z):
    z = z - z.mean(dim=0)
    batch_size = z.size(0)

    cov = (z.T @ z) / (batch_size - 1)
    diag = torch.eye(z.size(1), device=z.device)
    off_diag = cov * (1 - diag)

    loss = (off_diag ** 2).sum() / z.size(1)
    return loss

def train_model(model, num_epochs, trainloader, testloader, optimizer, scheduler, device, loss_fn, cycle_loss, model_save_path="barlow_twins.pth"):
    for epoch in range(num_epochs):
        con_train_loss_1 = []
        con_train_loss_2 = []
        train_loss = []
        train_cycle_loss = []
        train_var_loss = []
        train_cov_loss = []
        con_valid_loss_1 = []
        con_valid_loss_2 = []
        valid_cycle_loss = []
        valid_loss = []
        valid_var_loss = []
        valid_cov_loss = []
        model.train()
        for data in trainloader:
            optimizer.zero_grad()
            X = data[0].to(device)
            X_prime = data[1].to(device)
            X_prime_2 = data[2].to(device)
            B, T, C, H, W = X.shape
            X = X.reshape(B*T, C, H, W)
            X_prime = X_prime.reshape(B*T, C, H, W)
            X_prime_2 = X_prime_2.reshape(B*T, C, H, W)
            Z, Z_X = model(X)
            Z_prime, Z_prime_X = model(X_prime)
            Z_prime_2, Z_prime_2_X = model(X_prime_2)
            loss_1 = loss_fn(Z, Z_prime)
            loss_2 = loss_fn(Z, Z_prime_2)
            loss_cycle = cycle_loss(Z_prime - 2 * Z + Z_prime_2, torch.zeros_like(Z))

            var_loss = variance_loss(Z_X) + variance_loss(Z_prime_X) + variance_loss(Z_prime_2_X)
            cov_loss = covariance_loss(Z_X) + covariance_loss(Z_prime_X) + covariance_loss(Z_prime_2_X)


            loss_batch = loss_1 + loss_2 + loss_cycle + 1 * var_loss + 1 * cov_loss
            loss_batch.backward()
            optimizer.step()
            con_train_loss_1.append(loss_1.item())
            con_train_loss_2.append(loss_2.item())
            train_loss.append(loss_batch.item())
            train_cycle_loss.append(loss_cycle.item())
            train_var_loss.append(var_loss.item())
            train_cov_loss.append(cov_loss.item())

        model.eval()
        with torch.no_grad():
            for data in testloader:
                X = data[0].to(device)
                X_prime = data[1].to(device)
                X_prime_2 = data[2].to(device)
                B, T, C, H, W = X.shape
                X = X.reshape(B*T, C, H, W)
                X_prime = X_prime.reshape(B*T, C, H, W)
                X_prime_2 = X_prime_2.reshape(B*T, C, H, W)
                Z, Z_X = model(X)
                Z_prime, Z_prime_X = model(X_prime)
                Z_prime_2, Z_prime_2_X = model(X_prime_2)
                loss_1 = loss_fn(Z, Z_prime)
                loss_2 = loss_fn(Z, Z_prime_2)
                loss_cycle = cycle_loss(Z_prime - 2 * Z + Z_prime_2, torch.zeros_like(Z))
                var_loss = variance_loss(Z_X) + variance_loss(Z_prime_X) + variance_loss(Z_prime_2_X)
                cov_loss = covariance_loss(Z_X) + covariance_loss(Z_prime_X) + covariance_loss(Z_prime_2_X)


                loss_batch = loss_1 + loss_2 + loss_cycle + 1 * var_loss + 1 * cov_loss
                con_valid_loss_1.append(loss_1.item())
                con_valid_loss_2.append(loss_2.item())
                valid_loss.append(loss_batch.item())
                valid_cycle_loss.append(loss_cycle.item())
                valid_var_loss.append(var_loss.item())
                valid_cov_loss.append(cov_loss.item())

        torch.save(model, model_save_path)
        lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch: {epoch}\n"
            f"  Training:\n"
            f"    Contrastive Loss 1 : {np.mean(con_train_loss_1):.2f}\n"
            f"    Contrastive Loss 2 : {np.mean(con_train_loss_2):.2f}\n"
            f"    Cycle Loss         : {np.mean(train_cycle_loss):.2f}\n"
            f"    Variance Loss      : {np.mean(train_var_loss):.2f}\n"
            f"    Covariance Loss    : {np.mean(train_cov_loss):.2f}\n"
            f"    Total Loss         : {np.mean(train_loss):.2f}\n"
            f"  Validation:\n"
            f"    Contrastive Loss 1 : {np.mean(con_valid_loss_1):.2f}\n"
            f"    Contrastive Loss 2 : {np.mean(con_valid_loss_2):.2f}\n"
            f"    Cycle Loss         : {np.mean(valid_cycle_loss):.2f}\n"
            f"    Variance Loss      : {np.mean(valid_var_loss):.2f}\n"
            f"    Covariance Loss    : {np.mean(valid_cov_loss):.2f}\n"
            f"    Total Loss         : {np.mean(valid_loss):.2f}\n"
            f"  Learning Rate        : {lr:.2e}"
        )

        scheduler.step(np.mean(valid_loss))