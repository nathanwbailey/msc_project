import numpy as np
import pytorch_metric_learning.utils.loss_and_miner_utils as lmu
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
    model_save_path="barlow_twins.pth",
):
    for epoch in range(num_epochs):
        train_loss = []
        valid_loss = []
        model.train()
        for data in trainloader:
            optimizer.zero_grad()
            X = data[0].to(device)
            X_prime = data[1].to(device)
            X_prime_mask = data[2].to(device)
            B, T, C, H, W = X.shape
            X = X.reshape(B * T, C, H, W)
            X_prime = X_prime.reshape(B * T, C, H, W)
            X_prime_mask = X_prime_mask.reshape(B * T, C, H, W)
            Z = model(X)
            Z_prime = model(X_prime)
            Z_prime_mask = model(X_prime_mask)
            batch_size = X.shape[0]
            embeddings = torch.cat((Z, Z_prime, Z_prime_mask), dim=0)
            labels = torch.arange(batch_size)
            labels = torch.cat((labels, labels, labels), dim=0)
            loss_batch = loss_fn(embeddings, labels)
            loss_batch.backward()
            optimizer.step()
            train_loss.append(loss_batch.item())

        model.eval()
        with torch.no_grad():
            for data in testloader:
                X = data[0].to(device)
                X_prime = data[1].to(device)
                X_prime_mask = data[2].to(device)
                B, T, C, H, W = X.shape
                X = X.reshape(B * T, C, H, W)
                X_prime = X_prime.reshape(B * T, C, H, W)
                X_prime_mask = X_prime_mask.reshape(B * T, C, H, W)
                Z = model(X)
                Z_prime = model(X_prime)
                Z_prime_mask = model(X_prime_mask)
                batch_size = X.shape[0]
                embeddings = torch.cat((Z, Z_prime, Z_prime_mask), dim=0)
                labels = torch.arange(batch_size)
                labels = torch.cat((labels, labels, labels), dim=0)
                loss_batch = loss_fn(embeddings, labels)
                valid_loss.append(loss_batch.item())

        torch.save(model, model_save_path)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch: {epoch}, Train Loss: {np.mean(train_loss):.2f}, Validation Loss: {np.mean(valid_loss):.2f}, Learning Rate: {lr}"
        )
        scheduler.step(np.mean(valid_loss))
