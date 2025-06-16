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
            Z = model(X)
            Z_prime = model(X_prime)
            loss_batch = loss_fn(Z, Z_prime)
            loss_batch.backward()
            optimizer.step()
            train_loss.append(loss_batch.item())

        model.eval()
        with torch.no_grad():
            for data in testloader:
                X = data[0].to(device)
                X_prime = data[1].to(device)
                Z = model(X)
                Z_prime = model(X_prime)
                loss_batch = loss_fn(Z, Z_prime)
                valid_loss.append(loss_batch.item())

        torch.save(model, model_save_path)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch: {epoch}, Train Loss: {np.mean(train_loss):.2f}, Validation Loss: {np.mean(valid_loss):.2f}, Learning Rate: {lr}"
        )
        scheduler.step(np.mean(valid_loss))
