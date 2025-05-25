import torch
import numpy as np

def train_model(model, num_epochs, trainloader, testloader, optimizer, scheduler, device, loss_fn, model_save_path="barlow_twins.pth"):
    for epoch in range(num_epochs):
        train_loss = []
        valid_loss = []
        model.train()
        for data in trainloader:
            optimizer.zero_grad()
            X = data[0].to(device)
            X_prime = data[1].to(device)
            z1, z2 = model(X, X_prime)
            loss_batch = loss_fn(z1, z2)
            loss_batch.backward()
            optimizer.step()
            train_loss.append(loss_batch.item())

        model.train()
        with torch.no_grad():
            for data in testloader:
                X = data[0].to(device)
                X_prime = data[1].to(device)
                z1, z2 = model(X, X_prime)
                loss_batch = loss_fn(z1, z2)
                valid_loss.append(loss_batch.item())

        torch.save(model, model_save_path)
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch}, Train Loss: {np.mean(train_loss):.2f}, Validation Loss: {np.mean(valid_loss):.2f}, Learning Rate: {lr}')
        # scheduler.step(np.mean(valid_loss))