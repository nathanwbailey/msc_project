import torch
import numpy as np

def train_model(model, num_epochs, trainloader, testloader, optimizer, scheduler, device, loss_fn, cycle_loss, model_save_path="barlow_twins.pth", alpha_cycle=10):
    for epoch in range(num_epochs):
        con_train_loss_1 = []
        con_train_loss_2 = []
        train_loss = []
        train_cycle_loss = []
        con_valid_loss_1 = []
        con_valid_loss_2 = []
        valid_cycle_loss = []
        valid_loss = []
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
            loss_cycle = cycle_loss(Z_prime_X - 2 * Z_X + Z_prime_2_X, torch.zeros_like(Z_X))
            loss_batch = loss_1 + loss_2 + alpha_cycle * loss_cycle
            loss_batch.backward()
            optimizer.step()
            con_train_loss_1.append(loss_1.item())
            con_train_loss_2.append(loss_2.item())
            train_loss.append(loss_batch.item())
            train_cycle_loss.append(loss_cycle.item())

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
                loss_cycle = cycle_loss(Z_prime_X - 2 * Z_X + Z_prime_2_X, torch.zeros_like(Z_X))
                loss_batch = loss_1 + loss_2 + alpha_cycle * loss_cycle
                con_valid_loss_1.append(loss_1.item())
                con_valid_loss_2.append(loss_2.item())
                valid_loss.append(loss_batch.item())
                valid_cycle_loss.append(loss_cycle.item())

        torch.save(model, model_save_path)
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch}, Con Train Loss 1: {np.mean(con_train_loss_1):.2f}, Con Train Loss 2: {np.mean(con_train_loss_2):.2f}, Train Cycle Loss: {np.mean(train_cycle_loss):.2f}, Train Loss: {np.mean(train_loss):.2f}, Con Valid Loss 1: {np.mean(con_valid_loss_1):.2f}, Con Valid Loss 2: {np.mean(con_valid_loss_2):.2f}, Valid Cycle Loss: {np.mean(valid_cycle_loss):.2f}, Validation Loss: {np.mean(valid_loss):.2f}, Learning Rate: {lr}')
        scheduler.step(np.mean(valid_loss))