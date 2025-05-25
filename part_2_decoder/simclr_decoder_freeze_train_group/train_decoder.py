import torch
import numpy as np
import math

def train_decoder(model, num_epochs, trainloader, testloader, optimizer, scheduler, device, loss_fn_reconstruct, model_save_path="simclr.pth"):

    for epoch in range(num_epochs):
        total_train_loss = []
        total_valid_loss = []
        model.decoder.train()
        model.model.eval()
        model.model.encoder.eval()
        for data in trainloader:
            optimizer.zero_grad()
            X = data[4].to(device)
            X_masked = data[5].to(device)
            _, recon_masked = model(X_masked)
            loss_batch = loss_fn_reconstruct(recon_masked, X)
            loss_batch.backward()
            optimizer.step()
            total_train_loss.append(loss_batch.item())

        model.eval()
        with torch.no_grad():
            for data in testloader:
                X = data[4].to(device)
                X_masked = data[5].to(device)
                _, recon_masked = model(X_masked)
                loss_batch = loss_fn_reconstruct(recon_masked, X)
                total_valid_loss.append(loss_batch.item())


        torch.save(model, model_save_path)
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch}, Total Train Loss: {np.mean(total_train_loss):.2f}, Total Valid Loss: {np.mean(total_valid_loss):.2f}, Learning Rate: {lr}')
        scheduler.step(np.mean(total_valid_loss))