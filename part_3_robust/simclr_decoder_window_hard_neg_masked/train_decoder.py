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
            X = data[3].to(device)
            X_masked = data[4].to(device)
            B, T, C, H, W = X_masked.shape
            X_masked = X_masked.reshape(B*T, C, H, W)
            B, T, C, H, W = X.shape
            X = X.reshape(B*T, C, H, W)
            _, recon_masked = model(X_masked)
            loss_batch = loss_fn_reconstruct(recon_masked, X)
            loss_batch.backward()
            optimizer.step()
            total_train_loss.append(loss_batch.item())

        model.eval()
        with torch.no_grad():
            for data in testloader:
                X = data[3].to(device)
                X_masked = data[4].to(device)
                B, T, C, H, W = X_masked.shape
                X_masked = X_masked.reshape(B*T, C, H, W)
                B, T, C, H, W = X.shape
                X = X.reshape(B*T, C, H, W)
                _, recon_masked = model(X_masked)
                loss_batch = loss_fn_reconstruct(recon_masked, X)
                total_valid_loss.append(loss_batch.item())


        torch.save(model, model_save_path)
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch}, Total Train Loss: {np.mean(total_train_loss):.2f}, Total Valid Loss: {np.mean(total_valid_loss):.2f}, Learning Rate: {lr}')
        scheduler.step(np.mean(total_valid_loss))


def train_encoder_decoder(model, num_epochs, trainloader, testloader, optimizer, scheduler, device, loss_fn_contrastive, loss_fn_reconstruct, model_save_path="barlow_twins.pth", alpha=0.5):
    alpha_start = 1.0
    alpha_end = 0.0
    k = 0.01
    for epoch in range(num_epochs):
        alpha = alpha_end + (alpha_start - alpha_end) * math.exp(-k * epoch)
        total_train_loss = []
        recon_train_loss = []
        contrastive_train_loss = []
        total_valid_loss = []
        recon_valid_loss = []
        contrastive_valid_loss = []
        model.train()
        for data in trainloader:
            optimizer.zero_grad()
            X_sample = data[0].to(device)
            X_prime = data[1].to(device)
            X_prime_mask = data[2].to(device)
            X = data[3].to(device)
            X_masked = data[4].to(device)

            B, T, C, H, W = X_sample.shape
            X_sample = X_sample.reshape(B*T, C, H, W)
            X_prime = X_prime.reshape(B*T, C, H, W)
            X_prime_mask = X_prime_mask.reshape(B*T, C, H, W)
            X_masked = X_masked.reshape(B*T, C, H, W)
            B, T, C, H, W = X.shape
            X = X.reshape(B*T, C, H, W)


            Z, _ = model(X_sample)
            Z_prime, _ = model(X_prime)
            Z_prime_mask, _ = model(X_prime_mask)
            batch_size = X.shape[0]
            embeddings = torch.cat((Z, Z_prime, Z_prime_mask), dim=0)
            labels = torch.arange(batch_size)
            labels = torch.cat((labels, labels, labels), dim=0)
            loss_contrastive = loss_fn_contrastive(embeddings, labels)
            _, recon_masked = model(X_masked)
            loss_recon_X = loss_fn_reconstruct(recon_masked, X)
            loss_batch = alpha * loss_contrastive + (1-alpha) * (loss_recon_X)
            loss_batch.backward()
            optimizer.step()
            total_train_loss.append(loss_batch.item())
            recon_train_loss.append(loss_recon_X.item())
            contrastive_train_loss.append(loss_contrastive.item())

        model.eval()
        with torch.no_grad():
            for data in testloader:
                X_sample = data[0].to(device)
                X_prime = data[1].to(device)
                X_prime_mask = data[2].to(device)
                X = data[3].to(device)
                X_masked = data[4].to(device)

                B, T, C, H, W = X_sample.shape
                X_sample = X_sample.reshape(B*T, C, H, W)
                X_prime = X_prime.reshape(B*T, C, H, W)
                X_prime_mask = X_prime_mask.reshape(B*T, C, H, W)
                X_masked = X_masked.reshape(B*T, C, H, W)
                B, T, C, H, W = X.shape
                X = X.reshape(B*T, C, H, W)


                Z, _ = model(X_sample)
                Z_prime, _ = model(X_prime)
                Z_prime_mask, _ = model(X_prime_mask)
                batch_size = X.shape[0]
                embeddings = torch.cat((Z, Z_prime, Z_prime_mask), dim=0)
                labels = torch.arange(batch_size)
                labels = torch.cat((labels, labels, labels), dim=0)
                loss_contrastive = loss_fn_contrastive(embeddings, labels)
                _, recon_masked = model(X_masked)
                loss_recon_X = loss_fn_reconstruct(recon_masked, X)
                loss_batch = alpha * loss_contrastive + (1-alpha) * (loss_recon_X)
                total_valid_loss.append(loss_batch.item())
                recon_valid_loss.append(loss_recon_X.item())
                contrastive_valid_loss.append(loss_contrastive.item())

        torch.save(model, model_save_path)
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch}, Alpha: {alpha:.2f}, Total Train Loss: {np.mean(total_train_loss):.2f}, Contrastive Train Loss: {np.mean(contrastive_train_loss):.2f}, Recon Train Loss: {np.mean(recon_train_loss):.2f}, Total Valid Loss: {np.mean(total_valid_loss):.2f}, Contrastive Valid Loss: {np.mean(contrastive_valid_loss):.2f}, Recon Valid Loss: {np.mean(recon_valid_loss):.2f}, Learning Rate: {lr}')
        scheduler.step(np.mean(total_valid_loss))