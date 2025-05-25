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
            loss_batch = torch.tensor(0.0, device=X.device)
            for c in range(C):
                if c == 1 or c == 2:
                    loss_batch += 2 * loss_fn_reconstruct(recon_masked[:, c, :, :], X[:, c, :, :])
                else:
                    loss_batch += loss_fn_reconstruct(recon_masked[:, c, :, :], X[:, c, :, :])
            loss_batch.backward()
            optimizer.step()
            with torch.no_grad():
                loss_batch = loss_fn_reconstruct(recon_masked,X)
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
                loss_batch = torch.tensor(0.0, device=X.device)
                for c in range(C):
                    if c == 1 or c == 2:
                        loss_batch += 2 * loss_fn_reconstruct(recon_masked[:, c, :, :], X[:, c, :, :])
                    else:
                        loss_batch += loss_fn_reconstruct(recon_masked[:, c, :, :], X[:, c, :, :])
                loss_batch = loss_fn_reconstruct(recon_masked,X)
                total_valid_loss.append(loss_batch.item())


        torch.save(model, model_save_path)
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch}, Total Train Loss: {np.mean(total_train_loss):.2f}, Total Valid Loss: {np.mean(total_valid_loss):.2f}, Learning Rate: {lr}')
        scheduler.step(np.mean(total_valid_loss))

def train_encoder_decoder(model, num_epochs, trainloader, testloader, optimizer, scheduler, device, loss_fn_contrastive, loss_fn_reconstruct, cycle_loss, model_save_path="barlow_twins.pth", alpha=0.5):
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
            X_augment = data[0].to(device)
            X_prime_augment = data[1].to(device)
            X_prime_2 = data[2].to(device)
            X = data[3].to(device)
            X_masked = data[4].to(device)

            B, T, C, H, W = X_augment.shape
            X_augment = X_augment.reshape(B*T, C, H, W)
            X_prime_augment = X_prime_augment.reshape(B*T, C, H, W)
            X_prime_2 = X_prime_2.reshape(B*T, C, H, W)
            X_masked = X_masked.reshape(B*T, C, H, W)
            B, T, C, H, W = X.shape
            X = X.reshape(B*T, C, H, W)
            

            z1, _ =  model(X_augment)
            z2, _ =  model(X_prime_augment)
            Z_prime_2, _ = model(X_prime_2)
            _, recon_masked = model(X_masked)
            loss_cycle = cycle_loss(z2 - 2 * z1 + Z_prime_2, torch.zeros_like(z1))
            loss_contrastive = loss_fn_contrastive(z1, z2) + loss_fn_contrastive(z1, Z_prime_2) + loss_cycle
            loss_recon_X = torch.tensor(0.0, device=X.device)
            for c in range(C):
                if c == 1 or c == 2:
                    loss_recon_X += 2 * loss_fn_reconstruct(recon_masked[:, c, :, :], X[:, c, :, :])
                else:
                    loss_recon_X += loss_fn_reconstruct(recon_masked[:, c, :, :], X[:, c, :, :])

            loss_batch = alpha * loss_contrastive + (1-alpha) * (loss_recon_X)
            loss_batch.backward()
            optimizer.step()
            total_train_loss.append(loss_batch.item())
            with torch.no_grad():
                loss_recon_X = loss_fn_reconstruct(recon_masked,X)
            recon_train_loss.append(loss_recon_X.item())
            contrastive_train_loss.append(loss_contrastive.item())

        model.eval()
        with torch.no_grad():
            for data in testloader:
                X_augment = data[0].to(device)
                X_prime_augment = data[1].to(device)
                X_prime_2 = data[2].to(device)
                X = data[3].to(device)
                X_masked = data[4].to(device)

                B, T, C, H, W = X_augment.shape
                X_augment = X_augment.reshape(B*T, C, H, W)
                X_prime_augment = X_prime_augment.reshape(B*T, C, H, W)
                X_prime_2 = X_prime_2.reshape(B*T, C, H, W)
                X_masked = X_masked.reshape(B*T, C, H, W)
                B, T, C, H, W = X.shape
                X = X.reshape(B*T, C, H, W)

                z1, _ =  model(X_augment)
                z2, _ =  model(X_prime_augment)
                Z_prime_2, _ = model(X_prime_2)
                _, recon_masked = model(X_masked)
                loss_cycle = cycle_loss(z2 - 2 * z1 + Z_prime_2, torch.zeros_like(z1))
                loss_contrastive = loss_fn_contrastive(z1, z2) + loss_fn_contrastive(z1, Z_prime_2) + loss_cycle
                loss_recon_X = torch.tensor(0.0, device=X.device)
                for c in range(C):
                    if c == 1 or c == 2:
                        loss_recon_X += 2 * loss_fn_reconstruct(recon_masked[:, c, :, :], X[:, c, :, :])
                    else:
                        loss_recon_X += loss_fn_reconstruct(recon_masked[:, c, :, :], X[:, c, :, :])


                loss_batch = alpha * loss_contrastive + (1-alpha) * (loss_recon_X)
                total_valid_loss.append(loss_batch.item())
                loss_recon_X = loss_fn_reconstruct(recon_masked, X)
                recon_valid_loss.append(loss_recon_X.item())
                contrastive_valid_loss.append(loss_contrastive.item())

        torch.save(model, model_save_path)
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch}, Alpha: {alpha:.2f}, Total Train Loss: {np.mean(total_train_loss):.2f}, Contrastive Train Loss: {np.mean(contrastive_train_loss):.2f}, Recon Train Loss: {np.mean(recon_train_loss):.2f}, Total Valid Loss: {np.mean(total_valid_loss):.2f}, Contrastive Valid Loss: {np.mean(contrastive_valid_loss):.2f}, Recon Valid Loss: {np.mean(recon_valid_loss):.2f}, Learning Rate: {lr}')
        scheduler.step(np.mean(total_valid_loss))