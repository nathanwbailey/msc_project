import torch
import numpy as np
import math

def train_decoder(model, num_epochs, trainloader, testloader, optimizer, scheduler, device, loss_fn_reconstruct, loss_fn_reconstruct_2, model_save_path="simclr.pth"):

    for epoch in range(num_epochs):
        total_train_loss = []
        total_valid_loss = []
        recon_train_loss = []
        recon_train_loss_u = []
        recon_train_loss_v = []
        recon_valid_loss = []
        recon_valid_loss_u = []
        recon_valid_loss_v = []
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
            loss_recon = loss_fn_reconstruct(recon_masked, X)
            loss_recon_u = loss_fn_reconstruct_2(recon_masked[:, 1, :, :], X[:, 1, :, :])
            loss_recon_v = loss_fn_reconstruct_2(recon_masked[:, 2, :, :], X[:, 2, :, :])

            loss_batch = loss_recon + loss_recon_u + loss_recon_v
            loss_batch.backward()
            optimizer.step()
            recon_train_loss.append(loss_recon.item())
            recon_train_loss_u.append(loss_recon_u.item())
            recon_train_loss_v.append(loss_recon_v.item())
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
                loss_recon = loss_fn_reconstruct(recon_masked, X)
                loss_recon_u = loss_fn_reconstruct_2(recon_masked[:, 1, :, :], X[:, 1, :, :])
                loss_recon_v = loss_fn_reconstruct_2(recon_masked[:, 2, :, :], X[:, 2, :, :])

                loss_batch = loss_recon + loss_recon_u + loss_recon_v
                recon_valid_loss.append(loss_recon.item())
                recon_valid_loss_u.append(loss_recon_u.item())
                recon_valid_loss_v.append(loss_recon_v.item())
                total_valid_loss.append(loss_batch.item())


        torch.save(model, model_save_path)
        lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch: {epoch}, Learning Rate: {lr}\n"
            f"Train Losses -> Total: {np.mean(total_train_loss):.2f}, "
            f"Reconstruction: {np.mean(recon_train_loss):.2f}, "
            f"Reconstruction U Component: {np.mean(recon_train_loss_u):.2f}\n"
            f"Reconstruction V Component: {np.mean(recon_train_loss_v):.2f}\n"
            f"Valid Losses -> Total: {np.mean(total_valid_loss):.2f}, "
            f"Reconstruction: {np.mean(recon_valid_loss):.2f}, "
            f"Reconstruction U Component: {np.mean(recon_valid_loss_u):.2f}\n"
            f"Reconstruction V Component: {np.mean(recon_valid_loss_v):.2f}\n"
        )
        scheduler.step(np.mean(total_valid_loss))

def train_encoder_decoder(model, num_epochs, trainloader, testloader, optimizer, scheduler, device, loss_fn_contrastive, loss_fn_reconstruct, loss_fn_reconstruct_2, cycle_loss, model_save_path="barlow_twins.pth", alpha=0.5):
    alpha_start = 1.0
    alpha_end = 0.0
    k = 0.01
    for epoch in range(num_epochs):
        alpha = alpha_end + (alpha_start - alpha_end) * math.exp(-k * epoch)
        total_train_loss = []
        recon_train_loss = []
        recon_train_loss_u = []
        recon_train_loss_v = []
        contrastive_train_loss = []
        total_valid_loss = []
        recon_valid_loss = []
        recon_valid_loss_u = []
        recon_valid_loss_v = []
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

            loss_recon = loss_fn_reconstruct(recon_masked, X)
            loss_recon_u = loss_fn_reconstruct_2(recon_masked[:, 1, :, :], X[:, 1, :, :])
            loss_recon_v = loss_fn_reconstruct_2(recon_masked[:, 2, :, :], X[:, 2, :, :])

            loss_recon_X = loss_recon + loss_recon_u + loss_recon_v

            loss_batch = alpha * loss_contrastive + (1-alpha) * (loss_recon_X)
            loss_batch.backward()
            optimizer.step()
            total_train_loss.append(loss_batch.item())
            recon_train_loss.append(loss_recon.item())
            recon_train_loss_u.append(loss_recon_u.item())
            recon_train_loss_v.append(loss_recon_v.item())
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

                loss_recon = loss_fn_reconstruct(recon_masked, X)
                loss_recon_u = loss_fn_reconstruct_2(recon_masked[:, 1, :, :], X[:, 1, :, :])
                loss_recon_v = loss_fn_reconstruct_2(recon_masked[:, 2, :, :], X[:, 2, :, :])

                loss_recon_X = loss_recon + loss_recon_u + loss_recon_v

                loss_batch = alpha * loss_contrastive + (1-alpha) * (loss_recon_X)
                total_valid_loss.append(loss_batch.item())
                recon_valid_loss.append(loss_recon.item())
                recon_valid_loss_u.append(loss_recon_u.item())
                recon_valid_loss_v.append(loss_recon_v.item())
                contrastive_valid_loss.append(loss_contrastive.item())

        torch.save(model, model_save_path)
        lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch: {epoch}, Alpha: {alpha:.2f}, Learning Rate: {lr}\n"
            f"Train Losses -> Total: {np.mean(total_train_loss):.2f}, "
            f"Contrastive: {np.mean(contrastive_train_loss):.2f}, "
            f"Reconstruction: {np.mean(recon_train_loss):.2f}, "
            f"Reconstruction U Component: {np.mean(recon_train_loss_u):.2f}\n"
            f"Reconstruction V Component: {np.mean(recon_train_loss_v):.2f}\n"
            f"Valid Losses -> Total: {np.mean(total_valid_loss):.2f}, "
            f"Contrastive: {np.mean(contrastive_valid_loss):.2f}, "
            f"Reconstruction: {np.mean(recon_valid_loss):.2f}, "
            f"Reconstruction U Component: {np.mean(recon_valid_loss_u):.2f}\n"
            f"Reconstruction V Component: {np.mean(recon_valid_loss_v):.2f}\n"
        )
        scheduler.step(np.mean(total_valid_loss))