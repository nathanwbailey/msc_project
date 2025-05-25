import torch
import numpy as np
import math

def train_decoder(model, discriminator, num_epochs, trainloader, testloader, optimizer, discriminator_optimizer, gan_loss, scheduler, device, loss_fn_reconstruct, recon_alpha, model_save_path="simclr.pth"):

    for epoch in range(num_epochs):
        total_valid_loss = []
        recon_train_loss = []
        gen_train_loss = []
        discrim_train_loss = []
        model.decoder.train()
        model.model.eval()
        model.model.encoder.eval()
        for data in trainloader:
            X = data[3].to(device)
            X_masked = data[4].to(device)
            B, T, C, H, W = X_masked.shape
            X_masked = X_masked.reshape(B*T, C, H, W)
            B, T, C, H, W = X.shape
            X = X.reshape(B*T, C, H, W)
            discrim_loss, gen_loss, recon_loss = gan_step(model, discriminator, X_masked, X, optimizer, discriminator_optimizer, gan_loss, loss_fn_reconstruct)
            recon_train_loss.append(recon_loss)
            gen_train_loss.append(gen_loss)
            discrim_train_loss.append(discrim_loss)

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
        print(
            f"Epoch: {epoch} | LR: {lr:.2e}\n"
            f"  Recon Train Loss: {np.mean(recon_train_loss):.2f} Generator: {np.mean(gen_train_loss):.2f}, Discriminator: {np.mean(discrim_train_loss):.2f}\n"
            f"  Recon Validation Loss: {np.mean(total_valid_loss):.2f}, "
        )

        scheduler.step(np.mean(total_valid_loss))

def gan_step(model, discriminator, masked_X, X, generator_optimizer, discriminator_optimizer, gan_loss, loss_fn_reconstruct, recon_alpha=0.5):
    discriminator_optimizer.zero_grad()
    with torch.no_grad():
        _, recon_masked = model(masked_X)
    d_real = discriminator(X)
    d_fake = discriminator(recon_masked)
    real_loss = gan_loss(d_real, torch.ones_like(d_real))
    fake_loss = gan_loss(d_fake, torch.zeros_like(d_fake))
    loss = (real_loss + fake_loss)/2
    loss.backward()
    discriminator_optimizer.step()
    #Generator Step
    generator_optimizer.zero_grad()
    _, recon_masked = model(masked_X)
    d_fake = discriminator(recon_masked)
    recon_loss = loss_fn_reconstruct(recon_masked, X)
    gen_loss = 0.5*gan_loss(d_fake, torch.ones_like(d_fake)) + recon_loss
    gen_loss.backward()
    generator_optimizer.step()
    return loss.item(), gen_loss.item(), recon_loss.item()


def train_encoder_decoder(model, discriminator, num_epochs, trainloader, testloader, optimizer, discriminator_optimizer, scheduler, device, loss_fn_contrastive, loss_fn_reconstruct, cycle_loss, gan_loss, model_save_path="barlow_twins.pth", alpha=0.5):
    alpha_start = 1.0
    alpha_end = 0.0
    k = 0.01
    for epoch in range(num_epochs):
        alpha = alpha_end + (alpha_start - alpha_end) * math.exp(-k * epoch)
        total_train_loss = []
        recon_train_loss = []
        gen_train_loss = []
        discrim_train_loss = []
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

                loss_recon_X = loss_fn_reconstruct(recon_masked, X)

                loss_batch = alpha * loss_contrastive + (1-alpha) * (loss_recon_X)
                total_valid_loss.append(loss_batch.item())
                recon_valid_loss.append(loss_recon_X.item())
                contrastive_valid_loss.append(loss_contrastive.item())

        torch.save(model, model_save_path)
        lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch: {epoch} | Alpha: {alpha:.2f} | LR: {lr:.2e}\n"
            f"  Train Loss     - Total: {np.mean(total_train_loss):.2f}, Contrastive: {np.mean(contrastive_train_loss):.2f}, Recon: {np.mean(recon_train_loss):.2f}\n"
            f"  Validation Loss- Total: {np.mean(total_valid_loss):.2f}, Contrastive: {np.mean(contrastive_valid_loss):.2f}, Recon: {np.mean(recon_valid_loss):.2f}"
        )
        scheduler.step(np.mean(total_valid_loss))