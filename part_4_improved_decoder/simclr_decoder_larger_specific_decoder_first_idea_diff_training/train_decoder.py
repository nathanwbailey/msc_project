import torch
import numpy as np

def train_decoder(
    model, num_epochs, trainloader, testloader, optimizer, scheduler, device,
    loss_fn_reconstruct, model_save_path="simclr.pth"
):
    for epoch in range(num_epochs):
        train_losses, valid_losses = [], []

        model.decoder.train()
        model.model.eval()
        model.model.encoder.eval()

        # Training loop
        for data in trainloader:
            optimizer.zero_grad()
            X = data[3].to(device)
            X_masked = data[4].to(device)
            B, T, C, H, W = X_masked.shape
            X_masked = X_masked.reshape(B * T, C, H, W)
            B, T, C, H, W = X.shape
            X = X.reshape(B * T, C, H, W)

            _, recon_masked = model(X_masked)
            loss = sum(
                loss_fn_reconstruct(recon_masked[:, c, :, :], X[:, c, :, :])
                for c in range(C)
            )
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                batch_loss = loss_fn_reconstruct(recon_masked, X)
            train_losses.append(batch_loss.item())

        # Validation loop
        model.eval()
        with torch.no_grad():
            for data in testloader:
                X = data[3].to(device)
                X_masked = data[4].to(device)
                B, T, C, H, W = X_masked.shape
                X_masked = X_masked.reshape(B * T, C, H, W)
                B, T, C, H, W = X.shape
                X = X.reshape(B * T, C, H, W)

                _, recon_masked = model(X_masked)
                loss = sum(
                    loss_fn_reconstruct(recon_masked[:, c, :, :], X[:, c, :, :])
                    for c in range(C)
                )
                batch_loss = loss_fn_reconstruct(recon_masked, X)
                valid_losses.append(batch_loss.item())

        torch.save(model, model_save_path)
        lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch: {epoch}, "
            f"Train Loss: {np.mean(train_losses):.2f}, "
            f"Valid Loss: {np.mean(valid_losses):.2f}, "
            f"LR: {lr}"
        )
        scheduler.step(np.mean(valid_losses))


def train_encoder_decoder(
    model, num_epochs, trainloader_outer, testloader_outer, trainloader_inner, testloader_inner,
    optimizer_simclr, optimizer_mae, device, loss_fn_contrastive, loss_fn_reconstruct, cycle_loss,
    model_save_path="barlow_twins.pth", add_l1=False, l1_lambda=1e-6, add_l2=False, l2_lambda=1e-6
):
    for epoch in range(num_epochs):
        recon_train_losses, contrastive_train_losses = [], []
        recon_valid_losses, contrastive_valid_losses = [], []

        model.train()

        # Outer batch (contrastive training)
        for data_outer, data_inner in zip(trainloader_outer, trainloader_inner):
            # Un Freeze the projector
            for param in model.model.projector:
                param.requires_grad = True
            optimizer_simclr.zero_grad()
            X_aug = data_outer[0].to(device)
            X_prime_aug = data_outer[1].to(device)
            X_prime_2 = data_outer[2].to(device)

            B, T, C, H, W = X_aug.shape
            X_aug = X_aug.reshape(B * T, C, H, W)
            X_prime_aug = X_prime_aug.reshape(B * T, C, H, W)
            X_prime_2 = X_prime_2.reshape(B * T, C, H, W)

            z1, _ = model(X_aug)
            z2, _ = model(X_prime_aug)
            z3, _ = model(X_prime_2)

            loss_cycle = cycle_loss(z2 - 2 * z1 + z3, torch.zeros_like(z1))
            loss_contrastive = (
                loss_fn_contrastive(z1, z2)
                + loss_fn_contrastive(z1, z3)
                + loss_cycle
            )
            loss_contrastive.backward()
            optimizer_simclr.step()
            contrastive_train_losses.append(loss_contrastive.item())

            # Freeze the projector
            for param in model.model.projector:
                param.requires_grad = False
            optimizer_mae.zero_grad()
            X_prime_masked = data_inner[1].to(device)
            X_prime_2_masked = data_inner[2].to(device)
            X = data_inner[3].to(device)
            X_masked = data_inner[4].to(device)
            X_prime_recon = data_inner[5].to(device)
            X_prime_2_recon = data_inner[6].to(device)
            B, T, C, H, W = X_masked.shape
            X_masked = X_masked.reshape(B * T, C, H, W)
            X_prime_masked = X_prime_masked.reshape(B * T, C, H, W)
            X_prime_2_masked = X_prime_2_masked.reshape(B * T, C, H, W)
            B, T, C, H, W = X.shape
            X = X.reshape(B * T, C, H, W)
            X_prime_recon = X_prime_recon.reshape(B * T, C, H, W)
            X_prime_2_recon = X_prime_2_recon.reshape(B * T, C, H, W)

            _, X_recon = model(X_masked)
            _, x_prime_recon = model(X_prime_masked)
            _, x_prime_2_recon = model(X_prime_2_masked)

            loss_recon_X = sum(
                loss_fn_reconstruct(X_recon[:, c, :, :], X[:, c, :, :])
                for c in range(C)
            )
            loss_recon_X_prime = sum(
                loss_fn_reconstruct(x_prime_recon[:, c, :, :], X_prime_recon[:, c, :, :])
                for c in range(C)
            )
            loss_recon_X_prime_2 = sum(
                loss_fn_reconstruct(x_prime_2_recon[:, c, :, :], X_prime_2_recon[:, c, :, :])
                for c in range(C)
            )
            loss_recon = loss_recon_X + loss_recon_X_prime + loss_recon_X_prime_2
            if add_l1:
                l1_norm = sum(p.abs().sum() for p in model.decoder.parameters())
                loss_recon += l1_lambda * l1_norm

            if add_l2:
                l1_norm = sum((p ** 2).sum() for p in model.decoder.parameters())
                loss_recon += l2_lambda * l1_norm
            loss_recon.backward()
            optimizer_mae.step()
            batch_loss = loss_fn_reconstruct(X_recon, X)
            recon_train_losses.append(batch_loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            for data_outer, data_inner in zip(testloader_outer, testloader_inner):
                X_aug = data_outer[0].to(device)
                X_prime_aug = data_outer[1].to(device)
                X_prime_2 = data_outer[2].to(device)

                B, T, C, H, W = X_aug.shape
                X_aug = X_aug.reshape(B * T, C, H, W)
                X_prime_aug = X_prime_aug.reshape(B * T, C, H, W)
                X_prime_2 = X_prime_2.reshape(B * T, C, H, W)

                z1, _ = model(X_aug)
                z2, _ = model(X_prime_aug)
                z3, _ = model(X_prime_2)

                loss_cycle = cycle_loss(z2 - 2 * z1 + z3, torch.zeros_like(z1))
                loss_contrastive = (
                    loss_fn_contrastive(z1, z2)
                    + loss_fn_contrastive(z1, z3)
                    + loss_cycle
                )
                contrastive_valid_losses.append(loss_contrastive.item())

                X = data_inner[3].to(device)
                X_masked = data_inner[4].to(device)
                B, T, C, H, W = X_masked.shape
                X_masked = X_masked.reshape(B * T, C, H, W)
                B, T, C, H, W = X.shape
                X = X.reshape(B * T, C, H, W)

                _, recon_masked = model(X_masked)
                batch_loss = loss_fn_reconstruct(recon_masked, X)
                recon_valid_losses.append(batch_loss.item())

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_mae_state_dict': optimizer_mae.state_dict(),
            'optimizer_simclr_state_dict': optimizer_simclr.state_dict(),
        }, model_save_path)
        print(
            f"Epoch: {epoch}, "
            f"Contrastive Train Loss: {np.mean(contrastive_train_losses):.2f}, "
            f"Recon Train Loss: {np.mean(recon_train_losses):.2f}, "
            f"Contrastive Valid Loss: {np.mean(contrastive_valid_losses):.2f}, "
            f"Recon Valid Loss: {np.mean(recon_valid_losses):.2f}"
        )