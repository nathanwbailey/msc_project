import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
import os


def plot_image(xh, x_full, name):
    num_channels = xh.shape[0]
    _, axes = plt.subplots(2, num_channels, figsize=(4 * num_channels, 4))
    mode_labels = ["2m_temperature", "u_component_of_wind", "v_component_of_wind", "geopotential", "specific_humidity"]
    for i in range(num_channels):
        ax = axes[0, i]
        ax.imshow(x_full[i], cmap='coolwarm')
        ax.set_title(mode_labels[i])
        ax.axis('off')
    for i in range(num_channels):
        ax = axes[1, i]
        ax.imshow(xh[i], cmap='coolwarm')
        ax.set_title(mode_labels[i])
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(name)
    plt.close()

def train_diffusion_model(ddpm, num_epochs, device, optimizer, scheduler, encoder_model, decoder_model, trainloader, validloader, latent_dim, model_save_path='latent_model.pth'):
    for epoch in range(1, num_epochs):
        train_losses = []
        mse_losses = []
        ddpm.train()        
        for batch in trainloader:
            optimizer.zero_grad()
            x = batch[1]
            x = x.to(device)
            condition = batch[2].to(device)
            emb, _ = encoder_model(x)
            condition, _ = encoder_model(condition)
            loss, _, _ = ddpm(emb, condition)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        ddpm.eval()
        with torch.no_grad():
            os.makedirs(f"latent_saved/epoch_{epoch}", exist_ok=True)
            batch = random.choice(list(validloader))
            condition = batch[2].to(device)
            condition_plot = condition[0].cpu().numpy()
            # plot_image(condition_plot, name=f'latent_saved/epoch_{epoch}/conditional_latent_diffusion_input_masked_epoch_{epoch}.png')
            x_full = batch[0].to(device)
            x_full_plot = x_full[0].cpu().numpy()
            condition, _ = encoder_model(condition)
            xh = ddpm.sample(condition.shape[0], latent_dim, condition)
            xh = decoder_model(xh)
            xh_plot = xh[0].cpu().numpy()
            plot_image(xh_plot, x_full_plot, name=f'latent_saved/epoch_{epoch}/conditional_latent_diffusion_output_epoch_{epoch}.png')
            mse_loss = F.mse_loss(xh, x_full)
            mse_losses.append(mse_loss.item())
            print(f"MSE Loss Between Generated and Conditioned: {mse_loss.item():.4f}")
        
        torch.save(ddpm, model_save_path)
        print(f"Epoch: {epoch}, Loss: {np.mean(train_losses):.4f}")
        scheduler.step(np.mean(train_losses))
    return mse_losses, train_losses