import re
import matplotlib.pyplot as plt

# Path to your log file
log_file = "out_latent_model.log"

# Containers for the values
losses = []
mses = []

# Regular expressions to extract numbers
loss_pattern = re.compile(r"Epoch: \d+, Loss: ([\d.]+)")
mse_pattern = re.compile(r"MSE Loss Between Generated and Conditioned: ([\d.]+)")

# Read the log file and extract data
with open(log_file, "r") as f:
    for line in f:
        loss_match = loss_pattern.search(line)
        mse_match = mse_pattern.search(line)

        if loss_match:
            losses.append(float(loss_match.group(1)))
        if mse_match:
            mses.append(float(mse_match.group(1)))

# Epoch range
epochs = list(range(1, len(losses) + 1))

# Create a figure with two subplots (vertically stacked)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot Loss
ax1.plot(epochs, losses, label='Loss', color='tab:blue', linewidth=2)
ax1.set_ylabel('Loss')
ax1.set_title('Noise Prediction Loss vs. Epoch')

# Plot MSE with log scale
ax2.plot(epochs, mses, label='MSE (log scale)', color='tab:orange', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MSE')
ax2.set_title('Generative MSE vs. Epoch (Log Scale)')
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig('latent_model_losses.png')