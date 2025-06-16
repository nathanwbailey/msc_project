import re

import matplotlib.pyplot as plt


def parse_log_file(filepath):
    train_losses = []
    valid_losses = []
    with open(filepath, "r") as file:
        for line in file:
            match = re.search(
                r"Epoch: \d+, Train Loss: ([\d.]+), Valid Loss: ([\d.]+)",
                line,
            )
            if match:
                train_losses.append(float(match.group(1)))
                valid_losses.append(float(match.group(2)))
    return train_losses, valid_losses


log_file_2 = "latent_classification.log"
log_file_1 = "../autoencoder_hard_neg/latent_classification.log"

train1, valid1 = parse_log_file(log_file_1)
train2, valid2 = parse_log_file(log_file_2)

epochs1 = list(range(1, len(train1) + 1))
epochs2 = list(range(1, len(train2) + 1))

plt.figure(figsize=(10, 6))
plt.plot(epochs1, train1, label="Train Loss - Autoencoder")
plt.plot(epochs1, valid1, label="Valid Loss - Autoencoder")
plt.plot(epochs2, train2, label="Train Loss - SIMCLR", linestyle="--")
plt.plot(epochs2, valid2, label="Valid Loss - SIMCLR", linestyle="--")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss - AE vs SIMCLR")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("classification_losses.png")
