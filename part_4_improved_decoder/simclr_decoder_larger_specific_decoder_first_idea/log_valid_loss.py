import re

import matplotlib.pyplot as plt
import numpy as np


def extract_recon_valid_loss(log_path):
    losses = []
    epochs = []
    alphas = []

    pattern = re.compile(r"^\s*Recon Valid Loss:\s*([0-9]+\.[0-9]+)")
    pattern_alpha = re.compile(r"Alpha:\s*([0-9]+\.[0-9]+)")
    epoch = 0
    with open(log_path, "r") as f:
        for line in f:
            match_alpha = pattern_alpha.search(line)
            if match_alpha:
                alpha = float(match_alpha.group(1))
                alphas.append(alpha)
            match = pattern.search(line)
            if match:
                loss = float(match.group(1))
                epochs.append(epoch)
                losses.append(loss)
                epoch += 1

    return epochs, losses, alphas


def plot_loss(epochs, losses, alphas):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, color="blue", label="Recon Valid Loss")
    plt.plot(epochs, alphas, color="red", label="Alpha Value")
    y_min, y_max = plt.ylim()
    plt.yticks(np.linspace(y_min, y_max, 20))
    plt.axhline(
        y=0.15, color="purple", linestyle="--", label="Threshold = 0.15"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Recon Valid Loss Over Epochs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("valid_recon_loss.png")


if __name__ == "__main__":
    log_file = "out.log"
    epochs, losses, alphas = extract_recon_valid_loss(log_file)
    if losses:
        plot_loss(epochs, losses, alphas)
    else:
        print("No Recon Valid Loss entries found in the log file.")
