import re

import matplotlib.pyplot as plt
import numpy as np


def extract_recon_log_valid_loss(log_path):
    losses = []
    epochs = []
    alphas = []

    recon_log_pattern = re.compile(r"Recon \(log\):\s*([0-9]+\.[0-9]+)")
    alpha_pattern = re.compile(r"Alpha:\s*([0-9]+\.[0-9]+)")
    valid_block_start = re.compile(r"^\s*Valid Loss:")

    epoch = 0
    last_alpha = None

    with open(log_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # Capture alpha if found
        match_alpha = alpha_pattern.search(line)
        if match_alpha:
            last_alpha = float(match_alpha.group(1))

        # Detect start of Valid block
        if valid_block_start.search(line):
            # Scan ahead for Recon (log) in this block
            for j in range(
                i + 1, min(i + 10, len(lines))
            ):  # Look ahead a few lines
                match_recon_log = recon_log_pattern.search(lines[j])
                if match_recon_log:

                    loss = float(match_recon_log.group(1))
                    epochs.append(epoch)
                    losses.append(loss)
                    alphas.append(
                        last_alpha if last_alpha is not None else 0.0
                    )
                    epoch += 1
                    break  # Only take the first found in this block
    print(epochs)
    return epochs, losses, alphas


def plot_loss(epochs, losses, alphas):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, color="blue", label="Recon (log) Valid Loss")
    if any(a != 0.0 for a in alphas):
        plt.plot(epochs, alphas, color="red", label="Alpha Value")
    y_min, y_max = plt.ylim()
    plt.yticks(np.linspace(y_min, y_max, 20))
    plt.axhline(
        y=0.15, color="purple", linestyle="--", label="Threshold = 0.15"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Recon (log) Valid Loss Over Epochs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("valid_recon_log_loss.png")


if __name__ == "__main__":
    log_file = "out.log"
    epochs, losses, alphas = extract_recon_log_valid_loss(log_file)
    if losses:
        plot_loss(epochs, losses, alphas)
    else:
        print("No 'Recon (log)' entries found in the log file.")
