#!/usr/bin/env python3
import re
import matplotlib.pyplot as plt


LOG_FILE = "out.log"
OUT_PNG  = "recon_losses.png"


def parse_log(log_path):
    """
    Reads the log file and extracts:
      - epoch number
      - recon_train_loss
      - recon_valid_loss

    Returns three lists of equal length: epochs, train_losses, valid_losses.
    """
    # Regex to capture:
    #   Epoch: <int>, … Recon Train Loss: <float>, … Recon Valid Loss: <float>
    pattern = re.compile(
        r"Epoch:\s*(\d+).*?Recon Train Loss:\s*([0-9]*\.?[0-9]+).*?Recon Valid Loss:\s*([0-9]*\.?[0-9]+)"
    )

    epochs = []
    recon_train = []
    recon_valid = []

    with open(log_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                ep = int(match.group(1))
                rt = float(match.group(2))
                rv = float(match.group(3))
                epochs.append(ep)
                recon_train.append(rt)
                recon_valid.append(rv)

    return epochs, recon_train, recon_valid


def plot_losses(epochs, recon_train, recon_valid, out_path):
    """
    Plots Recon Train Loss and Recon Valid Loss over epochs.
    Saves the figure to out_path.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, recon_train, label="Train Loss")
    plt.plot(epochs, recon_valid, label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Reconstruction Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved plot to: {out_path}")


def main():
    epochs, recon_train, recon_valid = parse_log(LOG_FILE)

    if not epochs:
        print("No lines matching 'Recon Train Loss' / 'Recon Valid Loss' were found in the log.")
        return

    # In case the log isn’t strictly sorted by epoch:
    sorted_data = sorted(zip(epochs, recon_train, recon_valid), key=lambda x: x[0])
    epochs_sorted, recon_train_sorted, recon_valid_sorted = zip(*sorted_data)

    plot_losses(epochs_sorted, recon_train_sorted, recon_valid_sorted, OUT_PNG)


if __name__ == "__main__":
    main()
