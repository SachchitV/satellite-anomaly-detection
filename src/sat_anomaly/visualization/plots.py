"""Plotting helpers for training curves and reconstruction visualization."""

import matplotlib.pyplot as plt


def plot_training_history(train_losses, val_losses, save_path=None):
    """Plot training and validation loss curves on a log scale."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Training history plot saved to: {save_path}")

    plt.show()
