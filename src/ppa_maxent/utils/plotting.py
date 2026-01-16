import matplotlib.pyplot as plt
import numpy as np


def show_triplet(x_true: np.ndarray, d: np.ndarray, x_hat: np.ndarray, title: str = "MaxEnt Restoration") -> None:
    """
    Display (True, Degraded, Restored) side-by-side.
    """
    plt.figure(figsize=(12, 4))
    plt.suptitle(title)

    plt.subplot(1, 3, 1)
    plt.title("True")
    plt.imshow(x_true, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Degraded")
    plt.imshow(d, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Restored")
    plt.imshow(x_hat, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
