import numpy as np


def psnr(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR).
    """
    x64 = x.astype(np.float64, copy=False)
    y64 = y.astype(np.float64, copy=False)
    mse = np.mean((x64 - y64) ** 2)
    return 10.0 * np.log10((np.max(x64) ** 2 + eps) / (mse + eps))
