import numpy as np


def gaussian_kernel(size: int, sigma: float, dtype=np.float32) -> np.ndarray:
    """
    Normalized 2D Gaussian kernel.
    """
    ax = np.arange(-size // 2 + 1.0, size // 2 + 1.0, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    k = k / np.sum(k)
    return k.astype(dtype)


def effective_kernel_from_psf_icf(h: np.ndarray, c: np.ndarray, dtype=np.float32) -> np.ndarray:
    """
    Build effective kernel for A = H ∘ C.

    If H and C are convolution operators, then:
        A(x) = h * (c * x) = (h * c) * x

    So the effective kernel is the full convolution of c and h.

    Returns the FULL convolution kernel (shape = h+c-1).
    """
    sh = (c.shape[0] + h.shape[0] - 1, c.shape[1] + h.shape[1] - 1)
    C = np.fft.fft2(c.astype(np.float64), s=sh)
    H = np.fft.fft2(h.astype(np.float64), s=sh)
    k_eff = np.real(np.fft.ifft2(C * H))
    return k_eff.astype(dtype)


def kernel_energy(k: np.ndarray) -> float:
    """
    Energy of a kernel: ||k||_2^2
    """
    kk = k.astype(np.float64, copy=False)
    return float(np.sum(kk * kk))
