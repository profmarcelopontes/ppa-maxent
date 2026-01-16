import numpy as np

class FFTConvolution2D:
    """
    FFT-based 2D convolution operator with cached kernels.

    Supports:
    - forward:  A x
    - adjoint: A^T x

    Designed for repeated calls inside iterative solvers.
    """

    def __init__(self, kernel: np.ndarray, shape):
        self.shape = shape
        self.kernel = kernel

        self._prepare_kernels()

    def _prepare_kernels(self):
        """Pre-compute FFTs of kernel and its adjoint."""
        pad = np.zeros(self.shape, dtype=np.float32)

        kh, kw = self.kernel.shape
        pad[:kh, :kw] = self.kernel

        pad = np.roll(pad, -kh // 2, axis=0)
        pad = np.roll(pad, -kw // 2, axis=1)

        self.K = np.fft.rfftn(pad)
        self.K_adj = np.conj(self.K)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute A x."""
        X = np.fft.rfftn(x)
        y = np.fft.irfftn(X * self.K, s=self.shape)
        return y

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Compute A^T y."""
        Y = np.fft.rfftn(y)
        x = np.fft.irfftn(Y * self.K_adj, s=self.shape)
        return x
