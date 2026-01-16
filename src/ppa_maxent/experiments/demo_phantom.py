import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import resize

from ppa_maxent.core.kernels import gaussian_kernel, effective_kernel_from_psf_icf
from ppa_maxent.operators.fft_conv2d import FFTConvolution2D
from ppa_maxent.solvers.maxent_icf import maxent_icf_solver
from ppa_maxent.utils.metrics import psnr
from ppa_maxent.utils.plotting import show_triplet


def run_demo(
    N: int = 256,
    sigma_psf: float = 2.0,
    beta: float = 1.4,
    psnr_in: float = 45.0,
    seed: int = 0,
    chi_ratio_stop: float = 1.60,
):
    rng = np.random.default_rng(seed)

    # Ground truth
    x_true = resize(shepp_logan_phantom(), (N, N), anti_aliasing=True).astype(np.float32)
    x_true /= np.max(x_true)

    # PSF and ICF
    h = gaussian_kernel(21, sigma_psf, dtype=np.float32)
    c = gaussian_kernel(21, beta * sigma_psf, dtype=np.float32)

    # Effective kernel (full conv)
    k_eff = effective_kernel_from_psf_icf(h, c, dtype=np.float32)

    # Operator uses same-shape FFT convolution; kernel is embedded/padded internally
    Aop = FFTConvolution2D(kernel=k_eff, shape=x_true.shape)

    # Prior m
    m = np.full_like(x_true, float(np.mean(x_true)), dtype=np.float32)

    # Degrade
    x_blur = Aop.forward(x_true)

    # Noise (set sigma2 from target input PSNR)
    sigma2 = float(np.mean(x_blur.astype(np.float64) ** 2) / (10.0 ** (psnr_in / 10.0)))
    d = (x_blur + np.sqrt(sigma2) * rng.standard_normal(x_blur.shape)).astype(np.float32)

    # Restore
    x_hat, lam_final, chi2_final, q = maxent_icf_solver(
        d=d,
        m=m,
        sigma2=sigma2,
        A_forward=Aop.forward,
        A_adjoint=Aop.adjoint,
        chi_ratio_stop=chi_ratio_stop,
        verbose=True
    )

    # Report
    thresh = float(d.size)
    print("\n--- Results ---")
    print(f"Final lam      : {lam_final:.6e}")
    print(f"Final chi2     : {chi2_final:.6e} (thresh={thresh:.6e}) ratio={chi2_final/thresh:.3f}")
    print(f"q (scale)      : {q:.6e}")
    print(f"PSNR degraded  : {psnr(x_true, d):.3f} dB")
    print(f"PSNR restored  : {psnr(x_true, x_hat):.3f} dB")

    show_triplet(x_true, d, x_hat, title="MaxEnt + PPA (Package Structure)")


if __name__ == "__main__":
    run_demo()
