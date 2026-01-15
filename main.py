# main.py
"""
Maximum Entropy Image Restoration with ICF (Algorithm 1 aligned, practical fixes)

Keeps paper variable names:
x_hat, m, lam, chi2, thresh, q, v, p, k, tol, Q

Practical improvements to avoid "outer=67 and still running":
- Increase lam not only when ||p v|| < tol, but also when chi2 stagnates
- Stop when chi2 is close enough to thresh
- Safety max outer iterations
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from skimage.data import shepp_logan_phantom
from skimage.transform import resize


# ======================================================
# Utilities
# ======================================================

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    ax = np.arange(-size // 2 + 1.0, size // 2 + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return k / np.sum(k)


def psnr(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    mse = np.mean((x - y) ** 2)
    return 10.0 * np.log10((np.max(x) ** 2 + eps) / (mse + eps))


# ======================================================
# Forward model: A = H ∘ C  and adjoint
# ======================================================

def A(x: np.ndarray, h: np.ndarray, c: np.ndarray) -> np.ndarray:
    return fftconvolve(fftconvolve(x, c, mode="same"), h, mode="same")


def AT(x: np.ndarray, h: np.ndarray, c: np.ndarray) -> np.ndarray:
    return fftconvolve(
        fftconvolve(x, h[::-1, ::-1], mode="same"),
        c[::-1, ::-1], mode="same"
    )


# ======================================================
# MaxEnt terms: S, chi2, Q, dQdx
# ======================================================

def S(x_hat: np.ndarray, m: np.ndarray, eps: float = 1e-12) -> float:
    """
    Skilling relative entropy (PAD):
    S(x|m) = - Σ [ x log(x/m) - x + m ]
    """
    x = np.maximum(x_hat, eps)
    return -float(np.sum(x * np.log(x / m) - x + m))


def chi2(x_hat: np.ndarray, d: np.ndarray, h: np.ndarray, c: np.ndarray, sigma2: float) -> float:
    r = A(x_hat, h, c) - d
    return float(np.sum((r * r) / sigma2))


def Q(x_hat: np.ndarray, d: np.ndarray, h: np.ndarray, c: np.ndarray,
      m: np.ndarray, sigma2: float, lam: float) -> float:
    return S(x_hat, m) - lam * chi2(x_hat, d, h, c, sigma2)


def dQdx(x_hat: np.ndarray, d: np.ndarray, h: np.ndarray, c: np.ndarray,
         m: np.ndarray, sigma2: float, lam: float, eps: float = 1e-12) -> np.ndarray:
    """
    dS/dx = -log(x/m)
    d(chi2)/dx = 2 A^T(Ax - d)/sigma2
    """
    x = np.maximum(x_hat, eps)
    dS = -np.log(x / m)

    r = A(x_hat, h, c) - d
    dchi = 2.0 * AT(r / sigma2, h, c)

    return dS - lam * dchi


# ======================================================
# Algorithm 1 (paper names) + practical fixes
# ======================================================
def maxent_algorithm1(
    d: np.ndarray,
    h: np.ndarray,
    c: np.ndarray,
    m: np.ndarray,
    sigma2: float,
    *,
    k: float = 2.0,
    tol: float = 1e-5,
    thresh: float | None = None,
    lam_small: float = 1e-6,
    max_newton_steps: int = 80,
    max_backtrack: int = 50,
    max_outer: int = 120,
    chi_close_frac: float = 0.02,   # para quando chi2 estiver a 2% do alvo
    eps: float = 1e-12,
    verbose: bool = True,
) -> tuple[np.ndarray, float, float, float]:
    """
    Algorithm 1 (paper-aligned names) with the critical sign fix in the simplified Newton step.

    Key fix:
      v = (x_hat^{-1} + 2 * lam * q)^(-1) * dQ/dx
    not:
      (x_hat^{-1} - 2 * lam * q)^(-1)
    """
    if thresh is None:
        thresh = float(d.size)  # "usually N"

    # q: energy of effective kernel / sigma2
    k_eff = fftconvolve(c, h, mode="full")
    q = float(np.sum(k_eff * k_eff) / sigma2)

    x_hat = np.maximum(m.copy(), eps)
    lam = float(lam_small)

    chi2_val = chi2(x_hat, d, h, c, sigma2)

    for outer in range(max_outer):
        # stop if close enough (practical/statistical)
        if chi2_val <= thresh * (1.0 + chi_close_frac):
            if verbose:
                print(f"[outer {outer:02d}] STOP: lam={lam:.3e} chi2={chi2_val:.3e} thresh={thresh:.3e}")
            break

        # inner loop at fixed lam
        converged = False
        for _ in range(max_newton_steps):
            g = dQdx(x_hat, d, h, c, m, sigma2, lam, eps=eps)

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # CRITICAL FIX: PLUS sign
            denom = (1.0 / np.maximum(x_hat, eps)) + 2.0 * lam * q
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            denom = np.where(denom > eps, denom, eps)

            v = g / denom

            p = 1.0
            Qx = Q(x_hat, d, h, c, m, sigma2, lam)
            gv = float(np.sum(g * v))  # v^T g

            for _bt in range(max_backtrack):
                x_try = x_hat + p * v

                if np.any(x_try <= 0):
                    p *= 0.5
                    continue

                Q_try = Q(x_try, d, h, c, m, sigma2, lam)

                # Armijo-style sufficient increase
                if (Q_try - Qx) >= 0.5 * p * gv:
                    break

                p *= 0.5

            step = p * v
            x_hat = np.maximum(x_hat + step, eps)

            if np.linalg.norm(step.ravel()) < tol:
                converged = True
                break

        # after inner loop, evaluate chi2
        chi2_new = chi2(x_hat, d, h, c, sigma2)

        # λ update rule (paper-ish):
        # only increase λ when the inner loop "settled"
        if converged:
            lam = k * lam

        chi2_val = chi2_new

        if verbose:
            print(f"[outer {outer:02d}] lam={lam:.3e} chi2={chi2_val:.3e} (chi2/thresh={chi2_val/thresh:.3f})")

    return x_hat, lam, chi2_val, q



# ======================================================
# Demo data
# ======================================================

def build_demo_data(N: int, sigma_psf: float, beta: float, psnr_in: float, seed: int):
    rng = np.random.default_rng(seed)

    x_true = resize(shepp_logan_phantom(), (N, N), anti_aliasing=True)
    x_true = x_true / np.max(x_true)

    h = gaussian_kernel(21, sigma_psf)
    c = gaussian_kernel(21, beta * sigma_psf)

    m = np.full_like(x_true, np.mean(x_true))

    x_blur = A(x_true, h, c)

    sigma2 = float(np.mean(x_blur**2) / (10.0 ** (psnr_in / 10.0)))
    d = x_blur + np.sqrt(sigma2) * rng.standard_normal(x_blur.shape)

    return x_true, d, h, c, m, sigma2


def main():
    parser = argparse.ArgumentParser(description="MaxEnt restoration (Algorithm 1 aligned, practical)")
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--sigma_psf", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=1.4)
    parser.add_argument("--psnr_in", type=float, default=45.0)
    parser.add_argument("--seed", type=int, default=0)

    # paper-ish knobs
    parser.add_argument("--lam_small", type=float, default=1e-6)
    parser.add_argument("--k", type=float, default=2.0)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--thresh", type=float, default=None)

    # performance knobs
    parser.add_argument("--max_newton_steps", type=int, default=60)
    parser.add_argument("--max_outer", type=int, default=150)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    x_true, d, h, c, m, sigma2 = build_demo_data(
        N=args.N,
        sigma_psf=args.sigma_psf,
        beta=args.beta,
        psnr_in=args.psnr_in,
        seed=args.seed
    )

    thresh = float(args.thresh) if args.thresh is not None else float(d.size)

    x_hat, lam_final, chi2_final, q = maxent_algorithm1(
        d, h, c, m, sigma2,
        k=args.k,
        tol=args.tol,
        thresh=thresh,
        lam_small=args.lam_small,
        max_newton_steps=args.max_newton_steps,
        max_outer=args.max_outer,
        verbose=not args.quiet
    )

    print("\n--- Results ---")
    print(f"Final lam   : {lam_final:.6e}")
    print(f"Final chi2  : {chi2_final:.6e} (thresh={thresh:.6e})")
    print(f"q (scale)   : {q:.6e}")
    print(f"PSNR degraded: {psnr(x_true, d):.3f} dB")
    print(f"PSNR restored: {psnr(x_true, x_hat):.3f} dB")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.title("True"); plt.imshow(x_true, cmap="gray"); plt.axis("off")
    plt.subplot(1, 3, 2); plt.title("Degraded"); plt.imshow(d, cmap="gray"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.title("MaxEnt restored"); plt.imshow(x_hat, cmap="gray"); plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
