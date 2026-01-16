# src/ppa_maxent/core/functionals.py

import numpy as np
from typing import Callable


def entropy_skilling_pad(x_hat: np.ndarray, m: np.ndarray, eps: float = 1e-12) -> float:
    """
    Skilling relative entropy for PAD (positive additive distribution):

        S(x|m) = - Σ [ x log(x/m) - x + m ]

    This form enforces positivity and yields:
        dS/dx = -log(x/m)
    """
    x = np.maximum(x_hat, eps)
    return -float(np.sum(x * np.log(x / m) - x + m))


def chi2_awgn(x_hat: np.ndarray, d: np.ndarray, A_forward: Callable[[np.ndarray], np.ndarray], sigma2: float) -> float:
    """
    chi2 under AWGN:
        chi2(x) = Σ (Ax - d)^2 / sigma2
    """
    r = A_forward(x_hat) - d
    return float(np.sum((r.astype(np.float64) ** 2) / sigma2))


def Q_maxent(x_hat: np.ndarray, d: np.ndarray, m: np.ndarray,
            A_forward: Callable[[np.ndarray], np.ndarray], sigma2: float, lam: float) -> float:
    """
    MaxEnt functional to maximize:
        Q(x) = S(x|m) - lam * chi2(x)

    NOTE:
      This is the "classic" penalized MaxEnt used previously.
      Kept for backward-compatibility and comparison.
    """
    return entropy_skilling_pad(x_hat, m) - lam * chi2_awgn(x_hat, d, A_forward, sigma2)


def grad_Q_maxent(
    x_hat: np.ndarray,
    d: np.ndarray,
    m: np.ndarray,
    A_forward: Callable[[np.ndarray], np.ndarray],
    A_adjoint: Callable[[np.ndarray], np.ndarray],
    sigma2: float,
    lam: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Gradient of Q:

      dQ/dx = dS/dx - lam * d(chi2)/dx

    with:
      dS/dx = -log(x/m)
      d(chi2)/dx = 2 A^T(Ax - d) / sigma2
    """
    x = np.maximum(x_hat, eps).astype(np.float64, copy=False)
    mm = m.astype(np.float64, copy=False)

    # dS/dx
    dS = -np.log(x / mm)

    # residual
    r = A_forward(x_hat).astype(np.float64) - d.astype(np.float64)

    # d(chi2)/dx
    dchi = 2.0 * A_adjoint((r / sigma2).astype(np.float32)).astype(np.float64)

    return (dS - lam * dchi).astype(np.float64)


# ======================================================
# Option A: PPA-style quadratic penalty on chi2 violation
#   MINIMIZE: F_rho(x) = -S(x|m) + (rho/2) * max(0, chi2(x) - thresh)^2
# ======================================================

def F_penalty_chi2(
    x_hat: np.ndarray,
    d: np.ndarray,
    m: np.ndarray,
    A_forward: Callable[[np.ndarray], np.ndarray],
    sigma2: float,
    rho: float,
    thresh: float,
    eps: float = 1e-12,
) -> float:
    """
    Option A (PPA-style) objective to MINIMIZE:

        F_rho(x) = -S(x|m) + (rho/2) * max(0, chi2(x) - thresh)^2

    where:
      - S(x|m) is Skilling relative entropy (PAD)
      - chi2 is AWGN chi-square
      - thresh ~ N (number of measurements/pixels)
      - rho is penalty parameter
    """
    Sval = entropy_skilling_pad(x_hat, m, eps=eps)
    chi = chi2_awgn(x_hat, d, A_forward, sigma2)

    v = max(0.0, chi - thresh)  # hinge violation
    return (-Sval) + 0.5 * rho * (v * v)


def grad_F_penalty_chi2(
    x_hat: np.ndarray,
    d: np.ndarray,
    m: np.ndarray,
    A_forward: Callable[[np.ndarray], np.ndarray],
    A_adjoint: Callable[[np.ndarray], np.ndarray],
    sigma2: float,
    rho: float,
    thresh: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Gradient of Option A objective:

      F_rho(x) = -S(x|m) + (rho/2) * [max(0, chi2(x) - thresh)]^2

    Let:
      chi = chi2(x),
      v = max(0, chi - thresh).

    Then:
      ∇F = -∇S + rho * v * ∇chi2

    with:
      ∇S   = -log(x/m)   =>  -∇S = +log(x/m)
      ∇chi2 = 2 A^T(Ax-d)/sigma2
    """
    # -∇S = +log(x/m)
    x = np.maximum(x_hat, eps).astype(np.float64, copy=False)
    mm = m.astype(np.float64, copy=False)
    term_entropy = np.log(x / mm)

    # compute chi2 and hinge violation v
    r = A_forward(x_hat).astype(np.float64) - d.astype(np.float64)
    chi = float(np.sum((r * r) / sigma2))
    v = max(0.0, chi - thresh)

    # If feasible, penalty inactive
    if v == 0.0:
        return term_entropy.astype(np.float64)

    # ∇chi2
    dchi = 2.0 * A_adjoint((r / sigma2).astype(np.float32)).astype(np.float64)

    return (term_entropy + (rho * v) * dchi).astype(np.float64)
