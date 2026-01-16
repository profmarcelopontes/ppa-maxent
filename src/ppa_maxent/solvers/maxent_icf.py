# src/ppa_maxent/solvers/maxent_icf.py
#
# Option A (PPA-style): MINIMIZE
#   F_rho(x) = -S(x|m) + (rho/2) * max(0, chi2(x) - thresh)^2
#
# Key fix for PPA behavior:
# - Update rho based DIRECTLY on feasibility (chi2/thresh), not on inner-loop convergence.
# - Use NORMALIZED violation in the diagonal scaling to avoid exploding curvature.

import numpy as np

from ppa_maxent.core.functionals import (
    chi2_awgn,
    F_penalty_chi2,
    grad_F_penalty_chi2,
)


def maxent_icf_solver(
    d: np.ndarray,
    m: np.ndarray,
    sigma2: float,
    A_forward,
    A_adjoint,
    *,
    # PPA (Option A) knobs
    rho_small: float = 1e-6,
    k: float = 2.0,                 # multiplicative factor for rho growth
    tol: float = 1e-5,
    thresh: float | None = None,    # tau ~= N (number of pixels)
    chi_ratio_stop: float = 1.60,   # stop when chi2/thresh <= this
    # runtime knobs
    max_outer: int = 120,
    max_newton_steps: int = 40,
    max_backtrack: int = 25,
    rho_max: float = 1e2,
    # stability / scaling
    eps: float = 1e-12,
    q: float | None = None,         # scale factor approx of A^T A (energy)
    verbose: bool = True,
):
    """
    MaxEnt + ICF solver using PPA Option A (quadratic penalty on violation), as a MINIMIZATION.

    We solve:
        minimize_{x>0} F_rho(x) = -S(x|m) + (rho/2) * max(0, chi2(x) - thresh)^2

    where:
        S(x|m) : Skilling relative entropy (PAD form)
        chi2(x): AWGN chi-square
        thresh : tau (usually N = number of pixels/measurements)
        rho    : penalty parameter (PPA-style)

    Notes:
      - d, m are expected float32 images
      - internal math uses float64 for stability
      - A_forward and A_adjoint are typically FFT operators for speed
    """
    if thresh is None:
        thresh = float(d.size)

    x_hat = np.maximum(m.astype(np.float64, copy=False), eps)
    rho = float(rho_small)

    # Estimate q if not provided: use energy of impulse response (same-shaped kernel response)
    if q is None:
        impulse = np.zeros_like(d, dtype=np.float32)
        impulse[d.shape[0] // 2, d.shape[1] // 2] = 1.0
        k_eff_same = A_forward(impulse).astype(np.float64)
        q = float(np.sum(k_eff_same * k_eff_same) / sigma2)

    chi2_val = chi2_awgn(x_hat.astype(np.float32), d, A_forward, sigma2)

    # Keep a short history to detect stagnation (optional)
    ratio_hist: list[float] = [chi2_val / thresh]

    for outer in range(max_outer):
        ratio = chi2_val / thresh

        # Relaxed feasibility stop (practical)
        if ratio <= chi_ratio_stop:
            if verbose:
                print(f"[outer {outer:02d}] STOP: chi2/thresh={ratio:.3f} <= {chi_ratio_stop}")
            break

        # -------------------------
        # Inner minimization at fixed rho
        # -------------------------
        converged = False
        for _ in range(max_newton_steps):
            # Gradient of F_rho (Option A)
            g = grad_F_penalty_chi2(
                x_hat.astype(np.float32),
                d,
                m,
                A_forward,
                A_adjoint,
                sigma2,
                rho,
                thresh,
                eps=eps,
            )

            # Current chi2 and NORMALIZED violation for diagonal scaling
            chi2_here = chi2_awgn(x_hat.astype(np.float32), d, A_forward, sigma2)
            v = max(0.0, (chi2_here / thresh) - 1.0)  # normalized violation

            # Stable simplified diagonal model for MINIMIZATION:
            # - entropy contributes ~ (1/x)
            # - penalty curvature scales with rho * v and A^T A energy ~ q
            denom = (1.0 / np.maximum(x_hat, eps)) + 2.0 * rho * v * q
            denom = np.maximum(denom, 1e-18)

            # Descent direction: v_dir = - H^{-1} g
            v_dir = (-g / denom).astype(np.float64)

            p = 1.0
            Fx = F_penalty_chi2(
                x_hat.astype(np.float32),
                d,
                m,
                A_forward,
                sigma2,
                rho,
                thresh,
                eps=eps,
            )

            gv = float(np.sum(g * v_dir))  # should be <= 0 for descent direction

            # Backtracking (Armijo decrease + positivity)
            for _bt in range(max_backtrack):
                x_try = x_hat + p * v_dir

                if np.any(x_try <= 0):
                    p *= 0.5
                    continue

                F_try = F_penalty_chi2(
                    x_try.astype(np.float32),
                    d,
                    m,
                    A_forward,
                    sigma2,
                    rho,
                    thresh,
                    eps=eps,
                )

                # Minimization Armijo:
                if (F_try - Fx) <= 0.5 * p * gv:
                    break

                p *= 0.5

            step = p * v_dir
            x_hat = np.maximum(x_hat + step, eps)

            if np.linalg.norm(step.ravel()) < tol:
                converged = True
                break

        # Update chi2 and feasibility ratio
        chi2_new = chi2_awgn(x_hat.astype(np.float32), d, A_forward, sigma2)
        chi2_val = chi2_new
        ratio = chi2_val / thresh
        ratio_hist.append(ratio)

        # -------------------------
        # PPA update (CRITICAL FIX)
        # Increase rho based on violation (feasibility), not on inner-loop convergence.
        # -------------------------
        if ratio > chi_ratio_stop:
            rho = min(rho * k, rho_max)

        # Optional: if ratio stalls, push rho harder (helps escape flat regions)
        if len(ratio_hist) > 6:
            old = ratio_hist[-6]
            rel_impr = (old - ratio) / max(old, 1e-30)
            if rel_impr < 2e-4 and ratio > chi_ratio_stop:
                rho = min(rho * (k * 2.0), rho_max)

        if verbose:
            tag = " (converged)" if converged else ""
            print(f"[outer {outer:02d}] rho={rho:.3e} chi2={chi2_val:.3e} (chi2/thresh={ratio:.3f}){tag}")

    return x_hat.astype(np.float32), rho, chi2_val, q
