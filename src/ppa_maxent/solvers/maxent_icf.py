import numpy as np

from ppa_maxent.core.functionals import Q_maxent, chi2_awgn, grad_Q_maxent


def maxent_icf_solver(
    d: np.ndarray,
    m: np.ndarray,
    sigma2: float,
    A_forward,
    A_adjoint,
    *,
    # paper / penalty knobs
    lam_small: float = 1e-6,
    k: float = 2.0,
    tol: float = 1e-5,
    thresh: float | None = None,
    chi_ratio_stop: float = 1.60,
    # runtime knobs
    max_outer: int = 120,
    max_newton_steps: int = 40,
    max_backtrack: int = 25,
    lam_max: float = 1e2,
    # stability
    eps: float = 1e-12,
    q: float | None = None,
    verbose: bool = True,
):
    """
    MaxEnt solver aligned with Algorithm 1 (paper names) + PPA-guided penalty control.

    Variables (paper naming):
      x_hat, lam, chi2, thresh, q, v, p, k, tol

    Inputs:
      - A_forward(x): returns Ax
      - A_adjoint(y): returns A^T y

    Notes:
      - d, m are expected float32 images
      - internal math uses float64 for stability
    """
    if thresh is None:
        thresh = float(d.size)

    x_hat = np.maximum(m.astype(np.float64, copy=False), eps)
    lam = float(lam_small)

    # compute q if not provided (approx scale factor)
    # default: estimate from a delta impulse response
    if q is None:
        impulse = np.zeros_like(d, dtype=np.float32)
        impulse[d.shape[0] // 2, d.shape[1] // 2] = 1.0
        k_eff_same = A_forward(impulse).astype(np.float64)
        q = float(np.sum(k_eff_same * k_eff_same) / sigma2)

    chi2_val = chi2_awgn(x_hat.astype(np.float32), d, A_forward, sigma2)
    chi_hist = [chi2_val]

    for outer in range(max_outer):
        ratio = chi2_val / thresh

        if ratio <= chi_ratio_stop:
            if verbose:
                print(f"[outer {outer:02d}] STOP: chi2/thresh={ratio:.3f} <= {chi_ratio_stop}")
            break

        converged = False

        for _ in range(max_newton_steps):
            g = grad_Q_maxent(
                x_hat.astype(np.float32),
                d,
                m,
                A_forward,
                A_adjoint,
                sigma2,
                lam,
                eps=eps
            )

            # Stable simplified Newton diagonal (CRITICAL: PLUS)
            denom = (1.0 / np.maximum(x_hat, eps)) + 2.0 * lam * q
            denom = np.maximum(denom, 1e-18)

            v = (g / denom).astype(np.float64)

            p = 1.0
            Qx = Q_maxent(x_hat.astype(np.float32), d, m, A_forward, sigma2, lam)
            gv = float(np.sum(g * v))

            for _bt in range(max_backtrack):
                x_try = x_hat + p * v

                if np.any(x_try <= 0):
                    p *= 0.5
                    continue

                Q_try = Q_maxent(x_try.astype(np.float32), d, m, A_forward, sigma2, lam)

                # Armijo-type sufficient increase
                if (Q_try - Qx) >= 0.5 * p * gv:
                    break

                p *= 0.5

            step = p * v
            x_hat = np.maximum(x_hat + step, eps)

            if np.linalg.norm(step.ravel()) < tol:
                converged = True
                break

        chi2_new = chi2_awgn(x_hat.astype(np.float32), d, A_forward, sigma2)
        chi_hist.append(chi2_new)

        # ---- PPA-guided lambda update (knapsack mindset)
        # Increase penalty based on stagnation and feasibility gap.
        if converged:
            lam = min(lam * k, lam_max)

        # stagnation detection (if little improvement -> push penalty harder)
        if len(chi_hist) > 8:
            old = chi_hist[-8]
            rel_impr = (old - chi2_new) / max(old, 1e-30)
            if rel_impr < 2e-4:
                lam = min(lam * (k * 2.0), lam_max)

        chi2_val = chi2_new

        if verbose:
            print(f"[outer {outer:02d}] lam={lam:.3e} chi2={chi2_val:.3e} (chi2/thresh={chi2_val/thresh:.3f})")

    return x_hat.astype(np.float32), lam, chi2_val, q
