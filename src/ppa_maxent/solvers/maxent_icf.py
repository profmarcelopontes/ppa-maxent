# src/ppa_maxent/solvers/maxent_icf.py
#
# PPA Option A (quadratic hinge penalty on chi2 violation) with selectable inner solver:
# - inner_solver="mirror" (default): exponentiated-gradient mirror descent (KL geometry)
#   + exponent clipping
#   + flux normalization (default: "model" => sum(x)=sum(m); also "data" and "none")
#   + NEW: line-search on alpha (Armijo decrease) to prevent collapse/divergence
# - inner_solver="newton": simplified diagonal Newton + backtracking (robust baseline)
#
# Outer loop (PPA): increase rho based on feasibility ratio chi2/thresh and stop at chi_ratio_stop.

from __future__ import annotations

import numpy as np

from ppa_maxent.core.functionals import (
    chi2_awgn,
    F_penalty_chi2,
    grad_F_penalty_chi2,
)


def _mirror_descent_inner(
    x_hat: np.ndarray,
    d: np.ndarray,
    m: np.ndarray,
    sigma2: float,
    A_forward,
    A_adjoint,
    *,
    rho: float,
    thresh: float,
    q: float,
    beta: float,
    max_inner_steps: int,
    eps: float,
    flux_norm: str,
    exp_clip: float,
    # NEW: line search controls
    ls_max: int,
    c_armijo: float,
) -> np.ndarray:
    """
    KL-mirror descent (Exponentiated Gradient) inner solver for fixed rho:

        x <- x * exp(-alpha * grad)

    with:
        alpha0 = beta / max(1, ||grad||_1)

    Stabilizers:
      - exponent clipping: exp(clip(-alpha*g, -exp_clip, exp_clip))
      - flux normalization (projection-like scaling):
          "model": sum(x) = sum(m)   (DEFAULT)
          "data" : sum(x) = sum(d)
          "none" : no normalization
      - NEW: Armijo line-search on alpha to ensure F decreases:
          try alpha=alpha0; if F doesn't decrease enough, alpha <- alpha/2 (up to ls_max).
    """
    x = np.maximum(x_hat, eps).astype(np.float64, copy=False)

    flux_norm_l = flux_norm.lower()
    if flux_norm_l == "model":
        target_sum = float(np.sum(m.astype(np.float64, copy=False)))
    elif flux_norm_l == "data":
        target_sum = float(np.sum(d.astype(np.float64, copy=False)))
    elif flux_norm_l == "none":
        target_sum = None
    else:
        raise ValueError("flux_norm must be 'model', 'data', or 'none'")

    for _ in range(max_inner_steps):
        # gradient at current x
        g = grad_F_penalty_chi2(
            x.astype(np.float32),
            d,
            m,
            A_forward,
            A_adjoint,
            sigma2,
            rho,
            thresh,
            eps=eps,
        )  # float64

        g1 = float(np.sum(np.abs(g)))
        gamma = max(1.0, g1)
        alpha0 = beta / gamma

        Fx = F_penalty_chi2(
            x.astype(np.float32),
            d,
            m,
            A_forward,
            sigma2,
            rho,
            thresh,
            eps=eps,
        )

        # Line-search on alpha
        alpha = alpha0
        accepted = False

        for _ls in range(ls_max):
            z = np.clip(-alpha * g, -exp_clip, exp_clip)
            x_try = x * np.exp(z)
            x_try = np.maximum(x_try, eps)

            if target_sum is not None:
                s = float(np.sum(x_try))
                if s > 0.0:
                    x_try *= (target_sum / s)

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

            # Sufficient decrease. We use a simple decrease proxy based on ||g||_1:
            # require F_try <= F_x - c * alpha * ||g||_1
            if F_try <= Fx - c_armijo * alpha * g1:
                x = x_try
                accepted = True
                break

            alpha *= 0.5

        if not accepted:
            # Can't find a decreasing step => stop inner loop
            break

    return x.astype(np.float32)


def _newton_inner(
    x_hat: np.ndarray,
    d: np.ndarray,
    m: np.ndarray,
    sigma2: float,
    A_forward,
    A_adjoint,
    *,
    rho: float,
    thresh: float,
    q: float,
    max_newton_steps: int,
    max_backtrack: int,
    tol: float,
    eps: float,
) -> tuple[np.ndarray, bool]:
    """
    Simplified diagonal Newton + backtracking inner solver for fixed rho.
    Minimizes F_rho(x) with positivity enforced.
    """
    x = np.maximum(x_hat.astype(np.float64, copy=False), eps)
    converged = False

    for _ in range(max_newton_steps):
        g = grad_F_penalty_chi2(
            x.astype(np.float32),
            d,
            m,
            A_forward,
            A_adjoint,
            sigma2,
            rho,
            thresh,
            eps=eps,
        )

        chi2_here = chi2_awgn(x.astype(np.float32), d, A_forward, sigma2)
        v = max(0.0, (chi2_here / thresh) - 1.0)  # normalized violation for scaling

        denom = (1.0 / np.maximum(x, eps)) + 2.0 * rho * v * q
        denom = np.maximum(denom, 1e-18)

        v_dir = (-g / denom).astype(np.float64)  # descent direction

        p = 1.0
        Fx = F_penalty_chi2(x.astype(np.float32), d, m, A_forward, sigma2, rho, thresh, eps=eps)
        gv = float(np.sum(g * v_dir))

        for _bt in range(max_backtrack):
            x_try = x + p * v_dir
            if np.any(x_try <= 0):
                p *= 0.5
                continue

            F_try = F_penalty_chi2(
                x_try.astype(np.float32), d, m, A_forward, sigma2, rho, thresh, eps=eps
            )
            if (F_try - Fx) <= 0.5 * p * gv:
                break
            p *= 0.5

        step = p * v_dir
        x = np.maximum(x + step, eps)

        if np.linalg.norm(step.ravel()) < tol:
            converged = True
            break

    return x.astype(np.float32), converged


def maxent_icf_solver(
    d: np.ndarray,
    m: np.ndarray,
    sigma2: float,
    A_forward,
    A_adjoint,
    *,
    # Outer PPA (Option A)
    rho_small: float = 1e-6,
    k: float = 2.0,
    thresh: float | None = None,
    chi_ratio_stop: float = 1.60,
    rho_max: float = 1e2,
    max_outer: int = 120,
    # Inner solver selection
    inner_solver: str = "mirror",  # default = mirror
    # Mirror params
    beta0: float = 1e-2,           # safer default with line-search
    beta_decay: float = 1.0,       # recommended for stability (paper-like)
    max_inner_steps: int = 60,
    flux_norm: str = "model",
    exp_clip: float = 50.0,
    ls_max: int = 12,
    c_armijo: float = 1e-6,
    # Newton params (alternative)
    tol: float = 1e-5,
    max_newton_steps: int = 40,
    max_backtrack: int = 25,
    # stability / scaling
    eps: float = 1e-12,
    q: float | None = None,
    verbose: bool = True,
):
    """
    PPA Option A outer loop + selectable inner solver.

    Outer loop:
      - approximately solve min_x F_rho(x) for fixed rho
      - compute ratio r = chi2(x)/thresh
      - if r > chi_ratio_stop: rho <- min(k*rho, rho_max)
      - stop when r <= chi_ratio_stop
    """
    if thresh is None:
        thresh = float(d.size)

    x_hat = np.maximum(m.astype(np.float64, copy=False), eps).astype(np.float32)
    rho = float(rho_small)

    # q estimate if not provided: energy of impulse response
    if q is None:
        impulse = np.zeros_like(d, dtype=np.float32)
        impulse[d.shape[0] // 2, d.shape[1] // 2] = 1.0
        k_eff_same = A_forward(impulse).astype(np.float64)
        q = float(np.sum(k_eff_same * k_eff_same) / sigma2)

    chi2_val = chi2_awgn(x_hat, d, A_forward, sigma2)
    ratio_hist: list[float] = [chi2_val / thresh]

    for outer in range(max_outer):
        ratio = chi2_val / thresh
        if ratio <= chi_ratio_stop:
            if verbose:
                print(f"[outer {outer:02d}] STOP: chi2/thresh={ratio:.3f} <= {chi_ratio_stop}")
            break

        beta = beta0 if beta_decay <= 0 else float(beta0 / ((outer + 1) ** beta_decay))
        converged = False

        if inner_solver.lower() == "mirror":
            x_hat = _mirror_descent_inner(
                x_hat,
                d,
                m,
                sigma2,
                A_forward,
                A_adjoint,
                rho=rho,
                thresh=thresh,
                q=q,
                beta=beta,
                max_inner_steps=max_inner_steps,
                eps=eps,
                flux_norm=flux_norm,
                exp_clip=exp_clip,
                ls_max=ls_max,
                c_armijo=c_armijo,
            )
        elif inner_solver.lower() == "newton":
            x_hat, converged = _newton_inner(
                x_hat,
                d,
                m,
                sigma2,
                A_forward,
                A_adjoint,
                rho=rho,
                thresh=thresh,
                q=q,
                max_newton_steps=max_newton_steps,
                max_backtrack=max_backtrack,
                tol=tol,
                eps=eps,
            )
        else:
            raise ValueError("inner_solver must be 'mirror' or 'newton'")

        chi2_new = chi2_awgn(x_hat, d, A_forward, sigma2)
        chi2_val = chi2_new
        ratio = chi2_val / thresh
        ratio_hist.append(ratio)

        # PPA update: feasibility-driven
        if ratio > chi_ratio_stop:
            rho = min(rho * k, rho_max)

        # optional: if ratio stalls, push rho harder
        if len(ratio_hist) > 6:
            old = ratio_hist[-6]
            rel_impr = (old - ratio) / max(old, 1e-30)
            if rel_impr < 2e-4 and ratio > chi_ratio_stop:
                rho = min(rho * (k * 2.0), rho_max)

        if verbose:
            tag = " (converged)" if converged else ""
            if inner_solver.lower() == "mirror":
                extra = f" beta={beta:.3e} flux_norm={flux_norm} ls_max={ls_max}"
            else:
                extra = ""
            print(
                f"[outer {outer:02d}] rho={rho:.3e} chi2={chi2_val:.3e} "
                f"(chi2/thresh={ratio:.3f}){tag}{extra}"
            )

    return x_hat.astype(np.float32), rho, chi2_val, q
