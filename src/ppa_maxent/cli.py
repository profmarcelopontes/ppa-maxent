# src/ppa_maxent/cli.py

import argparse

from ppa_maxent.experiments.demo_phantom import run_demo


def main():
    parser = argparse.ArgumentParser(description="PPA-guided MaxEnt restoration (Option A)")

    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--sigma_psf", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=1.4)
    parser.add_argument("--psnr_in", type=float, default=45.0)
    parser.add_argument("--seed", type=int, default=0)

    # PPA / stopping
    parser.add_argument("--chi_ratio_stop", type=float, default=1.60)
    parser.add_argument("--rho_small", type=float, default=1e-6)
    parser.add_argument("--rho_max", type=float, default=1e2)
    parser.add_argument("--k", type=float, default=2.0)
    parser.add_argument("--max_outer", type=int, default=120)

    # Inner solver selection (DEFAULT: mirror)
    parser.add_argument("--inner_solver", choices=["mirror", "newton"], default="mirror")

    # Mirror descent params (stability knobs included)
    parser.add_argument("--beta0", type=float, default=1.0)
    parser.add_argument("--beta_decay", type=float, default=0.0)
    parser.add_argument("--max_inner_steps", type=int, default=40)
    parser.add_argument("--flux_norm", choices=["model", "data", "none"], default="model")
    parser.add_argument("--exp_clip", type=float, default=50.0)

    # Newton params (if inner_solver=newton)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--max_newton_steps", type=int, default=40)
    parser.add_argument("--max_backtrack", type=int, default=25)

    args = parser.parse_args()

    run_demo(
        N=args.N,
        sigma_psf=args.sigma_psf,
        beta=args.beta,
        psnr_in=args.psnr_in,
        seed=args.seed,
        chi_ratio_stop=args.chi_ratio_stop,
        # solver kwargs:
        inner_solver=args.inner_solver,
        rho_small=args.rho_small,
        rho_max=args.rho_max,
        k=args.k,
        max_outer=args.max_outer,
        beta0=args.beta0,
        beta_decay=args.beta_decay,
        max_inner_steps=args.max_inner_steps,
        flux_norm=args.flux_norm,
        exp_clip=args.exp_clip,
        tol=args.tol,
        max_newton_steps=args.max_newton_steps,
        max_backtrack=args.max_backtrack,
    )


if __name__ == "__main__":
    main()
