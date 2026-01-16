import argparse

from ppa_maxent.experiments.demo_phantom import run_demo


def main():
    parser = argparse.ArgumentParser(description="PPA-guided MaxEnt restoration (demo phantom)")
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--sigma_psf", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=1.4)
    parser.add_argument("--psnr_in", type=float, default=45.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chi_ratio_stop", type=float, default=1.60)

    args = parser.parse_args()

    run_demo(
        N=args.N,
        sigma_psf=args.sigma_psf,
        beta=args.beta,
        psnr_in=args.psnr_in,
        seed=args.seed,
        chi_ratio_stop=args.chi_ratio_stop
    )


if __name__ == "__main__":
    main()
