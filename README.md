PPA-MaxEnt: Maximum Entropy Image Restoration with Penalty Parameter Control

Overview
--------
This repository provides a fast, CPU-oriented implementation of Maximum Entropy (MaxEnt) image restoration with an Intrinsic Correlation Function (ICF). The solver combines a Penalty Parameter Algorithm (PPA) with an efficient inner optimization scheme and is designed for reproducible scientific experimentation.

The penalty strategy follows Option A (quadratic penalty) from the literature on convex separable optimization, while the inner loop solves a MaxEnt-regularized inverse problem arising in image deconvolution.

Key Features
------------
- MaxEnt objective:
    Q(x) = S(x | m) - λ χ²(x)
- Skilling relative entropy (PAD form) to enforce positivity
- AWGN data fidelity term (chi-square)
- FFT-based forward and adjoint operators for efficient CPU execution
- Penalty Parameter Algorithm (PPA) with feasibility-driven penalty updates
- Relaxed feasibility stopping rule based on χ² / N
- Modular inner solvers:
    - Newton (default, stable and recommended)
    - Mirror Descent (experimental)
- Command-line interface (CLI) entrypoint: `ppa`

Quick Start
-----------
1) Create and activate a virtual environment (Python 3.11 or newer recommended)

2) Install the project in editable mode:
   pip install -e .

3) Run with default settings (Newton inner solver):
   ppa

Recommended reproducible run:
   ppa --chi_ratio_stop 1.6 --seed 0

Inner Solver Options
--------------------
The algorithm supports two inner optimization strategies:

- Newton (default):
  A diagonal Newton / Cornwell-style update, robust for coupled quadratic data terms.
  This is the recommended option for image restoration problems.

  Example:
    ppa --inner_solver newton

- Mirror Descent (experimental):
  An exponentiated-gradient method inspired by mirror descent formulations used in
  separable knapsack problems. Due to the strong coupling induced by the imaging
  operator, this option is provided for research and comparison purposes only.

  Example:
    ppa --inner_solver mirror

Project Structure
-----------------
src/
  ppa_maxent/
    operators/     FFT-based forward and adjoint operators
    core/          Entropy and chi-square functionals
    solvers/       MaxEnt + ICF solver with PPA-guided control
    experiments/   Reproducible demo (phantom experiment)
    utils/         Metrics and plotting utilities

Author
------
Marcelo Pontes

## Citation

If you use this code in academic work, please cite the repository (tag **v0.1.0**):

### BibTeX

```bibtex
@software{pontes_ppa_maxent_2026,
  author  = {Marcelo Pontes},
  title   = {PPA-MaxEnt: Maximum Entropy Image Restoration with Penalty Parameter Control},
  year    = {2026},
  version = {v0.1.0},
  url     = {https://github.com/profmarcelopontes/ppa-maxent}
}
