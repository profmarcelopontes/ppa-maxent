PPA-MaxEnt: Maximum Entropy Image Restoration with Penalty Parameter Control

Overview
--------
This repository provides a fast CPU-oriented implementation of Maximum Entropy (MaxEnt) image restoration with an Intrinsic Correlation Function (ICF). The solver uses a simplified Newton (Cornwell-style) update and a Penalty Parameter Algorithm (PPA)-guided stopping rule based on chi-square feasibility (chi2 / N).

Key Features
------------
- MaxEnt objective: Q(x) = S(x|m) - lambda * chi2(x)
- Skilling relative entropy (PAD form) to enforce positivity
- AWGN data fidelity term (chi-square)
- FFT-based forward/adjoint operator for speed on CPU
- PPA-inspired penalty control and relaxed feasibility stopping criterion
- CLI entrypoint: `ppa`

Quick Start
-----------
1) Create and activate a virtual environment (Python 3.11 recommended)

2) Install the project in editable mode:
   pip install -e .

3) Run:
   ppa

Optional parameters:
   ppa --chi_ratio_stop 1.6 --seed 0

Project Structure
-----------------
src/
  ppa_maxent/
    operators/     FFT-based operators
    core/          Entropy and chi-square functionals
    solvers/       MaxEnt + ICF solver with PPA-guided control
    experiments/   Reproducible demo (phantom)
    utils/         Metrics and plotting

Author
------
Marcelo Pontes
