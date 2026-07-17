#!/usr/bin/env python
"""
Minimal end-to-end example: set up the input files for ONE finite-temperature
AFQMC calculation of the 4x4 Hubbard model, ready to run with the `safire`
executable.

  setup_benchmarks.write_ft_wfn         -> finite-T trial wavefunction (HDF5)
  setup_benchmarks.write_hubbard_ham_ft -> Hubbard Hamiltonian (HDF5)
  generate_benchmarks.build_json        -> safire input file (ftafqmc.json)

It creates a single run directory (example_mu_<mu>/) containing the three
files safire needs.  See README.md for what to do next.
"""
import os
from contextlib import contextmanager
from pathlib import Path

from setup_benchmarks import write_ft_wfn, write_hubbard_ham_ft, build_json

# ---------------------------------------------------------------------------
# System / method parameters
# ---------------------------------------------------------------------------
Lx, Ly = 4, 4              # 4x4 square lattice -> 16 sites
U       = 4.0              # on-site Hubbard repulsion (in units of the hopping t)
mu      = -1.0             # chemical potential
muT     = -1.0             # trial chemical potential (will be overwritten if tune_mu=True)
nt      = 100              # number of imaginary-time slices
dt      = 0.02             # imaginary-time step  ->  beta = nt*dt
wtype   = "collinear"      # spin symmetry of walkers / wavefunction / Hamiltonian
hst_type = "discrete_spin" # Hubbard-Stratonovich decomp. options are:
                           # discrete_spin, discrete_charge, continuous spin, continuous charge

sweeps = 30            # number of sweeps
nwlks_per_mpi = 50     # number of walkers per mpi task
nPop = 10              # pop. control interval
nOrtho = 10            # stabilization interval
seed = 42              # seed for rng

tune_mu = False        # parameter to tune trial chemical potential to match target dens.
                       # if tune_mu=True, you should provide target_dens to write_ft_wfn)
target_dens = 0.0      # otherwise target_dens defaults to 0.0

@contextmanager
def working_dir(path):
    """Temporarily chdir into path, restoring the original cwd on exit."""
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def main():
    beta = nt * dt
    mu_str = str(mu).replace(".", "_")
    run_dir = Path(f"example_mu_{mu_str}")
    run_dir.mkdir(exist_ok=True)

    print(f"=== finite-T AFQMC setup: {Lx}x{Ly} Hubbard, U={U}, mu={mu} ===")
    print(f"beta = nt*dt = {beta}")
    print(f"writing inputs into {run_dir}/\n")

    with working_dir(run_dir):
        # 1) Hubbard Hamiltonian (also returns the hopping-only one-body matrix)
        ham_file, one_body = write_hubbard_ham_ft(
            Lx, Ly, U,
            mu=mu,
            htype=wtype,
            hst_type=hst_type
        )

        # 2) finite-temperature trial wavefunction, built from the same hopping
        #    one-body matrix as the Hamiltonian
        wfn_file = write_ft_wfn(
            nt, dt, one_body,
            muT=muT,
            wtype=wtype,
            tune_mu=tune_mu,
            target_dens=target_dens,
            print_nT=True,   # print the trial density as a sanity check
        )

        # 3) safire input file
        with open("ftafqmc.json", "w") as jf:
            jf.write(build_json(wfn_file, ham_file, dt, nt, sweeps, nwlks_per_mpi, nPop, nOrtho, seed))

    print(f"\nDone. {run_dir}/ now contains:")
    print(f"  - {wfn_file}   (trial wavefunction)")
    print(f"  - {ham_file}   (Hamiltonian)")
    print(f"  - ftafqmc.json                            (safire input)")
    print("\nNext: stage the `safire` executable and submit run.sh from inside")
    print(f"      {run_dir}/  (see README.md).")


if __name__ == "__main__":
    main()
