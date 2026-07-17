"""Generate safire finite-T AFQMC inputs for a grid of chemical potentials.

Running this module as a script loops over `mu_targets`, and for each entry
creates a directory `mu_<value>/` containing the trial wavefunction, the Hubbard
Hamiltonian, and the `ftafqmc.json` input file, and a SLURM submission script. 
The script will also submit the jobs if submit_jobs = True.
Edit the parameter block below to change the physical system, then run:

    python generate_benchmarks.py

To change the system, edit the module-level parameters here (they apply to every
mu in the sweep). To run a single point instead, see example_single_point.py.
"""
import json
import os
from contextlib import contextmanager
from pathlib import Path
import shutil
import subprocess
from setup_benchmarks import write_ft_wfn, write_hubbard_ham_ft

# ---------------------------------------------------------------------------
# System / method parameters -- these apply to every mu in the sweep.
# ---------------------------------------------------------------------------
Lx, Ly = 4, 4          # lattice dimensions; nsites = Lx*Ly
U = 4.0                # on-site Hubbard U, in units of the hopping t
nt = 500               # imaginary-time slices; beta = nt*dt
dt = 0.01              # imaginary-time step (Trotter step)
wtype = "collinear"    # spin symmetry: "collinear" or "noncollinear"
hst_type = "discrete_spin" # Hubbard-Stratonovich decomp. options are:
                           # discrete_spin, discrete_charge, continuous spin, continuous charge

sweeps = 30            # number of sweeps
nwlks_per_mpi = 50     # number of walkers per mpi task
nPop = 10              # pop. control interval
nOrtho = 10            # stabilization interval
seed = 42              # seed for rng

# If True, ignore the mu column of mu_targets and instead solve for the trial
# chemical potential muT that reproduces each paired target density. If False,
# the mu value is used directly and the density column is ignored.
tune_mu = True

submit_jobs = False # set to True for automated job submission

# (mu, target_density) pairs. With tune_mu=False only mu is used, and muT=mu; with
# tune_mu=True the trial muT is tuned to the paired target density.
mu_targets = [
    (-2.0, 1.0),
    (-1.8, 0.99),
    (-1.6, 0.97),
    (-1.4, 0.94),
    (-1.2, 0.89),
    (-1.0, 0.83),
    (-0.8, 0.77),
    (-0.6, 0.71),
    (-0.4, 0.67),    
    (-0.2, 0.64),
    ( 0.0, 0.63)
]

@contextmanager
def working_dir(path):
    """Temporarily chdir into path, restoring the original cwd on exit."""
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)

import re

def build_json(wfn_file: str,
               ham_file: str,
               dt: float,
               nt: int,
               sweeps: int = 30,
               wlks_per_mpi: int = 30,
               nPop: int = 10,
               nOrtho: int = 10,
               seed: int = 42):
    """Build the text of the ftafqmc.json input file safire reads.

    Parameters
    ----------
    wfn_file, ham_file : str
        Filenames returned by write_ft_wfn and write_hubbard_ham_ft. They are
        referenced by name, so the JSON must live in the same directory.
    dt, nt : float, int
        Imaginary-time step and number of slices, written as `timestep` and
        `steps`. Keep these consistent with the wavefunction/Hamiltonian.
    sweeps : int, optional
        Number of measurement sweeps.
    wlks_per_mpi : int, optional
        Walkers per MPI rank (written as `n_walkers_per_mpi_task`); trades
        statistics against memory/cost.
    nPop : int, optional
        Population-control interval: how often to reweight the walker population
        (written as `population_control_interval`).
    nOrtho : int, optional
        Walker re-orthogonalization / stabilization interval (written as
        `walker_ortho_interval`).
    seed : int, optional
        RNG seed.

    Returns
    -------
    str
        The JSON document as text (see Notes on why it isn't returned as a dict).

    """
    config = {
        "ftafqmc": {
            "project": {
                "id": "ftafqmc",
                "series": 0,
            },
            "execute": {
                "walker_set": {
                    "walker_type": f"{wtype}",
                },
                "wavefunction": {
                    "filename": wfn_file,
                },
                "hamiltonian": {
                    "filename": ham_file,
                },
                "timestep": dt,
                "steps": nt,
                "sweeps": sweeps,
                "n_walkers_per_mpi_task": wlks_per_mpi,
                "population_control_interval": nPop,
                "walker_ortho_interval": nOrtho,
                "seed": seed,
                "estimator__0": {"name": "energy", "print_components": "true"},
                "estimator__1": {"name": "mixed", "onerdm": {"name": "onerdm"}},
                "estimator__2": {"name": "mixed", "twordm": {"name": "twordm"}},
            },
        }
    }

    text = json.dumps(config, indent=2)
    # Collapse the placeholder keys ("estimator__0", "estimator__1", ...)
    # back to the literal repeated key the executable expects.
    text = re.sub(r'"estimator__\d+"', '"estimator"', text)
    return text

def main():
    """Loop over mu_targets and write inputs into one mu_<value>/ dir per point."""
    for mu, dens in mu_targets:
        if tune_mu:
            print(f"\n=== mu = {mu}, target density = {dens} ===")
        else:
            print(f"\n=== mu = {mu} ===")

        beta = f"{nt*dt:.3f}".rstrip("0").rstrip(".")
        if "." in beta: beta = beta.replace(".", "_")
        mu_str = str(mu).replace(".", "_")
        run_dir = Path(f"Beta{beta}/mu_{mu_str}")
        run_dir.mkdir(parents=True,exist_ok=True)
        shutil.copy("run.sh",run_dir)
        
        with working_dir(run_dir):
            ham_file, one_body = write_hubbard_ham_ft(
                Lx, Ly, U,
                mu=mu,
                htype=wtype,
                hst_type=hst_type
            )

            wfn_file = write_ft_wfn(
                nt, dt, one_body,
                muT=mu,
                wtype=wtype,
                tune_mu=tune_mu,
                target_dens=dens,
                print_mu_loop=True,
                print_nT=True,
            )

            config = build_json(wfn_file, ham_file, dt, nt,
                                sweeps, nwlks_per_mpi, nPop, nOrtho, seed)
            with open("ftafqmc.json", "w") as jf:
                jf.write(config)

        print(f"wrote files into {run_dir}/")

        if(submit_jobs):
            result = subprocess.run(
                ["sbatch", "run.sh"],
                cwd=run_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            print(result.stdout)
        
if __name__ == "__main__":
    main()
