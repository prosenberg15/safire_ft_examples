#!/usr/bin/env python3
"""
Set up FT-AFQMC runs over a list of chemical potentials for the
Be atom in the MIDI basis to reproduce independent calculations from Table 1 of
JCTC 2018, 14, 4722.

For each mu: make a directory, generate the mu-shifted Hamiltonian and the
matching fixed-reference trial inside it, and write ftafqmc.json.
"""
import os
import json
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from pyscf import gto, scf
from afqmctools.utils.pyscf_utils import load_from_pyscf_chk_mol
from afqmctools.hamiltonian.mol import write_hamil_mol

# Reuse the trial writer from the combined setup script.
from utils import write_ft_wfn_mol


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
n_electrons = 4
atom_chkfile = "Be_midi.chk"

# Timestep, number of timeslices, inverse temp.
dt       = 0.02
nt       = 500
beta = f"{nt*dt:.3f}".rstrip("0").rstrip(".")
if "." in beta: beta = beta.replace(".", "_")

# Chemical potentials to scan (Hamiltonian mu).
mu_values = np.round(np.arange(-0.01, 0.01, 0.002), 4)

wtype = "collinear"


@contextmanager
def working_dir(path):
    """Temporarily chdir into path, restoring the original cwd on exit."""
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def build_json(wfn_file, ham_file, dt, nt):
    """Build the ftafqmc JSON text with three repeated 'estimator' keys."""
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
                "sweeps": 10,
                "n_walkers_per_mpi_task": 30,
                "population_control_interval": 10,
                "walker_ortho_interval": 10,
                "seed": 42,
                "estimator__0": {"name": "energy", "print_components": "true"},
                "estimator__1": {"name": "mixed", "onerdm": {"name": "onerdm"}},
                "estimator__2": {"name": "mixed", "twordm": {"name": "twordm"}},
            },
        }
    }
    text = json.dumps(config, indent=2)
    text = re.sub(r'"estimator__\d+"', '"estimator"', text)
    return text


def build_mean_field():
    """One-time RHF mean field in the MIDI basis; writes the chkfile."""
    mol = gto.M(
        atom  = "Be 0. 0. 0.",
        spin  = 0,
        basis = "MIDI",
    )
    mf = scf.RHF(mol).newton()
    mf.chkfile = atom_chkfile
    mf.kernel()
    mo1 = mf.stability()[0]
    dm1 = mf.make_rdm1(mo1, mf.mo_occ)
    mf = mf.run(dm1)
    mf.stability()
    mf.kernel()
    return mol, mf


def main():
    # Mean field is shared across all mu; build it once at top level.
    mol, mf = build_mean_field()
    nbasis_1s = len(mf.mo_energy)
    S = mol.intor("int1e_ovlp_sph")
    target_dens = n_electrons / nbasis_1s

    # Absolute path to the chkfile so it resolves from inside each run dir.
    chk_abs = str(Path(atom_chkfile).resolve())

    for mu in mu_values:
        print(f"\n=== mu = {mu} ===")
        mu_str = str(mu).replace(".", "_").replace("-", "m")
        run_dir = Path(f"Beta{beta}/mu_{mu_str}")
        run_dir.mkdir(parents=True,exist_ok=True)

        with working_dir(run_dir):
            # --- fixed-reference trial (tuned to N = n_electrons) ---
            wfn_file = write_ft_wfn_mol(
                mf, nt=nt, dt=dt, wtype=wtype,
                tune_mu=True, target_dens=target_dens,
                print_mu_loop=False, print_nT=True, basis="mo",
            )

            # --- mu-shifted Hamiltonian ---
            scf_data = load_from_pyscf_chk_mol(chkfile=chk_abs)
            hcore = np.array(scf_data["hcore"], copy=True)
            if hcore.shape[-2:] != S.shape:
                raise ValueError("hcore/overlap shape mismatch; check conventions.")
            hcore = hcore - mu * S
            scf_data["hcore"] = hcore

            ham_file = f"afqmc_be_midi_mu{mu_str}.h5"
            write_hamil_mol(scf_data, hamil_file=ham_file,
                            chol_cut=1e-6, verbose=True)

            # --- JSON ---
            config = build_json(wfn_file, ham_file, dt, nt)
            with open("ftafqmc.json", "w") as jf:
                jf.write(config)

        print(f"wrote files into {run_dir}/")


if __name__ == "__main__":
    main()
