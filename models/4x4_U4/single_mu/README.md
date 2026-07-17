# Finite-temperature AFQMC with `safire`

This directory shows how to set up and run a **finite-temperature (finite-T)
AFQMC** calculation of the Hubbard model end to end: build the input files with
the Python helpers here, then run them with the `safire` executable.

The scripts prepare a benchmark calculation to match Fig. 1 of [Phys. Rev. B 99 045108](https://journals.aps.org/prb/pdf/10.1103/PhysRevB.99.045108), i.e. the **4×4 square-lattice Hubbard model** at `U = 4t`, inverse temperature `β = nt·dt = 2`, at a single chemical potential `μ = -1`.

---

## 1. Files

| File | Role |
|------|------|
| `utils.py` | Small functions to compute the Green's function, tune the trial chemical-potential, and manage some I/O tasks. |
| `setup_benchmarks.py` | Contains routines to write input files for `safire`: `write_ft_wfn` (trial wavefunction) and `write_hubbard_ham_ft` (Hamiltonian). |
| `example_single_mu.py` | Writes the input files for a single `μ` (i.e. not a loop over `μ`). |
| `analysis.py` | A simple example analysis script to reproduce the results from Fig. 1 of [Phys. Rev. B 99 045108](https://journals.aps.org/prb/pdf/10.1103/PhysRevB.99.045108).
| `run.sh` | Example submission script for Rusty. |

---

## 2. SAFIRE and afqmctools installation

The `safire` executable can be built following the instructions on the [github](https://github.com/SFQMC/SAFIRE/tree/overhaul), make sure you clone and build the **overhaul** branch. You should also be able to find instructions there for installing `afqmctools`, which is required to write the wavefunction and hamiltonian input files.

Here is a simple set of instructions that should work if you are installing on Rusty:

### Building SAFIRE
```bash
git clone https://github.com/SFQMC/SAFIRE.git
cd SAFIRE
git checkout overhaul

mkdir build; cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="."
make -j 10
```

The `safire` exectuable should now be in the `/path/to/safire/build/bin` directory.

### Installing afqmctools
```bash
cd /path/to/venvs          # move to directory where you keep virtual envs
python -m venv afqmctools  # make virtual environment for afqmctools
source afqmctools/bin/activate # activate venv
cd /path/to/safire/utils   # move to utils directory in SAFIRE repo you cloned
pip install .              # install afqmctools
```

You should now be able to run the small example.

---

## 3. A simple example

### Step 1 — generate the input files

To set up the example calculation you can run:

```bash
python example_single_mu.py
```

This creates `example_mu_-1_0/` containing three files:

- `wfn_collinear_ft_N16_Beta2_0_nt100_muT-1_0.h5` — finite-T trial wavefunction
- `hamil_4x4_U4_0_mu-1_0_collinear.h5` — Hubbard Hamiltonian
- `ftafqmc.json` — the `safire` input file

The script prints `ntot_trial`, the trial-wavefunction density, as a quick
sanity check.

For this example the parameters are chosen to be:

| parameter | value |
|---|---|
|`Lx` | 4|
|`Ly` | 4|                                                                               
|`U`  | 4.0|                                                                             
|`nt` | 100|    
|`dt` | 0.02|
| `μ` | -1|
|`tune_mu` | `False` |      
|`wtype` | `'collinear'`|
|`hst_type` | `'discrete_spin'`|      
|`sweeps` |30 |             
|`n_walkers_per_mpi_task` | 50 |
|`population_control_interval` | 10 |
|`walker_ortho_interval` |10 |
|`seed` | 42 |

The inverse temperature `β` is determined by the choice of `nt` and `dt`, i.e. `β = nt·dt`. Because `tune_mu = False`, the trial chemical potential will be set by hand, in this case equal to the many-body chemical potential. When `tune_mu = True` the trial chemical potential will be tuned to match a given `target_dens`.   

### Step 2 — run

To run the simulation, change to the directory where the input files were written and copy the Slurm submission script (or use your own). You will also need a copy of safire in the run directory (or your run script will have to point to it):

```bash
cd example_mu_-1_0
cp ../run.sh .
sbatch run.sh
```

When the job finishes, the run directory will contain:

- `ftafqmc.s000.scalar.dat` — scalar estimators (energies)
- `ftafqmc.s000.stat.h5` — accumulated observables (1- and 2-body RDMs in this case)
- `out` — log with run info (parameters, timing, etc.)

### Step 3 — analyze

Analyze the results by changing back to the parent directory and running:

```bash
cd ..
python analysis.py example_mu_-1_0
```

This prints the energy, and optionally the density, density-density correlation, and spin-spin correlation.

---

## 4. Setting parameters

The following parameters can be easily modified:

| Type | Parameters |
|------|------------|
| System parameters | `Lx`, `Ly`, `U`, `μ`, `tune_mu`, `target_dens` |
| MC parameters |  `nt`, `dt`, `sweeps`, `n_walkers_per_mpi_task`, `population_control_interval`, `walker_ortho_interval`, `seed` |

To change them, edit the parameter block at the top of `example_single_mu.py`
(or `generate_benchmarks.py` for a loop over `μ`).  If you wish to change further MC parameters, or add others, you can directly edit `ftafqmc.json`. 
 Included below is a brief summary of the relevant MC parameters (a more exhaustive list can be found in the `safire` documentation).

---

## 5. MC parameters for `safire` (in `ftafqmc.json`)

| Parameter name | Meaning |
|---------|---------|
| `timestep` (`dt`), `steps` (`nt`) | imaginary-time step and count; `β = nt·dt` |
| `walker_type` | spin symmetry, from `wtype` (`collinear` / `noncollinear`) |
| `sweeps` | number of measurement sweeps |
| `n_walkers_per_mpi_task` | walkers per MPI rank |
| `population_control_interval`, `walker_ortho_interval` | pop. control, stabilization intervals |
---

## 6. Summary

With this, you should be able to set up and run calculations on the Hubbard model, and analyze the results. To change common parameters, edit the parameters block at the top of `example_single_mu.py` (or `generate_benchmarks.py` if you want to loop over `μ`). An overview of the workflow is:

```bash
python example_single_mu.py        # writes input files into example_mu_<mu_value>
cd example_mu_-1_0                    # change to directory with input files
cp ../run.sh .                        # copy Slurm submission script
sbatch run.sh                         # submit
cd ../                                # change to parent directory
python analysis.py example_mu_-1_0    # run analysis script
```

The remaining sections provide further detail on the scripts to generate the input files.

## 7. Function reference

Each function also has a docstring in the source
(`help(write_ft_wfn)` etc.). To change the parameters you can edit
the parameter block at the top of `example_single_mu.py` (or
`generate_benchmarks.py`) — the functions below read those values.

### `setup_benchmarks.py`

#### `write_ft_wfn(nt, dt, one_body, muT=0.0, wtype='collinear', tune_mu=False, target_dens=0.0, print_mu_loop=False, print_nT=False) -> str`

Writes the finite-temperature **trial wavefunction** (in the factored `U·D·V` form, see note below) to an HDF5 file and returns its
filename.

| Parameter | What to change it for |
|-----------|-----------------------|
| `nt`, `dt` | Temperature: `β = nt·dt`. |
| `one_body` | The hopping-only one-body matrix, i.e. the **second return value of `write_hubbard_ham_ft`**. It sets the trial hopping and, via its shape, the system size. Pass it straight through so the trial matches the Hamiltonian. |
| `muT` | Trial chemical potential. When `tune_mu=False`, set by hand, when `tune_mu=True`, `muT` is tuned to match `target_dens`. |
| `wtype` | `'collinear'` (RHF/UHF) or `'noncollinear'` (GHF). Must match `write_hubbard_ham_ft`'s `htype` and the JSON `walker_type`. |
| `tune_mu`, `target_dens` | Set `tune_mu=True` to solve for the trial `muT` that reproduces `target_dens`. |
| `print_mu_loop`, `print_nT` | Diagnostics: print the mu-search iterations / the trial density. |

Returns e.g. `wfn_collinear_ft_N16_Beta2_0_nt100_muT-1_0.h5` (the `N16` encodes `nsites`).

#### `write_hubbard_ham_ft(Lx, Ly, U, mu=0.0, htype='collinear', hst_type='discrete_spin') -> (str, one_body)`

Builds a **Hubbard Hamiltonian** with `afqmctools` and
writes it to HDF5. Returns a tuple: the HDF5 filename **and** the hopping-only
one-body matrix (the `tij` term *before* the `mu` shift is folded in). Feed that
matrix straight into `write_ft_wfn`'s `one_body` argument so the trial uses the
same hopping as the Hamiltonian.

| Parameter | What to change it for |
|-----------|-----------------------|
| `Lx`, `Ly` | System size (must match the wavefunction). |
| `U` | On-site interaction strength (units of `t`). Main interaction knob. |
| `mu` | Chemical potential, added as `mu·I` to the one-body term (match the wavefunction `mu`). |
| `htype` | Spin symmetry; must match `wtype`. |
| `hst_type` | Hubbard–Stratonovich decomposition (`'discrete_spin'`, `'discrete_charge'`, `'continuous_spin'`, `'continuous_charge'`). |

Two things are set **inside** the function rather than as arguments — edit the
source to change them:
- **Hopping** `hopping = [1.0, 0.0]` = `[t, t']`. Add `t'`/further neighbors for
  non-nearest-neighbor models.
- **Lattice / boundaries** via the `get_lattice(...)` call (PBC in both
  directions; a commented `type="honeycomb"` shows how to switch lattices).
- `nelec` is fixed to half filling and acts only as the reference sector; the
  physical filling in a finite-T run is set by `mu` and `β`, not `nelec`.

The filename, e.g. `hamil_4x4_U4_0_mu-1_0_collinear.h5` is returned
alongside the one-body matrix as `(fname, one_body)`.

#### `build_json(wfn_file, ham_file, dt, nt, sweeps=30, wlks_per_mpi=30, nPop=10, nOrtho=10, seed=42) -> str`

Returns the text of `ftafqmc.json`. The main **Monte Carlo / sampling knobs are
arguments** — `sweeps`, `wlks_per_mpi` (`n_walkers_per_mpi_task`), `nPop`
(`population_control_interval`), `nOrtho` (`walker_ortho_interval`), and `seed` —
so the drivers pass them in from their parameter blocks. Only `walker_type`
(from `wtype`) and the three `estimator` blocks (energy, 1-RDM, 2-RDM) remain
hard-coded in the config dict; edit those in the source. See §5 for what each
does.

#### helpers
- `SymType` — enum mapping `'collinear'`/`'noncollinear'` ↔ the integer safire
  stores in the wavefunction `dims` field.
- `write_sparse_mat(f, path, mat)` — writes a CSR matrix into an HDF5 group in
  safire's sparse layout. You should not need to call it directly.

---

## 9. Form of trial wavefunction

In this script, the trial propagator is constructed from the non-interacting Hamiltonian:
```math
\mathbf{B}_T = e^{-\Delta\tau(\mathbf{H}_0+\mu_T\mathbf{I})}
```
The non-interacting Hamiltonian can be diagonalized:
```math
   \mathbf{H}_0 = \mathbf{U} \boldsymbol{\Lambda} \mathbf{U}^\dagger, \,\,\,\boldsymbol{\Lambda}=\textrm{diag}(\lambda_1,\dots,\lambda_{N_s})
```
which yields,
```math
    \mathbf{B}_T = \mathbf{U} e^{-\Delta\tau(\boldsymbol{\Lambda}+\mu_T\mathbf{I})} \mathbf{U}^\dagger
```
We include a scale factor $\xi\equiv 0.5(\lambda_\textrm{min}+\lambda_\textrm{max})$, and then write the propagator as,
```math
    \mathbf{B}_T = \mathbf{V}_L \mathbf{D}_L \mathbf{U}_L,
```
with,
```math
\begin{aligned}
    \mathbf{V}_L &= \mathbf{U} \\
    \mathbf{D}_L &= e^{-\Delta\tau(\boldsymbol{\Lambda}+\mu_T\mathbf{I}+\xi)} \\
    \mathbf{U}_L &= \mathbf{U}^\dagger
\end{aligned}
```
