# Finite-temperature AFQMC: Be atom in the MIDI basis

This example applies **finite-temperature (finite-T) AFQMC** to a molecular
system — the beryllium atom in the MIDI Gaussian basis. It scans over chemical potential `μ`, writing a full set of `safire` inputs into a run directory for each `μ`.

The setup allows you to reproduce the  results in Table 1
of [J. Chem. Theory Comput. 2018, 14, 4722](https://doi.org/10.1021/acs.jctc.8b00569).

This README assumes you have already worked through the lattice examples. Start
with the [`single_mu` README](../../models/4x4_U4/single_mu/README.md) for
installing `safire` and `afqmctools`, the meaning of the Monte Carlo knobs in
`ftafqmc.json`, and the form of the finite-T trial wavefunction. The
[`mu_scan` README](../../models/4x4_U4/mu_scan/README.md) covers the
scan-over-`μ` structure. This README only describes what is different for a
molecular calculation.

---

## 1. What is different from the lattice examples

| Aspect | Lattice (Hubbard) | Molecular (this example) |
|--------|-------------------|--------------------------|
| One-body / two-body integrals | Built by hand from the lattice (`write_hubbard_ham_ft`). | Generated from a **PySCF mean field** and written with `afqmctools`' `write_hamil_mol`. |
| Trial wavefunction | Diagonal hopping matrix. | Built from the RHF molecular orbitals (`write_ft_wfn_mol`, `basis='mo'`). |
| Chemical-potential shift | `μ·I` folded into the one-body term. | `hcore → hcore − μ·S`, i.e. shifted by `μ` times the AO **overlap** matrix `S`. |
| Mean field | none | An RHF calculation (`Be_midi.chk`) built **once** and shared across all `μ`. |

---

## 2. Files

| File | Role |
|------|------|
| `setup.py` | Builds the shared RHF mean field, then loops over `μ`, writing inputs into `Beta{β}/mu_{μ}/`. |
| `utils.py` | `write_ft_wfn_mol` (molecular finite-T trial) and the sparse-matrix I/O helper. |
| `run.sh` | Example Slurm submission script for Rusty. |
| `../analysis_formatted.py` | Analyzes a single run directory (energy components and density). |

---

## 3. Workflow

```bash
# edit the parameter block at the top of setup.py for the desired system
python setup.py                                   # mean field + inputs for every mu
# ... submit the jobs (see Step 2) and wait for the safire runs to finish ...
python ../analysis.py Beta10/mu_m0_5    # analyze one run directory
```

> **Note:** This example runs using the MIDI basis, which can be installed via pip:
 ```bash 
    pip install basis_set_exchange
```

### Step 1 — generate

```bash
python setup.py
```

This first runs a one-time RHF calculation for Be in the MIDI basis (with a
stability check), writing `Be_midi.chk`. The trial filling is then tuned to the
neutral atom (`n_electrons = 4`). For each `μ` in `mu_values` the script creates

```
Beta{β}/mu_{μ}/
├── wfn_be_midi_*.h5           # fixed-reference finite-T trial
├── afqmc_be_midi_mu*.h5       # mu-shifted molecular Hamiltonian
└── ftafqmc.json               # safire input file
```

where `β = nt·dt` (e.g. `nt=500, dt=0.02` → `Beta10/`). The `μ` value is encoded
in the directory name with `.` replaced by `_` and a leading `-` replaced by `m`
(e.g. `μ = -0.5` → `mu_m0_5/`).

Unlike the lattice `mu_scan` script, `setup.py` does **not** submit jobs — it
only writes inputs. The main parameters live in the block at the top of
`setup.py`:

| Parameter | Meaning |
|-----------|---------|
| `n_electrons` | Target electron number the trial filling is tuned to. |
| `dt`, `nt` | Imaginary-time step and slice count; `β = nt·dt`. |
| `mu_values` | The grid of Hamiltonian chemical potentials to scan. |
| `wtype` | Spin symmetry (`collinear` / `noncollinear`). |
| `atom_chkfile` | Filename for the shared PySCF chkfile. |

The remaining Monte Carlo knobs (`sweeps`, `n_walkers_per_mpi_task`, etc.) are
set inside `build_json` in `setup.py`; see the
[`single_mu` README](../../models/4x4_U4/single_mu/README.md) for what each does.

### Step 2 — run

Copy the submission script and the `safire` executable into each run directory
(or point `run.sh` at your build) and submit:

```bash
cd Beta10/mu_m0_5
cp /path/to/run.sh .
cp /path/to/safire .        # or give path to safire
sbatch run.sh
cd -
```

When a job finishes its directory contains the usual `safire` outputs:
`ftafqmc.s000.scalar.dat` (energies), `ftafqmc.s000.stat.h5` (RDMs), and `out`
(log).

### Step 3 — analyze

`analysis.py` takes a single run directory as its argument and prints
the total, interaction, and kinetic energies plus the total density:

```bash
python ../analysis.py Beta10/mu_m0_5
```

Run it once per `μ` directory (or wrap it in a shell loop) to build up the scan.

---

## 4. Changing parameters

Edit the parameter block at the top of `setup.py` for the system and
temperature, and `mu_values` for the scan grid. The set up should work with a different molecule
or basis, just edit the `gto.M(...)` call in `build_mean_field`. Other Monte Carlo
settings can be changed in `build_json` (or directly in each `ftafqmc.json`), as
described in the [`single_mu` README](../../models/4x4_U4/single_mu/README.md).
