# Finite-temperature AFQMC: benchmark 4x4 U=4

This directory extends the [single-point example](../single_mu/README.md) to a
**scan over chemical potential** `μ`. For each `μ` it writes a full set of
`safire` inputs into its own run directory, (optionally) submits the jobs,
collects the results into a table, and plots them against reference
determinant QMC (DQMC) data.

The setup allows you to reproduce the benchmark calculations presented in Fig. 1 of
[Phys. Rev. B 99 045108](https://journals.aps.org/prb/pdf/10.1103/PhysRevB.99.045108):
the **4×4 square-lattice Hubbard model** at `U = 4t`, scanned across `μ` at two
inverse temperatures, `β = 2` and `β = 5`.

If this is your first time with these examples, start with the
[`single_mu` README](../single_mu/README.md) — it covers installing `safire`
and `afqmctools`, the meaning of several important parameters in the `safire` input file,
`ftafqmc.json`, as well as the input-file helpers (`write_ft_wfn`,
`write_hubbard_ham_ft`), and the form of the trial wavefunction.

---

## 1. Files

| File | Role |
|------|------|
| `generate_benchmarks.py` | Loops over `μ`, writing inputs into `Beta{β}/mu_{μ}/` and optionally submitting the jobs. This is the scan analogue of `example_single_mu.py`. |
| `analysis.py` | Collects every `mu_*/` run in a `Beta{β}/` directory into a single `props_vs_mu` table. |
| `plot_benchmarks.py` | Plots the FT-AFQMC tables against the bundled DQMC reference data. |
| `setup_benchmarks.py`, `utils.py` | Input-file helpers, same roles as in `single_mu` (see that README). |
| `run.sh` | Example Slurm submission script for Rusty. |
| `props_vs_mu_Beta2_DQMC`, `props_vs_mu_Beta5_DQMC` | Reference DQMC data for `β = 2` and `β = 5`. |

---

## 2. Workflow

```bash
# edit parameters at the top of generate_benchmarks.py to target correct physical system
# e.g. for β = 2, with nt = 200, dt = 0.01, tune_mu = False, the workflow is as follows:
python generate_benchmarks.py   # write inputs for all mu values, optionally submit jobs
# once runs have concluded
cd Beta2
python ../analysis.py              # collect results into props_vs_mu
cd ..
python plot_benchmarks.py       # plot vs. DQMC reference
```
The workflow for `β = 5` is similar, for example just set `nt = 500`, `dt = 0.01`. To match the results from [Phys. Rev. B 99 045108](https://journals.aps.org/prb/pdf/10.1103/PhysRevB.99.045108), set `tune_mu = True`, with the target densities already defined in the `generate_benchmarks.py`.  

### Step 1 — generate (and optionally submit)

```bash
python generate_benchmarks.py
```

The parameter block at the top of `generate_benchmarks.py` applies to **every**
`μ` in the scan; the individual points come from the `mu_targets` list of
`(μ, target_density)` pairs. For each point the script creates

```
Beta{β}/mu_{μ}/
├── wfn_collinear_ft_*.h5   # trial wavefunction
├── hamil_4x4_U4_*.h5       # Hubbard Hamiltonian
├── ftafqmc.json            # safire input file
└── run.sh                  # copied submission script
```

where `β = nt·dt` (e.g. `nt=500, dt=0.01` → `Beta5/`).

Two flags control the scan:

| Flag | Effect |
|------|--------|
| `tune_mu` | If `True`, the trial chemical potential `muT` is tuned to reproduce each paired `target_density`; if `False`, `muT = μ` and the density column is ignored. |
| `submit_jobs` | If `True`, the script runs `sbatch run.sh` in each run directory automatically. If `False`, submit them yourself. |

As described above, to produce both temperatures for the plot, run the script twice, changing `nt`
so that `β = nt·dt` gives 2 and 5 (e.g. `nt=200` and `nt=500` at `dt=0.01`).

### Step 2 — analyze

Once the runs have finished, collect them into a table. `analysis.py` globs the
`mu_*/` directories in the **current** working directory, so run it from inside
each `Beta{β}/` directory:

```bash
cd Beta5
python ../analysis.py
cd ..
```

This reads each run's `ftafqmc.s000.scalar.dat` (energies) and
`ftafqmc.s000.stat.h5` (1- and 2-body RDMs) and writes a `props_vs_mu` table
with, per `μ`: total / interaction / kinetic energy, density, nearest-neighbor
density-density correlation, and nearest-neighbor spin-spin correlation (each
with error bars).

### Step 3 — plot

```bash
python plot_benchmarks.py
```

This reads `Beta2/props_vs_mu` and `Beta5/props_vs_mu`, overlays the DQMC
reference files, and writes `4x4_U4_FTAFQMC_vs_DQMC.png` (energy, density, and
the two correlators, with relative-error insets). The dataset list and the
`MU_SHIFT` (the constant offset between the `safire` and DQMC `μ` conventions)
are set at the top of the script — adjust them if you change which temperatures
you run.

---

## 3. Changing parameters

Similarly to the single `μ` case, the parameters can be changed at the top of
`generate_benchmarks.py`. See the [`single_mu` README](../single_mu/README.md) for the list and the meaning of the relevant parameters. This script also includes a list of chemical potentials and target densities, `mu_targets`, and a flag to tune the trial chemical potential, `tune_mu`, as well as a flag that allows you to submit jobs automatically, `submit_jobs`.
