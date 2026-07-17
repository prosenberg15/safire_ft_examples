#!/bin/bash
## ---------------------------------------------------------------------------
## Slurm submission script for a single finite-T AFQMC (safire) calculation.
## Copy this file into a run directory (e.g. example_mu_-1_0/) together with
## the `safire` executable, then submit with:  sbatch run.sh
## ---------------------------------------------------------------------------

## Computing resources
#SBATCH --nodes=1 --exclusive
#SBATCH --ntasks-per-node=64      ## for the [ccq], [gen] and [preempt] subclusters

## Partition and subcluster
#SBATCH --partition=ccq
#SBATCH -C ib-icelake

## Wall-clock limit
#SBATCH --time=1:00:00

## Job name and log files
#SBATCH --job-name=ftafqmc_example
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

## --- Software environment -------------------------------------------------
## These must match the modules the `safire` executable was compiled against.
module swap modules modules/2.4-20250724
module load slurm
module load cmake/3.31.6
module load gcc/13.3.0
module load openmpi/4.1.8
module load hdf5/mpi-1.12.3
module load intel-oneapi-mkl/2024.2.2
module load boost/1.87.0
module load python/3.11.11

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

## --- Run ------------------------------------------------------------------
## `safire` is compiled elsewhere; stage a copy (or symlink) into this dir.
mpirun /mnt/home/prosenberg/Develop/SAFIRE/build/cpu/bin/safire --filenames ftafqmc.json &> out
