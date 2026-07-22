#!/bin/bash

## The computing resources we need
#SBATCH --nodes=1 --exclusive
#SBATCH --ntasks-per-node=64     ## For the [ccq], [gen] and [preempt] subcluster 

## The submitted partition of the job
#SBATCH --partition=ccq

## The subclusters on the rusty
#SBATCH -C ib-icelake

## The limited computation time of the job
#SBATCH --time=5:00:00

## The name of present job
#SBATCH --job-name=run_full_test

## The standard output for the job
#SBATCH --output=slurm-%j.out

## The Error output file for the job
#SBATCH --error=slurm-%j.err

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

mpirun ./safire --filenames ftafqmc.json &> out
