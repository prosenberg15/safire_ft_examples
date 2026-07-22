# SAFIRE finite-T examples and notes

This repository contains examples demonstrating how to run simple finite-T calculations
with the SAFIRE code. A good starting point is `models/4x4_U4/single_mu`, which explains the workflow from setup to analysis for a calculation on the 4x4 Hubbard model.

The 12x12 model directory already contains example results, as well as benchmarks from the existing FT-AFQMC code. Calculations for more complicated models can be set up
in the same manner as the model calculations demonstrated here. The trial wavefunction could simply be constructed with the one-body term
of the Hamiltonian.

The quantum chemistry example comes from the trivial benchmark on the Be atom in https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.8b00569. There are additional ED
results in the supplementary material of that paper for future benchmarks that can be set up using the same machinery demonstrated here.

#
# TO-DO

1. Polish tools to set up calculations (models, solids, and quantum chemistry), incorporate into afqmctools where appropriate
2. Test more models, more complicated models, & quantum chemistry, solids workflow
3. Performance testing CPU/GPU
4. Find better solution for stabilization routine on GPU:
	- `orthogonalize_wQR` --> copies data to host for GPU runs
	-  batched QRP, SVD not implemented or efficient on GPU
5. Find better solution for half-rotated Green's functions at finite-T:
	- For GS calculations, the trial wavefunction is needed to compute the full Green's  function from the half-rotated Green's function
	- At finite-T the full Green's function is always computed directly 
	- `WavefunctionFactory`passes an array of identity matrices to`getHamiltonianOperations` in place of a trial wavefunction to reuse existing code, this should be replaced with something that avoids the unnecessary contractions
6. Add routines to average measurements at different imaginary times, compute dynamical Green's functions
