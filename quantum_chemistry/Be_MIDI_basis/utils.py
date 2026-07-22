# utility functions to set up calculations

import numpy as np
import h5py as h5
import scipy.sparse as sps
from scipy.linalg import eigh, inv, det
from scipy.linalg.lapack import zgetrf, zgetri

def to_complex(array:np.array):
    """Convert from numpy.complex128 to internal complex format                                                                                                                   
    """
    if array.dtype != np.complex128:
        array = array.astype(np.complex128,casting='same_kind')
    shape = array.shape
    return np.ascontiguousarray(array).view(np.float64).reshape(shape+(2,))

def splitDmat(D,Dscale,NT=0,mu=0,dtau=0):
    Dmin = np.zeros(len(D))
    Dmax = np.zeros(len(D))

    ksi = np.log(D.real) + Dscale.real - NT*dtau*mu
    for i in range(len(D)):
        if(ksi[i] > 0):
            Dmin[i] = 1.0
            if(ksi[i] >= 32*np.log(10.0)):
                Dmax[i] = 0.0
            else:
                Dmax[i] = np.exp(-ksi[i])
        else:
            Dmax[i] = 1.0
            if(ksi[i] <= -32*np.log(10.0)):
                Dmin[i] = 0.0
            else:
                Dmin[i] = np.exp(ksi[i])

    return Dmin, Dmax
    
def computeG(UR,DR,VR,UL,DL,VL,sclL,sclR,muL,muR):
    DRmin, DRmax_inv = splitDmat(DR,sclR)
    DLmin, DLmax_inv = splitDmat(DL,sclL)

    DRmax = np.zeros(len(DRmax_inv))
    DLmax = np.zeros(len(DLmax_inv))    
    for i in range(len(DLmax_inv)):
        if(DRmax_inv[i] > 1e-8):
            DRmax[i] = 1.0/DRmax_inv[i]
        if(DLmax_inv[i] > 1e-8):
            DLmax[i] = 1.0/DLmax_inv[i]
    
    A = np.dot(np.dot(np.diag(DRmax_inv),np.dot(inv(UR),inv(UL))),np.diag(DLmax_inv))
    B = np.dot(np.dot(np.diag(DRmin),np.dot(VR,VL)),np.diag(DLmin))
    
    T, Piv, info = zgetrf(A+B)
    invAB, info = zgetri(T, Piv)

    logdetUR = np.log(det(UR)+1j*0)
    logdetUL = np.log(det(UL)+1j*0)
    
    logdetAB = 0.0
    logdetDLmax = 0.0
    logdetDRmax = 0.0
    for i in range(T.shape[0]):
        logdetDLmax = logdetDLmax + np.log(DLmax[i])
        logdetDRmax = logdetDRmax + np.log(DRmax[i])
        if(Piv[i] == i):
            logdetAB = logdetAB + np.log(T[i,i])
        else:
            logdetAB = logdetAB + np.log(-T[i,i])

    ptl = logdetAB+logdetDLmax+logdetDRmax + logdetUR + logdetUL

    G = np.dot(np.dot(np.dot(inv(UL),np.diag(DLmax_inv)),invAB),np.dot(np.diag(DRmax_inv),inv(UR)))
    ptl2 = det(np.dot(UR,np.diag(DRmax)))*det(A+B)*det(np.dot(np.diag(DLmax),UL))
    
    return np.eye(G.shape[0])-np.transpose(G), ptl

def tune_mu_trial(target_dens,beta,UL,DL,VL,print_iters=False,wtype='collinear'):
    Ns = UL.shape[0]
    UR = np.eye(Ns)
    DR = np.ones(Ns)
    VR = np.eye(Ns)
    tol = 1e-3
    dens = -100.0
    direc = 1
    old_direc = 1
    dmuT = 0.5
    old_muT  = 0.0
    new_muT  = 0.0
    niter = 0
    nitermax = 200
    DL0 = DL
    diff = abs(dens-target_dens)
    while diff > tol and niter < nitermax:
        DL = DL0*np.exp(-beta*np.array(Ns*[new_muT]))
        G, pt = computeG(UR,DR,VR,UL,DL,VL,0.0,0.0,0.0,0.0)
        if(wtype=='collinear'):
            dens = 2*(np.trace(G).real/Ns) #factor of 2 assumes up = down
        elif(wtype=='noncollinear' or wtype=='non-collinear'):
            dens = (np.trace(G).real/Ns)
        diff = abs(dens-target_dens)        
        if(print_iters):
            print(f"iter {niter+1}")
            print(f"=======================================")
            print(f"muT = {new_muT:7.5f},  dens = {dens:7.5f},  diff = {dens-target_dens:7.5f}\n")
        if(dens > target_dens):
            direc = 1
        else:
            direc = -1
        if(direc != old_direc and niter !=0):
            dmuT = dmuT/2.0
        old_direc = direc
        old_muT = new_muT
        new_muT = old_muT + direc*dmuT
        niter = niter + 1

    return old_muT

# ----------------------------------------------------------------------
# Trial-writer helpers
# ----------------------------------------------------------------------
def write_sparse_mat(f, path, mat):
    f.create_dataset(f"{path}/dims", data=[mat.shape[0], mat.shape[1], mat.nnz])
    f.create_dataset(f"{path}/data_", data=to_complex(mat.data))
    f.create_dataset(f"{path}/jdata_", data=mat.indices)
    f.create_dataset(f"{path}/pointers_begin_", data=mat.indptr[:-1])
    f.create_dataset(f"{path}/pointers_end_", data=mat.indptr[1:])


def write_ft_wfn_mol(mf, nt, dt, mu=0.0, wtype='collinear',
                     tune_mu=False, target_dens=0.0,
                     print_mu_loop=False, print_nT=False, basis='mo'):
    """
    Finite-T trial (UL, DL, VL), built from a molecular mean field `mf`.

    basis : 'mo'  -> trial H is diag(mf.mo_energy)  (consistent with afqmc.h5)
            'oao' -> trial H is the Fock matrix in the orthonormalized AO basis
    """
    beta = nt * dt

    # --- Build the trial one-body Hamiltonian H (one spin block) ---
    if basis == 'mo':
        H0 = np.diag(mf.mo_energy)
    elif basis == 'oao':
        Sloc = mf.get_ovlp()
        F = mf.get_fock()
        s, U = eigh(Sloc)
        X = U @ np.diag(1.0 / np.sqrt(s)) @ U.conj().T   # S^{-1/2}
        H0 = X.conj().T @ F @ X
    else:
        raise ValueError("basis must be 'mo' or 'oao'")

    nb1 = H0.shape[0]
    if wtype.lower() == 'noncollinear':
        nbasis = 2 * nb1
        H = np.zeros((nbasis, nbasis), dtype=H0.dtype)
        H[:nb1, :nb1] = H0
        H[nb1:, nb1:] = H0
        nspin = 1
    else:  # collinear
        nbasis = nb1
        H = H0
        nspin = 2

    # --- Eigendecomposition: B = exp(-tau H) = VL DL UL, VL=M, UL=M^dagger ---
    lbda, M = eigh(H)
    ksi = -0.5 * (lbda.min() + lbda.max())   # spectral centering for stability

    UL = np.conj(np.transpose(M))
    VL = M
    UR = np.eye(nbasis)
    DR = np.eye(nbasis)
    VR = np.eye(nbasis)

    # --- Tune mu_T against the un-shifted reference, or use mu directly ---
    if tune_mu:
        DL0 = np.exp(-beta * (lbda + ksi))
        muT = tune_mu_trial(target_dens, beta, UL, DL0, VL, print_mu_loop)
    else:
        muT = mu

    # --- Time-resolved DL over slices 0..nt ---
    DL = np.zeros((nt + 1, nbasis))
    for i in range(nt + 1):
        DL[i, :] = np.exp(-(beta - i * dt) * (lbda + ksi + muT))

    if print_nT:
        G, pt = computeG(UR, DR.diagonal(), VR, UL, DL[0, :], VL,
                         0.0, 0.0, 0.0, 0.0)
        dens = (np.trace(G) / nb1).real
        if wtype.lower() == 'collinear':
            dens = 2 * dens
        print(f"    filling_trial = {dens:.6f}  (n_elec ~ {dens * nb1:.4f})")

    UL = sps.csr_matrix(UL)
    DL = sps.csr_matrix(DL)
    VL = sps.csr_matrix(VL)

    beta_str = str(beta).replace(".", "_")
    muT_str = str(muT).replace(".", "_").replace("-", "m")
    fname = f"wfn_be_midi_Beta{beta_str}_nt{nt}_muT{muT_str}.h5"

    with h5.File(fname, "w") as f:
        f.create_group("Wavefunction/NOMSD")
        f.create_dataset("Wavefunction/NOMSD/UR_alpha", data=to_complex(UR))
        f.create_dataset("Wavefunction/NOMSD/DR_alpha", data=to_complex(DR))
        f.create_dataset("Wavefunction/NOMSD/VR_alpha", data=to_complex(VR))
        if nspin > 1:
            f.create_dataset("Wavefunction/NOMSD/UR_beta", data=to_complex(UR))
            f.create_dataset("Wavefunction/NOMSD/DR_beta", data=to_complex(DR))
            f.create_dataset("Wavefunction/NOMSD/VR_beta", data=to_complex(VR))
        for i in range(nspin):
            for name, mat in (("UL", UL), ("DL", DL), ("VL", VL)):
                write_sparse_mat(f, f"Wavefunction/NOMSD/{name}_{i}", mat)
        wfn_value = 2 if wtype.lower() == 'collinear' else 3
        f.create_dataset("Wavefunction/NOMSD/dims",
                         data=np.array([nb1, nt + 1, 0, wfn_value, 1],
                                       dtype=np.int_))
        f.create_dataset("Wavefunction/NOMSD/ci_coeffs",
                         data=to_complex(np.array([1.0], dtype=complex)))
    return fname
