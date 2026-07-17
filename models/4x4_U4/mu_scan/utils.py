import numpy as np
from scipy.linalg import inv, det
from scipy.linalg.lapack import zgetrf, zgetri

def to_complex(array:np.array):
    """Convert from numpy.complex128 to internal complex format                                                                                                                   
    """
    if array.dtype != np.complex128:
        array = array.astype(np.complex128,casting='same_kind')
    shape = array.shape
    return np.ascontiguousarray(array).view(np.float64).reshape(shape+(2,))

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


