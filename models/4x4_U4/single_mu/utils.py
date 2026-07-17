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

def buildHubbardmat(Lx,Ly):
    t = 1.0
    Ns = Lx*Ly
    H = np.zeros((Ns,Ns))

    for i in range(Ns):
        # +y-direction
        # top row
        if(i>=Lx*(Ly-1)):
            H[i,i-Lx*(Ly-1)] = H[i,i-Lx*(Ly-1)] -t
        else:
            H[i,i+Lx] = H[i,i+Lx] -t
        # -y-direction        
        # bottom row
        if(i<=Lx-1):
            H[i,i+Lx*(Ly-1)] = H[i,i+Lx*(Ly-1)] -t
        else:
            H[i,i-Lx] = H[i,i-Lx] -t
            
        # +x-direction
        # right-most column
        if((i+1)%Lx==0):
            H[i,i-(Lx-1)] = H[i,i-(Lx-1)] -t
        else:
            H[i,i+1] = H[i,i+1] -t
        # -x-direction
        # left-most column
        if(i%Lx==0):
            H[i,i+Lx-1] = H[i,i+Lx-1] -t
        else:
            H[i,i-1] = H[i,i-1] -t

    return H

def buildHubbardmat(Lx,Ly,kpt=[0.0,0.0]):
    t = 1.0
    Ns = Lx*Ly
    H = np.zeros((Ns,Ns),dtype=complex)

    for i in range(Ns):
        # +y-direction
        # top row
        if(i>=Lx*(Ly-1)):
            H[i,i-Lx*(Ly-1)] = H[i,i-Lx*(Ly-1)] -t*np.exp(1j*kpt[1]/Ly)
        else:
            H[i,i+Lx] = H[i,i+Lx] -t*np.exp(1j*kpt[1]/Ly)
        # -y-direction        
        # bottom row
        if(i<=Lx-1):
            H[i,i+Lx*(Ly-1)] = H[i,i+Lx*(Ly-1)] -t*np.exp(-1j*kpt[1]/Ly)
        else:
            H[i,i-Lx] = H[i,i-Lx] -t*np.exp(-1j*kpt[1]/Ly)
            
        # +x-direction
        # right-most column
        if((i+1)%Lx==0):
            H[i,i-(Lx-1)] = H[i,i-(Lx-1)] -t*np.exp(1j*kpt[0]/Lx)
        else:
            H[i,i+1] = H[i,i+1] -t*np.exp(1j*kpt[1]/Lx)
        # -x-direction
        # left-most column
        if(i%Lx==0):
            H[i,i+Lx-1] = H[i,i+Lx-1] -t*np.exp(-1j*kpt[0]/Lx)
        else:
            H[i,i-1] = H[i,i-1] -t*np.exp(-1j*kpt[0]/Lx)

    return H

def buildHubbardmat_soc(Lx, Ly, kpt=[0.0, 0.0], lam=0.0, soc='rashba'):
    """
    Build the 2*Ns x 2*Ns hopping matrix for the square lattice Hubbard
    model with Rashba or Dresselhaus spin-orbit coupling.

    Basis ordering (block form):
        [ H_up_up    H_up_dn ]
        [ H_dn_up    H_dn_dn ]
    each block Ns x Ns, Ns = Lx*Ly.

    Spin-diagonal blocks: ordinary nearest-neighbor hopping (-t).
    Spin-off-diagonal blocks: SOC spin-flip hopping with amplitude lam.

    Rashba (continuum H_R = lam (sigma_x k_y - sigma_y k_x)):
        +x bond:  -i lam sigma_y      -x bond:  +i lam sigma_y
        +y bond:  +i lam sigma_x      -y bond:  -i lam sigma_x

    Dresselhaus (H_D = lam (sigma_x k_x - sigma_y k_y)):
        +x bond:  -i lam sigma_x      -x bond:  +i lam sigma_x
        +y bond:  +i lam sigma_y      -y bond:  -i lam sigma_y

    Parameters
    ----------
    Lx, Ly : int
    kpt    : [kx, ky] twist/boundary phases
    lam    : SOC strength
    soc    : 'rashba' or 'dresselhaus'

    Returns
    -------
    H : (2*Ns, 2*Ns) complex ndarray (Hermitian)
    """
    t = 1.0
    Ns = Lx * Ly

    # Spin-diagonal hopping (your existing single-spin builder)
    h = buildHubbardmat(Lx, Ly, kpt)

    H = np.zeros((2 * Ns, 2 * Ns), dtype=complex)
    H[0:Ns,    0:Ns]    = h   # up-up
    H[Ns:2*Ns, Ns:2*Ns] = h   # dn-dn

    # Pauli matrix spin components used by each bond direction.
    # We assemble the four spin-block entries explicitly:
    #   uu, ud, du, dd  for a given Pauli matrix * (i * sign)
    #
    # sigma_x = [[0,1],[1,0]]   -> (ud, du) = (1, 1)
    # sigma_y = [[0,-i],[i,0]]  -> (ud, du) = (-i, i)
    #
    # The hopping carries a factor (i * lam * sign) * sigma, where
    # 'sign' is +1 for +x/+y and -1 for -x/-y.

    if soc == 'rashba':
        # +x: -i lam sigma_y ; +y: +i lam sigma_x
        sx_dir = ('y', -1.0)   # for x-bonds: sigma_y, prefactor -1
        sy_dir = ('x', +1.0)   # for y-bonds: sigma_x, prefactor +1
    elif soc == 'dresselhaus':
        # +x: -i lam sigma_x ; +y: +i lam sigma_y
        sx_dir = ('x', -1.0)
        sy_dir = ('y', +1.0)
    else:
        raise ValueError("soc must be 'rashba' or 'dresselhaus'")

    def pauli_entries(which):
        # returns (ud, du) entries of the Pauli matrix
        if which == 'x':
            return 1.0, 1.0
        elif which == 'y':
            return -1j, 1j

    # off-diagonal spin blocks
    Hud = np.zeros((Ns, Ns), dtype=complex)  # up-dn
    Hdu = np.zeros((Ns, Ns), dtype=complex)  # dn-up

    xwhich, xpref = sx_dir
    ywhich, ypref = sy_dir
    ud_x, du_x = pauli_entries(xwhich)
    ud_y, du_y = pauli_entries(ywhich)

    for i in range(Ns):
        # ----- x-direction bonds -----
        # +x neighbor (with periodic wrap and twist phase, matching buildHubbardmat)
        if (i + 1) % Lx == 0:
            jp = i - (Lx - 1)
            phase_p = np.exp(1j * kpt[0] / Lx)
        else:
            jp = i + 1
            phase_p = np.exp(1j * kpt[0] / Lx)
        # -x neighbor
        if i % Lx == 0:
            jm = i + Lx - 1
            phase_m = np.exp(-1j * kpt[0] / Lx)
        else:
            jm = i - 1
            phase_m = np.exp(-1j * kpt[0] / Lx)

        # amplitude = i * lam * xpref * sign(direction) * phase
        amp_p = 1j * lam * xpref * (+1.0) * phase_p   # +x
        amp_m = 1j * lam * xpref * (-1.0) * phase_m   # -x
        Hud[i, jp] += amp_p * ud_x
        Hdu[i, jp] += amp_p * du_x
        Hud[i, jm] += amp_m * ud_x
        Hdu[i, jm] += amp_m * du_x

        # ----- y-direction bonds -----
        # +y neighbor
        if i >= Lx * (Ly - 1):
            jp = i - Lx * (Ly - 1)
            phase_p = np.exp(1j * kpt[1] / Ly)
        else:
            jp = i + Lx
            phase_p = np.exp(1j * kpt[1] / Ly)
        # -y neighbor
        if i <= Lx - 1:
            jm = i + Lx * (Ly - 1)
            phase_m = np.exp(-1j * kpt[1] / Ly)
        else:
            jm = i - Lx
            phase_m = np.exp(-1j * kpt[1] / Ly)

        amp_p = 1j * lam * ypref * (+1.0) * phase_p   # +y
        amp_m = 1j * lam * ypref * (-1.0) * phase_m   # -y
        Hud[i, jp] += amp_p * ud_y
        Hdu[i, jp] += amp_p * du_y
        Hud[i, jm] += amp_m * ud_y
        Hdu[i, jm] += amp_m * du_y

    H[0:Ns,    Ns:2*Ns] = Hud
    H[Ns:2*Ns, 0:Ns]    = Hdu

    return H

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


