import numpy as np
import h5py as h5
from utils import to_complex, computeG, tune_mu_trial
from scipy.linalg import eigh
import scipy.sparse as sps
from afqmctools.systems.lattice import get_lattice
from afqmctools.hamiltonian.model.builder import HamiltonianBuilder
from afqmctools.hamiltonian.model.ham_class import HamiltonianComponent, SpinSymm
import afqmctools.utils.io as io
from enum import IntEnum

class SymType(IntEnum):
    """Spin symmetry of the wavefunction/Hamiltonian. The integer value is what
    safire expects in the wavefunction `dims` field; use from_str to convert the
    'collinear'/'noncollinear' strings used throughout the drivers."""
    COLLINEAR = 2
    NONCOLLINEAR = 3

    @classmethod
    def from_str(cls, name: str) -> "SymType":
        """Map 'collinear'/'noncollinear' (any case) to the enum member."""
        try:
            return cls[name.strip().upper()]
        except KeyError:
            valid = ", ".join(m.name.lower() for m in cls)
            raise ValueError(
                f"unknown spin symmetry {name!r}, expected one of: {valid}"
            ) from None
    
def write_sparse_mat(f, path, mat):
    """Write a scipy CSR matrix into HDF5 group `path` in safire's sparse layout
    (dims, complex data, column indices, and begin/end row pointers)."""
    f.create_dataset(f"{path}/dims", data=[mat.shape[0], mat.shape[1], mat.nnz])
    f.create_dataset(f"{path}/data_", data=to_complex(mat.data))
    f.create_dataset(f"{path}/jdata_", data=mat.indices)
    f.create_dataset(f"{path}/pointers_begin_", data=mat.indptr[:-1])
    f.create_dataset(f"{path}/pointers_end_", data=mat.indptr[1:])

def write_ft_wfn(nt: int,
                 dt: float,
                 one_body,
                 muT: float = 0.0,
                 wtype: str = 'collinear',
                 tune_mu: bool = False,
                 target_dens: float=0.0,
                 print_mu_loop: bool = False,
                 print_nT: bool = False) -> str:

    """Write a finite-temperature trial wavefunction to an HDF5 file.

    The trial is the non-interacting (U=0) density matrix of the Hubbard
    tight-binding Hamiltonian, stored in the U*D*V factored form safire reads
    (the "NOMSD" group). It is built once and reused as the importance-sampling
    trial for the finite-T AFQMC run.

    Parameters
    ----------
    nt : int
        Number of imaginary-time slices. The wavefunction stores nt+1 diagonal
        factors D (one per slice).
    dt : float
        Imaginary-time step. Together with nt this sets the inverse temperature
        beta = nt*dt. Lower the temperature by increasing nt (keeping dt fixed
        and small enough that the Trotter error is acceptable); raise it by
        decreasing nt.
    one_body : scipy.sparse matrix or numpy.ndarray
        The non-interacting one-body (hopping) matrix used to build the trial,
        and the source of the system sizes (nsites, nbasis) inferred from its
        shape. Pass the one-body term produced by afqmctools' HamiltonianBuilder
        (e.g. ``builder.hamiltonian.get_one_body()``, or the second return value
        of write_hubbard_ham_ft) so the trial uses the same hopping as the
        Hamiltonian. This must be the hopping-only term *without* any mu shift,
        since muT is applied separately in the imaginary-time weighting.
        Expected in the builder's spin layout: (2*nsites, nsites) for collinear
        (up block stacked on down block; a plain (nsites, nsites) matrix is also
        accepted) or (2*nsites, 2*nsites) for noncollinear (block diagonal).
    muT : float, optional
        Trial chemical potential used to weight the imaginary-time propagator
        when tune_mu is False. Ignored when tune_mu is True.
    wtype : {'collinear', 'noncollinear'}, optional
        Spin symmetry. 'collinear' uses an nsites basis with equal up/down
        channels; 'noncollinear' uses the full 2*nsites basis. Must match the
        htype passed to write_hubbard_ham_ft and the walker_type in the JSON.
    tune_mu : bool, optional
        If True, solve for the trial chemical potential muT that reproduces
        target_dens (via utils.tune_mu_trial) instead of using mu.
    target_dens : float, optional
        Target filling for the mu search (only used when tune_mu is True). For
        collinear this is the total density per site (0..2; 1.0 = half filling).
    print_mu_loop : bool, optional
        Print each iteration of the tune_mu bisection search.
    print_nT : bool, optional
        Print the resulting trial density ntot_trial as a sanity check.

    Returns
    -------
    str
        The name of the HDF5 file written, encoding nsites, beta, nt and muT,
        e.g. ``wfn_collinear_ft_N16_Beta2_0_nt100_muT-1_0.h5``. This name is
        passed straight into build_json.
    """
    beta = nt*dt

    wfn_type = SymType.from_str(wtype)

    # Trial H from the non-interacting (hopping) one-body matrix produced by
    # afqmctools. Densify it if sparse, derive the system sizes from its shape,
    # and reduce it to the nbasis x nbasis block that eigh expects. In the
    # builder's native layouts the number of columns equals nbasis (the trial
    # dimension): collinear is (2*nsites, nsites), noncollinear is
    # (2*nsites, 2*nsites).
    H1 = one_body.toarray() if sps.issparse(one_body) else np.asarray(one_body)

    if wfn_type is SymType.NONCOLLINEAR:
        if H1.shape[0] != H1.shape[1] or H1.shape[1] % 2 != 0:
            raise ValueError(
                f"one_body has shape {H1.shape}, expected a square "
                "(2*nsites, 2*nsites) matrix for noncollinear"
            )
        nbasis = H1.shape[1]
        nsites = nbasis // 2
        H = H1
    else:  # COLLINEAR
        nsites = H1.shape[1]
        nbasis = nsites
        if H1.shape[0] not in (nsites, 2*nsites):
            raise ValueError(
                f"one_body has shape {H1.shape}, expected "
                f"({2*nsites}, {nsites}) or ({nsites}, {nsites}) for collinear"
            )
        # collinear stacks the up block on the down block; the trial uses a
        # single nsites channel, so take the up block
        H = H1[:nsites, :]

    UR = np.eye(nbasis)
    DR = np.eye(nbasis)
    VR = np.eye(nbasis)

    lbda, M = eigh(H)

    ksi = -0.5*(lbda.min()+lbda.max())

    UL = np.conj(np.transpose(M))
    DL = np.zeros((nt+1,nbasis))
    VL = M
    
    if tune_mu:
        DL0 = np.exp(-beta*(lbda+ksi))
        muT = tune_mu_trial(target_dens,beta,UL,DL0,VL,print_mu_loop)
    for i in range(nt+1):
        DL[i,:] = np.exp(-(beta-i*dt)*(lbda+ksi+muT))

    if print_nT:
        G, pt = computeG(UR,DR.diagonal(),VR,UL,DL[0,:],VL,0.0,0.0,0.0,0.0)

        ntot = (np.trace(G)/nsites).real
        if(wfn_type is SymType.COLLINEAR):
            ntot = 2*ntot
        
        print(f"ntot_trial = {ntot}")

    UL = sps.csr_matrix(UL)
    DL = sps.csr_matrix(DL)
    VL = sps.csr_matrix(VL)

    beta_str = str(beta).replace(".","_")
    muT_str = str(muT).replace(".","_")
    fname = f"wfn_{wfn_type.name.lower()}_ft_N{nsites}_Beta{beta_str}_nt{nt}_muT{muT_str}.h5"
    print(f"wavefunction file: {fname}")
    
    with h5.File(fname,"w") as f:
        f.create_group("Wavefunction/NOMSD")
        
        nspin = 2 if wfn_type is SymType.COLLINEAR else 1
        f.create_dataset("Wavefunction/NOMSD/UR_alpha",data=to_complex(UR))
        f.create_dataset("Wavefunction/NOMSD/DR_alpha",data=to_complex(DR))
        f.create_dataset("Wavefunction/NOMSD/VR_alpha",data=to_complex(VR))

        if nspin > 1:
            f.create_dataset("Wavefunction/NOMSD/UR_beta",data=to_complex(UR))
            f.create_dataset("Wavefunction/NOMSD/DR_beta",data=to_complex(DR))
            f.create_dataset("Wavefunction/NOMSD/VR_beta",data=to_complex(VR))

        for i in range(nspin):
            for name, mat in (("UL", UL), ("DL", DL), ("VL", VL)):
                write_sparse_mat(f, f"Wavefunction/NOMSD/{name}_{i}", mat)
        
        f.create_dataset("Wavefunction/NOMSD/dims",data=np.array([nsites,nt+1,0,wfn_type.value,1],dtype=np.int_))
        f.create_dataset("Wavefunction/NOMSD/ci_coeffs",data=to_complex(np.array([1.0],dtype=complex)))

    return fname
        
def write_hubbard_ham_ft(Lx: int,
                         Ly: int,
                         U:  float,
                         mu: float = 0.0,
                         htype: str = 'collinear',
                         hst_type: str = 'discrete_spin'):
    """Write the Hubbard Hamiltonian to an HDF5 file for finite-T calculations.

    Builds a periodic square-lattice Hubbard model with afqmctools and writes it
    in the format safire reads. The chemical potential is folded into the
    one-body term as a uniform diagonal shift.

    Parameters
    ----------
    Lx, Ly : int
        Lattice dimensions; nsites = Lx*Ly. Periodic boundaries (PBC) are used
        in both directions (see the get_lattice call to change this or switch to
        another lattice, e.g. honeycomb).
    U : float
        On-site Hubbard repulsion, in units of the hopping t. This is the main
        interaction knob.
    mu : float, optional
        Chemical potential, added as U_shift = mu * I to the one-body matrix.
    htype : {'collinear', 'noncollinear'}, optional
        Spin symmetry; must match the wtype used for the wavefunction.

    Returns
    -------
    fname : str
        The name of the HDF5 file written, e.g.
        ``hamil_4x4_U4_0_mu-1_0_collinear.h5``. Passed into build_json.
    one_body : scipy.sparse.csr_array
        The hopping-only one-body matrix (the tij term *before* the mu shift is
        folded in), in the builder's spin layout. Pass this straight into
        write_ft_wfn's one_body argument so the trial uses the same hopping as
        the Hamiltonian. mu is deliberately excluded because write_ft_wfn
        applies muT separately in the imaginary-time weighting.

    Notes
    -----
    - Hopping is set by ``hopping = [1.0, 0.0]`` = [t, t']. Set t' (and further
      neighbors) here for non-nearest-neighbor models.
    - ``nelec`` is fixed to half filling (nsites//2, nsites//2). For finite-T /
      grand-canonical runs this is the reference particle-number sector; filling
      is controlled physically through mu (and beta), not nelec.
    """
    lattice = get_lattice(
        params=dict(
            L1 = Lx,
            L2 = Ly,
            #type = "honeycomb",
            boundary1 = "PBC",
            boundary2 = "PBC"
        )
    )

    nbasis = lattice.N_sites
    nelec = (nbasis//2,nbasis//2)

    ham_type = SymType.from_str(htype)
    spin_symm = SpinSymm[ham_type.name]

    if ham_type is SymType.NONCOLLINEAR:
        nelec  = tuple(2*n for n in nelec)
        nbasis = 2*nbasis
    
    # list of hopping parameters is interpreted as follows:
    #    hopping[0] is 't'
    #    hopping[1] is 't^{prime}'
    #    ....
    #    hopping[n-1] is 't^{n}'
    hopping = [1.0,0.0]

    builder = HamiltonianBuilder(
        lattice=lattice,
        spin_symm=spin_symm
    )

    # add standard Hubbard terms
    builder.nth_neighbor_hopping(hopping)

    # capture the hopping-only one-body (before the mu shift is folded in below)
    # so it can be handed to write_ft_wfn as the trial's one-body matrix
    one_body = builder.hamiltonian.get_one_body()

    builder.onsite_hubbard(U,hst_type=hst_type)
    #builder.onsite_hubbard(U,hst_type="continuous_spin")

    one_body_matrix = sps.eye(nbasis,format="csr") * mu
    if ham_type is SymType.COLLINEAR:
        # the convention is to stack the spin-up hopping matrix on the spin-down hopping matrix
        one_body_matrix = sps.vstack([one_body_matrix,one_body_matrix],format="csr")

    custom_one_body = HamiltonianComponent(
        csr_array=one_body_matrix,
        model_type='one_body',
        spin_symm=spin_symm
    )

    # manually add the custom term
    builder.hamiltonian["tij"] = custom_one_body

    builder.finalize()

    hamiltonian = builder.hamiltonian

    mu_str = str(mu).replace(".","_")
    U_str = str(U).replace(".","_")
    
    fname = f"hamil_{Lx}x{Ly}_U{U_str}_mu{mu_str}_{ham_type.name.lower()}.h5"

    print(f"hamiltonian file: {fname}")

    # write hamiltonian to file
    io.write_model_hamiltonian(
        hamiltonian=hamiltonian,
        fname=fname,
        nelec=nelec
    )

    return fname, one_body
