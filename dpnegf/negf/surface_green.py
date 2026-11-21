import numpy as np
import scipy.linalg as SLA
import logging
import torch

log = logging.getLogger(__name__)

SELF_ENERGY_METHODS = frozenset({
    "Sancho-Rubio", "Lopez-Sancho", "Boundary-Schur", "GEP",
})
SURFACE_GREEN_METHODS = frozenset({"Sancho-Rubio", "Lopez-Sancho", "GEP"})


def validate_self_energy_method(method):
    if not isinstance(method, str) or method not in SELF_ENERGY_METHODS:
        choices = ", ".join(sorted(SELF_ENERGY_METHODS))
        raise ValueError(f"Unknown self-energy solver {method!r}; expected one of: {choices}.")
    return method


def selfEnergy(hL, hLL, sL, sLL, ee, hDL=None, sDL=None, etaLead=1e-8, Bulk=False,
                    E_ref=0.0, dtype=np.complex128, device='cpu', method='Lopez-Sancho', numba_jit=None,
                    h10=None, s10=None, boundary_plan=None):
    '''calculates the self-energy and surface Green's function for a given  Hamiltonian and overlap matrix.

    Parameters
    ----------
    hL
        Hamiltonian matrix for one principal layer in Lead
    hLL
        Hamiltonian matrix between the most nearby principal layers in Lead
    sL
        Overlap matrix for one principal layer in Lead
    sLL
        Overlap matrix between the most nearby principal layers in Lead
    ee
        the given energy
    hDL
        Hamiltonian matrix between the lead and the device.
    sDL
        Overlap matrix between the lead and the device.
    etaLead
        A small imaginary number that is used to avoid the singularity of the surface Green's function.
    Bulk, optional
        Ignore it, please.
    chemiPot
        the chemical potential of the lead.
    dtype
        the data type of the tensors used in the calculations.
    device
        The "device" parameter specifies the device on which the calculations will be performed. It can be
        set to 'cpu' for CPU computation or 'cuda' for GPU computation.
    method
        ``"Sancho-Rubio"``/``"Lopez-Sancho"`` return the full surface Green
        function. ``"Boundary-Schur"`` computes only the contact-projected
        self-energy and returns ``None`` for the Green-function result.
    numba_jit
        A boolean flag that indicates whether to use Numba's Just-In-Time (JIT) compilation for the surface Green's function calculation.
    h10, s10
        Optional precomputed ``conj(hLL.T)`` and ``conj(sLL.T)``. Passed in as
        C-contiguous complex128 numpy arrays from ``_precompute_lead_kdata`` so
        the surface-Green core does not recompute them per energy.

    Returns
    -------
        Two values: self-energy and surface Green's function. The latter is
        ``None`` for ``method="Boundary-Schur"`` because its reduced Green
        function is not the full principal-layer surface Green's function.

    '''
    validate_self_energy_method(method)

    hL = convert_to_numpy(hL)
    sL = convert_to_numpy(sL)
    hLL = convert_to_numpy(hLL)
    sLL = convert_to_numpy(sLL)
    ee = convert_to_numpy(ee)
    if hDL is not None:
        hDL = convert_to_numpy(hDL)
    if sDL is not None:
        sDL = convert_to_numpy(sDL)
    E_ref = convert_to_numpy(E_ref)
    if h10 is not None:
        h10 = convert_to_numpy(h10)
    if s10 is not None:
        s10 = convert_to_numpy(s10)

    if not isinstance(ee, np.ndarray):
        eeshifted = np.array(ee, dtype=dtype) + E_ref
    else:
        eeshifted = ee + E_ref

    eeshifted = eeshifted.item()

    if method == 'Boundary-Schur':
        if hDL is None or sDL is None:
            raise ValueError("Boundary-Schur requires lead-device H/S contacts.")
        from dpnegf.negf.boundary_self_energy import boundary_self_energy

        sigma, _ = boundary_self_energy(
            hL=hL, hLL=hLL, sL=sL, sLL=sLL, hDL=hDL, sDL=sDL,
            energy=ee, E_ref=E_ref, etaLead=etaLead, numba_jit=numba_jit,
            h10=h10, s10=s10, plan=boundary_plan,
        )
        return sigma.to(device=device), None

    if hDL is None:
        ESH = (eeshifted * sL - hL)
        SGF = surface_green(hL, hLL, sL, sLL, eeshifted + 1j * etaLead, method,
                            numba_jit=numba_jit, h10=h10, s10=s10)

        if Bulk:
            Sig = np.linalg.inv(SGF)
        else:
            Sig = ESH - np.linalg.inv(SGF)
    else:
        a, b = hDL.shape
        SGF = surface_green(hL, hLL, sL, sLL, eeshifted + 1j * etaLead, method,
                            numba_jit=numba_jit, h10=h10, s10=s10)

        Sig = (eeshifted*sDL-hDL) @ SGF[:b,:b] @ (eeshifted*sDL.conj().T-hDL.conj().T)

    Sig = torch.tensor(Sig, dtype=torch.complex128, device=device)
    SGF = torch.tensor(SGF, dtype=torch.complex128, device=device)

    return Sig, SGF


_numba_available = False


try:
    from numba import njit

    def _make_surface_green_numba_core(fastmath):
        # Lazy signature: dispatcher compiles one specialization per input
        # layout, so both writable arrays (parent process / tests) and
        # joblib-memmapped readonly arrays (loky workers, see
        # `_pack_lead_matrices` in lead_property.py) match without falling
        # back to the scipy core.
        @njit(fastmath=fastmath)
        def _core(H, h01, S, s01, h10, s10, ee):

            N = H.shape[0]
            alpha = h10 - ee * s10
            beta = h01 - ee * s01
            eps = H.copy()
            epss = H.copy()

            eS = ee * S  # loop invariant

            # preallocated scratch buffer reused every iteration
            RHS = np.empty((N, 2 * N), dtype=np.complex128)

            converged = False
            iteration = 0
            while not converged:
                iteration += 1
                # one LU solve with stacked RHS = [alpha | beta]
                RHS[:, :N] = alpha
                RHS[:, N:] = beta
                sol = np.linalg.solve(eS - eps, RHS)

                # Two batched GEMMs: AS = alpha @ [tmpa | tmpb], BS = beta @ [tmpa | tmpb].
                # Splits into the four (N,N)@(N,N) products the recursion needs, but the
                # ZGEMM "A" panel is packed once per merged call instead of twice.
                AS = alpha @ sol
                BS = beta @ sol

                # next iter uses these as GEMM "A" operand; copy so they are C-contig
                alpha = np.ascontiguousarray(AS[:, :N])
                beta = np.ascontiguousarray(BS[:, N:])

                # AS[:, N:] = alpha_prev @ tmpb; BS[:, :N] = beta_prev @ tmpa
                eps += AS[:, N:] + BS[:, :N]
                epss += BS[:, :N]

                # Cheaper probe: max(max|alpha|, max|beta|) < 1e-40 is equivalent to the old
                # max(|alpha|+|beta|) < 1e-40 up to a factor of 2, which is trivially covered.
                LopezConvTest = max(np.max(np.abs(alpha)), np.max(np.abs(beta)))

                if LopezConvTest < 1.0e-40:
                    gs = np.linalg.inv(eS - epss)

                    test = eS - H - (ee * s01 - h01) @ gs @ (ee * s10 - h10)
                    myConvTest = np.max(np.abs((test @ gs) - np.eye(H.shape[0], dtype=h01.dtype)))

                    if myConvTest < 3.0e-5:
                        converged = True
                        if myConvTest > 1.0e-8: # warning threshold
                            return gs, 1, myConvTest, ee.real
                        else:
                            return gs, 0, 0, 0
                    else:
                        raise ArithmeticError

                if iteration >= 101:
                    raise RuntimeError

            return gs
        return _core

    _surface_green_numba_core = _make_surface_green_numba_core(fastmath=True)
    # Regression-test sibling: identical body, fastmath disabled. Not called by
    # production code; kept so `tests/test_surface_green_fastmath.py` can compare.
    _surface_green_numba_core_nofastmath = _make_surface_green_numba_core(fastmath=False)

    _numba_available = True
    log.info("Numba is available and JIT functions are compiled.")

except (ImportError, Exception) as e:
    log.warning(f"Numba acceleration is not available. Falling back to pure NumPy. Error: {e}")
    _numba_available = False

# Scipy-based implementation of the surface Green's function calculation
def _surface_green_scipy_core(H, h01, S, s01, h10, s10, ee):
    N = H.shape[0]
    alpha = h10 - ee * s10
    beta = h01 - ee * s01

    eps = H.copy()
    epss = H.copy()

    eS = ee * S  # loop invariant

    # preallocated scratch buffer reused every iteration
    RHS = np.empty((N, 2 * N), dtype=np.complex128)

    converged = False
    iteration = 0
    while not converged:
        iteration += 1
        # one LU solve with stacked RHS = [alpha | beta]
        RHS[:, :N] = alpha
        RHS[:, N:] = beta
        sol = SLA.solve(eS - eps, RHS,
                        overwrite_a=True, overwrite_b=True, check_finite=False)

        # Same batched-GEMM structure as the numba core.
        AS = alpha @ sol
        BS = beta @ sol

        alpha = np.ascontiguousarray(AS[:, :N])
        beta = np.ascontiguousarray(BS[:, N:])

        eps += AS[:, N:] + BS[:, :N]
        epss += BS[:, :N]

        LopezConvTest = max(np.max(np.abs(alpha)), np.max(np.abs(beta)))

        if LopezConvTest < 1.0e-40:
            gs = np.linalg.inv(eS - epss)

            test = eS - H - (ee * s01 - h01) @ gs @ (ee * s10 - h10)
            myConvTest = np.max(np.abs((test @ gs) - np.eye(N, dtype=h01.dtype)))

            if myConvTest < 3.0e-5:
                converged = True
                if myConvTest > 1.0e-8:
                    log.warning(f"Lopez-scheme not-so-well converged at E = {ee.real:.4f} eV: {myConvTest}")
            else:
                log.error(f"Lopez-Sancho {myConvTest:.8f} Error: gs iteration {iteration}")
                raise ArithmeticError("Criteria not met. Please check output...")

        if iteration >= 101:
            log.error("Lopez-scheme not converged after 100 iteration.")
            raise RuntimeError("Lopez-scheme not converged.")

    return gs


def surface_green(H, h01, S, s01, ee,
                  method='Lopez-Sancho',
                  numba_jit=None,
                  h10=None, s10=None):
    '''calculate surface green function
    At this stage, we realized Lopez-Sancho scheme and  GEP scheme.
    However, GEP scheme is not so stable, and we strongly recommended  to implement the Lopez-Sancho scheme.

    ``h10`` / ``s10`` (``conj(h01.T)`` / ``conj(s01.T)``) may be passed in from
    the precomputed lead pack to avoid re-doing the transpose+conj per energy.
    They are derived on the fly if omitted.

    '''

    if method not in SURFACE_GREEN_METHODS:
        choices = ", ".join(sorted(SURFACE_GREEN_METHODS))
        raise ValueError(f"Unknown surface-Green solver {method!r}; expected one of: {choices}.")

    # default: use numba whenever it compiled successfully
    if numba_jit is None:
        numba_jit = _numba_available

    if h10 is None:
        h10 = np.ascontiguousarray(np.conj(h01.T))
    if s10 is None:
        s10 = np.ascontiguousarray(np.conj(s01.T))

    if method == 'GEP':
        gs = calcg0(ee, H, S, h01, s01)
        return gs
    else: # Lopez-Sancho scheme
        if numba_jit and _numba_available:
            try:
                # normalize ee to numpy complex128 to match the @njit signature
                ee = np.complex128(ee)
                log.debug("surface_green: using numba core")
                gs, conv_flag, conv_test, e_real = _surface_green_numba_core(H, h01, S, s01, h10, s10, ee)
                if conv_flag == 1:
                    log.warning(f"Lopez-Sancho scheme not-so-well converged at E = {e_real:.4f} eV: {conv_test}")
                return gs
            except Exception as e:
                log.error(f"Numba JIT function failed at runtime. Falling back to NumPy. Error: {e}")
                return _surface_green_scipy_core(H, h01, S, s01, h10, s10, ee)
        else:
            log.debug("surface_green: using scipy core")
            return _surface_green_scipy_core(H, h01, S, s01, h10, s10, ee)
                



def calcg0(ee, h00, s00, h01, s01):
    '''The `calcg0` function calculates the surface Green's function for a specific |k> , ref. Euro Phys J B 62, 381 (2008)
        Inverse of : NOTE, setup for "right" lead.
        e-h00 -h01  ...
        -h10  e-h11 ...
         .
         .
         .

    Parameters
    ----------
    ee
        The parameter `ee` represents the energy value for which the surface Green's function is
    calculated. It is a complex number that determines the energy of the state being considered.
    h00
        hamiltonian matrix within principal layer
    s00
        overlap matrix within principal layer
    h01
        hamiltonian matrix between two adject principal layers
    s01
        overlap matrix between two adject principal layers
    
    Returns
    -------
        Surface Green's function `g00`.
    
    ''' 
    
    NN = h00.shape[0]
    ee = ee.real + max(ee.imag, 1e-8) * 1.0j

    # Solve generalized eigen-problem
    a, b = np.zeros((2 * NN, 2 * NN), dtype=h00.dtype), np.zeros((2 * NN, 2 * NN), dtype=h00.dtype)
    
    a[0:NN, 0:NN] = ee * s00 - h00
    a[0:NN, NN:2 * NN] = -np.eye(NN, dtype=h00.dtype)
    a[NN:2 * NN, 0:NN] = h01.conj().T - ee * s01.conj().T
    b[0:NN, 0:NN] = h01 - ee * s01
    b[NN:2 * NN, NN:2 * NN] = np.eye(NN, dtype=h00.dtype)


    ev, evec = SLA.eig(a=a, b=b)

    ipiv = np.where(np.abs(ev) < 1.)[0]
    ev, evec = ev[ipiv], evec[:NN, ipiv].T

    # Normalize evec
    norm = np.sqrt(np.diag(evec @ evec.conj().T)) 
    evec = np.diag(1.0 / norm) @ evec

    # E^+ Lambda_+ (E^+)^-1 --->>> g00
    EP = evec.T
    FP = EP @ np.diag(ev) @ np.linalg.inv(EP.conj().T @ EP) @ EP.conj().T
    g00 = np.linalg.inv(ee * s00 - h00 - (h01 - ee * s01) @ FP)
    
    g00 = iterative_gf_numpy(ee, g00, h00, h01, s00, s01, iter=3)

    err = np.max(np.abs(g00 - np.linalg.inv(ee * s00 - h00 - \
                                            (h01 - ee * s01) @ g00 @ (h01.conj().T - ee * s01.conj().T))))
    if err > 1.0e-8:
        print("WARNING: not-so-well converged for RIGHT electrode at E = {0} eV:".format(ee.real), err)
    
    return g00

def iterative_gf_numpy(ee, gs, h00, h01, s00, s01, iter=1):
    '''
    NumPy-based rewrite of the PyTorch iterative_gf function.
    '''
    # 将输入张量转换为 NumPy 数组
    gs = np.array(gs)
    h00 = np.array(h00)
    h01 = np.array(h01)
    s00 = np.array(s00)
    s01 = np.array(s01)
    ee = np.array(ee)
    
    for i in range(iter):
        gs_new = ee*s00 - h00 - (ee * s01 - h01) @ gs @ (ee * s01.conj().T - h01.conj().T)
        gs = np.linalg.pinv(gs_new)
    
    return gs

def iterative_simple_numpy(ee, h00, h01, s00, s01, iter_max=1000):
    '''
    NumPy-based rewrite of the PyTorch iterative_simple function.
    '''
    # 将输入张量转换为 NumPy 数组
    h00 = np.array(h00)
    h01 = np.array(h01)
    s00 = np.array(s00)
    s01 = np.array(s01)
    ee = np.array(ee)
    
    gs = np.linalg.inv(ee*s00 - h00)
    diff_gs = 1
    iteration = 0
    while diff_gs > 1e-8:
        iteration += 1
        gs_prev = gs.copy()
        
        term = (ee * s01 - h01) @ gs_prev @ (ee * s01.conj().T - h01.conj().T)
        gs = np.linalg.inv(ee*s00 - h00 - term)
        
        diff_gs = np.max(np.abs(gs - gs_prev))
        
        if iteration > iter_max:
            log.warning("iterative_simple not converged after 1000 iteration.")
            break
            
    return gs

def convert_to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, (int, float, complex, np.number)):
        return np.array(data)
