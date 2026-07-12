import torch
from typing import List
from dpnegf.negf.surface_green import selfEnergy
import logging
import os
from dpnegf.utils.constants import Boltzmann, eV2J
import numpy as np
from dpnegf.negf.bloch import Bloch
import ase
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
import h5py
import glob
import psutil
import time


log = logging.getLogger(__name__)

# """The data output of the intermidiate result should be like this:
# {each kpoint
#     "e_mesh":[],
#     "emap":[]
#     "se":[se(e0), se(e1),...], 
#     "sgf":[...e...]
# }
# There will be a kmap outside like: {(0,0,0):1, (0,1,2):2}, to locate which file it is to reads.
# """


class LeadProperty(object):
    '''
    The Lead class represents a lead in a structure and provides methods for calculating the self energy
    and gamma for the lead.

    Property
    ----------
    hamiltonian
        hamiltonian of the whole structure.
    structure
        structure of the lead.
    tab
        lead tab.
    voltage
        voltage of the lead.
    results_path
        output  path.
    kBT
        Boltzmann constant times temperature.
    efermi
        Fermi energy.
    mu
        chemical potential of the lead.
    gamma
        the broadening function of the isolated energy level of the device
    HL 
        hamiltonian within principal layer
    HLL 
        hamiiltonian between two adjacent principal layers
    HDL 
        hamiltonian between principal layer and device
    SL SLL and SDL 
        the overlap matrix, with the same meaning as HL HLL and HDL.
    

    Method
    ----------
    self_energy
        calculate  the self energy and surface green function at the given kpoint and energy.
    sigma2gamma
        calculate the Gamma function from the self energy.

    '''
    def __init__(self, tab, hamiltonian, structure, results_path, voltage,
                 structure_leads_fold:ase.Atoms=None,bloch_sorted_indice:torch.Tensor=None, useBloch: bool=False,
                    bloch_factor: List[int]=[1,1,1],bloch_R_list:List=None,
                    e_T=300, efermi:float=0.0, E_ref:float=None) -> None:
        self.hamiltonian = hamiltonian
        self.structure = structure
        self.tab = tab
        self.voltage = voltage
        self.results_path = results_path
        self.kBT = Boltzmann * e_T / eV2J
        self.e_T = e_T
        self.efermi = efermi
        if E_ref is None:
            self.E_ref = efermi
        else:
            self.E_ref = E_ref
        self.chemiPot_lead = efermi - voltage
        self.kpoint = None
        self.voltage_old = None
        
        
        self.useBloch = useBloch
        self.bloch_factor = bloch_factor
        self.bloch_sorted_indice = bloch_sorted_indice
        self.bloch_R_list = bloch_R_list
        self.structure_leads_fold = structure_leads_fold
        if self.useBloch:
            assert self.bloch_sorted_indice is not None
            assert self.bloch_R_list is not None
            assert self.bloch_factor is not None
            assert self.structure_leads_fold is not None

    def self_energy(self, kpoint, energy, 
                    eta_lead: float=1e-5,
                    method: str="Lopez-Sancho",
                    save_path: str=None, 
                    save_format: str="h5",
                    se_info_display: bool=False,
                    HS_inmem: bool=True):
        '''calculate and loads the self energy and surface green function at the given kpoint and energy.
        
        Parameters
        ----------
        kpoint
            the coordinates of a specific point in the Brillouin zone. 
        energy
            specific energy value.
        eta_lead : 
            the broadening parameter for calculating lead surface green function.
        method : 
            specify the method for calculating the self energy. At this stage it only supports "Lopez-Sancho".
        save :
            whether to save the self energy. 
        save_path :
            the path to save the self energy. If not specified, the self energy will be saved in the results_path.
        se_info_display :
            whether to display the information of the self energy calculation.   
        HS_inmem :
            whether to store the Hamiltonian and overlap matrix in memory. Default is False.     
        '''
        assert len(np.array(kpoint).reshape(-1)) == 3
        # according to given kpoint and e_mesh, calculating or loading the self energy and surface green function to self.
        if not isinstance(energy, torch.Tensor):
            energy = torch.tensor(energy) # Energy relative to Ef
        
        save_path = self._get_save_path(kpoint, energy, save_format, save_path)
        # log.info(f"Self energy save path: {save_path}")

        # Try load
        if os.path.isfile(save_path):
            if se_info_display:
                log.info(f"Loading {self.tab} self-energy from {save_path}")
            self.se = self._load_self_energy(save_path, kpoint, energy, save_format)
            return
        
        # If not loaded, just compute
        if se_info_display:
            log.info(f"Computing {self.tab} self-energy (method={method}) "
                    f"at k={kpoint}, E={energy.item():.6f}")

        self.se = self.self_energy_cal( kpoint, 
                                        energy,
                                        eta_lead=eta_lead, 
                                        method=method,
                                        HS_inmem=HS_inmem,
                                        se_numba_jit=None)

    def _get_save_path(self, kpoint, energy, save_format: str, save_path: str = None):
        """
        Generate the save path for self-energy files.

        Parameters
        ----------
        kpoint : array-like
            The k-point (length 3).
        energy : torch.Tensor or float
            Energy value.
        save_format : str
            File format, supports "pth" or "h5".
        save_path : str, optional
            User-specified save path. If None, use default under results_path/self_energy.

        Returns
        -------
        str
            Full path to the save file.
        """
        # Ensure kpoint is array for string formatting
        kx, ky, kz = np.asarray(kpoint, dtype=float).reshape(3)
        energy_val = energy.item() if hasattr(energy, "item") else float(energy)

        # Case 1: User provided save_path
        if save_path is not None:
            # If it's a directory, append default filename
            if os.path.isdir(save_path):
                if save_format == "pth":
                    return os.path.join(save_path,
                                        f"se_{self.tab}_k{kx:.4f}_{ky:.4f}_{kz:.4f}_E{energy_val:.6f}.pth")
                elif save_format == "h5":
                    if self.tab == "lead_L":
                        return os.path.join(save_path, "self_energy_leadL.h5")
                    elif self.tab == "lead_R":
                        return os.path.join(save_path, "self_energy_leadR.h5")
                else:
                    raise ValueError(f"Unsupported save_format {save_format}")
            return save_path  # direct file path given by user

        # Case 2: Default path under results_path
        parent_dir = os.path.join(self.results_path, "self_energy")
        os.makedirs(parent_dir, exist_ok=True)

        if save_format == "pth":
            return os.path.join(parent_dir,
                                f"se_{self.tab}_k{kx:.4f}_{ky:.4f}_{kz:.4f}_E{energy_val:.6f}.pth")

        elif save_format == "h5":
            if self.tab == "lead_L":
                return os.path.join(parent_dir, "self_energy_leadL.h5")
            elif self.tab == "lead_R":
                return os.path.join(parent_dir, "self_energy_leadR.h5")
            else:
                raise ValueError(f"Unsupported tab {self.tab} for h5 save.")

        else:
            raise ValueError(f"Unsupported save_format {save_format}, only 'pth' and 'h5' are supported.")

    @staticmethod
    def _load_self_energy(save_path: str, kpoint, energy, save_format: str):
        """
        Load self-energy from file.

        Parameters
        ----------
        save_path : str
            Path to the saved self-energy file.
        kpoint : array-like
            The k-point (length 3).
        energy : torch.Tensor or float
            Energy value.
        save_format : str
            File format, supports "pth" or "h5".

        Returns
        -------
        torch.Tensor
            Loaded self-energy tensor.
        """
        if save_format == "pth":
            try:
                se = torch.load(save_path, weights_only=False)
            except Exception as e:
                raise IOError(f"Failed to load self-energy from {save_path} (pth format).") from e

        elif save_format == "h5":
            try:
                data = read_from_hdf5(save_path, kpoint, energy)
                se = torch.as_tensor(data, dtype=torch.complex128)  # self-energy should be complex
            except KeyError as e:
                kx, ky, kz = np.asarray(kpoint, dtype=float).reshape(3)
                ev = energy.item() if hasattr(energy, "item") else float(energy)
                raise KeyError(
                    f"Cannot find self-energy in {save_path} "
                    f"for k=({kx:.4f},{ky:.4f},{kz:.4f}), E={ev:.6f}"
                ) from e
            except Exception as e:
                raise IOError(f"Failed to read HDF5 self-energy from {save_path}.") from e

        else:
            raise ValueError(f"Unsupported save_format {save_format}, only 'pth' and 'h5' are supported.")

        return se


    def self_energy_cal(self, 
                        kpoint, 
                        energy, 
                        eta_lead: float=1e-5,
                        method: str="Lopez-Sancho",
                        se_numba_jit=None,
                        HS_inmem: bool=True):
        """
        Calculates the self-energy for a lead in a quantum transport calculation.
        This method computes the self-energy matrix for a given k-point and energy, 
        using either the standard or Bloch-based approach depending on the object's configuration.
        Parameters
        ----------
        kpoint : array-like
            The k-point in reciprocal space at which to calculate the self-energy.
        energy : float or torch.Tensor
            The energy value at which to evaluate the self-energy.
        eta_lead : float, optional
            Small imaginary part added to the energy for numerical stability (default: 1e-5).
        method : str, optional
            The method used for self-energy calculation (default: "Lopez-Sancho").
        HS_inmem : bool, optional
            If False, deletes Hamiltonian and overlap matrices from memory after calculation (default: True).
            This is useful for large systems to save memory.
        Returns
        -------
        se : torch.Tensor
            The calculated self-energy matrix for the specified k-point and energy.
        Notes
        -----
        - If `useBloch` is True, the calculation unfolds the k-points and applies Bloch phase factors.
        - The method caches Hamiltonian and overlap matrices for efficiency unless `HS_inmem` is False.
        - The shape of the returned self-energy matrix is consistent with the reduced Hamiltonian blocks.
        """      
        subblocks = self.hamiltonian.get_hs_device(kpoint, only_subblocks=True)
        # calculate self energy
        if not self.useBloch:
            if  not hasattr(self, "HL") \
                or abs(self.voltage_old-self.voltage)>1e-6 \
                or max(abs(self.kpoint-torch.tensor(kpoint)))>1e-6:

                self.HLk, self.HLLk, self.HDLk, self.SLk, self.SLLk, self.SDLk \
                    = self.hamiltonian.get_hs_lead(kpoint, tab=self.tab, v=self.voltage)
                self.voltage_old = self.voltage
                self.kpoint = torch.tensor(kpoint)

            HDL_reduced, SDL_reduced = self.HDL_reduced(self.HDLk, self.SDLk,subblocks)
            
            self.se, _ = selfEnergy(
                ee=energy,
                hL=self.HLk,
                hLL=self.HLLk,
                sL=self.SLk,
                sLL=self.SLLk,
                hDL=HDL_reduced,
                sDL=SDL_reduced,             #TODO: check chemiPot settiing is correct or not
                E_ref=self.E_ref,
                etaLead=eta_lead, 
                method=method,
                numba_jit=se_numba_jit
            )

            # torch.save(self.se, os.path.join(self.results_path, f"se_nobloch_k{kpoint[0]}_{kpoint[1]}_{kpoint[2]}_{energy}.pth"))
        
        else:
            if not hasattr(self, "HL") \
                or abs(self.voltage_old-self.voltage)>1e-6 \
                or max(abs(self.kpoint-torch.tensor(kpoint)))>1e-6:
                self.kpoint = torch.tensor(kpoint)
                self.voltage_old = self.voltage

            bloch_unfolder = Bloch(self.bloch_factor)
            kpoints_bloch = bloch_unfolder.unfold_points(self.kpoint.tolist())
            sgf_k = []
            m_size = self.bloch_factor[1]*self.bloch_factor[0]
            for k_bloch in kpoints_bloch:
                k_bloch = torch.tensor(k_bloch)
                self.HLk, self.HLLk, self.HDLk, self.SLk, self.SLLk, self.SDLk \
                    = self.hamiltonian.get_hs_lead(k_bloch, tab=self.tab, v=self.voltage)
                
                _, sgf = selfEnergy(
                    ee=energy,
                    hL=self.HLk,
                    hLL=self.HLLk,
                    sL=self.SLk,
                    sLL=self.SLLk,            #TODO: check chemiPot settiing is correct or not
                    E_ref=self.E_ref,  # temmporarily change to self.efermi for the case in which applying lead bias to corresponding to Nanotcad
                    etaLead=eta_lead, 
                    method=method,
                    numba_jit=se_numba_jit
                )
                phase_factor_m = torch.zeros([m_size,m_size],dtype=torch.complex128)
                for i in range(m_size):
                    for j in range(m_size):
                        if i == j:
                            phase_factor_m[i,j] = 1
                        else:
                            phase_factor_m[i,j] = torch.exp(torch.tensor(1j)*2*torch.pi*torch.dot(self.bloch_R_list[j]-self.bloch_R_list[i],k_bloch))  
                phase_factor_m = phase_factor_m.contiguous()
                sgf = sgf.contiguous()
                sgf_k.append(torch.kron(phase_factor_m,sgf)) 
             

            sgf_k = torch.sum(torch.stack(sgf_k),dim=0)/len(sgf_k)
            sgf_k = sgf_k[self.bloch_sorted_indice,:][:,self.bloch_sorted_indice]
            b = self.HDLk.shape[1] # size of lead hamiltonian

            # reduce the Hamiltonian and overlap matrix based on the non-zero range of HDL
            HDL_reduced, SDL_reduced = self.HDL_reduced(self.HDLk, self.SDLk,subblocks) 
            if not isinstance(energy, torch.Tensor):
                eeshifted = torch.scalar_tensor(energy, dtype=torch.complex128) + self.E_ref
            else:
                eeshifted = energy + self.E_ref
            self.se = (eeshifted*SDL_reduced-HDL_reduced) @ sgf_k[:b,:b] @ (eeshifted*SDL_reduced.conj().T-HDL_reduced.conj().T)
            # In subblocks case, the self energy shape of left/right lead should be consistent with subblocks[0] and subblocks[-1]
        if not HS_inmem:
            del self.HLk, self.HLLk, self.HDLk, self.SLk, self.SLLk, self.SDLk

        return self.se

    @staticmethod
    def HDL_reduced(HDL: torch.Tensor, SDL: torch.Tensor, subblocks: np.ndarray) -> torch.Tensor:
        '''This function takes in Hamiltonian/Overlap matrix between lead and device and reduces 
        it based on the subblocks results or non-zero range of the Hamiltonian matrix.

            When the device part has only one orbital, the Hamiltonian matrix is not reduced.
        
        Parameters
        ----------
        HDL : torch.Tensor
            HDL is a torch.Tensor representing the Hamiltonian matrix between the first principal layer and the device.
        SDL : torch.Tensor
            SDL is a torch.Tensor representing the overlap matrix between the first principal layer and the device.
        
        Returns
        -------
        HDL_reduced, SDL_reduced
            The reduced Hamiltonian and overlap matrix.
        
        '''
        assert len(HDL.shape) == 2, "The shape of HDL should be 2."
        assert len(SDL.shape) == 2, "The shape of SDL should be 2."
        assert HDL.shape == SDL.shape, "The shape of HDL and SDL should be the same."

        HDL_nonzero_range = (HDL.nonzero().min(dim=0).values, HDL.nonzero().max(dim=0).values)
        if subblocks is None:
            cut_range = HDL_nonzero_range
        else:
            cut_range = ((subblocks[-1],subblocks[-1]), (subblocks[0],subblocks[0]))
        # HDL_nonzero_range is a tuple((min_row,min_col),(max_row,max_col))
        if HDL.shape[0] == 1: # Only 1 orbital in the device
            HDL_reduced = HDL
            SDL_reduced = SDL
        elif HDL_nonzero_range[0][0] > 0: # Right lead
            if subblocks is None:
                HDL_reduced = HDL[cut_range[0][0]:, :]
                SDL_reduced = SDL[cut_range[0][0]:, :]
            else:
                HDL_reduced = HDL[-1*cut_range[0][0]:, :]
                SDL_reduced = SDL[-1*cut_range[0][0]:, :]
        else: # Left lead
            if subblocks is None:
                HDL_reduced = HDL[:cut_range[1][0]+1, :]
                SDL_reduced = SDL[:cut_range[1][0]+1, :]
            else:
                HDL_reduced = HDL[:cut_range[1][0], :]
                SDL_reduced = SDL[:cut_range[1][0], :]

        return HDL_reduced, SDL_reduced


    def sigmaLR2Gamma(self, se):
        '''calculate the Gamma function from the self energy.
        
        Gamma function is the broadening function of the isolated energy level of the device.

        Parameters
        ----------
        se
            The parameter "se" represents self energy, a complex matrix.
        
        Returns
        -------
        Gamma
            The Gamma function, Gamma = 1j(se-se^dagger).
        
        '''
        return 1j * (se - se.mH)
    
    def fermi_dirac(self, x) -> torch.Tensor:
        return 1 / (1 + torch.exp((x - self.chemiPot_lead)/ self.kBT))
    
    @property
    def gamma(self):
        return self.sigmaLR2Gamma(self.se)


def _estimate_worker_memory(lead_L, lead_R, kpoint=None, temp_allocation_factor=3.0):
    """
    Estimate memory (in bytes) needed per joblib worker for self-energy calculation.

    The estimation separates two components:
    1. Base overhead: Fixed memory for Python process and imported libraries
       (Python interpreter, NumPy, SciPy, PyTorch, DeePTB, etc.)
    2. Computation memory: Dynamic memory for matrices and intermediate calculations,
       scaled by a factor to account for temporary allocations during surface Green's
       function iteration, matrix products, and LAPACK/BLAS workspace.

    Parameters
    ----------
    lead_L, lead_R : LeadProperty
        Lead objects containing Hamiltonian data.
    kpoint : array-like, optional
        A sample k-point to use for fetching Hamiltonian matrices. If None, uses [0, 0, 0].
    temp_allocation_factor : float
        Multiplier for computation memory to account for intermediate arrays created
        during surface Green's function iteration and matrix operations. Default 3.0.

    Returns
    -------
    int
        Estimated memory in bytes per worker.
    """
    # Base overhead for Python process + libraries (interpreter, numpy, scipy, torch, etc.)
    base_overhead = 100 * 1024 * 1024  # 100 MB

    matrix_bytes = 0

    if kpoint is None:
        kpoint = [0.0, 0.0, 0.0]  # use Gamma point if not provided

    # Estimate from lead Hamiltonian matrices using get_hs_lead method
    # Each complex128 element = 16 bytes
    for lead in [lead_L, lead_R]:
        try:
            # get_hs_lead returns: (hL, hLL, hDL, sL, sLL, sDL)
            hL, hLL, hDL, sL, sLL, sDL = lead.hamiltonian.get_hs_lead(
                kpoint=kpoint, tab=lead.tab, v=lead.voltage
            )
            # Sum up memory for all matrices
            for tensor in [hL, hLL, hDL, sL, sLL, sDL]:
                if tensor is not None:
                    if hasattr(tensor, 'numel'):  # PyTorch tensor
                        matrix_bytes += tensor.numel() * 16  # complex128
                    elif hasattr(tensor, 'nbytes'):  # NumPy array
                        matrix_bytes += tensor.nbytes
        except Exception as e:
            log.warning(f"Could not estimate matrix memory from {lead.tab}: {e}"
                        " Using fallback matrix estimate: 100 MB this lead.")
            # Fallback: assume 100 MB per lead
            matrix_bytes += 100 * 1024 * 1024
            
    # Total estimate: base overhead + scaled computation memory
    computation_memory = matrix_bytes * temp_allocation_factor
    total_estimate = base_overhead + int(computation_memory)

    return total_estimate


def _get_safe_n_workers(lead_L, lead_R, requested_n_workers=-1, max_memory_fraction=0.9,
                        min_workers=1, kpoint=None, cpu_budget=None):
    """
    Calculate safe number of parallel workers based on available system memory.

    Parameters
    ----------
    lead_L, lead_R : LeadProperty
        Lead objects for memory estimation.
    requested_n_workers : int
        User-requested n_workers. -1 means auto-detect.
    max_memory_fraction : float
        Maximum fraction of available memory to use. Default 0.9.
    min_workers : int
        Minimum number of workers to use. Default 1.
    kpoint : array-like, optional
        A sample k-point for fetching Hamiltonian matrices to estimate memory.
    cpu_budget : int or None
        Total CPU cores the self-energy pool is allowed to size against.
        If None, uses os.cpu_count().

    Returns
    -------
    int
        Safe number of parallel workers.
    """
    cpu_count = cpu_budget if cpu_budget is not None else os.cpu_count()
    if cpu_count is None or cpu_count < 1:
        cpu_count = 1
        log.warning("os.cpu_count() returned None or invalid value. Defaulting to 1 CPU core.")

    available_memory = psutil.virtual_memory().available
    memory_per_worker = _estimate_worker_memory(lead_L, lead_R, kpoint=kpoint)

    # Calculate max workers that fit in available memory
    if memory_per_worker <= 0:
        log.warning(f"Memory estimation returned non-positive value. Using min_workers={min_workers}.")
        return min_workers

    # Calculate max workers that fit in available memory
    max_workers_by_memory = int((available_memory * max_memory_fraction) / memory_per_worker)
    max_workers_by_memory = int((available_memory * max_memory_fraction) / memory_per_worker)
    max_workers_by_memory = max(max_workers_by_memory, min_workers)

    # Cap by CPU count
    max_workers = min(max_workers_by_memory, cpu_count)

    safe_n_worker = 0
    # check requested_n_workers is a number
    if not isinstance(requested_n_workers, int):
        log.warning(f"Requested n_workers={requested_n_workers} is not an integer. \n"
                    f"Using min_workers={min_workers}.")
        safe_n_worker = min_workers

    if requested_n_workers == -1:
        safe_n_worker = max_workers
    elif requested_n_workers == 0:
        log.warning(f"Requested n_workers=0 is invalid. Using min_workers={min_workers}.")
        safe_n_worker = min_workers
    elif requested_n_workers > 0:
        if requested_n_workers > max_workers:
            log.warning(f"Requested n_workers={requested_n_workers} may exceed available memory. "
                       f"Limiting to {max_workers} workers "
                       f"(available: {available_memory / 1e9:.1f} GB, "
                       f"est. per worker: {memory_per_worker / 1e9:.1f} GB)")
            safe_n_worker = max_workers
        else:
            safe_n_worker = requested_n_workers
    else:
        # Negative values other than -1: joblib interprets as (cpu_count + 1 + n_workers)
        effective_n_workers = max(cpu_count + 1 + requested_n_workers, min_workers)
        safe_n_worker = min(effective_n_workers, max_workers)

    log.info(f"Estimated safe n_workers={safe_n_worker} based on available memory.")
    return safe_n_worker


def _autotune_blas_threads(leadL_pack, sample_kpoint, sample_energy, eta_lead,
                            n_workers, cpu_count, se_numba_jit, requested=None):
    """Pick per-worker BLAS threads by timing the real surface-green code path.

    The Lopez-Sancho loop in ``surface_green._surface_green_{numba,scipy}_core``
    (surface_green.py) issues repeated complex128 ``solve`` on shape ``(N, 2N)``
    and several ``(N, N)`` matmuls per iteration. The payoff from BLAS threading
    is hardware-dependent (OpenBLAS vs MKL, cache size, N), so instead of a
    static table we probe the actual `_compute_self_energy_from_pack` at a few
    candidate thread counts and keep the fastest. Runs once per
    ``compute_all_self_energy`` call, in the parent process, before workers fan
    out. Cost is bounded to ``2 * len(candidates)`` surface-green calls at the
    sample ``(k, E)``.

    Parameters
    ----------
    leadL_pack : dict
        Precomputed lead pack (see `_precompute_lead_kdata`). Only the left
        lead is timed; the right lead has the same principal-layer size in
        normal use, so measuring one halves the probe cost.
    sample_kpoint, sample_energy : array-like, float
        The (k, E) pair used for the probe. Take from the grid actually being
        computed so N and dtype match production.
    eta_lead : float
        Same broadening as production.
    n_workers : int
        Number of joblib workers about to be launched; bounds per-worker CPU.
    cpu_count : int
        Total CPU budget.
    se_numba_jit : bool or None
        Same flag workers will see. The first call absorbs numba JIT compile;
        the timed second call measures steady-state.
    requested : int or None
        User override. When a positive int, skip the bench and clamp to the
        per-worker budget.

    Returns
    -------
    int
        BLAS threads to give each loky worker via ``threadpool_limits``.
    """
    n_workers = max(int(n_workers), 1)
    per_worker_budget = max(int(cpu_count) // n_workers, 1)

    if requested is not None:
        if not isinstance(requested, int) or requested < 1:
            log.warning(f"Requested blas_threads={requested} is not a positive int. "
                        "Falling back to autotune.")
        else:
            if requested > per_worker_budget:
                log.warning(f"Requested blas_threads={requested} exceeds per-worker "
                            f"CPU budget ({per_worker_budget} = cpu_count // n_workers). "
                            f"Clamping to {per_worker_budget}.")
            threads = min(requested, per_worker_budget)
            log.info(f"BLAS threads per worker: {threads} (user-requested).")
            return threads

    if per_worker_budget == 1:
        log.info("BLAS threads per worker: 1 (no CPU budget left for BLAS threading).")
        return 1

    if sample_kpoint is None or not leadL_pack.get("kdata"):
        log.info("BLAS threads per worker: 1 (empty sample; skipping autotune).")
        return 1

    sample_dim = _sample_principal_layer_dim(leadL_pack, sample_kpoint)
    if sample_dim < 64:
        log.info(f"BLAS threads per worker: 1 (N={sample_dim} < 64; skipping autotune).")
        return 1

    candidates = sorted({1, 2, 4, 8, per_worker_budget})
    candidates = [c for c in candidates if c <= per_worker_budget]

    # Wall-clock cap for the whole probe. stop between
    # candidates (not mid-call) and pick the best-so-far.
    max_autotune_seconds = 90.0
    bench_start = time.perf_counter()

    timings = {}
    for t in candidates:
        if time.perf_counter() - bench_start > max_autotune_seconds:
            log.warning(f"BLAS autotune exceeded {max_autotune_seconds:.0f}s wall-clock cap; "
                        f"stopping after {len(timings)}/{len(candidates)} candidates.")
            break
        try:
            with threadpool_limits(limits=t, user_api='blas'):
                _compute_self_energy_from_pack(
                    leadL_pack, sample_kpoint, sample_energy,
                    eta_lead, se_numba_jit=se_numba_jit,
                )
                start = time.perf_counter()
                _compute_self_energy_from_pack(
                    leadL_pack, sample_kpoint, sample_energy,
                    eta_lead, se_numba_jit=se_numba_jit,
                )
                elapsed = time.perf_counter() - start
        except Exception as e:
            log.warning(f"BLAS autotune t={t} failed ({e!r}); skipping candidate.")
            continue
        timings[t] = elapsed
        log.info(f"BLAS autotune t={t}: {elapsed * 1e3:.1f} ms")

    if not timings:
        log.warning("BLAS autotune found no working candidate; falling back to 1.")
        return 1

    best = min(timings, key=timings.get)
    log.info(f"BLAS threads per worker: {best} "
             f"(autotuned, N={sample_dim}, n_workers={n_workers}, cpu_count={cpu_count}).")
    return best


def _sample_principal_layer_dim(pack, sample_kpoint):
    """Return the principal-layer Hamiltonian dimension N from a precomputed
    lead pack, handling both non-Bloch and Bloch branches. Returns 0 if the
    sample entry cannot be located, letting callers fall back to a safe
    default."""
    if sample_kpoint is None:
        return 0
    try:
        entry = pack["kdata"][_k_key(sample_kpoint)]
    except (KeyError, TypeError):
        return 0
    if "HLk" in entry:
        HLk = entry["HLk"]
    else:
        bloch_entries = entry.get("bloch_entries") or []
        if not bloch_entries:
            return 0
        HLk = bloch_entries[0]["HLk"]
    return int(HLk.shape[0])


def compute_all_self_energy(eta, lead_L, lead_R, kpoints_grid, energy_grid,
                            self_energy_save_path=None, ek_batch_size=200,
                            cpu_budget=None, n_workers=-1, se_numba_jit=None, blas_threads=None):
    """
    Computes and saves self-energy matrices for all combinations of k-points and energy values
    for left and right leads.

    The self-energy calculations are performed in parallel batches, and results are saved as HDF5 files.
    Temporary files are merged into final output files for each lead.

    Parameters
    ----------
    eta : float
        Small imaginary part added to energy for numerical stability.
    lead_L : Lead
        lead object containing Left lead Hamiltonian and results path.
    lead_R : Lead
        lead object containing Right lead Hamiltonian and results path.
    kpoints_grid : array-like
        List or array of k-points to compute self-energy for.
    energy_grid : array-like
        List or array of energy values to compute self-energy for.
    self_energy_save_path : str or None, optional
        Directory to save self-energy files. If None, uses lead_L's results_path.
    ek_batch_size : int, optional
        Number of (k, e) tasks per parallel batch. Default is 200.
    cpu_budget : int or None, optional
        Total CPU cores the self-energy pool is allowed to size against.
        If None, uses os.cpu_count().
    n_workers : int, optional
        Number of parallel joblib workers to use. Default is -1 (auto-select
        based on `cpu_budget` and available memory).
    se_numba_jit : bool or None, optional
        Boolean flag controlling whether to use the Numba-accelerated surface Green's function core.
        If None, Numba will be used when available. Default is None.
    blas_threads : int or None, optional
        BLAS threads to give each worker. None (default) autotunes by timing
        `_compute_self_energy_from_pack` across a few candidate thread counts and picking the fastest.
        Pass an int to force that value; it is still clamped to `cpu_budget // n_workers` to avoid
        oversubscription.

    Returns
    -------
    None
        Results are saved to disk as HDF5 files.
    """
    if self_energy_save_path is None:
        if lead_L.results_path != lead_R.results_path:
            log.warning("The results_path of lead_L and lead_R are different. "
                        "Self energy files will be saved in lead_L's results_path.")
        self_energy_save_path = os.path.join(lead_L.results_path, "self_energy")

    # Calculate safe number of workers based on available memory
    # Use first k-point for memory estimation
    sample_kpoint = kpoints_grid[0] if len(kpoints_grid) > 0 else None
    safe_n_workers = _get_safe_n_workers(lead_L, lead_R,
                                         requested_n_workers=n_workers,
                                         kpoint=sample_kpoint,
                                         cpu_budget=cpu_budget)
    if n_workers == -1:
        log.info(f"Auto-detected safe n_workers={safe_n_workers} based on available memory")
    elif safe_n_workers < n_workers:
        log.info(f"Adjusted n_workers from {n_workers} to {safe_n_workers} due to memory constraints")

    # Precompute all k-dependent matrices in the parent so workers receive
    # only plain tensors (the hamiltonian holds a torch.jit.ScriptFunction-
    # bearing model that loky/cloudpickle cannot serialize).
    leadL_pack = _precompute_lead_kdata(lead_L, kpoints_grid)
    leadR_pack = _precompute_lead_kdata(lead_R, kpoints_grid)

    # Choose BLAS threads-per-worker by autotuning on the real code path at the
    # sample (k, E). Cheaper than a hardware-agnostic table and always correct
    # for the actual N / BLAS backend the workers will run under.
    cpu_count = cpu_budget if cpu_budget is not None else os.cpu_count()
    sample_energy = energy_grid[0] if len(energy_grid) > 0 else 0.0
    blas_threads_per_worker = _autotune_blas_threads(
        leadL_pack, sample_kpoint, sample_energy, eta,
        safe_n_workers, cpu_count, se_numba_jit, requested=blas_threads,
    )

    total_tasks = [(k, e) for k in kpoints_grid for e in energy_grid]
    # Capture the parent's log level so loky workers (which start with a clean
    # logging state and the WARNING default) can match it when they reinit.
    parent_log_level = logging.getLogger().getEffectiveLevel()
    if len(total_tasks) <= ek_batch_size:
        Parallel(n_jobs=min(safe_n_workers, len(total_tasks)), backend="loky")(
            delayed(_self_energy_worker_blas)(k, e, eta, leadL_pack, leadR_pack,
                                               self_energy_save_path, se_numba_jit,
                                               parent_log_level, blas_threads_per_worker)
            for k, e in total_tasks
        )
    else:
        for i in range(0, len(total_tasks), ek_batch_size):
            batch = total_tasks[i:i+ek_batch_size]
            Parallel(n_jobs=min(safe_n_workers, len(batch)), backend="loky")(
                delayed(_self_energy_worker_blas)(k, e, eta, leadL_pack, leadR_pack,
                                                   self_energy_save_path, se_numba_jit,
                                                   parent_log_level, blas_threads_per_worker)
                for k, e in batch
            )


    save_path_L = os.path.join(self_energy_save_path, "self_energy_leadL.h5")
    save_path_R = os.path.join(self_energy_save_path, "self_energy_leadR.h5")

    merge_hdf5_files(self_energy_save_path, save_path_L, pattern="tmp_leadL_*.h5")
    merge_hdf5_files(self_energy_save_path, save_path_R, pattern="tmp_leadR_*.h5")


def _k_key(k):
    """Canonical tuple key for caching by k-point."""
    arr = np.asarray(k, dtype=float).reshape(3)
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _precompute_lead_kdata(lead, kpoints_grid):
    """Fetch every k-dependent quantity a lead needs during self-energy
    calculation, so workers can run without holding a reference to the
    hamiltonian (whose model contains torch.jit.ScriptFunction objects
    that cannot be pickled across loky processes).

    Parameters
    ----------
    lead : LeadProperty
    kpoints_grid : iterable of length-3 array-likes

    Returns
    -------
    dict
        Plain (pickleable) data: per-k Hamiltonian/overlap matrices and
        device subblocks, plus the small scalars used by `selfEnergy`
        and the Bloch unfolding.
    """
    pack = {
        "tab": lead.tab,
        "voltage": lead.voltage,
        "E_ref": lead.E_ref,
        "useBloch": lead.useBloch,
        "bloch_factor": lead.bloch_factor,
        "bloch_R_list": lead.bloch_R_list,
        "bloch_sorted_indice": lead.bloch_sorted_indice,
        "kdata": {},
    }

    bloch_unfolder = Bloch(lead.bloch_factor) if lead.useBloch else None
    seen = set()
    for k in kpoints_grid:
        key = _k_key(k)
        if key in seen:
            continue
        seen.add(key)

        subblocks = lead.hamiltonian.get_hs_device(k, only_subblocks=True)

        if not lead.useBloch:
            HLk, HLLk, HDLk, SLk, SLLk, SDLk = lead.hamiltonian.get_hs_lead(
                k, tab=lead.tab, v=lead.voltage
            )
            pack["kdata"][key] = _pack_lead_matrices(HLk, HLLk, HDLk, SLk, SLLk, SDLk, subblocks)
        else:
            kpoints_bloch = bloch_unfolder.unfold_points(list(np.asarray(k, dtype=float).reshape(3)))
            bloch_entries = []
            for k_bloch in kpoints_bloch:
                kb_tensor = torch.tensor(k_bloch)
                HLk, HLLk, HDLk, SLk, SLLk, SDLk = lead.hamiltonian.get_hs_lead(
                    kb_tensor, tab=lead.tab, v=lead.voltage
                )
                entry = _pack_lead_matrices(HLk, HLLk, HDLk, SLk, SLLk, SDLk, subblocks=None)
                entry["k_bloch"] = kb_tensor
                bloch_entries.append(entry)
            pack["kdata"][key] = {
                "subblocks": subblocks,
                "bloch_entries": bloch_entries,
            }

    return pack


def _to_numpy_c128(x):
    """Detach torch tensors and cast to a C-contiguous complex128 numpy array.

    Workers receive these plain arrays so `selfEnergy` / `surface_green` skip the
    per-energy torch→numpy round-trip, and the numba core can consume them
    directly without the `@njit` boundary re-inferring layout.
    """
    if isinstance(x, torch.Tensor):
        arr = x.detach().numpy()
    else:
        arr = np.asarray(x)
    return np.ascontiguousarray(arr, dtype=np.complex128)


def _pack_lead_matrices(HLk, HLLk, HDLk, SLk, SLLk, SDLk, subblocks):
    """Build a per-k pack entry.

    HLk/HLLk/SLk/SLLk go into the surface-Green core as numpy — convert them
    once here so workers don't repeat the torch→numpy round-trip per energy,
    and precompute ``h10 = conj(HLLk.T)`` / ``s10 = conj(SLLk.T)`` for the same
    reason. HDLk/SDLk stay as torch tensors because `LeadProperty.HDL_reduced`
    and the Bloch-branch matmul below use torch ops on them.
    """
    HLk_np = _to_numpy_c128(HLk)
    HLLk_np = _to_numpy_c128(HLLk)
    SLk_np = _to_numpy_c128(SLk)
    SLLk_np = _to_numpy_c128(SLLk)
    entry = {
        "HLk": HLk_np, "HLLk": HLLk_np, "HDLk": HDLk,
        "SLk": SLk_np, "SLLk": SLLk_np, "SDLk": SDLk,
        "h10": np.ascontiguousarray(np.conj(HLLk_np.T)),
        "s10": np.ascontiguousarray(np.conj(SLLk_np.T)),
    }
    if subblocks is not None:
        entry["subblocks"] = subblocks
    return entry


def _compute_self_energy_from_pack(pack, k, e, eta_lead, method="Lopez-Sancho", se_numba_jit=None):
    """Pure-function port of LeadProperty.self_energy_cal that operates on
    the dict produced by _precompute_lead_kdata. Mirrors the math in
    lead_property.py self_energy_cal (non-Bloch and Bloch branches)."""
    if not isinstance(e, torch.Tensor):
        energy = torch.tensor(e)
    else:
        energy = e

    entry = pack["kdata"][_k_key(k)]
    subblocks = entry["subblocks"]
    E_ref = pack["E_ref"]

    if not pack["useBloch"]:
        HDL_reduced, SDL_reduced = LeadProperty.HDL_reduced(
            entry["HDLk"], entry["SDLk"], subblocks
        )
        se, _ = selfEnergy(
            ee=energy,
            hL=entry["HLk"],
            hLL=entry["HLLk"],
            sL=entry["SLk"],
            sLL=entry["SLLk"],
            hDL=HDL_reduced,
            sDL=SDL_reduced,
            E_ref=E_ref,
            etaLead=eta_lead,
            method=method,
            numba_jit=se_numba_jit,
            h10=entry["h10"], s10=entry["s10"],
        )
        return se

    sgf_k = []
    bloch_factor = pack["bloch_factor"]
    m_size = bloch_factor[1] * bloch_factor[0]
    last_HDLk = None
    last_SDLk = None
    for be in entry["bloch_entries"]:
        k_bloch = be["k_bloch"]
        last_HDLk, last_SDLk = be["HDLk"], be["SDLk"]
        _, sgf = selfEnergy(
            ee=energy,
            hL=be["HLk"],
            hLL=be["HLLk"],
            sL=be["SLk"],
            sLL=be["SLLk"],
            E_ref=E_ref,
            etaLead=eta_lead,
            method=method,
            numba_jit=se_numba_jit,
            h10=be["h10"], s10=be["s10"],
        )
        phase_factor_m = torch.zeros([m_size, m_size], dtype=torch.complex128)
        bloch_R_list = pack["bloch_R_list"]
        for i in range(m_size):
            for j in range(m_size):
                if i == j:
                    phase_factor_m[i, j] = 1
                else:
                    phase_factor_m[i, j] = torch.exp(
                        torch.tensor(1j) * 2 * torch.pi
                        * torch.dot(bloch_R_list[j] - bloch_R_list[i], k_bloch)
                    )
        phase_factor_m = phase_factor_m.contiguous()
        sgf = sgf.contiguous()
        sgf_k.append(torch.kron(phase_factor_m, sgf))

    sgf_k = torch.sum(torch.stack(sgf_k), dim=0) / len(sgf_k)
    sorted_idx = pack["bloch_sorted_indice"]
    sgf_k = sgf_k[sorted_idx, :][:, sorted_idx]
    b = last_HDLk.shape[1]

    HDL_reduced, SDL_reduced = LeadProperty.HDL_reduced(last_HDLk, last_SDLk, subblocks)
    if not isinstance(energy, torch.Tensor):
        eeshifted = torch.scalar_tensor(energy, dtype=torch.complex128) + E_ref
    else:
        eeshifted = energy + E_ref
    se = (eeshifted * SDL_reduced - HDL_reduced) @ sgf_k[:b, :b] \
         @ (eeshifted * SDL_reduced.conj().T - HDL_reduced.conj().T)
    return se


def _init_worker_logging(level):
    """Attach a console handler to the root logger inside a loky worker.

    Loky workers start with no logging configuration, so log records emitted
    from `selfEnergy` / `surface_green` are silently dropped. Reuse the
    parent's formatter (CFORMATTER + _AppFilter from dpnegf.utils.loggers)
    so worker output matches parent output, but skip the version banner.

    Idempotent: returns immediately if a handler is already attached, so
    persistent loky workers pay the cost only once.
    """
    root_log = logging.getLogger()
    if root_log.handlers:
        return
    from dpnegf.utils.loggers import CFORMATTER, _AppFilter
    root_log.setLevel(level)
    ch = logging.StreamHandler()
    ch.setFormatter(CFORMATTER)
    ch.setLevel(level)
    ch.addFilter(_AppFilter())
    root_log.addHandler(ch)


def _self_energy_worker_pure(k, e, eta, leadL_pack, leadR_pack, self_energy_save_path, se_numba_jit, log_level):
    """joblib worker replacement that takes only pickleable packs."""
    _init_worker_logging(log_level)
    save_tmp_L = os.path.join(self_energy_save_path, f"tmp_leadL_k{k[0]}_{k[1]}_{k[2]}_E{e:.8f}.h5")
    save_tmp_R = os.path.join(self_energy_save_path, f"tmp_leadR_k{k[0]}_{k[1]}_{k[2]}_E{e:.8f}.h5")

    seL = _compute_self_energy_from_pack(leadL_pack, k, e, eta, se_numba_jit=se_numba_jit)
    seR = _compute_self_energy_from_pack(leadR_pack, k, e, eta, se_numba_jit=se_numba_jit)

    write_to_hdf5(save_tmp_L, k, e, seL)
    write_to_hdf5(save_tmp_R, k, e, seR)


def _self_energy_worker_blas(k, e, eta, leadL_pack, leadR_pack, self_energy_save_path,
                              se_numba_jit, log_level, blas_threads=1):
    """Loky entry point that pins this worker's BLAS/LAPACK runtime to a bounded
    thread count, then delegates to `_self_energy_worker_pure`.

    Each loky worker is a separate process whose BLAS library would otherwise
    autodetect every physical core, leading to N_workers * N_cores threads
    contending for N_cores cores. The Lopez-Sancho iteration in
    `surface_green._surface_green_{numba,scipy}_core` issues repeated
    `solve` / `inv` / matmul calls whose payoff from BLAS threading depends on
    the principal-layer dimension N: single-threaded already wins for small N,
    but multi-threaded solve/GEMM helps for larger N when the CPU budget left
    over from the memory-driven `n_workers` cap is not zero.

    `blas_threads` is chosen by `_autotune_blas_threads` in the parent and
    passed in per call; default 1 preserves the historical behavior for any
    external caller.
    """
    with threadpool_limits(limits=blas_threads, user_api='blas'):
        return _self_energy_worker_pure(
            k, e, eta, leadL_pack, leadR_pack,
            self_energy_save_path, se_numba_jit, log_level,
        )


def self_energy_worker(k, e, eta, lead_L, lead_R, self_energy_save_path, se_numba_jit=None):
    """
    Calculates the self-energy for left and right leads at a given k-point and energy,
    and saves the results to HDF5 files.

    Parameters
    ----------
    k : array-like
        The k-point in reciprocal space, typically a 3-element array or list.
    e : float
        The energy value at which to calculate the self-energy.
    eta : float
        A small imaginary part added to the energy for numerical stability.
    lead_L : object
        The left lead object, which must implement a `self_energy_cal` method.
    lead_R : object
        The right lead object, which must implement a `self_energy_cal` method.
    self_energy_save_path : str
        Directory path where the self-energy HDF5 files will be saved.

    Returns
    -------
    None
        The function saves the calculated self-energies to files and does not return anything.
    """

    save_tmp_L = os.path.join(self_energy_save_path, f"tmp_leadL_k{k[0]}_{k[1]}_{k[2]}_E{e:.8f}.h5")
    save_tmp_R = os.path.join(self_energy_save_path, f"tmp_leadR_k{k[0]}_{k[1]}_{k[2]}_E{e:.8f}.h5")

    seL = lead_L.self_energy_cal(kpoint=k, energy=e, eta_lead=eta, se_numba_jit=se_numba_jit)
    seR = lead_R.self_energy_cal(kpoint=k, energy=e, eta_lead=eta, se_numba_jit=se_numba_jit)

    write_to_hdf5(save_tmp_L, k, e, seL)
    write_to_hdf5(save_tmp_R, k, e, seR)


def write_to_hdf5(h5_path, k, e, se):
    with h5py.File(h5_path, "a") as f:
        group_name = f"E_{e:.8f}"
        dset_name = f"k_{k[0]}_{k[1]}_{k[2]}"
        grp = f.require_group(group_name)
        # if dset_name in grp:
        #     log.warning(f"Dataset {dset_name} already exists in group {group_name}. Skipping it.")
        grp.create_dataset(dset_name, data=se.cpu().numpy(), compression="gzip")
        f.flush()



def read_from_hdf5(h5_path, k, e):
    with h5py.File(h5_path, "r") as f:
        group_name = f"E_{e:.8f}"
        dset_name = f"k_{k[0]}_{k[1]}_{k[2]}"
        if group_name in f and dset_name in f[group_name]:
            return f[group_name][dset_name][:]
        else:
            raise KeyError(f"Data for kpoint {k} and energy {e} not found.")



def merge_hdf5_files(tmp_dir, output_path, pattern, remove=True):

    tmp_paths = sorted(glob.glob(os.path.join(tmp_dir, pattern)))
    if not tmp_paths:
        raise ValueError(f"No files matched pattern '{pattern}' in '{tmp_dir}'")

    log.info(f"Merging {len(tmp_paths)} tmp self energy files into {output_path}")

    with h5py.File(output_path, 'a') as fout:
        for path in tmp_paths:
            with h5py.File(path, 'r') as fin:
                for group_name in fin:
                    fin_group = fin[group_name]
                    fout_group = fout.require_group(group_name)

                    for dset_name in fin_group:
                        if dset_name in fout_group:
                            # log.warning(f"Dataset '{dset_name}' already exists in group '{group_name}'. Skipping.")
                            continue
                        fin_group.copy(dset_name, fout_group)

    log.info("Merge complete.")

    if remove:
        for path in tmp_paths:
            try:
                os.remove(path)
                # log.info(f"Deleted tmp file: {path}")
            except Exception as e:
                log.warning(f"Failed to delete {path}: {e}")


def _has_saved_self_energy(root: str) -> bool:
        from pathlib import Path
        p = Path(root) if root is not None else None
        if p is None or not p.exists():
            return False
        
        patterns = ("*.h5", "*.pth")
        for pat in patterns:
            if any(p.rglob(pat)):
                return True
        return False