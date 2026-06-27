from dpnegf.negf.recursive_green_cal import recursive_gf
import logging
import torch
import os
from typing import Union
from dpnegf.negf.negf_utils import update_kmap, update_temp_file,gauss_xw, leggauss
from dpnegf.negf.density import Ozaki
from dpnegf.utils.constants import Boltzmann, eV2J,pi
import numpy  as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt



"""
a Device object for calculating the Green's function, current, density of states, local density of states, and local current.
"""
log = logging.getLogger(__name__)


def _build_s_in_batched(hd, seinL, seinR, idx0, idy0, idx1, idy1):
    '''Allocate per-block [B, n_q, n_q] zeros and inject the corner self-energy slices.

    Mirrors the scalar construction in cal_green_function (the seinL/seinR fed in
    here are the already-scaled 1j*(seL - seL.mH) * f tensors of shape [B,n,n]).
    '''
    B = seinL.shape[0]
    s_in = [torch.zeros((B,) + tuple(blk.shape), dtype=torch.complex128, device=blk.device) for blk in hd]
    s_in[0][:, :idx0, :idy0] = s_in[0][:, :idx0, :idy0] + seinL[:, :idx0, :idy0]
    s_in[-1][:, -idx1:, -idy1:] = s_in[-1][:, -idx1:, -idy1:] + seinR[:, -idx1:, -idy1:]
    return s_in


class DeviceProperty(object):
    '''Device object for NEGF calculation

        a device object for NEGF (Non-Equilibrium Green's Function)
        calculations, which includes methods for computing Green's functions, calculating current, density
        of states, local density of states, and more.
        
        Property
        ----------
        greenfuncs
            a dictionary that contains the Green's function and its related variables.
        hamiltonian
             the Hamiltonian matrix of a system. 
        structure
            an object of the "ase.Atoms" class. 
        results_path
            a string that specifies the path where the results of thecalculations will be saved.    
        e_T 
            electron temperature in Kelvin.
        efermi
            the Fermi energy level. 
        mu
            the chemical potential of the device.
        dos
            the density of states (DOS) with spin multiplicity.
        ldos    
            the local density of states (LDOS) with spin multiplicity.
        current
            the current between the left and right leads.
        lcurrent
            the local current between different atoms.
        tc
            trasmission coefficient.
        various Green's functions tags
            see the docstring of the RGF class for details.

        Methods
        -------
        set_leadLR
            initialize the left and right lead in Device object
        cal_green_function
            computes the Green's function for a given energy and k-point in device.
        _cal_current_
            calculate the current based on the voltage difference
        _cal_current_nscf_
            calculates the non self consistent field (nscf) current.
        fermi_dirac
            calculates the Fermi-Dirac distribution function for a given energy.
        _cal_tc_
            calculate the transmission coefficient
        _cal_dos_
            calculate the density of states
        _cal_ldos_
            calculate the local density of states
        _cal_local_current_
            calculate the local current between different atoms
        _cal_density_
            calculate density matrix     
        
    '''    
    def __init__(self, hamiltonian, structure, results_path, e_T=300,
                 efermi: dict=None, chemiPot: dict=None, E_ref: float=None,
                 rgf_device: Union[str, torch.device]="cpu") -> None:
        self.greenfuncs = 0
        self.hamiltonian = hamiltonian
        self.structure = structure # ase Atoms
        self.results_path = results_path
        self.cdtype = torch.complex128
        if isinstance(rgf_device, str):
            rgf_device = torch.device(rgf_device)
        self.rgf_device = rgf_device
        self.kBT = Boltzmann * e_T / eV2J
        self.e_T = e_T
        # self.efermi = efermi
        self.chemiPot = chemiPot
        if E_ref is None:
            self.E_ref = efermi
            log.info(f"Using efermi as E_ref in DeviceProperty: {self.E_ref}")
        else:
            self.E_ref = E_ref

        self.kpoint = None  # kpoint for cal_green_function
        self.newK_flag = None # whether the kpoint is new or not in cal_green_function
        self.newV_flag = None # whether the voltage is new or not in cal_green_function
    
    def set_leadLR(self, lead_L, lead_R):
        '''initialize the left and right lead in Device object
        
        Parameters
        ----------
        lead_L
            the  lead obeject corresponding to the left lead
        lead_R
            the lead object corresponding to the right lead
        mu
            the chemical potential of the device
        
        '''
        self.lead_L = lead_L
        self.lead_R = lead_R
      # self.mu = self.efermi - 0.5*(self.lead_L.voltage + self.lead_R.voltage) # temporarily for NanoTCAD


    def cal_green_function(self, energy, kpoint, eta_device=0., block_tridiagonal=True, Vbias=None,
                           HS_inmem:bool=True, need_lesser:bool=False, need_greater:bool=False, need_gr_lc:bool=False):
        ''' computes the Green's function for a given energy and k-point in device.

        the tags used here to identify different Green's functions follows the NEGF theory 
        developed by Supriyo Datta in his book "Quantum Transport: Atom to Transistor". 
        The detials are listed in dpnegf/negf/recursive_green_cal.py docstring.
        
        Parameters
        ----------
        energy
            the energy at which the Green's function is evaluated.
        kpoint
            the k-point in the Brillouin zone.
        eta_device
            a float that represents the broadening factor used in the calculation of the Green's function.
            It is used to avoid the divergence of the Green's function at the poles of the Hamiltonian.
        block_tridiagonal
            A boolean parameter that shows whether the Hamiltonian matrix is block tridiagonal or not. 
            If set to True, the Hamiltonian matrix is assumed to have a block tridiagonal structure, 
            which can lead to computational efficiency in certain cases.
        HS_inmem
            A boolean parameter that shows whether the Hamiltonian/overlap is stored in memory after finishing 
            cal_green_function or not, which is important for large-scale calculations.
        need_lesser
            A boolean parameter that indicates whether the lesser Green's function is needed in the calculation.
            The lesser Green's function is used to calculate the electron density and current density.
        need_greater
            A boolean parameter that indicates whether the greater Green's function is needed in the calculation.
            The greater Green's function is used to calculate the hole density and phase-breaking scattering.
        need_gr_lc
            A boolean parameter that indicates whether the last column blocks of the retarded Green's function are needed.
        '''
        assert len(np.array(kpoint).reshape(-1)) == 3
        energy = torch.as_tensor(energy, dtype=torch.complex128, device=self.rgf_device)
        if energy.ndim == 0:
            energy = energy.reshape(1)
        assert energy.ndim == 1, f"energy must be 0-d, scalar, or 1-D [B]; got shape {tuple(energy.shape)}"
        B = energy.shape[0]
        batched_mode = B > 1

        self.block_tridiagonal = block_tridiagonal
        if self.kpoint is None or abs(self.kpoint - torch.tensor(kpoint)).sum() > 1e-5:
            self.kpoint = torch.tensor(kpoint)
            self.newK_flag = True
        else:
            self.newK_flag = False


        # if V is not None:
        #     HD_ = self.attachPotential(HD, SD, V)
        # else:
        #     HD_ = HD
        if  hasattr(self, "V"):
            self.oldV = self.V
        else:
            self.oldV = None

        if Vbias is None:
            if os.path.exists(os.path.join(self.results_path, "POTENTIAL.pth")):
                self.V = torch.load(os.path.join(self.results_path, "POTENTIAL.pth"))
            # elif abs([self.chemiPot[lead] - self.efermi[lead] for lead in ["lead_L","lead_R"]]).max() > 1e-7:
            #     self.V = torch.tensor(self.efermi - self.chemiPot)
            else:
                self.V = torch.tensor(0.)
        else:
            self.V = Vbias

        assert torch.is_tensor(self.V)
        self.V = self.V.to(self.rgf_device)
        if not self.oldV is None:
            if torch.abs(self.V - self.oldV).sum() > 1e-5:
                self.newV_flag = True
            else:
                self.newV_flag = False
        else:
            self.newV_flag = True  # for the first time to run cal_green_function in Poisson-NEGF SCF

        if (not (hasattr(self, "hd") and hasattr(self, "sd"))) or (self.newK_flag or self.newV_flag):
            self.hd, self.sd, self.hl, self.su, self.sl, self.hu = self.hamiltonian.get_hs_device(self.kpoint, self.V, block_tridiagonal)
            # TODO: if all blocks transferred to GPU, OOM may happen for large systems. 
            # Optimization should be implemented here.
            self.hd = [b.to(self.rgf_device) for b in self.hd]
            self.sd = [b.to(self.rgf_device) for b in self.sd]
            self.hl = [b.to(self.rgf_device) for b in self.hl]
            self.su = [b.to(self.rgf_device) for b in self.su]
            self.sl = [b.to(self.rgf_device) for b in self.sl]
            self.hu = [b.to(self.rgf_device) for b in self.hu]


        tags = ["g_trans","gr_lc", \
               "grd", "grl", "gru", "gr_left", \
               "gnd", "gnl", "gnu", "gin_left", \
               "gpd", "gpl", "gpu", "gip_left"]

        seL = self.lead_L.se.to(self.rgf_device)
        seR = self.lead_R.se.to(self.rgf_device)
        if batched_mode:
            assert seL.ndim == 3 and seR.ndim == 3, f"In batched mode, the self-energy should have shape [B,n,n], but got {seL.shape} and {seR.shape}"
        else:
            assert seL.ndim == 2 and seR.ndim == 2, f"In non-batched mode, the self-energy should have shape [n,n], but got {seL.shape} and {seR.shape}"
           
        s01, s02 = self.hd[0].shape  # The shape of the first H block
        se01, se02 = seL.shape[-2], seL.shape[-1]   # last two dims work for [n,n] and [B,n,n]
        s11, s12 = self.hd[-1].shape
        se11, se12 = seR.shape[-2], seR.shape[-1]
        idx0, idy0 = min(s01, se01), min(s02, se02)
        idx1, idy1 = min(s11, se11), min(s12, se12)
        if block_tridiagonal:
            # Based on the block tridiagonal algorithm, the shape of the self-energy should be
            # equal to or lesser than the corresponding Hamiltonian block
            if se01 > s01 or se02 > s02:
                log.warning(f"The shape of left self-energy ({se01},{se02}) is larger than\
                             the first Hamiltonian block ({s01},{s02}).")
                raise ValueError("Left Lead Self Energy size is larger than the first Hamiltonian Block.")
            if se11 > s11 or se12 > s12:
                log.warning(f"The shape of right self-energy ({se11},{se12}) is larger than\
                             the last Hamiltonian block ({s11},{s12}).")
                raise ValueError("Right Lead Self Energy size is larger than the last Hamiltonian Block.")

        green_funcs = {}

        if need_lesser:
            # Fluctuation-Dissipation theorem; only build s_in when the lesser GF is consumed
            if batched_mode:
                fL = self.lead_L.fermi_dirac(energy + self.E_ref).reshape(B, 1, 1)
                fR = self.lead_R.fermi_dirac(energy + self.E_ref).reshape(B, 1, 1)
                seinL = 1j * (seL - seL.mH) * fL
                seinR = 1j * (seR - seR.mH) * fR
                s_in = _build_s_in_batched(self.hd, seinL, seinR, idx0, idy0, idx1, idy1)
            else:
                seinL = 1j*(seL-seL.conj().T) * self.lead_L.fermi_dirac(energy+self.E_ref).reshape(-1)
                seinR = 1j*(seR-seR.conj().T) * self.lead_R.fermi_dirac(energy+self.E_ref).reshape(-1)
                s_in = [torch.zeros(i.shape, dtype=torch.complex128, device=self.rgf_device) for i in self.hd]
                s_in[0][:idx0,:idy0] = s_in[0][:idx0,:idy0] + seinL[:idx0,:idy0]
                s_in[-1][-idx1:,-idy1:] = s_in[-1][-idx1:,-idy1:] + seinR[-idx1:,-idy1:]
        else:
            s_in = 0

        # gr_left is only consumed inside the lesser/greater forward pass of the
        # kernel. If neither is active, the per-block list would sit on the GPU
        # unread; ask the kernel to drop it so its slots are freed mid-sweep.
        keep_gr_left = bool(need_lesser or need_greater)
        ans = recursive_gf(energy, hl=self.hl, hd=self.hd, hu=self.hu,
                            sd=self.sd, su=self.su, sl=self.sl,
                            left_se=seL, right_se=seR, seP=None, s_in=s_in,
                            s_out=None, eta=eta_device, E_ref=self.E_ref,
                            need_lesser=need_lesser, need_greater=need_greater,
                            need_gr_lc=need_gr_lc, keep_gr_left=keep_gr_left)
            # green shape [[g_trans, grd, grl,...],[g_trans, ...]]
        
        for t in range(len(tags)):
            green_funcs[tags[t]] = ans[t]

        self.greenfuncs = green_funcs

        if not HS_inmem:
            del self.hd, self.sd, self.hl, self.su, self.sl, self.hu

        # self.green = update_temp_file(update_fn=fn, file_path=GFpath, ee=ee, tags=tags, info="Computing Green's Function")

    def release_greenfuncs(self):
        '''Drop the Green's-function dict so the underlying rgf_device storage
        can be freed before the next energy chunk. H/S blocks are kept resident
        (they are k,V-dependent, not energy-dependent). The runner is
        responsible for restoring scalar lead.se references before calling
        this, so any batched [B,n,n] copies become collectable too.'''
        self.greenfuncs = 0
        if isinstance(self.rgf_device, torch.device) and self.rgf_device.type == "cuda":
            torch.cuda.empty_cache()

    def _cal_current_(self, espacing):
        '''calculate the current based on the voltage difference 

        At this stage, this method only supports the calculation of the current in the 
        non-self-consistent field (nscf) calculation. 
        
        So this function is not used.
        
        Parameters
        ----------
        espacing
            the spacing between energy grid points. It is used to determine the number of grid points 
            in the energy range defined by `xl` and `xu`.
        
        '''
        v_L = self.lead_L.voltage
        v_R = self.lead_R.voltage

        # check the energy grid satisfied the requirement
        xl = min(v_L, v_R)-4*self.kBT
        xu = max(v_L, v_R)+4*self.kBT

        self.__CURRENT__ = simpson(y=(self.lead_L.fermi_dirac(self.ee+self.E_ref) 
                                    - self.lead_R.fermi_dirac(self.ee+self.E_ref)) * self.tc, x=self.ee)

    def _cal_current_nscf_(self, energy_grid, tc):
        '''calculates the non self consistent field (nscf) current.

        Parameters
        ----------
        ee
            unit energy grid points in NEGF calculation
        tc
            Transmission calculated at zero bias voltage
        
        Returns
        -------
        vv
            voltage range
        cc
            calculated current

        '''
        if abs(self.lead_L.efermi-self.lead_R.efermi)<5e-4:
            log.warning(msg="The Fermi energy of the left and right leads should be equal in nscf current calculation.")
        efermi = self.lead_L.efermi
        f = lambda x,mu: 1 / (1 + torch.exp((x - mu) / self.kBT))

        emin = energy_grid.min()
        emax = energy_grid.max()
        vmin = emin + 4*self.kBT
        vmax = emax - 4*self.kBT
        vm = 0.5 * (vmin+vmax)
        vmid = vm - vmin
        
        vv = torch.linspace(start=0., end=vmid, steps=int(vmid / 0.1)+1) * 2
        cc = []

        for dv in vv * 0.5:
            I = simpson(y=(f(energy_grid+efermi, efermi-vm+dv) 
                           -f(energy_grid+efermi, efermi-vm-dv)) * tc, x=energy_grid)
            cc.append(I)

        return vv, cc


    def _cal_tc_(self):
        '''calculate the transmission coefficient

        Returns
        -------
           tc is the transmission coefficient

        '''

        g_trans = self.g_trans
        batched = g_trans.ndim == 3
        tx, ty = g_trans.shape[-2], g_trans.shape[-1]
        gammaL_full = self.lead_L.gamma.to(self.rgf_device)
        gammaR_full = self.lead_R.gamma.to(self.rgf_device)
        lx = gammaL_full.shape[-2]
        rx = gammaR_full.shape[-2]
        x0 = min(lx, tx)
        x1 = min(rx, ty)

        gL_shape = (g_trans.shape[0], tx, tx) if batched else (tx, tx)
        gR_shape = (g_trans.shape[0], ty, ty) if batched else (ty, ty)
        gammaL = torch.zeros(size=gL_shape, dtype=self.cdtype, device=self.rgf_device)
        gammaR = torch.zeros(size=gR_shape, dtype=self.cdtype, device=self.rgf_device)
        if batched:
            gammaL[:, :x0, :x0] = gammaL[:, :x0, :x0] + gammaL_full[:, :x0, :x0]
            gammaR[:, -x1:, -x1:] = gammaR[:, -x1:, -x1:] + gammaR_full[:, -x1:, -x1:]
        else:
            gammaL[:x0, :x0] += gammaL_full[:x0, :x0]
            gammaR[-x1:, -x1:] += gammaR_full[-x1:, -x1:]

        tc = (gammaL @ g_trans @ gammaR @ g_trans.mH).diagonal(dim1=-2, dim2=-1).real.sum(-1)

        return tc
    
    def _cal_dos_(self):
        ''' calculates the density of states (DOS) using a given set of diagonal blocks.
        
        Returns
        -------
            DOS with spin multiplicity
        '''
        dos = 0
        #TODO: transfer cal_dos to static method for any k and energy
        if (not(hasattr(self, "hd") and hasattr(self, "sd"))) or (self.newK_flag or self.newV_flag):
            self.hd, self.sd, self.hl, self.su, self.sl, self.hu = \
                self.hamiltonian.get_hs_device(self.kpoint, self.V, self.block_tridiagonal)
            # defensive .to(self.device) in case the blocks came back from a cached/legacy path on CPU.
            self.hd = [b.to(self.rgf_device) for b in self.hd]
            self.sd = [b.to(self.rgf_device) for b in self.sd]
            self.hl = [b.to(self.rgf_device) for b in self.hl]
            self.su = [b.to(self.rgf_device) for b in self.su]
            self.sl = [b.to(self.rgf_device) for b in self.sl]
            self.hu = [b.to(self.rgf_device) for b in self.hu]

        for jj in range(len(self.grd)):
            if not self.block_tridiagonal or len(self.gru) == 0:
                temp = self.grd[jj] @ self.sd[jj] # taking each diagonal block with all energy e together
            else:
                if jj == 0:
                    temp = self.grd[jj] @ self.sd[jj] + self.gru[jj] @ self.sl[jj]
                elif jj == len(self.grd)-1:
                    temp = self.grd[jj] @ self.sd[jj] + self.grl[jj-1] @ self.su[jj-1]
                else:
                    temp = self.grd[jj] @ self.sd[jj] + self.grl[jj-1] @ self.su[jj-1] + self.gru[jj] @ self.sl[jj]
            dos -= temp.imag.diagonal(dim1=-2, dim2=-1).sum(-1) / pi
        return dos * 2

    def _cal_ldos_(self):
        ''' calculates the local density of states (LDOS) for a given Hamiltonian and k-point.
        
        Returns
        -------
            LDOS with spin multiplicity
        
        '''
        ldos = []
        # sd = self.hamiltonian.get_hs_device(kpoint=self.kpoint, V=self.V, block_tridiagonal=self.block_tridiagonal)[1]
        for jj in range(len(self.grd)):
            if not self.block_tridiagonal or len(self.gru) == 0:
                temp = self.grd[jj] @ self.sd[jj] # taking each diagonal block with all energy e together
            else:
                if jj == 0:
                    temp = self.grd[jj] @ self.sd[jj] + self.gru[jj] @ self.sl[jj]
                elif jj == len(self.grd)-1:
                    temp = self.grd[jj] @ self.sd[jj] + self.grl[jj-1] @ self.su[jj-1]
                else:
                    temp = self.grd[jj] @ self.sd[jj] + self.grl[jj-1] @ self.su[jj-1] + self.gru[jj] @ self.sl[jj]
            ldos.append(-temp.imag.diagonal(dim1=-2, dim2=-1) / pi) # [n_q] or [B, n_q]

        ldos = torch.cat(ldos, dim=-1).contiguous()

        norbs = [0]+self.norbs_per_atom
        accmap = np.cumsum(norbs)
        ldos = torch.stack([ldos[..., accmap[i]:accmap[i+1]].sum(-1) for i in range(len(accmap)-1)], dim=-1)

        # return ldos*2
        return ldos*2

    def _cal_local_current_(self):
        '''calculate the local current between different atoms

        At this stage, local current calculation only support non-block-triagonal format Hamiltonian

        Returns
        -------
            the local current

        '''
        # TODO(batched-energy): vectorize then batch — currently expects scalar-E gnd[0] (2-D).
        # current only support non-block-triagonal format
        v_L = self.lead_L.voltage
        v_R = self.lead_R.voltage

        # check the energy grid satisfied the requirement
        
        na = len(self.norbs_per_atom)
        local_current = torch.zeros(na, na, device=self.rgf_device)
        hd = self.hamiltonian.get_hs_device(kpoint=self.kpoint, V=self.V, block_tridiagonal=self.block_tridiagonal)[0][0]
        hd = hd.to(self.rgf_device)  # defensive .to(self.device) in case the block came back from a cached/legacy path on CPU.

        for i in range(na):
            for j in range(na):
                if i != j:
                    id = self.get_index(i)
                    jd = self.get_index(j)
                    ki = hd[id[0]:id[1], jd[0]:jd[1]] @ (1j*self.gnd[0][jd[0]:jd[1],id[0]:id[1]])
                    kj = hd[jd[0]:jd[1], id[0]:id[1]] @ (1j*self.gnd[0][id[0]:id[1],jd[0]:jd[1]])
                    local_current[i,j] = ki.real.diag().sum() - kj.real.diag().sum()
        
        return local_current.contiguous()
    
    def _cal_density_(self, dm_options):
        ''' calculate the density matrix
        
        Parameters
        ----------
        dm_options
            a dictionary that contains options for the `Ozaki` class. It is used  to initialize 
            an instance of the `Ozaki` class with the specified options. The `Ozaki` class is then
            used to calculate the density matrix
        
        Returns
        -------
            the variables DM_eq and DM_neq.
        
        '''
        dm = Ozaki(**dm_options)
        DM_eq, DM_neq = dm.integrate(deviceprop=self.rgf_device, kpoint=self.kpoint)

        return DM_eq, DM_neq
    
    # @property
    # def current_nscf(self):
    #     return self._cal_current_nscf_()


    @property
    def dos(self):
        return self._cal_dos_()
        
    @property
    def current(self):
        return self._cal_current_()
    
    @property
    def ldos(self):
        return self._cal_ldos_()

    @property
    def tc(self):
        return self._cal_tc_()
        
    @property
    def lcurrent(self):
        return self._cal_local_current_()


    @property
    def g_trans(self):
        return self.greenfuncs["g_trans"] # [n,n]
    @property
    def gr_lc(self): # last column of Gr
        return self.greenfuncs["gr_lc"]  
    @property
    def grd(self):
        return self.greenfuncs["grd"] # [[n,n]]
    
    @property
    def grl(self):
        return self.greenfuncs["grl"]
    
    @property
    def gru(self):
        return self.greenfuncs["gru"]
    
    @property
    def gr_left(self):
        return self.greenfuncs["gr_left"]
    
    @property
    def gnd(self):
        return self.greenfuncs["gnd"]
    
    @property
    def gnl(self):
        return self.greenfuncs["gnl"]
    
    @property
    def gnu(self):
        return self.greenfuncs["gnu"]
    
    @property
    def gin_left(self):
        return self.greenfuncs["gin_left"]
    
    @property
    def gpd(self):
        return self.greenfuncs["gpd"]
    
    @property
    def gpl(self):
        return self.greenfuncs["gpl"]
    
    @property
    def gpu(self):
        return self.greenfuncs["gpu"]
    
    @property
    def gip_left(self):
        return self.greenfuncs["gip_left"]
    
    @property
    def norbs_per_atom(self):
        return self.hamiltonian.device_norbs

    @property
    def positions(self):
        return self.structure.positions
    
    def get_index(self, iatom):
        '''returns the start and end indices of orbitals for a specific atom in a system.
        
        Parameters
        ----------
        iatom
            the index of the atom for which we want to calculate the start and end orbital indices.
        
        Returns
        -------
            a list containing the start and end orbital indices for a specific atom in a system.
        
        '''
        start = sum(self.norbs_per_atom[:iatom])
        end = start + self.norbs_per_atom[iatom]

        return [start, end]
    
    def get_index_block(self, iatom):
        pass