from typing import Dict, Union
from dargs import Argument, Variant
import logging
from numbers import Number


log = logging.getLogger(__name__)


def common_options():
    doc_device = "The device to run the calculation, choose among `cpu` and `cuda[:int]`, Default: `cpu`"
    doc_dtype = """The digital number's precison, choose among:
                    Default: `float32`
                        - `float32`: indicating torch.float32
                        - `float64`: indicating torch.float64
                """

    doc_seed = "The random seed used to initialize the parameters and determine the shuffling order of datasets. Default: `3982377700`"
    doc_basis = "The atomic orbitals used to construct the basis. e.p. {'A':['2s','2p','s*'],'B':'[3s','3p']}"
    doc_overlap = "Whether to calculate the overlap matrix. Default: False"

    args = [
        Argument("basis", dict, optional=False, doc=doc_basis),
        Argument("overlap", bool, optional=True, default=False, doc=doc_overlap),
        Argument("device", str, optional = True, default="cpu", doc = doc_device),
        Argument("dtype", str, optional = True, default="float32", doc = doc_dtype),
        Argument("seed", int, optional=True, default=3982377700, doc=doc_seed),
    ]

    doc_common_options = ""

    return Argument("common_options", dict, optional=False, sub_fields=args, sub_variants=[], doc=doc_common_options)




def tbtrans_negf():
    doc_scf = "Whether to run a self-consistent (Poisson-NEGF) loop. Default: False (non-self-consistent transport)."
    doc_block_tridiagonal = ("Whether to block-tridiagonalize the device Hamiltonian for "
                             "faster recursive Green's function inversion. Default: False.")
    doc_ele_T = "Electronic temperature in Kelvin. Required."
    doc_unit = "Energy unit used for the energy grid and outputs. Choose 'Hartree' (default) or 'eV'."
    doc_scf_options = "SCF mixing options (only used when `scf` is true)."
    doc_stru_options = "Structure partition and k-mesh options: device region, left/right leads, PBC, k-mesh."
    doc_poisson_options = "Poisson solver options (only used when `scf` is true)."
    doc_sgf_solver = "Surface Green's function solver. 'Sancho-Rubio' (default) or 'Lopez-Sancho'."
    doc_espacing = "Energy grid spacing in `unit`. Required."
    doc_emin = "Lower bound of the energy grid in `unit`. Required."
    doc_emax = "Upper bound of the energy grid in `unit`. Required."
    doc_e_fermi = "Device Fermi level in `unit`. Required."
    doc_eta_lead = "Small imaginary broadening added to the lead energy. Default: 1e-5."
    doc_eta_device = "Small imaginary broadening added to the device energy. Default: 0."
    doc_out_dos = "Whether to write density of states. Default: False."
    doc_out_tc = "Whether to write transmission coefficient. Default: False."
    doc_out_current = "Whether to write self-consistent current. Default: False."
    doc_out_current_nscf = "Whether to write non-self-consistent current. Default: False."
    doc_out_ldos = "Whether to write local density of states. Default: False."
    doc_out_density = "Whether to write electron density. Default: False."
    doc_out_lcurrent = "Whether to write local current. Default: False."
    doc_density_options = "Density integration options (Ozaki or Fiori method)."
    doc_out_potential = "Whether to write the self-consistent electrostatic potential. Default: False."

    return [
        Argument("scf", bool, optional=True, default=False, doc=doc_scf),
        Argument("block_tridiagonal", bool, optional=True, default=False, doc=doc_block_tridiagonal),
        Argument("ele_T", [float, int], optional=False, doc=doc_ele_T),
        Argument("unit", str, optional=True, default="Hartree", doc=doc_unit),
        Argument("scf_options", dict, optional=True, default={}, sub_fields=[], sub_variants=[scf_options()], doc=doc_scf_options),
        Argument("stru_options", dict, optional=False, sub_fields=stru_options(), doc=doc_stru_options),
        Argument("poisson_options", dict, optional=True, default={}, sub_fields=[], sub_variants=[poisson_options()], doc=doc_poisson_options),
        Argument("sgf_solver", str, optional=True, default="Sancho-Rubio", doc=doc_sgf_solver),
        Argument("espacing", [int, float], optional=False, doc=doc_espacing),
        Argument("emin", [int, float], optional=False, doc=doc_emin),
        Argument("emax", [int, float], optional=False, doc=doc_emax),
        Argument("e_fermi", [int, float], optional=False, doc=doc_e_fermi),
        Argument("density_options", dict, optional=True, default={}, sub_fields=[], sub_variants=[density_options()], doc=doc_density_options),
        Argument("eta_lead", [int, float], optional=True, default=1e-5, doc=doc_eta_lead),
        Argument("eta_device", [int, float], optional=True, default=0., doc=doc_eta_device),
        Argument("out_dos", bool, optional=True, default=False, doc=doc_out_dos),
        Argument("out_tc", bool, optional=True, default=False, doc=doc_out_tc),
        Argument("out_density", bool, optional=True, default=False, doc=doc_out_density),
        Argument("out_potential", bool, optional=True, default=False, doc=doc_out_potential),
        Argument("out_current", bool, optional=True, default=False, doc=doc_out_current),
        Argument("out_current_nscf", bool, optional=True, default=False, doc=doc_out_current_nscf),
        Argument("out_ldos", bool, optional=True, default=False, doc=doc_out_ldos),
        Argument("out_lcurrent", bool, optional=True, default=False, doc=doc_out_lcurrent)
    ]




def negf():
    doc_scf = "Whether to run a self-consistent (Poisson-NEGF) loop. Default: False (non-self-consistent transport)."
    doc_block_tridiagonal = ("Whether to block-tridiagonalize the device Hamiltonian for "
                             "faster recursive Green's function inversion. Default: False.")
    doc_ele_T = "Electronic temperature in Kelvin. Required."
    doc_unit = "Energy unit used for the energy grid and outputs. Choose 'Hartree' (default) or 'eV'."
    doc_scf_options = "SCF mixing options (only used when `scf` is true)."
    doc_stru_options = "Structure partition and k-mesh options: device region, left/right leads, PBC, k-mesh."
    doc_poisson_options = "Poisson solver options (only used when `scf` is true)."
    doc_espacing = "Energy grid spacing in `unit`. Required."
    doc_emin = "Lower bound of the energy grid in `unit`. Required."
    doc_emax = "Upper bound of the energy grid in `unit`. Required."
    doc_e_fermi = ("Device Fermi level in `unit`. Optional; if None, computed from the "
                   "lead Fermi level and applied to the device.")
    doc_eta_lead = "Small imaginary broadening added to the lead energy. Default: 1e-5."
    doc_eta_device = "Small imaginary broadening added to the device energy. Default: 0."

    doc_self_energy_options = ("Self-energy stage options: SGF solver, numba JIT, "
                               "on-disk cache, and CPU parallelism (joblib workers, "
                               "BLAS threads, ek batching).")
    doc_rgf_options = ("Recursive Green's function stage options: compute device "
                       "('cpu' or 'cuda') and energy-loop chunk size. Independent "
                       "from the self-energy pool, which always runs on CPU.")
    doc_hs_cache = "On-disk cache of the device Hamiltonian and overlap matrix."
    doc_density_options = ""
    doc_output_options = "Which physical quantities to write out at the end of the run."

    return [
        Argument("scf", bool, optional=True, default=False, doc=doc_scf),
        Argument("block_tridiagonal", bool, optional=True, default=False, doc=doc_block_tridiagonal),
        Argument("plot_blocks", bool, optional=True, default=False, doc="Whether to plot the block tridiagonalization process"),
        Argument("ele_T", [float, int], optional=False, doc=doc_ele_T),
        Argument("unit", str, optional=True, default="Hartree", doc=doc_unit),
        Argument("hs_cache", dict, optional=True, default={}, sub_fields=hs_cache_options(), doc=doc_hs_cache),
        Argument("scf_options", dict, optional=True, default={}, sub_fields=[], sub_variants=[scf_options()], doc=doc_scf_options),
        Argument("stru_options", dict, optional=False, sub_fields=stru_options(), doc=doc_stru_options),
        Argument("poisson_options", dict, optional=True, default={}, sub_fields=[], sub_variants=[poisson_options()], doc=doc_poisson_options),
        Argument("self_energy_options", dict, optional=True, default={}, sub_fields=self_energy_options(), doc=doc_self_energy_options),
        Argument("espacing", [int, float], optional=False, doc=doc_espacing),
        Argument("emin", [int, float], optional=False, doc=doc_emin),
        Argument("emax", [int, float], optional=False, doc=doc_emax),
        Argument("rgf_options", dict, optional=True, default={}, sub_fields=rgf_options_group(), doc=doc_rgf_options),
        Argument("e_fermi", [int, float], optional=True, default=None ,doc=doc_e_fermi),
        Argument("density_options", dict, optional=True, default={}, sub_fields=[], sub_variants=[density_options()], doc=doc_density_options),
        Argument("eta_lead", [int, float], optional=True, default=1e-5, doc=doc_eta_lead),
        Argument("eta_device", [int, float], optional=True, default=0., doc=doc_eta_device),
        Argument("output_options", dict, optional=True, default={}, sub_fields=output_options(), doc=doc_output_options),
    ]


def output_options():
    return [
        Argument("dos", bool, optional=True, default=False, doc="Write density of states."),
        Argument("tc", bool, optional=True, default=False, doc="Write transmission coefficient."),
        Argument("density", bool, optional=True, default=False, doc="Write electron density."),
        Argument("potential", bool, optional=True, default=False, doc="Write electrostatic potential."),
        Argument("current", bool, optional=True, default=False, doc="Write self-consistent current."),
        Argument("current_nscf", bool, optional=True, default=False, doc="Write non-self-consistent current."),
        Argument("ldos", bool, optional=True, default=False, doc="Write local density of states."),
        Argument("lcurrent", bool, optional=True, default=False, doc="Write local current."),
    ]


def self_energy_cache_options():
    return [
        Argument("use_saved", bool, optional=True, default=False,
                 doc="Whether to load self-energy from an on-disk HDF5 cache instead of recomputing."),
        Argument("save_path", [str, None], optional=True, default=None,
                 doc="Directory that holds (or will hold) the self-energy HDF5 files. "
                     "If None, defaults to `<results_path>/self_energy`."),
    ]


def self_energy_parallel_options():
    doc_n_workers = ("Number of joblib workers used to parallelize the self-energy "
                     "sweep over (k, E). -1 (default) auto-selects based on the CPU "
                     "and memory budget (see `_get_safe_n_workers`).")
    doc_cpu_budget = ("Total CPU cores the self-energy pool is allowed to size against. "
                      "None (default) uses os.cpu_count(). This bounds both `n_workers` "
                      "and `blas_threads` so BLAS-thread * worker <= cpu_budget.")
    doc_blas_threads = ("BLAS/LAPACK threads per worker inside the Lopez-Sancho loop. "
                       "None (default) auto-tunes by timing a sample (k, E) at 1, 2, 4, "
                       "8, and cpu_budget/n_workers threads and keeping the fastest.")
    doc_ek_batch = ("Batch size for (k, E) tasks handed to joblib. Larger = fewer "
                    "batches, more memory. Default 200.")
    return [
        Argument("n_workers", [int, None], optional=True, default=-1, doc=doc_n_workers),
        Argument("cpu_budget", [int, None], optional=True, default=None, doc=doc_cpu_budget),
        Argument("blas_threads", [int, None], optional=True, default=None, doc=doc_blas_threads),
        Argument("ek_batch_size", int, optional=True, default=200, doc=doc_ek_batch),
    ]


def self_energy_options():
    doc_solver = ("Surface Green's function solver. 'Sancho-Rubio' (default) or "
                  "'Lopez-Sancho'.")
    doc_numba_jit = ("Whether to JIT-compile the surface Green's function core with "
                     "numba. None (default) uses numba when available.")
    doc_info_display = ("Whether to log per-(k, E) self-energy solver info. Verbose; "
                        "off by default.")
    doc_cache = "On-disk cache of the self-energy for reuse across runs."
    doc_parallel = "CPU parallelism knobs for the self-energy sweep."
    return [
        Argument("solver", str, optional=True, default="Sancho-Rubio", doc=doc_solver),
        Argument("numba_jit", [bool, None], optional=True, default=None, doc=doc_numba_jit),
        Argument("info_display", bool, optional=True, default=False, doc=doc_info_display),
        Argument("cache", dict, optional=True, default={}, sub_fields=self_energy_cache_options(), doc=doc_cache),
        Argument("parallel", dict, optional=True, default={}, sub_fields=self_energy_parallel_options(), doc=doc_parallel),
    ]


def rgf_options_group():
    doc_device = ("Device used only for the RGF (recursive Green's function) step. "
                  "'cpu' (default) or 'cuda'. Hamiltonian initialization always runs "
                  "on CPU (it is the memory-heavy phase). Self-energy always runs on "
                  "CPU because it goes through joblib + numba.")
    doc_e_batch = ("Number of energy points solved in one batched RGF call. None "
                   "(default) auto-picks from free CUDA memory on 'cuda'; on 'cpu' "
                   "the full grid is used.")
    return [
        Argument("device", str, optional=True, default="cpu", doc=doc_device),
        Argument("e_batch_size", [int, float, None], optional=True, default=None, doc=doc_e_batch),
    ]


def hs_cache_options():
    return [
        Argument("use_saved", bool, optional=True, default=False,
                 doc="Whether to load the device Hamiltonian and overlap from an on-disk cache."),
        Argument("save_path", [str, None], optional=True, default=None,
                 doc="Path to the saved Hamiltonian / overlap file. Required when use_saved is true."),
    ]

def stru_options():
    doc_kmesh = "K-mesh used for the device transverse Brillouin-zone sampling, [nkx, nky, nkz]. Default: [1, 1, 1]."
    doc_pbc = ("Periodic boundary condition applied to the device, list of three booleans "
               "for x/y/z. Default: [False, False, False].")
    doc_device = "Device region: atom index range and sorting."
    doc_lead_L = "Left lead: atom index range, voltage, k-mesh for lead Fermi-level, Bloch expansion."
    doc_lead_R = "Right lead: atom index range, voltage, k-mesh for lead Fermi-level, Bloch expansion."
    doc_gamma_center = "Whether the k-mesh is Gamma-centered. Default: True."
    doc_time_reversal_symmetry = "Whether to enforce time-reversal symmetry when folding the k-mesh. Default: True."
    doc_e_fermi_smearing = "Smearing method for lead Fermi-level determination. Default: 'FD' (Fermi-Dirac)."
    doc_eig_solver = "Eigenvalue solver used for lead Fermi-level calculation. Default: 'torch'."
    doc_nel_atom = "Number of valence electrons per element, e.g. {'C': 4}. Required for SCF or when computing lead Fermi-level."
    return [
        Argument("device", dict, optional=False, sub_fields=device(), doc=doc_device),
        Argument("lead_L", dict, optional=False, sub_fields=lead(), doc=doc_lead_L),
        Argument("lead_R", dict, optional=False, sub_fields=lead(), doc=doc_lead_R),
        Argument("kmesh", list, optional=True, default=[1,1,1], doc=doc_kmesh),
        Argument("pbc", list, optional=True, default=[False, False, False], doc=doc_pbc),
        Argument("gamma_center", list, optional=True, default=True, doc=doc_gamma_center),
        Argument("time_reversal_symmetry", list, optional=True, default=True, doc=doc_time_reversal_symmetry),
        Argument("e_fermi_smearing", str, optional=True, default="FD", doc=doc_e_fermi_smearing),
        Argument("eig_solver", str, optional=True, default="torch", doc=doc_eig_solver),
        Argument("nel_atom", [dict,None], optional=True, default=None, doc=doc_nel_atom)
    ]

def device():
    doc_id = "Atom index range of the device region, e.g. '32-64' (0-indexed, half-open)."
    doc_sort = "Whether to sort atoms in the device region along the transport axis. Default: True."

    return [
        Argument("id", str, optional=False, doc=doc_id),
        Argument("sort", bool, optional=True, default=True, doc=doc_sort)
    ]

def lead():
    doc_id = "Atom index range of the lead region, e.g. '0-32' (0-indexed, half-open)."
    doc_voltage = "Applied bias voltage on this lead, in `unit`. Required."
    doc_useBloch = "Whether to use Bloch expansion to reduce lead unit cell size. Default: False."
    doc_bloch_factor = "Bloch factor [nx, ny, nz] for the lead unit cell. Default: [1, 1, 1]."
    doc_kmesh_lead_Ef = "K-mesh used to compute the lead Fermi level, [nkx, nky, nkz]."
    doc_charge = "Doping charge on the lead, used when computing the lead Fermi level. Default: 0.0."
    return [
        Argument("id", str, optional=False, doc=doc_id),
        Argument("voltage", [int, float], optional=False, doc=doc_voltage),
        Argument("useBloch", bool, optional=True, default=False, doc=doc_useBloch),
        Argument("bloch_factor", list, optional=True, default=[1,1,1], doc=doc_bloch_factor),
        Argument("kmesh_lead_Ef", list, optional=True, doc=doc_kmesh_lead_Ef),
        Argument("charge", [int, float], optional=True, default=0.0, doc=doc_charge)
    ]

def scf_options():
    doc_mode = "SCF mixing algorithm. Currently only 'PDIIS' is supported."
    doc_PDIIS = "PDIIS (Periodic-DIIS) mixing parameters."

    return Variant("mode", [
        Argument("PDIIS", dict, PDIIS(), doc=doc_PDIIS)
        ], optional=True, default_tag="PDIIS", doc=doc_mode)

def PDIIS():
    doc_mixing_period = "How many iterations between DIIS extrapolations. Default: 3."
    doc_step_size = "Linear mixing step size between DIIS steps. Default: 0.05."
    doc_n_history = "Number of previous residuals kept for the DIIS subspace. Default: 6."
    doc_abs_err = "Absolute convergence tolerance on the SCF residual. Default: 1e-6."
    doc_rel_err = "Relative convergence tolerance on the SCF residual. Default: 1e-4."
    doc_max_iter = "Maximum SCF iterations. Default: 100."

    return [
        Argument("mixing_period", int, optional=True, default=3, doc=doc_mixing_period),
        Argument("step_size", [int, float], optional=True, default=0.05, doc=doc_step_size),
        Argument("n_history", int, optional=True, default=6, doc=doc_n_history),
        Argument("abs_err", [int, float], optional=True, default=1e-6, doc=doc_abs_err),
        Argument("rel_err", [int, float], optional=True, default=1e-4, doc=doc_rel_err),
        Argument("max_iter", int, optional=True, default=100, doc=doc_max_iter)
    ]

def poisson_options():
    doc_solver = ("Poisson solver backend. 'fmm' (default) uses the free-space Fast Multipole "
                  "Method; 'pyamg' uses an algebraic-multigrid finite-difference solver on a "
                  "grid; 'scipy' uses SciPy's sparse linear algebra on a grid.")
    doc_fmm = "FMM solver options."
    doc_pyamg = "pyamg (algebraic multigrid) solver options: grid, gates, dielectric and doped regions."
    doc_scipy = "SciPy sparse solver options: grid, gates, dielectric and doped regions."
    return Variant("solver", [
        Argument("fmm", dict, fmm(), doc=doc_fmm),
        Argument("pyamg", dict, pyamg(), doc=doc_pyamg),
        Argument("scipy", dict, scipy(), doc=doc_scipy)
    ], optional=True, default_tag="fmm", doc=doc_solver)

def density_options():
    doc_method = ("Density integration method. 'Ozaki' (default) uses the Ozaki contour with "
                  "Matsubara-pole expansion; 'Fiori' uses real-axis integration with Gauss "
                  "quadrature (used with the Fiori/effective-mass approach).")
    doc_Ozaki = "Ozaki-contour options."
    doc_Fiori = "Fiori real-axis integration options."
    return Variant("method", [
        Argument("Ozaki", dict, Ozaki(), doc=doc_Ozaki),
        Argument("Fiori", dict, Fiori(), doc=doc_Fiori)
    ], optional=True, default_tag="Ozaki", doc=doc_method)

def Ozaki():
    doc_M_cut = "Number of Ozaki poles kept. Default: 30."
    doc_R = "Radius of the semicircular contour in `unit`. Default: 1e6."
    doc_n_gauss = "Number of Gauss-Legendre points on the equilibrium contour. Default: 10."
    return [
        Argument("R", [int, float], optional=True, default=1e6, doc=doc_R),
        Argument("M_cut", int, optional=True, default=30, doc=doc_M_cut),
        Argument("n_gauss", int, optional=True, default=10, doc=doc_n_gauss),
    ]

def Fiori():
    doc_n_gauss = "Number of Gauss quadrature points along the real-axis integration. Default: 100."
    doc_integrate_way = "Integration strategy for the Fiori method: 'direct' or 'gauss'. Default: 'direct'."
    return [
        Argument("integrate_way", int, optional=True, default='direct', doc=doc_integrate_way),
        Argument("n_gauss", int, optional=True, default=100, doc=doc_n_gauss)
    ]

def fmm():
    doc_err = "Requested FMM accuracy on the Coulomb sum. Default: 1e-5."

    return [
        Argument("err", [int, float], optional=True, default=1e-5, doc=doc_err)
    ]

def pyamg():
    doc_err = "Error target passed to the AMG solver. Default: 1e-5."
    doc_tolerance = "Convergence tolerance of the AMG linear solve. Default: 1e-7."
    doc_grid = "Real-space grid used to discretize the Poisson equation."
    doc_gate = "Dirichlet boundary condition for a metallic gate (x/y/z range and applied voltage)."
    doc_dielectric = ("Rectangular region with a fixed relative permittivity. Additional regions "
                       "can be added via `dielectric_region2` ... `dielectric_region6`.")
    doc_doped = "Rectangular region with a fixed doping charge density."
    doc_max_iter = "Maximum AMG iterations. Default: 100."
    doc_mix_rate = "Underrelaxation between successive Poisson solves. Default: 0.25."
    doc_poisson_dtype = "Floating precision of the Poisson solver ('float32' or 'float64'). Default: 'float64'."
    return [
        Argument("err", [int, float], optional=True, default=1e-5, doc=doc_err),
        Argument("tolerance", [int, float], optional=True, default=1e-7, doc=doc_tolerance),
        Argument("max_iter", int, optional=True, default=100, doc=doc_max_iter),
        Argument("mix_rate", int, optional=True, default=0.25, doc=doc_mix_rate),
        Argument("poisson_dtype", str, optional=True, default='float64', doc=doc_poisson_dtype),
        Argument("grid", dict, optional=False, sub_fields=grid(), doc=doc_grid),
        Argument("gate_top", dict, optional=True, sub_fields=Dirichlet_BC(), doc=doc_gate),
        Argument("gate_bottom", dict, optional=True, sub_fields=Dirichlet_BC(), doc=doc_gate),
        Argument("gate_left", dict, optional=True, sub_fields=Dirichlet_BC(), doc=doc_gate),
        Argument("gate_right", dict, optional=True, sub_fields=Dirichlet_BC(), doc=doc_gate),
        Argument("dielectric_region", dict, optional=False, sub_fields=dielectric(), doc=doc_dielectric),
        *[
            Argument(f"dielectric_region{i}", dict, optional=True, sub_fields=dielectric(), doc=doc_dielectric)
            for i in range(2, 7)
        ],
        Argument("doped_region", dict, optional=False, sub_fields=doped(), doc=doc_doped)
    ]

def scipy():
    doc_err = "Residual threshold on the potential update. Default: 1e-5."
    doc_tolerance = "Convergence tolerance of the sparse linear solve. Default: 1e-7."
    doc_grid = "Real-space grid used to discretize the Poisson equation."
    doc_gate = "Dirichlet boundary condition for a metallic gate (x/y/z range and applied voltage)."
    doc_dielectric = ("Rectangular region with a fixed relative permittivity. Additional regions "
                       "can be added via `dielectric_region2` ... `dielectric_region6`.")
    doc_doped = "Rectangular region with a fixed doping charge density."
    doc_max_iter = "Maximum Poisson iterations. Default: 100."
    doc_mix_rate = "Underrelaxation between successive Poisson solves. Default: 0.25."
    doc_poisson_dtype = "Floating precision of the Poisson solver ('float32' or 'float64'). Default: 'float64'."
    return [
        Argument("err", [int, float], optional=True, default=1e-5, doc=doc_err),
        Argument("tolerance", [int, float], optional=True, default=1e-7, doc=doc_tolerance),
        Argument("max_iter", int, optional=True, default=100, doc=doc_max_iter),
        Argument("mix_rate", int, optional=True, default=0.25, doc=doc_mix_rate),
        Argument("poisson_dtype", str, optional=True, default='float64', doc=doc_poisson_dtype),
        Argument("with_Dirichlet_leads", bool, optional=True, default=False, doc="Whether to use Dirichlet boundary condition for leads"),
        Argument("grid", dict, optional=True, sub_fields=grid(), doc=doc_grid),
        Argument("gate_top", dict, optional=True, sub_fields=Dirichlet_BC(), doc=doc_gate),
        Argument("gate_bottom", dict, optional=True, sub_fields=Dirichlet_BC(), doc=doc_gate),
        Argument("gate_left", dict, optional=True, sub_fields=Dirichlet_BC(), doc=doc_gate),
        Argument("gate_right", dict, optional=True, sub_fields=Dirichlet_BC(), doc=doc_gate),
        Argument("lead_L", dict, optional=True, sub_fields=Dirichlet_BC(), doc=doc_gate),
        Argument("lead_R", dict, optional=True, sub_fields=Dirichlet_BC(), doc=doc_gate),
        Argument("dielectric_region", dict, optional=True, sub_fields=dielectric(), doc=doc_dielectric),
        *[
            Argument(f"dielectric_region{i}", dict, optional=True, sub_fields=dielectric(), doc=doc_dielectric)
            for i in range(2, 7)
        ],
        Argument("doped_region1", dict, optional=True, sub_fields=doped(), doc=doc_doped),
        Argument("doped_region2", dict, optional=True, sub_fields=doped(), doc=doc_doped)
    ]

def grid():
    doc_xrange = "Grid range along x as 'start:stop:n' (Angstrom, n points)."
    doc_yrange = "Grid range along y as 'start:stop:n' (Angstrom, n points)."
    doc_zrange = "Grid range along z as 'start:stop:n' (Angstrom, n points)."
    return [
        Argument("x_range", str, optional=False, doc=doc_xrange),
        Argument("y_range", str, optional=False, doc=doc_yrange),
        Argument("z_range", str, optional=False, doc=doc_zrange),
    ]

def Dirichlet_BC():
    doc_xrange = "Range along x as 'x0:x1' (Angstrom)."
    doc_yrange = "Range along y as 'y0:y1' (Angstrom)."
    doc_zrange = "Range along z as 'z0:z1' (Angstrom)."
    doc_voltage = "Applied voltage on this Dirichlet boundary in `unit`. Default: None."
    return [
        Argument("x_range", str, optional=False, doc=doc_xrange),
        Argument("y_range", str, optional=False, doc=doc_yrange),
        Argument("z_range", str, optional=False, doc=doc_zrange),
        Argument("voltage", [int, float], optional=True, doc=doc_voltage, default=None)
    ]

def dielectric():
    doc_xrange = "Range along x as 'x0:x1' (Angstrom)."
    doc_yrange = "Range along y as 'y0:y1' (Angstrom)."
    doc_zrange = "Range along z as 'z0:z1' (Angstrom)."
    doc_permittivity = "Relative permittivity of the region."
    return [
        Argument("x_range", str, optional=False, doc=doc_xrange),
        Argument("y_range", str, optional=False, doc=doc_yrange),
        Argument("z_range", str, optional=False, doc=doc_zrange),
        Argument("relative permittivity", [int, float], optional=False, doc=doc_permittivity)
    ]

def doped():
    doc_xrange = "Range along x as 'x0:x1' (Angstrom)."
    doc_yrange = "Range along y as 'y0:y1' (Angstrom)."
    doc_zrange = "Range along z as 'z0:z1' (Angstrom)."
    doc_charge = "Fixed doping charge (electrons per unit cell of the region)."
    return [
        Argument("x_range", str, optional=False, doc=doc_xrange),
        Argument("y_range", str, optional=False, doc=doc_yrange),
        Argument("z_range", str, optional=False, doc=doc_zrange),
        Argument("charge", [int, float], optional=False, doc=doc_charge)
    ]

def run_options():
    doc_task = "Which transport task to run and its detailed options. See `task_options` below."
    doc_structure = "Path to the structure file (xyz/vasp/etc.). Overrides the CLI `--structure` flag when set."
    doc_gui = "Whether to launch the GUI. Default: False."
    doc_device = ("The torch device used to build the Hamiltonian and run the calculation. "
                  "Choose 'cpu' or 'cuda[:int]'. Default: None (use the device stored in the model ckpt).")
    doc_dtype = ("Floating precision used at build time. 'float32' or 'float64'. "
                  "Default: None (use the dtype stored in the model ckpt).")
    doc_pbc = ("Overall periodic boundary condition of the input structure. "
                 "Bool applies to all three axes; a list of three bools sets x/y/z individually. "
                 "Default: None (inferred from the structure file).")

    args = [
        Argument("task_options", dict, sub_fields=[], optional=True, sub_variants=[task_options()], doc = doc_task),
        Argument("structure", [str,None], optional=True, default=None, doc = doc_structure),
        Argument("pbc", [None, bool, list], optional=True, doc=doc_pbc, default=None),
        Argument("use_gui", bool, optional=True, default=False, doc = doc_gui),
        Argument("device", [str,None], optional = True, default=None, doc = doc_device),
        Argument("dtype", [str,None], optional = True, default=None, doc = doc_dtype),
        AtomicData_options_sub()
    ]

    return Argument("run_op", dict, args)

def normalize_run(data):

    _migrate_legacy_negf_task_options(data)

    run_op = run_options()
    data = run_op.normalize_value(data)
    run_op.check_value(data, strict=True)

    return data


# (flat legacy key)  -> tuple describing where it now lives inside task_options.
# Tuples of length 1 = top-level under task_options; length 2 = one sub-dict; etc.
_NEGF_LEGACY_KEY_MAP = {
    "sgf_solver":            ("self_energy_options", "solver"),
    "se_numba_jit":          ("self_energy_options", "numba_jit"),
    "se_info_display":       ("self_energy_options", "info_display"),
    "use_saved_se":          ("self_energy_options", "cache", "use_saved"),
    "self_energy_save_path": ("self_energy_options", "cache", "save_path"),
    "n_cpus":                ("self_energy_options", "parallel", "cpu_budget"),
    "rgf_device":            ("rgf_options", "device"),
    "e_batch_size":          ("rgf_options", "e_batch_size"),
    "use_saved_HS":          ("hs_cache", "use_saved"),
    "saved_HS_path":         ("hs_cache", "save_path"),
    "out_dos":               ("output_options", "dos"),
    "out_tc":                ("output_options", "tc"),
    "out_density":           ("output_options", "density"),
    "out_potential":         ("output_options", "potential"),
    "out_current":           ("output_options", "current"),
    "out_current_nscf":      ("output_options", "current_nscf"),
    "out_ldos":              ("output_options", "ldos"),
    "out_lcurrent":          ("output_options", "lcurrent"),
}


def _migrate_legacy_negf_task_options(data):
    """Rewrite pre-refactor flat NEGF keys into the new nested shape in place.

    The NEGF `task_options` block used to expose parallelism, RGF device,
    self-energy caching, and HS caching as top-level flat keys. They now live
    under `self_energy_options`, `rgf_options`, and `hs_cache`. To avoid
    breaking existing input files, this function inspects `task_options`
    before strict-schema check, and for each legacy key it finds:

      * emits `log.warning` naming the new location,
      * moves the value into the correct nested sub-dict (creating parents),
      * deletes the flat key so the strict check sees the new shape only.

    Only runs when `task_options.task == 'negf'`. Silent no-op otherwise.
    """
    if not isinstance(data, dict):
        return
    task_options = data.get("task_options")
    if not isinstance(task_options, dict):
        return
    if task_options.get("task") != "negf":
        return

    for legacy_key, path in _NEGF_LEGACY_KEY_MAP.items():
        if legacy_key not in task_options:
            continue
        value = task_options.pop(legacy_key)
        new_location = "task_options." + ".".join(path)
        log.warning(
            f"NEGF task_options: '{legacy_key}' is deprecated; "
            f"use '{new_location}' instead. "
            f"Migrating value automatically for this run."
        )

        cursor = task_options
        for parent in path[:-1]:
            existing = cursor.get(parent)
            if not isinstance(existing, dict):
                existing = {}
                cursor[parent] = existing
            cursor = existing
        leaf = path[-1]
        if leaf in cursor:
            log.warning(
                f"NEGF task_options: both legacy '{legacy_key}' and "
                f"'{new_location}' are set; keeping the value from '{new_location}'."
            )
        else:
            cursor[leaf] = value

def task_options():
    doc_task = '''The string define the task to conduct:
                    - `negf`: for non-equilibrium green function calculation.
                    - `tbtrans_negf`: for non-equilibrium green function calculation with tbtrans.
                '''

    return Variant("task", [
            Argument("negf", dict, negf()),
            Argument("tbtrans_negf", dict, tbtrans_negf()),
        ],optional=False, doc=doc_task)


def AtomicData_options_sub():
    doc_r_max = "Cutoff radius (Angstrom) for bond construction in the TB model. Can be a float or a per-species dict."
    doc_er_max = "Cutoff radius for the environment of each site (env-correction models). Default: None."
    doc_oer_max = "Cutoff radius for onsite environment; only used by nnsk models in strain or NRL mode. Default: None."
    doc_pbc = "Periodic boundary condition, bool or list of three bools for x/y/z."

    args = [
        Argument("r_max", [float, int, dict], optional=False, doc=doc_r_max, default=4.0),
        Argument("er_max", [float, int, dict], optional=True, doc=doc_er_max, default=None),
        Argument("oer_max", [float, int, dict], optional=True, doc=doc_oer_max,default=None)
    ]

    return Argument("AtomicData_options", dict, optional=True, sub_fields=args, sub_variants=[], doc="", default=None)


def format_cuts(rcut: Union[Dict[str, Number], Number], decay_w: Number, nbuffer: int) -> Union[Dict[str, Number], Number]:
    if not isinstance(decay_w, Number) or decay_w <= 0:
        raise ValueError("decay_w should be a positive number")

    buffer_addition = decay_w * nbuffer

    if isinstance(rcut, dict):
        return {key: value + buffer_addition for key, value in rcut.items()}
    elif isinstance(rcut, Number):
        return rcut + buffer_addition
    else:
        raise TypeError("rcut should be a dict or a number")

def get_cutoffs_from_model_options(model_options):
    """
    Extract cutoff values from the provided model options.

    This function retrieves the cutoff values `r_max`, `er_max`, and `oer_max` from the `model_options`
    dictionary. It handles different model types such as `embedding`, `nnsk`, and `dftbsk`, ensuring
    that the appropriate cutoff values are provided and valid.

    Parameters:
    model_options (dict): A dictionary containing model configuration options. It may include keys
                          like `embedding`, `nnsk`, and `dftbsk` with their respective cutoff values.

    Returns:
    tuple: A tuple containing the cutoff values (`r_max`, `er_max`, `oer_max`).

    Raises:
    ValueError: If neither `r_max` nor `rc` is provided in `model_options` for embedding.
    AssertionError: If `r_max` is provided outside the `nnsk` or `dftbsk` context when those models are used.

    Logs:
    Error messages if required cutoff values are missing or incorrectly provided.
    """
    r_max, er_max, oer_max = None, None, None
    if model_options.get("embedding",None) is not None:
        # switch according to the embedding method
        embedding = model_options.get("embedding")
        if embedding["method"] == "se2":
            er_max = embedding["rc"]
        elif embedding["method"] in ["slem", "lem"]:
            r_max = embedding["r_max"]
        else:
            log.error("The method of embedding have not been defined in get cutoff functions")
            raise NotImplementedError("The method of embedding have not been defined in get cutoff functions")

    if model_options.get("nnsk", None) is not None:
        assert r_max is None, "r_max should not be provided in outside the nnsk for training nnsk model."
        if model_options["nnsk"]["hopping"].get("rs",None) is not None:
            # 其他方法在模型公式中，已经包含了 +5w 的范围，所以这里为了保险额外加上3w 的范围;
            # 对于两个特例，powerlaw 和 varTang96 的情况，为了和旧版存档的兼容, 模型公式的本身并没有 +5w 的范围，所以这里额外加上8w 的范围。
            if model_options["nnsk"]["hopping"]['method'] in ["powerlaw","varTang96"]:
                r_max = format_cuts(model_options["nnsk"]["hopping"]["rs"], model_options["nnsk"]["hopping"]["w"], 8)
            else:
                r_max = format_cuts(model_options["nnsk"]["hopping"]["rs"], model_options["nnsk"]["hopping"]["w"], 3)

        if model_options["nnsk"]["onsite"].get("rs",None) is not None:
            if  model_options["nnsk"]["onsite"]['method'] == "strain" and model_options["nnsk"]["hopping"]['method'] in ["powerlaw","varTang96"]:
                oer_max = format_cuts(model_options["nnsk"]["onsite"]["rs"], model_options["nnsk"]["onsite"]["w"], 8)
            else:
                oer_max = format_cuts(model_options["nnsk"]["onsite"]["rs"], model_options["nnsk"]["onsite"]["w"], 3)

    elif model_options.get("dftbsk", None) is not None:
        assert r_max is None, "r_max should not be provided other than the dftbsk param section for training dftbsk model."
        r_max = model_options["dftbsk"].get("r_max")

    else:
        # not nnsk not dftbsk, must be only env or E3. the embedding should be provided.
        assert model_options.get("embedding",None) is not None

    return r_max, er_max, oer_max
