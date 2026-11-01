"""Model-independent solver for a batch of non-SCF RGF energies."""

import logging
import os

import torch

from dpnegf.negf.device_property import DeviceProperty
from dpnegf.negf.lead_property import LeadProperty
from dpnegf.utils.hs_cache import CachedHamiltonian

log = logging.getLogger(__name__)


def rgf_solver_options(runner):
    """Select numerical/output settings; never serialize the NEGF runner."""
    return {name: getattr(runner, name) for name in (
        "e_batch_size", "eta_lead", "eta_device", "block_tridiagonal",
        "sgf_solver", "self_energy_save_path", "se_cache_format", "se_info_display",
        "gf_cal_mode", "need_trans_eigs", "trans_eigs_options",
        "out_trans_eigvals", "out_trans_eigchannel", "out_dos", "out_tc",
        "out_current_nscf", "out_ldos",
    )}


class _CachedLead(LeadProperty):
    """Read the prepared self-energy cache without running a lead solver."""

    def self_energy(self, kpoint, energy, *, save_path, save_format, **kwargs):
        path = self._get_save_path(kpoint, energy, save_format, save_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"RGF worker requires prepared self-energy cache: {path}")
        self.se = self._load_self_energy(path, kpoint, energy, save_format)


class RGFSolver:
    """Own only the device properties and numerical settings needed for RGF."""

    def __init__(self, deviceprop, options):
        self.deviceprop = deviceprop
        self.options = options
        self.rgf_device = torch.device(deviceprop.rgf_device)
        self.e_batch_size = options["e_batch_size"]
        self.block_tridiagonal = options["block_tridiagonal"]
        self.need_trans_eigs = options.get("need_trans_eigs", False)
        self.out_trans_eigchannel = options.get("out_trans_eigchannel", False)
        self.trans_eigs_options = options.get("trans_eigs_options", {})

    @classmethod
    def from_cache(cls, spec, device):
        """Build from saved H/S files, loaded in this process on first use.

        The specification carries a disk cache path and metadata, with no H/S
        tensors or model-backed Hamiltonian from the parent process.
        """
        options = spec["options"]
        ham = CachedHamiltonian(**spec["hamiltonian"], device=device)
        prop = DeviceProperty(ham, structure=None, rgf_device=device, **spec["device"])
        prop.set_leadLR(**{
            name: _CachedLead(tab=name, hamiltonian=ham, structure=None,
                              results_path=prop.results_path, **values)
            for name, values in spec["leads"].items()
        })
        return cls(prop, options)

    @torch.no_grad()
    def compute_rgf_chunk(self, e_chunk, k, ik, Vbias, energy_offset=0, out=None):
        """Solve one energy batch and append CPU outputs in energy order."""
        out = {} if out is None else out
        prop, options = self.deviceprop, self.options
        e_batch_size = len(e_chunk)
        log.info(
            f"computing green's functions on {self.rgf_device} for chunk "
            f"e=[{float(e_chunk[0]):>6.3f}..{float(e_chunk[-1]):>6.3f}], B={e_batch_size}"
        )
        seL_list, seR_list = [], []
        try:
            for e in e_chunk:
                for ll in ("lead_L", "lead_R"):
                    getattr(prop, ll).self_energy(
                        energy=e,
                        kpoint=k,
                        eta_lead=options["eta_lead"],
                        method=options["sgf_solver"],
                        save_path=options["self_energy_save_path"],
                        save_format=options["se_cache_format"],
                        se_info_display=options["se_info_display"]
                        )
                seL_list.append(prop.lead_L.se)
                seR_list.append(prop.lead_R.se)

            if e_batch_size > 1:
                prop.lead_L.se = torch.stack(seL_list, dim=0)
                prop.lead_R.se = torch.stack(seR_list, dim=0)
            # else: leave the per-E [n,n] in place — preserves scalar contract exactly.

            prop.cal_green_function(
                energy=e_chunk, kpoint=k,
                eta_device=options["eta_device"],
                block_tridiagonal=self.block_tridiagonal,
                Vbias=Vbias,
                need_lesser=False,
                need_greater=False,
                need_gr_lc=False, # set to False for memory saving, can be set to True for lead spectral function G^r * \Gamma * G^a
                gf_cal_mode=options["gf_cal_mode"],
                need_trans_eigs=options["need_trans_eigs"],
                )

            if options["need_trans_eigs"]:
                prop.store_trans_eigs(
                    k, ik, out, trans_eigs_options=options["trans_eigs_options"],
                    out_trans_eigvals=options["out_trans_eigvals"],
                    out_trans_eigchannel=options["out_trans_eigchannel"],
                    eta_lead=options["eta_lead"], energy_offset=energy_offset)
            if options["out_dos"]:
                out.setdefault('DOS', {}).setdefault(str(k), []).append(prop.dos.reshape(-1).cpu())
            if options["out_tc"] or options["out_current_nscf"]:
                out.setdefault('T_k', {}).setdefault(str(k), []).append(prop.tc.reshape(-1).cpu())
            if options["out_ldos"]:
                ldos_chunk = prop.ldos
                if ldos_chunk.ndim == 1:  # scalar-E chunk → [na]
                    ldos_chunk = ldos_chunk.unsqueeze(0)
                out.setdefault('LDOS', {}).setdefault(str(k), []).append(ldos_chunk.cpu())

        finally:
            prop.release_greenfuncs()
            if seL_list:
                prop.lead_L.se = seL_list[-1]
            if seR_list:
                prop.lead_R.se = seR_list[-1]
            del seL_list, seR_list

        return out
