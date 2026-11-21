"""Density integration must preserve the configured lead solver."""

import numpy as np
import pytest
import torch

from dpnegf.negf import density as density_module
from dpnegf.negf.density import Fiori, Ozaki


class _Lead:
    def __init__(self, voltage=0.0):
        self.voltage = voltage
        self.mu = 0.0
        self.se = torch.zeros((1, 1), dtype=torch.complex128)
        self.gamma = torch.zeros((1, 1), dtype=torch.complex128)
        self.calls = []

    def self_energy(self, **kwargs):
        self.calls.append(("cached", kwargs))
        self.se = torch.zeros((1, 1), dtype=torch.complex128)

    def self_energy_cal(self, **kwargs):
        self.calls.append(("direct", kwargs))
        self.se = torch.zeros((1, 1), dtype=torch.complex128)
        return self.se

    def fermi_dirac(self, energy):
        return torch.tensor(0.5)


class _Device:
    def __init__(self):
        self.kBT = 0.025
        self.mu = 0.0
        self.lead_L = _Lead()
        self.lead_R = _Lead()
        self.grd = [torch.zeros((2, 2), dtype=torch.complex128)]
        self.g_trans = torch.zeros((2, 2), dtype=torch.complex128)

    def cal_green_function(self, **kwargs):
        pass


def test_ozaki_passes_solver_to_all_lead_energy_points(monkeypatch):
    monkeypatch.setattr(
        density_module, "ozaki_residues",
        lambda M_cut: (np.array([1.0]), np.array([0.25])),
    )
    device = _Device()
    density = Ozaki(R=0.5, M_cut=1, n_gauss=2)

    density.integrate(
        deviceprop=device,
        kpoint=[0.0, 0.0, 0.0],
        self_energy_method="Boundary-Schur",
    )

    for lead in (device.lead_L, device.lead_R):
        assert len(lead.calls) == 2
        assert all(call[1]["method"] == "Boundary-Schur" for call in lead.calls)


@pytest.mark.parametrize("with_dirichlet_leads, expected_path", [
    (False, "direct"),
    (True, "cached"),
])
def test_fiori_passes_solver_to_selected_self_energy_path(
        with_dirichlet_leads, expected_path):
    device = _Device()
    density = Fiori(integrate_way="direct", e_grid=torch.tensor([0.0, 1.0]))
    density.integrate_range = torch.tensor([0.25])
    density.pre_factor = torch.tensor([1.0])
    kpoint = [0.0, 0.0, 0.0]
    free_charge = {str(kpoint): torch.zeros(1, dtype=torch.complex128)}

    density.density_integrate_Fiori(
        e_grid=torch.tensor([0.0, 1.0]),
        kpoint=kpoint,
        Vbias=None,
        block_tridiagonal=False,
        subblocks=None,
        integrate_way="direct",
        deviceprop=device,
        device_atom_norbs=[1],
        potential_at_atom=torch.tensor([-1.0]),
        with_Dirichlet_leads=with_dirichlet_leads,
        free_charge=free_charge,
        E_ref=0.0,
        self_energy_method="Boundary-Schur",
    )

    for lead in (device.lead_L, device.lead_R):
        assert len(lead.calls) == 1
        path, kwargs = lead.calls[0]
        assert path == expected_path
        assert kwargs["method"] == "Boundary-Schur"
