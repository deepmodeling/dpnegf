"""Regression coverage for the contact-projected Boundary-Schur solver."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import importlib

from dpnegf.negf.boundary_self_energy import (
    boundary_self_energy,
    prepare_boundary_schur_plan,
)
from dpnegf.negf import lead_property
from dpnegf.negf.surface_green import selfEnergy
from dpnegf.utils.argcheck import normalize_run

boundary_solver_module = importlib.import_module("dpnegf.negf.boundary_self_energy")


def _make_lead_problem(contact_with_rest=False, full_boundary=False):
    rng = np.random.default_rng(41)
    n = 7
    raw = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    h = (raw + raw.conj().T) * 0.035
    h[np.diag_indices(n)] += np.linspace(-0.4, 0.5, n)

    raw = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    s = np.eye(n) + (raw + raw.conj().T) * 0.001

    h01 = np.zeros((n, n), dtype=np.complex128)
    s01 = np.zeros_like(h01)
    if full_boundary:
        h01[np.diag_indices(n)] = 0.08
    else:
        # C={0,2}, R={4,6}; orbitals 1,3,5 are eliminable interiors.
        c = np.array([0, 2])
        r = np.array([4, 6])
        h01[np.ix_(r, c)] = np.array([[0.26, 0.04j], [-0.03, 0.21]])
        s01[np.ix_(r, c)] = np.array([[0.002, 0.0003j], [-0.0002, 0.001]])

    hdl = np.zeros((3, n), dtype=np.complex128)
    sdl = np.zeros_like(hdl)
    c = np.array([0, 2])
    hdl[:, c] = np.array([[0.35, 0.02j], [0.04, 0.29], [-0.01j, 0.18]])
    sdl[:, c] = np.array([[0.008, 0.001j], [0.001, 0.006], [0.0005j, 0.004]])
    if contact_with_rest:
        hdl[0, 3] = 0.025
        sdl[1, 3] = 0.0004
    return h, h01, s, s01, hdl, sdl


@pytest.mark.parametrize("energy", [-0.37, 0.16 + 0.003j, 0.72])
def test_two_endpoint_boundary_schur_matches_full_sancho(energy):
    h, h01, s, s01, hdl, sdl = _make_lead_problem()
    plan = prepare_boundary_schur_plan(h01, s01, hdl, sdl)
    assert plan.boundary.size == 4
    assert plan.internal.size == 3
    assert plan.has_two_endpoints

    expected, _ = selfEnergy(
        h, h01, s, s01, torch.tensor(energy, dtype=torch.complex128), hdl, sdl,
        E_ref=0.11, etaLead=0.015, method="Sancho-Rubio", numba_jit=False,
    )
    actual, diagnostics = boundary_self_energy(
        h, h01, s, s01, hdl, sdl, energy, E_ref=0.11, etaLead=0.015,
        numba_jit=False, plan=plan,
    )

    assert diagnostics["solver"] == "boundary-schur-two-endpoint"
    assert diagnostics["fixed_point_residual"] < 3e-5
    assert diagnostics["internal_residual"] < 1e-12
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def test_boundary_solver_public_dispatch_returns_sigma_only():
    h, h01, s, s01, hdl, sdl = _make_lead_problem()
    expected, _ = selfEnergy(
        h, h01, s, s01, 0.23 + 0.004j, hdl, sdl, E_ref=0.07, etaLead=0.02,
        method="Sancho-Rubio", numba_jit=False,
    )
    actual, surface_green = selfEnergy(
        h, h01, s, s01, 0.23 + 0.004j, hdl, sdl, E_ref=0.07, etaLead=0.02,
        method="Boundary-Schur", numba_jit=False,
    )
    assert surface_green is None
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def test_additional_contact_orbital_uses_boundary_iteration():
    h, h01, s, s01, hdl, sdl = _make_lead_problem(contact_with_rest=True)
    plan = prepare_boundary_schur_plan(h01, s01, hdl, sdl)
    assert plan.boundary.size == 5
    assert plan.internal.size == 2
    assert not plan.has_two_endpoints

    expected, _ = selfEnergy(
        h, h01, s, s01, -0.12, hdl, sdl, E_ref=0.03, etaLead=0.01,
        method="Sancho-Rubio", numba_jit=False,
    )
    actual, diagnostics = boundary_self_energy(
        h, h01, s, s01, hdl, sdl, -0.12, E_ref=0.03, etaLead=0.01,
        numba_jit=False, plan=plan,
    )
    assert diagnostics["solver"] == "boundary-schur-boundary"
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def test_energy_cancellation_does_not_change_structural_partition():
    h, _, s, s01, hdl, sdl = _make_lead_problem()
    h01 = 0.25 * s01
    plan = prepare_boundary_schur_plan(h01, s01, hdl, sdl)
    assert plan.boundary.size == 4
    assert plan.internal.size == 3
    assert plan.has_two_endpoints

    expected, _ = selfEnergy(
        h, h01, s, s01, 0.25, hdl, sdl, etaLead=0.0,
        method="Sancho-Rubio", numba_jit=False,
    )
    actual, diagnostics = boundary_self_energy(
        h, h01, s, s01, hdl, sdl, 0.25, etaLead=0.0,
        numba_jit=False, plan=plan,
    )
    assert diagnostics["solver"] == "boundary-schur-two-endpoint"
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def test_no_dimension_reduction_falls_back_to_sancho():
    h, h01, s, s01, hdl, sdl = _make_lead_problem(full_boundary=True)
    expected, _ = selfEnergy(
        h, h01, s, s01, 0.31, hdl, sdl, etaLead=0.01,
        method="Sancho-Rubio", numba_jit=False,
    )
    actual, diagnostics = boundary_self_energy(
        h, h01, s, s01, hdl, sdl, 0.31, etaLead=0.01, numba_jit=False,
    )
    assert diagnostics["solver"] == "Sancho-Rubio"
    assert "no effective dimension reduction" in diagnostics["fallback_reason"]
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def test_bad_two_endpoint_solution_fails_independent_fixed_point_check(monkeypatch):
    h, h01, s, s01, hdl, sdl = _make_lead_problem()
    plan = prepare_boundary_schur_plan(h01, s01, hdl, sdl)

    def incorrect_endpoint_solution(d_bulk, d_surface, alpha, beta):
        size = d_bulk.shape[0]
        return np.zeros((size, size), dtype=np.complex128), 1, 0.0

    monkeypatch.setattr(boundary_solver_module, "_decimate", incorrect_endpoint_solution)
    expected, _ = selfEnergy(
        h, h01, s, s01, 0.16, hdl, sdl, E_ref=0.11, etaLead=0.015,
        method="Sancho-Rubio", numba_jit=False,
    )
    actual, diagnostics = boundary_self_energy(
        h, h01, s, s01, hdl, sdl, 0.16, E_ref=0.11, etaLead=0.015,
        numba_jit=False, plan=plan,
    )

    assert diagnostics["solver"] == "Sancho-Rubio"
    assert "fixed-point residual" in diagnostics["fallback_reason"]
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def test_unclassified_runtime_error_is_not_silently_fallback(monkeypatch):
    h, h01, s, s01, hdl, sdl = _make_lead_problem()
    plan = prepare_boundary_schur_plan(h01, s01, hdl, sdl)

    def unexpected_runtime_error(*args, **kwargs):
        raise RuntimeError("unexpected implementation failure")

    monkeypatch.setattr(boundary_solver_module, "_decimate", unexpected_runtime_error)
    with pytest.raises(RuntimeError, match="unexpected implementation failure"):
        boundary_self_energy(
            h, h01, s, s01, hdl, sdl, 0.16, E_ref=0.11, etaLead=0.015,
            numba_jit=False, plan=plan,
        )


def test_unknown_self_energy_solver_is_rejected():
    h, h01, s, s01, hdl, sdl = _make_lead_problem()
    with pytest.raises(ValueError, match="Unknown self-energy solver"):
        selfEnergy(
            h, h01, s, s01, 0.16, hdl, sdl,
            method="boundary-schur", numba_jit=False,
        )


def test_normalized_config_rejects_unknown_legacy_solver():
    config_path = (
        Path(__file__).parent
        / "data/test_negf/test_negf_run/negf_chain_new.json"
    )
    with config_path.open() as stream:
        config = json.load(stream)
    config["task_options"]["sgf_solver"] = "boundary-schur"

    with pytest.raises(ValueError, match="Unknown self-energy solver"):
        normalize_run(config)


def test_precomputed_pack_uses_boundary_schur_for_selected_solver():
    h, h01, s, s01, hdl, sdl = _make_lead_problem()

    class Hamiltonian:
        def get_hs_device(self, kpoint, only_subblocks):
            assert only_subblocks
            return [hdl.shape[0]]

        def get_hs_lead(self, kpoint, tab, v):
            assert v == 0
            return (
                torch.as_tensor(h), torch.as_tensor(h01), torch.as_tensor(hdl),
                torch.as_tensor(s), torch.as_tensor(s01), torch.as_tensor(sdl),
            )

    lead = lead_property.LeadProperty(
        "lead_L", Hamiltonian(), None, ".", voltage=0.0, E_ref=0.11,
    )
    kpoint = [0.0, 0.0, 0.0]
    pack = lead_property._precompute_lead_kdata(lead, [kpoint])
    lead_property._prepare_boundary_schur_plans(pack)

    actual = lead_property._compute_self_energy_from_pack(
        pack, kpoint, 0.16, 0.015, method="Boundary-Schur", se_numba_jit=False,
    )
    expected, _ = selfEnergy(
        h, h01, s, s01, torch.tensor(0.16), hdl, sdl, E_ref=0.11, etaLead=0.015,
        method="Sancho-Rubio", numba_jit=False,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)


def test_lead_property_caches_boundary_plan_for_repeated_energies():
    h, h01, s, s01, hdl, sdl = _make_lead_problem()

    class Hamiltonian:
        def __init__(self):
            self.lead_reads = 0
            self.contact_metadata = {}

        def get_hs_device(self, kpoint, only_subblocks):
            return [hdl.shape[0]]

        def get_hs_lead(self, kpoint, tab, v):
            self.lead_reads += 1
            return tuple(torch.as_tensor(x) for x in (h, h01, hdl, s, s01, sdl))

    hamiltonian = Hamiltonian()
    lead = lead_property.LeadProperty(
        "lead_L", hamiltonian, None, ".", voltage=0.0, E_ref=0.11,
    )
    kpoint = [0.0, 0.0, 0.0]
    first = lead.self_energy_cal(
        kpoint, 0.16, method="Boundary-Schur", se_numba_jit=False,
    )
    second = lead.self_energy_cal(
        kpoint, 0.41, method="Boundary-Schur", se_numba_jit=False,
    )
    assert hamiltonian.lead_reads == 1

    expected_second, _ = selfEnergy(
        h, h01, s, s01, 0.41, hdl, sdl, E_ref=0.11, etaLead=1e-5,
        method="Sancho-Rubio", numba_jit=False,
    )
    torch.testing.assert_close(second, expected_second, rtol=1e-10, atol=1e-12)
    assert first.shape == second.shape == expected_second.shape
