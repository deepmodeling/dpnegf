"""CPU RGF worker configuration and thread-limit contracts."""

import json
from pathlib import Path

import pytest
import torch

from dpnegf.utils import parrallel
from dpnegf.utils.argcheck import normalize_run


def test_cpu_rgf_option_defaults_and_validation(monkeypatch):
    monkeypatch.setattr(parrallel, "cpu_count", lambda: 8)
    defaults = parrallel.validate_rgf_options({})
    assert defaults == {
        "cpu_workers": 1, "cpu_threads": None,
        "startup_timeout": 600.0, "sync_timeout": 600.0,
        "task_timeout": 3000.0, "max_pending_results": None,
        "e_batch_size": None,
    }
    assert parrallel.validate_rgf_options({"cpu_workers": 4})["cpu_workers"] == 4
    assert parrallel.validate_rgf_options({"cpu_workers": 2, "cpu_threads": 4})[
        "cpu_threads"] == 4

    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="cpu_workers"):
            parrallel.validate_rgf_options({"cpu_workers": value})
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="cpu_threads"):
            parrallel.validate_rgf_options({"cpu_threads": value})


def test_cpu_rgf_rejects_conflicts_scft_and_budget_overflow(monkeypatch):
    monkeypatch.setattr(parrallel, "cpu_count", lambda: 4)
    with pytest.raises(ValueError, match="scf=false"):
        parrallel.validate_rgf_options({"cpu_workers": 2}, scf=True)
    with pytest.raises(ValueError, match="cannot be combined"):
        parrallel.validate_rgf_options({"cpu_workers": 2, "device": "cuda:0"})
    with pytest.raises(ValueError, match="cannot be combined"):
        parrallel.validate_rgf_options({"cpu_threads": 2, "devices": ["cuda:0"]})
    with pytest.raises(ValueError, match=r"cpu_workers \* cpu_threads"):
        parrallel.validate_rgf_options({"cpu_workers": 2, "cpu_threads": 3})
    with pytest.raises(ValueError, match="e_batch_size=1"):
        parrallel.validate_rgf_options({"cpu_workers": 2, "e_batch_size": 2})


def test_normalized_cpu_workers_resolve_to_scalar_tasks(monkeypatch):
    monkeypatch.setattr(parrallel, "cpu_count", lambda: 4)
    data = Path(__file__).parent / "data/test_negf/test_negf_run/negf_chain_new.json"
    run_options = json.loads(data.read_text())
    run_options["task_options"]["rgf_options"] = {"cpu_workers": 2}

    normalized = normalize_run(run_options)["task_options"]["rgf_options"]
    assert normalized["cpu_workers"] == 2
    assert normalized["e_batch_size"] == 1


def test_negf_constructor_accepts_numpy_integer_batch(monkeypatch, tmp_path):
    import numpy as np
    from types import SimpleNamespace
    from contextlib import nullcontext
    import importlib

    negf_module = importlib.import_module("dpnegf.runner.NEGF")
    device_structure = SimpleNamespace(positions=np.asarray([[0.0, 0.0, 0.0]]))

    class FakeHamiltonianInit:
        def __init__(self, **kwargs):
            self.self_energy_save_path = str(tmp_path)
            self.self_energy_cache_format = "h5"
            self.atom_norbs = [1]
            self.device_id = (0, 1)

        def initialize(self, **kwargs):
            return (
                device_structure,
                {"lead_L": None, "lead_R": None},
                {"lead_L": None, "lead_R": None},
                {"lead_L": None, "lead_R": None},
                {"lead_L": None, "lead_R": None},
            )

    class FakeDeviceProperty:
        def __init__(self, *args, **kwargs):
            self.E_ref = 0.0

        def set_leadLR(self, **kwargs):
            pass

    monkeypatch.setattr(negf_module, "NEGFHamiltonianInit", FakeHamiltonianInit)
    monkeypatch.setattr(negf_module, "DeviceProperty", FakeDeviceProperty)
    monkeypatch.setattr(negf_module, "LeadProperty", lambda **kwargs: object())
    monkeypatch.setattr(negf_module, "_profile_to_html", lambda _path: nullcontext())
    monkeypatch.setattr(
        negf_module.NEGF, "update_atomicdata_options",
        staticmethod(lambda *args, **kwargs: {}),
    )

    runner = negf_module.NEGF(
        model=SimpleNamespace(device="cpu"), structure=None, ele_T=300,
        scf=False, rgf_options={"e_batch_size": np.int64(1)},
        output_options={"tc": True},
        e_fermi=0.0,
        results_path=str(tmp_path),
        density_options={"method": "Fiori", "integrate_way": "direct"},
        stru_options={
            "pbc": [False, False, False], "kmesh": [1, 1, 1],
            "gamma_center": True, "time_reversal_symmetry": False,
            "lead_L": {"useBloch": False, "voltage": 0},
            "lead_R": {"useBloch": False, "voltage": 0},
        },
        poisson_options={},
        energy_grid={"method": "uniform", "emin": -1, "emax": 1, "num_points": 3},
    )
    assert runner.e_batch_size == 1
    assert type(runner.e_batch_size) is int


@pytest.mark.parametrize("field", ["startup_timeout", "sync_timeout", "task_timeout"])
@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf")])
def test_invalid_rgf_timeouts_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        parrallel.validate_rgf_options({field: value})


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_pending_window_is_rejected(value):
    with pytest.raises(ValueError, match="max_pending_results"):
        parrallel.validate_rgf_options({"max_pending_results": value})


def test_thread_limits_restore_torch_after_success_and_failure():
    original = torch.get_num_threads()
    with parrallel._rgf_cpu_thread_limits(1):
        assert torch.get_num_threads() == 1
    assert torch.get_num_threads() == original

    with pytest.raises(RuntimeError, match="expected"):
        with parrallel._rgf_cpu_thread_limits(1):
            assert torch.get_num_threads() == 1
            raise RuntimeError("expected")
    assert torch.get_num_threads() == original
