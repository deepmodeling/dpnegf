"""Energy-parallel RGF: numerical equivalence, ordering and state isolation."""

import importlib
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
import torch

from dpnegf.runner.NEGF import NEGF
from dpnegf.utils import parrallel
from dpnegf.utils.parrallel import _parallel_rgf_energy_loop, _rgf_devices
from dpnegf.utils.argcheck import normalize_run


DATA = Path(__file__).parent / "data/test_negf/test_negf_run"


@pytest.mark.parametrize("devices", [
    [], "cuda:0", [0, 1], ["cpu"], ["cuda"],
    ["cuda:0", "cuda:0"], ["cuda:2"],
])
def test_invalid_devices(monkeypatch, devices):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError):
        _rgf_devices({"devices": devices})


def test_device_options_and_scf_guard(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert _rgf_devices({}) == [torch.device("cpu")]
    assert _rgf_devices({"device": "cuda"}) == [torch.device("cuda")]
    options = {"device": "cpu", "devices": ["cuda:1", "cuda:0"]}
    assert _rgf_devices(options) == [torch.device("cuda:1"), torch.device("cuda:0")]
    with pytest.raises(ValueError, match="scf=false"):
        NEGF(model=SimpleNamespace(device="cpu"), structure=None, ele_T=300,
             scf=True, rgf_options=options,
             energy_grid={"method": "uniform", "emin": -1, "emax": 1, "num_points": 3})


def test_scheduler_overlaps_workers_and_preserves_order(monkeypatch):
    runner = NEGF.__new__(NEGF)
    runner.uni_grid = torch.arange(7)
    runner.rgf_devices = [torch.device("cpu"), torch.device("cpu")]
    runner.out = {}
    second_finished = Event()
    assignments = []
    active = set()

    def make_worker(runner, device):
        slot = len(assignments)
        assignments.append([])

        def compute(energies, k, ik, bias, offset):
            assert slot not in active, "a worker must not run overlapping batches"
            active.add(slot)
            try:
                assignments[slot].append(offset)
                if offset == 0:
                    assert second_finished.wait(10), "workers did not overlap"
                elif offset == 2:
                    second_finished.set()
                worker.out = {"T_k": {str(k): [energies.clone()]}}
            finally:
                active.remove(slot)

        worker = SimpleNamespace(rgf_device=device, e_batch_size=2,
                                 _compute_rgf_chunk=compute)
        return worker

    monkeypatch.setattr(parrallel, "_make_rgf_worker", make_worker)
    _parallel_rgf_energy_loop(runner, [0, 0, 0], 0, None)
    torch.testing.assert_close(torch.cat(runner.out["T_k"]["[0, 0, 0]"]), runner.uni_grid)
    assert assignments == [[0, 4], [2, 6]]


def test_scheduler_propagates_failure(monkeypatch):
    runner = NEGF.__new__(NEGF)
    runner.uni_grid = torch.arange(3)
    runner.rgf_devices = [torch.device("cpu"), torch.device("cpu")]
    runner.out = {}

    def fail(*args):
        raise RuntimeError("worker failed")

    monkeypatch.setattr(parrallel, "_make_rgf_worker", lambda runner, device: SimpleNamespace(
        rgf_device=device, e_batch_size=1, _compute_rgf_chunk=fail))
    with pytest.raises(RuntimeError, match="worker failed"):
        _parallel_rgf_energy_loop(runner, [0, 0, 0], 0, None)


def test_auto_batches_use_every_selected_device(monkeypatch):
    runner = NEGF.__new__(NEGF)
    runner.uni_grid = torch.arange(5)
    runner.rgf_devices = [torch.device("cpu")] * 4
    runner.out = {}
    assignments = []

    def make_worker(runner, device):
        offsets = []
        assignments.append(offsets)

        def compute(energies, k, ik, bias, offset):
            offsets.append(offset)
            worker.out = {"T_k": {str(k): [energies.clone()]}}

        worker = SimpleNamespace(rgf_device=device, e_batch_size=None,
                                 _compute_rgf_chunk=compute)
        return worker

    monkeypatch.setattr(parrallel, "_make_rgf_worker", make_worker)
    _parallel_rgf_energy_loop(runner, [0, 0, 0], 0, None)
    assert assignments == [[0, 4], [1], [2], [3]]
    torch.testing.assert_close(torch.cat(runner.out["T_k"]["[0, 0, 0]"]), runner.uni_grid)


@pytest.mark.parametrize("backend", [
    "cpu_workers",
    pytest.param("cuda", marks=pytest.mark.skipif(
        torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")),
])
@pytest.mark.parametrize("mode,batch", [("full", 2), ("transmission_only", 1), ("dense", None)])
def test_parallel_runner_matches_serial(tmp_path, monkeypatch, backend, mode, batch):
    from dpnegf.entrypoints.run import run

    module = importlib.import_module("dpnegf.runner.NEGF")
    raw = json.loads((DATA / "negf_chain_new.json").read_text())
    raw["structure"] = str(DATA / "chain.vasp")
    raw["device"] = "cpu"
    task = raw["task_options"]
    for key in list(task):
        if key.startswith("out_") or key in ("emin", "emax", "espacing", "sgf_solver"):
            task.pop(key)
    task.update(
        block_tridiagonal=mode != "dense",
        btd_initialization="dense" if mode == "dense" else "direct",
        energy_grid={"method": "uniform", "emin": -.07, "emax": .07, "num_points": 7},
        output_options=({"tc": True} if mode == "transmission_only" else {
            "tc": True, "dos": True, "ldos": True,
            "trans_eigvals": True, "trans_eigchannel": True}),
        trans_eigs_options={"n_channels": 1},
        rgf_options={"device": "cpu", "e_batch_size": batch},
        self_energy_options={"numba_jit": False,
                             "parallel": {"n_workers": 1, "blas_threads": 1},
                             "cache": {"save_path": str(tmp_path / "se")}},
    )

    def execute(name):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(raw))
        run(str(path), str(DATA / "nnsk_C_new.json"), str(DATA / "chain.vasp"),
            str(tmp_path / name), None)
        return torch.load(tmp_path / name / "results/negf.out.pth", weights_only=False)

    serial = execute("serial")
    task["self_energy_options"]["cache"]["use_saved"] = True
    # Compare the scheduling using identical Hamiltonian and self-energy caches.
    task["hs_cache"] = {"use_saved": True, "save_path": str(tmp_path / "serial/results")}
    task["rgf_options"]["devices"] = ["cuda:0", "cuda:1"]
    # Schema must accept the new option before any device mocking.
    assert normalize_run(raw)["task_options"]["rgf_options"]["devices"] == ["cuda:0", "cuda:1"]
    if backend == "cpu_workers":
        original = module._rgf_devices
        monkeypatch.setattr(module, "_rgf_devices", lambda opts: (
            [torch.device("cpu"), torch.device("cpu")]
            if opts.get("devices") else original(opts)))

    seen_devices = []
    compute = NEGF._compute_rgf_chunk

    def check_device(self, *args, **kwargs):
        try:
            return compute(self, *args, **kwargs)
        finally:
            assert self.deviceprop.greenfuncs == 0
            assert self.deviceprop._trans_eigs_state is None
            assert all(block.device == self.rgf_device for block in self.deviceprop.hd)
            seen_devices.append(self.rgf_device)

    monkeypatch.setattr(NEGF, "_compute_rgf_chunk", check_device)
    parallel = execute("parallel")
    assert set(seen_devices) == set(module._rgf_devices(task["rgf_options"]))
    k = str(serial["k"][0])
    for key in ("T_avg", "energy_grid"):
        torch.testing.assert_close(parallel[key], serial[key], atol=1e-9, rtol=1e-8)
    for key in ("T_k", "DOS", "LDOS"):
        if key in serial:
            torch.testing.assert_close(parallel[key][k], serial[key][k], atol=1e-9, rtol=1e-8)
    if mode != "transmission_only":
        for key in ("trans_eigvals", "trans_eigchannel"):
            assert len(parallel[key][k]) == 7
            for index, (a, b) in enumerate(zip(parallel[key][k], serial[key][k])):
                torch.testing.assert_close(a["energy"], b["energy"])
                if key == "trans_eigchannel":
                    assert a["path"] == b["path"]
                    assert a["path"].endswith(f"e{index:08d}.pth")
                    a = torch.load(tmp_path / "parallel/results" / a["path"], weights_only=False)
                    b = torch.load(tmp_path / "serial/results" / b["path"], weights_only=False)
                    ca, cb = a["coefficients"], b["coefficients"]
                    torch.testing.assert_close(ca @ ca.mH, cb @ cb.mH, atol=1e-8, rtol=1e-7)
                torch.testing.assert_close(a["tau_all"], b["tau_all"], atol=1e-9, rtol=1e-8)
