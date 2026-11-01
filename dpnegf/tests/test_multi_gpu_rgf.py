"""Energy-parallel RGF: numerical equivalence, ordering and state isolation."""

import importlib
import json
from pathlib import Path
import multiprocessing as mp
import os
import time
from types import SimpleNamespace

import pytest
import torch

from dpnegf.runner.NEGF import NEGF
from dpnegf.utils import parrallel
from dpnegf.utils.parrallel import RGFProcessPool, RGFWorkerError, _rgf_devices
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


def _protocol_worker(connection, device, spec):
    """Real spawn target for scheduling and process-failure tests."""
    send, recv = parrallel._send_packet, parrallel._recv_packet
    mode = spec.get("mode", "normal")
    if mode == "startup_error":
        send(connection, ("error", "ValueError", "startup failed", "test traceback"))
        connection.close()
        return
    send(connection, ("ready", os.getpid()))
    try:
        while True:
            command, args = recv(connection)
            if command == "close":
                return
            if command == "chunk_size":
                send(connection, ("chunk_size", spec.get("batch", args[0])))
                continue
            index, energies, k, ik, bias, offset = args
            if mode in ("error", "exit"):
                if offset == 0:
                    # The earlier worker never returns; its peer must still be observed.
                    spec["started"].set()
                    spec["never"].wait()
                else:
                    assert spec["started"].wait(10)
                    if mode == "exit":
                        os._exit(7)
                    send(connection, ("error", "RuntimeError", "worker failed", "test traceback"))
                    return
            if mode == "overlap" and ik == 0:
                if offset == 0:
                    assert spec["second_finished"].wait(10), "workers did not overlap"
                elif offset == 2:
                    spec["second_finished"].set()
            send(connection, ("result", index, {
                "T_k": {str(k): [energies]},
                "assignment": {str(k): [(offset, os.getpid(), mp.get_start_method())]},
            }))
    finally:
        connection.close()


def test_spawn_pool_overlaps_orders_and_reuses_workers(monkeypatch):
    monkeypatch.setattr(parrallel, "_rgf_worker_main", _protocol_worker)
    spec = {"mode": "overlap", "batch": 2,
            "second_finished": mp.get_context("spawn").Event()}
    with RGFProcessPool(["cpu", "cpu"], spec) as pool:
        pids = pool.worker_pids.copy()
        assert len(set(pids)) == 2
        assert os.getpid() not in pids
        out = {}
        for ik, k in enumerate(([0, 0, 0], [0.5, 0, 0])):
            pool.run_energy_loop(torch.arange(7), k, ik, None, out)
            assert pool.worker_pids == pids
            torch.testing.assert_close(torch.cat(out["T_k"][str(k)]), torch.arange(7))
            assert out["assignment"][str(k)] == [
                (offset, pids[index % 2], "spawn")
                for index, offset in enumerate((0, 2, 4, 6))]
        assert all(process.is_alive() for process in pool.processes)
    assert pool.closed
    assert all(not process.is_alive() and process.exitcode == 0 for process in pool.processes)
    assert all(connection.closed for connection in pool.connections)
    with pytest.raises(RuntimeError, match="closed"):
        pool.run_energy_loop(torch.arange(1), [0, 0, 0], 0, None, {})


@pytest.mark.parametrize("mode", ["error", "exit"])
def test_failure_terminates_hung_peer(monkeypatch, mode):
    monkeypatch.setattr(parrallel, "_rgf_worker_main", _protocol_worker)
    context = mp.get_context("spawn")
    pool = RGFProcessPool(["cpu", "cpu"], {
        "mode": mode, "batch": 1, "started": context.Event(), "never": context.Event()})
    started = time.monotonic()
    with pytest.raises(RGFWorkerError, match="worker failed|disconnected|code 7"):
        pool.run_energy_loop(torch.arange(3), [0, 0, 0], 0, None, {})
    assert time.monotonic() - started < 10
    assert pool.closed
    assert all(not process.is_alive() for process in pool.processes)
    assert all(connection.closed for connection in pool.connections)


def test_startup_failure_cleans_up_workers(monkeypatch):
    monkeypatch.setattr(parrallel, "_rgf_worker_main", _protocol_worker)
    children_before = {process.pid for process in mp.active_children()}
    with pytest.raises(RGFWorkerError, match="startup failed"):
        RGFProcessPool(["cpu", "cpu"], {"mode": "startup_error"})
    assert {process.pid for process in mp.active_children()} == children_before


def test_real_cache_solver_reports_missing_cache_and_reaps_workers(tmp_path):
    # A worker can start and report a numerical/cache error using only these
    # settings, without a model, NEGF runner, or model-backed Hamiltonian.
    spec = {
        "default_dtype": torch.float64,
        "options": {"e_batch_size": 1, "block_tridiagonal": True,
                    "eta_lead": 1e-5, "sgf_solver": "Sancho-Rubio",
                    "self_energy_save_path": str(tmp_path),
                    "se_cache_format": "h5", "se_info_display": False},
        "hamiltonian": {"cache_path": str(tmp_path), "device_norbs": [1]},
        "device": {"results_path": str(tmp_path), "e_T": 300,
                   "chemiPot": {"lead_L": 0., "lead_R": 0.}, "E_ref": 0.},
        "leads": {name: {"voltage": 0., "e_T": 300, "efermi": 0., "E_ref": 0.}
                  for name in ("lead_L", "lead_R")},
    }
    pool = RGFProcessPool(["cpu", "cpu"], spec)
    with pytest.raises(RGFWorkerError, match="FileNotFoundError.*prepared self-energy cache"):
        pool.run_energy_loop(torch.arange(3), [0, 0, 0], 0, None, {})
    assert pool.closed
    assert all(not process.is_alive() for process in pool.processes)


def test_auto_batches_use_every_selected_device(monkeypatch):
    monkeypatch.setattr(parrallel, "_rgf_worker_main", _protocol_worker)
    with RGFProcessPool(["cpu"] * 4, {}) as pool:
        out = {}
        k = [0, 0, 0]
        pool.run_energy_loop(torch.arange(5), k, 0, None, out)
        assert out["assignment"][str(k)] == [
            (index, pool.worker_pids[index % 4], "spawn") for index in range(5)]
        torch.testing.assert_close(torch.cat(out["T_k"][str(k)]), torch.arange(5))


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

    if mode == "full":
        # Two transverse k points exercise H/S refresh and pool reuse in NEGF itself.
        task["stru_options"].update(pbc=[True, False, False], kmesh=[2, 1, 1])

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

    pools = []
    original_init = RGFProcessPool.__init__

    def record_pool(self, devices, spec):
        # Nothing in this specification can carry the runner or a neural model.
        assert set(spec) == {"default_dtype", "options", "hamiltonian", "device", "leads"}
        assert set(spec["hamiltonian"]) == {
            "cache_path", "device_norbs", "input_orbital_mapping"}
        original_init(self, devices, spec)
        pools.append(self)

    calls = []
    original_loop = RGFProcessPool.run_energy_loop

    def record_loop(self, grid, k, ik, bias, out):
        calls.append((id(self), tuple(self.worker_pids), ik))
        return original_loop(self, grid, k, ik, bias, out)

    monkeypatch.setattr(RGFProcessPool, "__init__", record_pool)
    monkeypatch.setattr(RGFProcessPool, "run_energy_loop", record_loop)
    parallel = execute("parallel")
    assert len(pools) == 1
    assert len(set(pools[0].worker_pids)) == 2
    assert pools[0].devices == module._rgf_devices(task["rgf_options"])
    assert all(not process.is_alive() and process.exitcode == 0
               for process in pools[0].processes)
    assert len(serial["k"]) == (2 if mode == "full" else 1)
    assert calls == [(id(pools[0]), tuple(pools[0].worker_pids), ik)
                     for ik in range(len(serial["k"]))]
    for key in ("T_avg", "energy_grid"):
        torch.testing.assert_close(parallel[key], serial[key], atol=1e-9, rtol=1e-8)
    for kpoint in serial["k"]:
        k = str(kpoint)
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
