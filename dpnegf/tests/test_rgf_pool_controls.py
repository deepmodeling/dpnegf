"""Timeout, compact-message and dynamic scheduling regression tests."""

import gc
import multiprocessing as mp
import os
import queue
import struct
import threading
import time
import weakref

import pytest
import torch

from dpnegf.negf.device_property import DeviceProperty
from dpnegf.utils import parrallel
from dpnegf.utils.parrallel import RGFProcessPool, RGFWorkerError
from dpnegf.utils.rgf_shared import SharedHS


def _controls_worker(connection, device, spec):
    send, recv = parrallel._send_packet, parrallel._recv_packet
    attachments = []
    if spec.get("startup_delay"):
        time.sleep(spec["startup_delay"])
    send(connection, ("ready", os.getpid()))
    try:
        while True:
            command, args = recv(connection)
            if command == "close":
                return
            if command == "shared_hs":
                attachment = SharedHS.attach(args)
                attachments.append(attachment)
                spec["attached"].set()
                if spec.get("mode") == "shared_sync_timeout":
                    spec["never"].wait()
                send(connection, ("shared_hs", args["version"], args["name"]))
                continue
            if command == "chunk_size":
                if spec.get("mode") == "sync_timeout":
                    spec["never"].wait()
                send(connection, ("chunk_size", spec.get("batch", 1)))
                continue

            index, energies, k, ik, bias, offset, shared_version = args
            if spec.get("compact"):
                assert energies.untyped_storage().nbytes() == energies.numel() * energies.element_size()
            if spec.get("seen_tasks") is not None:
                spec["seen_tasks"].put((bias, shared_version))
            if spec.get("mode") == "task_timeout":
                spec["never"].wait()
            if spec.get("mode") == "late_error" and offset == 1:
                assert spec["first_merged"].wait(5)
                send(connection, ("error", "RuntimeError", "late task failed", "test traceback"))
                return
            if spec.get("mode") == "partial_message":
                connection._send(struct.pack("!i", 128))
                connection._send(b"partial")
                spec["never"].wait()
            if spec.get("mode") == "dynamic":
                if offset == 0:
                    assert spec["later_task"].wait(5)
                elif offset == 2:
                    spec["later_task"].set()
            if spec.get("seen") is not None:
                spec["seen"].put((offset, os.getpid()))
            send(connection, ("result", index, {
                "T_k": {str(k): [energies]},
                "worker_pid": {str(k): [os.getpid()]},
            }))
    finally:
        for attachment in attachments:
            attachment.close()
        connection.close()


def _pool(monkeypatch, spec, **options):
    monkeypatch.setattr(parrallel, "_rgf_worker_main", _controls_worker)
    return RGFProcessPool(["cpu"] * options.pop("workers", 2), spec, **options)


def test_receiver_releases_consumed_packet_while_waiting(monkeypatch):
    pool = RGFProcessPool.__new__(RGFProcessPool)
    pool.messages = queue.Queue()
    pool._messages_lock = threading.Lock()
    pool.closed = False
    payload = torch.empty(1024)
    payload_ref = weakref.ref(payload)
    packets = [("result", 0, {"payload": payload})]
    del payload
    wait_for_stop = threading.Event()

    def receive(_connection):
        if packets:
            return packets.pop()
        wait_for_stop.wait(5)
        raise EOFError

    monkeypatch.setattr(parrallel, "_recv_packet", receive)
    receiver = threading.Thread(target=pool._receiver_main, args=(0, object()))
    receiver.start()
    event = pool.messages.get(timeout=2)
    assert event[1][0] == "result"
    del event
    gc.collect()
    assert payload_ref() is None

    wait_for_stop.set()
    receiver.join(timeout=2)
    assert not receiver.is_alive()


def test_terminate_clears_messages_and_pending_state():
    pool = RGFProcessPool.__new__(RGFProcessPool)
    pool.messages = queue.Queue()
    pool._messages_lock = threading.Lock()
    pool.pending = {0: ("task", 0.0, 10.0, {"index": 0})}
    pool.closed = False
    pool.processes = []
    pool.connections = []
    pool.receiver_threads = []
    pool.process_watchers = []
    pool._deviceprop = None
    pool._owner_regions = []
    pool._shared_hs = None

    payload = torch.empty(1024)
    payload_ref = weakref.ref(payload)
    pool.messages.put((0, ("result", 0, payload), None, time.monotonic()))
    del payload

    pool.terminate()
    gc.collect()
    assert pool.messages.empty()
    assert pool.pending == {}
    assert payload_ref() is None

    pool._queue_message((0, ("process_exit", 7), None, time.monotonic()))
    assert pool.messages.empty()


def test_terminate_reports_threads_that_remain_alive():
    class StillAliveThread:
        name = "rgf-recv-stuck"
        ident = 123

        def __init__(self):
            self.join_timeouts = []
            self.checked = False

        def join(self, timeout):
            self.join_timeouts.append(timeout)

        def is_alive(self):
            self.checked = True
            return True

    pool = RGFProcessPool.__new__(RGFProcessPool)
    pool.messages = queue.Queue()
    pool._messages_lock = threading.Lock()
    pool.pending = {}
    pool.closed = False
    pool.processes = []
    pool.connections = []
    pool.receiver_threads = [StillAliveThread()]
    pool.process_watchers = []
    pool._deviceprop = None
    pool._owner_regions = []
    pool._shared_hs = None

    with pytest.raises(RGFWorkerError, match=r"threads alive: rgf-recv-stuck\(ident=123\)"):
        pool.terminate()
    assert pool.receiver_threads[0].join_timeouts == [1]
    assert pool.receiver_threads[0].checked
    assert pool.messages.empty()
    assert pool.pending == {}


def test_fast_worker_gets_later_energy_before_first_task_finishes(monkeypatch):
    context = mp.get_context("spawn")
    spec = {"mode": "dynamic", "batch": 1,
            "later_task": context.Event(), "seen": context.Queue()}
    with _pool(monkeypatch, spec, max_pending_results=4) as pool:
        out = {}
        pool.run_energy_loop(torch.arange(6), [0, 0, 0], 0, None, out)
        assert torch.equal(torch.cat(out["T_k"]["[0, 0, 0]"]), torch.arange(6))
        seen = [spec["seen"].get(timeout=1) for _ in range(6)]
        later_task = next(pid for offset, pid in seen if offset == 2)
        assert later_task == pool.worker_pids[1]
        assert len({offset for offset, _pid in seen}) == 6
    spec["seen"].close()
    spec["seen"].join_thread()


def test_energy_slice_sent_to_worker_has_compact_storage(monkeypatch):
    with _pool(monkeypatch, {"batch": 1, "compact": True}) as pool:
        backing = torch.arange(20, dtype=torch.float64)
        grid = backing[::2]
        pool.run_energy_loop(grid, [0, 0, 0], 0, None, {})


@pytest.mark.parametrize("window", [1, 2, 4])
def test_pending_window_preserves_order_and_tail_batch(monkeypatch, window):
    with _pool(monkeypatch, {"batch": 1}, max_pending_results=window) as pool:
        out = {}
        pool.run_energy_loop(torch.arange(5), [0, 0, 0], 0, None, out)
        torch.testing.assert_close(
            torch.cat(out["T_k"]["[0, 0, 0]"]), torch.arange(5))


def test_failed_energy_scan_discards_staged_outputs(monkeypatch):
    context = mp.get_context("spawn")
    spec = {"mode": "late_error", "batch": 1,
            "first_merged": context.Event()}
    pool = _pool(monkeypatch, spec, workers=2)
    original_merge = RGFProcessPool._merge_outputs
    caller_out = {"existing": {"entry": ["keep"]}}

    def signal_first_merge(target, k, outputs):
        original_merge(target, k, outputs)
        if target is not caller_out:
            spec["first_merged"].set()

    monkeypatch.setattr(RGFProcessPool, "_merge_outputs", staticmethod(signal_first_merge))
    try:
        with pytest.raises(RGFWorkerError, match="late task failed"):
            pool.run_energy_loop(torch.arange(3), [0.0, 0.0, 0.0], 0, None, caller_out)
        assert caller_out == {"existing": {"entry": ["keep"]}}
    finally:
        pool.terminate()


@pytest.mark.parametrize("mode,phase", [
    ("task_timeout", "task"),
    ("sync_timeout", "sync"),
    ("partial_message", "task"),
])
def test_worker_deadlines_reap_hung_or_partial_workers(monkeypatch, mode, phase):
    context = mp.get_context("spawn")
    spec = {"mode": mode, "batch": 1, "never": context.Event()}
    pool = _pool(
        monkeypatch, spec, workers=1, startup_timeout=3,
        sync_timeout=0.2, task_timeout=0.2,
    )
    started = time.monotonic()
    with pytest.raises(RGFWorkerError, match=rf"RGF {phase} timeout"):
        pool.run_energy_loop(torch.arange(3), [0, 0, 0], 0, None, {})
    assert time.monotonic() - started < 5
    assert pool.closed
    assert all(not process.is_alive() for process in pool.processes)
    assert all(connection.closed for connection in pool.connections)
    assert all(not thread.is_alive() for thread in pool.receiver_threads)


def test_startup_deadline_stops_worker_that_never_becomes_ready(monkeypatch):
    monkeypatch.setattr(parrallel, "_rgf_worker_main", _controls_worker)
    with pytest.raises(RGFWorkerError, match="RGF startup timeout"):
        RGFProcessPool(["cpu"], {"startup_delay": 1.0}, startup_timeout=0.2)


class _SharedHamiltonian:
    def __init__(self):
        self.parts = (
            [torch.eye(2, dtype=torch.complex128)],
            [torch.eye(2, dtype=torch.complex128)],
            [], [], [], [],
        )
        self.device_norbs = [1, 1]

    def get_hs_device(self, kpoint, potential, block_tridiagonal, device=None):
        return tuple([[block.to(device) for block in group] for group in self.parts])

    def get_orbital_map(self):
        return {"status": "unknown", "orbital_map": None}


def test_shared_mapping_timeout_unlinks_owner_region(monkeypatch, tmp_path):
    context = mp.get_context("spawn")
    spec = {"mode": "shared_sync_timeout", "never": context.Event(),
            "attached": context.Event()}
    monkeypatch.setattr(parrallel, "_rgf_worker_main", _controls_worker)
    published_names = []
    original_publish = SharedHS.publish.__func__

    def record_publish(cls, *args, **kwargs):
        region = original_publish(cls, *args, **kwargs)
        published_names.append(region.descriptor["name"])
        return region

    monkeypatch.setattr(SharedHS, "publish", classmethod(record_publish))
    deviceprop = DeviceProperty.__new__(DeviceProperty)
    deviceprop.greenfuncs = 0
    deviceprop.hamiltonian = _SharedHamiltonian()
    deviceprop.rgf_device = torch.device("cpu")
    deviceprop.results_path = str(tmp_path)
    deviceprop.kpoint = None
    pool = RGFProcessPool(
        ["cpu"], spec, sync_timeout=0.2, task_timeout=1,
        share_cpu_hs=True,
    )
    pool._deviceprop = deviceprop

    with pytest.raises(RGFWorkerError, match="RGF sync timeout.*shared H/S version 1"):
        pool.run_energy_loop(torch.arange(2), [0.0, 0.0, 0.0], 0, None, {})
    assert spec["attached"].is_set()
    assert pool.closed and pool._owner_regions == []
    assert all(hasattr(deviceprop, name) for name in ("hd", "sd", "hl", "hu", "sl", "su"))
    assert not hasattr(deviceprop, "_shared_hs_owner")
    deviceprop.prepare_hamiltonian([0.0, 0.0, 0.0], True, None)
    assert len(published_names) == 1
    with pytest.raises(FileNotFoundError):
        SharedHS.attach({
            "name": published_names[0], "blocks": {}, "context": {}, "version": 1,
        })


def test_shared_tasks_send_context_version_instead_of_bias(monkeypatch, tmp_path):
    context = mp.get_context("spawn")
    spec = {"batch": 1, "attached": context.Event(), "seen_tasks": context.Queue()}
    deviceprop = DeviceProperty.__new__(DeviceProperty)
    deviceprop.greenfuncs = 0
    deviceprop.hamiltonian = _SharedHamiltonian()
    deviceprop.rgf_device = torch.device("cpu")
    deviceprop.results_path = str(tmp_path)
    deviceprop.kpoint = None
    bias = torch.tensor([0.15, 0.25])

    with _pool(monkeypatch, spec, workers=1, share_cpu_hs=True) as pool:
        pool._deviceprop = deviceprop
        pool.run_energy_loop(torch.arange(2), [0.0, 0.0, 0.0], 0, bias, {})
        sent_bias, version = spec["seen_tasks"].get(timeout=2)
        assert sent_bias is None
        assert version == 1
    assert all(hasattr(deviceprop, name) for name in ("hd", "sd", "hl", "hu", "sl", "su"))
    assert not hasattr(deviceprop, "_shared_hs_owner")
    deviceprop.prepare_hamiltonian([0.0, 0.0, 0.0], True, bias)

    second_spec = {
        "batch": 1, "attached": context.Event(), "seen_tasks": context.Queue(),
    }
    with _pool(monkeypatch, second_spec, workers=1, share_cpu_hs=True) as pool:
        pool._deviceprop = deviceprop
        pool.run_energy_loop(torch.arange(1), [0.0, 0.0, 0.0], 0, bias, {})
        sent_bias, version = second_spec["seen_tasks"].get(timeout=2)
        assert sent_bias is None
        assert version == 1
    spec["seen_tasks"].close()
    spec["seen_tasks"].join_thread()
    second_spec["seen_tasks"].close()
    second_spec["seen_tasks"].join_thread()


class _UncopyableMapping:
    def __deepcopy__(self, memo):
        raise RuntimeError("mapping copy failed")


def test_shared_publish_mapping_failure_releases_region(monkeypatch, tmp_path):
    monkeypatch.setattr(parrallel, "_rgf_worker_main", _controls_worker)
    published_names = []
    original_publish = SharedHS.publish.__func__

    def record_publish(cls, *args, **kwargs):
        region = original_publish(cls, *args, **kwargs)
        published_names.append(region.descriptor["name"])
        return region

    monkeypatch.setattr(SharedHS, "publish", classmethod(record_publish))
    deviceprop = DeviceProperty.__new__(DeviceProperty)
    deviceprop.greenfuncs = 0
    deviceprop.rgf_device = torch.device("cpu")
    deviceprop.results_path = str(tmp_path)
    deviceprop.kpoint = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)
    deviceprop.V = torch.tensor(0.0)
    deviceprop.block_tridiagonal = True
    deviceprop._orbital_mapping_snapshot = _UncopyableMapping()
    (deviceprop.hd, deviceprop.sd, deviceprop.hl,
     deviceprop.hu, deviceprop.sl, deviceprop.su) = _SharedHamiltonian().parts

    pool = RGFProcessPool(
        ["cpu"], {"attached": mp.get_context("spawn").Event()}, share_cpu_hs=True)
    pool._deviceprop = deviceprop
    with pytest.raises(RuntimeError, match="mapping copy failed"):
        pool.run_energy_loop(torch.arange(1), [0.0, 0.0, 0.0], 0, None, {})
    assert pool.closed and pool._owner_regions == []
    assert len(published_names) == 1
    with pytest.raises(FileNotFoundError):
        SharedHS.attach({"name": published_names[0], "blocks": {}, "context": {}, "version": 1})
