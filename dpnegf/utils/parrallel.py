"""Functions and classes for the parallel environment."""

import copy
import os
import logging
import multiprocessing as mp
import math
import pickle
import queue
import threading
import time
import traceback
from contextlib import contextmanager
from numbers import Integral, Real
from typing import Any, List, Tuple
import datetime

import numpy as np
import torch
from joblib import cpu_count
from threadpoolctl import threadpool_info, threadpool_limits

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


__all__ = ["MPIEnv"]

log = logging.getLogger(__name__)


def _rgf_devices(options):
    """Resolve an optional explicit list of logical CUDA devices."""
    devices = options.get("devices")
    if devices is None:
        return [torch.device(options.get("device", "cpu"))]
    if not isinstance(devices, list) or not devices:
        raise ValueError("rgf_options.devices must be a non-empty list of CUDA devices.")
    resolved = []
    for value in devices:
        if not isinstance(value, str):
            raise ValueError("rgf_options.devices entries must be strings such as 'cuda:0'.")
        device = torch.device(value)
        if device.type != "cuda" or device.index is None:
            raise ValueError("rgf_options.devices requires explicit CUDA indices, e.g. 'cuda:0'.")
        if device in resolved:
            raise ValueError(f"Duplicate RGF device: {device}.")
        if device.index >= torch.cuda.device_count():
            raise ValueError(f"RGF CUDA device is unavailable: {device}.")
        resolved.append(device)
    return resolved


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"rgf_options.{name} must be a positive integer.")
    return int(value)


def _timeout_value(value, name):
    if value is None:
        return None
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f"rgf_options.{name} must be a positive finite number or None.")
    return float(value)


def _rgf_cpu_options(options):
    workers = options.get("cpu_workers", 1)
    workers = _positive_integer(workers, "cpu_workers")
    threads = options.get("cpu_threads")
    if threads is not None:
        threads = _positive_integer(threads, "cpu_threads")
    return {"cpu_workers": workers, "cpu_threads": threads}


def _rgf_pool_options(options):
    window = options.get("max_pending_results")
    if window is not None:
        _positive_integer(window, "max_pending_results")
    return {
        "startup_timeout": _timeout_value(options.get("startup_timeout", 600.0), "startup_timeout"),
        "sync_timeout": _timeout_value(options.get("sync_timeout", 600.0), "sync_timeout"),
        "task_timeout": _timeout_value(options.get("task_timeout", 3000.0), "task_timeout"),
        "max_pending_results": window,
    }


def validate_rgf_options(options, scf=False):
    """Normalize and cross-check CPU worker and process-pool settings."""
    options = dict(options or {})
    cpu = _rgf_cpu_options(options)
    pool = _rgf_pool_options(options)
    workers, threads = cpu["cpu_workers"], cpu["cpu_threads"]
    explicit_devices = options.get("devices") is not None
    device = torch.device(options.get("device", "cpu"))

    if workers > 1 and scf:
        raise ValueError("Multiple CPU RGF workers require scf=false.")
    if (workers != 1 or threads is not None) and (explicit_devices or device.type == "cuda"):
        raise ValueError("CPU RGF worker options cannot be combined with CUDA or rgf_options.devices.")

    if workers > 1 or threads is not None:
        effective_threads = threads if threads is not None else 1
        budget = cpu_count()
        if workers * effective_threads > budget:
            raise ValueError(
                f"rgf_options.cpu_workers * cpu_threads requests "
                f"{workers * effective_threads} threads, exceeding the available CPU budget {budget}."
            )

    batch = options.get("e_batch_size")
    if batch is not None and (
        isinstance(batch, bool) or not isinstance(batch, Real)
        or not math.isfinite(batch) or batch < 1 or int(batch) != batch
    ):
        raise ValueError("rgf_options.e_batch_size must be a positive integer or None.")
    if workers > 1 and batch not in (None, 1):
        raise ValueError("Multiple CPU RGF workers require e_batch_size=1 or None.")
    return {
        **cpu, **pool,
        "e_batch_size": None if batch is None else int(batch),
    }


@contextmanager
def _rgf_cpu_thread_limits(threads):
    """Temporarily limit Torch and native CPU pools for one RGF phase."""
    if threads is None:
        yield
        return
    original = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        with threadpool_limits(limits=threads):
            yield
    finally:
        torch.set_num_threads(original)


def _auto_chunk_size(runner, n_grid):
    """Pick a chunk size from free CUDA memory when the user didn't set
    ``e_batch_size``. Returns the full grid length on CPU. If the CUDA
    memory query or geometry estimate fails, logs the cause and returns 1.

    Call only after prepare_hamiltonian() has made the current k/bias H/S
    resident. The free-memory query then excludes those persistent tensors.
    Per-energy RGF peak (post per-slot-release, complex128) approximated as
        bytes_per_E ~= C * K * n_max**2 * 16
    with C bundling the live tensors in the worst backward-sweep slot
    (grd full + grl + gru full + decaying gr_left tail + gU + transients
    and allocator headroom). C=14 with a 0.5x free-memory budget; conservative
    because without expandable_segments the allocator can't defragment on
    demand, and on real workloads (CNT10/long6) the previous 10x / 0.7
    combination still picked a chunk that OOM'd on a 15.77 GiB V100.

    Eigen outputs additionally retain batched gr_left/self-energy snapshots.
    Reserve a separate, batch-independent workspace for contact eigh/SVD and
    thin RHS/channel coefficients: eigen outputs are streamed one energy at
    a time while the entire RGF batch remains resident. Contact block sizes
    bound the unknown broadening ranks conservatively.
    """
    rgf_dev = runner.rgf_device
    if not (isinstance(rgf_dev, torch.device) and rgf_dev.type == "cuda"):
        return n_grid
    try:
        sizes = [int(b.shape[-1]) for b in runner.deviceprop.hd]
        n_max = max(sizes)
        K = len(sizes)
        per_e = 14 * K * (n_max ** 2) * 16
        if min(sizes) <= 0 or per_e <= 0:
            raise ValueError(f"Non-positive per-energy memory estimate: {per_e}")
        eig_workspace = 0
        if getattr(runner, "need_trans_eigs", False):
            left, right = sizes[0], sizes[-1]
            boundary_sq = left**2 + right**2
            # Retained gr_left and copied self-energy state scale with B.
            per_e += 16 * (sum(n**2 for n in sizes) + 2 * boundary_sq)
            options = runner.trans_eigs_options
            incoming = right if options.get("incoming", "left") == "right" else left
            # Gamma factors, contact Green/K matrices, eigh/SVD scratch and
            # streaming contact solves; include orbital permutation buffers.
            eig_workspace = 16 * (8 * (boundary_sq + left * right) + 4 * n_max * incoming)
            eig_workspace += 3 * sum(sizes) * 8
            if runner.out_trans_eigchannel:
                selected = min(left, right)
                if options.get("n_channels") is not None:
                    selected = min(selected, options["n_channels"])
                elif options.get("channel_indices") is not None:
                    selected = min(selected, len(options["channel_indices"]))
                # Forward/back substitution responses, coefficients, residuals;
                # original order also materializes concatenated/reordered copies.
                copies = 6 if options.get("return_format", "blocks") == "original" else 4
                eig_workspace += 16 * copies * sum(sizes) * selected
        free_bytes, _total = torch.cuda.mem_get_info(rgf_dev)
        budget = int(0.5 * free_bytes)
        b = max(1, min(n_grid, max(0, budget - eig_workspace) // per_e))
    except Exception as exc:
        log.error(
            "Failed to determine CUDA auto e_batch_size on %s; "
            "falling back to e_batch_size=1 (%s: %s)",
            rgf_dev, type(exc).__name__, exc, exc_info=True,
        )
        return 1
    log.info(
        f"auto e_batch_size={b} (free={free_bytes/2**30:.2f} GiB, "
        f"per_E~={per_e/2**20:.1f} MiB, "
        f"eig_workspace~={eig_workspace/2**20:.1f} MiB, K={K}, n_max={n_max})"
    )
    return b


@torch.no_grad()
def _rgf_chunk_size(runner, n_grid, k, Vbias=None):
    runner.deviceprop.prepare_hamiltonian(k, runner.block_tridiagonal, Vbias)
    cap = _auto_chunk_size(runner, n_grid)
    chunk = runner.e_batch_size if runner.e_batch_size is not None else cap
    if runner.e_batch_size is not None and runner.rgf_device.type == "cuda" and chunk > cap:
        log.warning("user e_batch_size=%s exceeds CUDA auto-cap=%s on %s",
                    chunk, cap, runner.rgf_device)
    log.info("Using e_batch_size=%s on %s for %s energy points",
             chunk, runner.rgf_device, n_grid)
    return chunk


def _cpu_payload(value):
    """Keep CUDA tensors and storage handles out of process messages."""
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_payload(item) for item in value)
    return value


def _send_packet(connection, packet):
    # Explicit bytes use ordinary pickle, avoiding torch's multiprocessing
    # reducers and their shared-storage/file-descriptor lifetime requirements.
    connection.send_bytes(pickle.dumps(packet, protocol=pickle.HIGHEST_PROTOCOL))


def _recv_packet(connection):
    return pickle.loads(connection.recv_bytes())


def _rgf_worker_main(connection, device, spec):
    """Spawn entry point: build a solver and serve synchronized RGF tasks."""
    solver = None
    try:
        torch.set_default_dtype(spec["default_dtype"])
        device = torch.device(device)
        worker_threads = (1 if device.type == "cuda" else
                          spec.get("options", {}).get("worker_threads", 1))
        torch.set_num_threads(worker_threads)
        torch.set_num_interop_threads(1)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        from dpnegf.negf.rgf_solver import RGFSolver

        solver = RGFSolver.from_cache(spec, device)
        prop = solver.deviceprop
        with threadpool_limits(limits=worker_threads):
            _send_packet(connection, ("ready", {
                "pid": os.getpid(), "device": str(device),
                "torch_intra_threads": torch.get_num_threads(),
                "torch_inter_threads": torch.get_num_interop_threads(),
                "threadpools": threadpool_info(),
            }))
            while True:
                command, args = _recv_packet(connection)
                if command == "close":
                    break
                if command == "shared_hs":
                    descriptor = args
                    prop.install_shared_hs(descriptor)
                    _send_packet(connection, (
                        "shared_hs", descriptor["version"], descriptor["name"]))
                elif command == "chunk_size":
                    per_device, k, bias = args
                    if device.type == "cpu" and getattr(prop, "_shared_hs_worker", False):
                        batch_size = 1
                    else:
                        batch_size = _rgf_chunk_size(
                            solver, per_device, k, bias)
                    _send_packet(connection, ("chunk_size", batch_size))
                elif command == "solve":
                    index, energies, k, ik, bias, offset, shared_version = args
                    outputs = solver.compute_rgf_chunk(
                        energies, k, ik, bias, offset,
                        shared_hs_version=shared_version,
                    )
                    _send_packet(connection, ("result", index, _cpu_payload(outputs)))
                else:
                    raise ValueError(f"Unknown RGF worker command: {command}")
    except (EOFError, BrokenPipeError):
        pass  # Parent closed or terminated the pool.
    except BaseException as exc:
        try:
            _send_packet(connection, ("error", type(exc).__name__, str(exc), traceback.format_exc()))
        except (OSError, EOFError):
            pass
    finally:
        if solver is not None:
            try:
                solver.deviceprop.detach_shared_hs()
            except (BufferError, OSError):
                pass
        connection.close()


class RGFWorkerError(RuntimeError):
    """A spawned solver failed or exited before returning its result."""


class RGFProcessPool:
    """Persistent spawn workers with bounded, timeout-aware energy scheduling."""

    def __init__(self, devices, spec, *, startup_timeout=600.0,
                 sync_timeout=600.0, task_timeout=3000.0,
                 max_pending_results=None, share_cpu_hs=False,
                 cpu_threads=None):
        if not devices:
            raise ValueError("An RGF process pool requires at least one device.")
        self.devices = [torch.device(device) for device in devices]
        self.startup_timeout = _timeout_value(startup_timeout, "startup_timeout")
        self.sync_timeout = _timeout_value(sync_timeout, "sync_timeout")
        self.task_timeout = _timeout_value(task_timeout, "task_timeout")
        if max_pending_results is None:
            self.max_pending_results = 2 * len(self.devices)
        else:
            self.max_pending_results = _positive_integer(
                max_pending_results, "max_pending_results")
        self.share_cpu_hs = bool(share_cpu_hs)
        self.cpu_threads = cpu_threads
        self.processes = []
        self.connections = []
        self.receiver_threads = []
        self.process_watchers = []
        self.messages = queue.Queue()
        self._messages_lock = threading.Lock()
        self.pending = {}
        self.worker_pids = []
        self.worker_info = []
        self.closed = False
        self._deviceprop = None
        self.block_tridiagonal = spec.get("options", {}).get("block_tridiagonal", True)
        self._shared_hs = None
        self._owner_regions = []
        context = mp.get_context("spawn")
        try:
            for device in self.devices:
                parent, child = context.Pipe()
                process = context.Process(target=_rgf_worker_main,
                                          args=(child, str(device), spec))
                started = time.monotonic()
                self._set_pending(len(self.processes), "startup", started,
                                  self.startup_timeout, {})
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
                self.connections.append(parent)
                self.processes.append(process)
                receiver = threading.Thread(
                    target=self._receiver_main, args=(len(self.connections) - 1, parent),
                    name=f"rgf-recv-{process.pid}", daemon=True,
                )
                receiver.start()
                self.receiver_threads.append(receiver)
                watcher = threading.Thread(
                    target=self._watch_process_exit,
                    args=(len(self.processes) - 1, process),
                    name=f"rgf-exit-{process.pid}", daemon=True,
                )
                watcher.start()
                self.process_watchers.append(watcher)
            ready = {}
            while len(ready) < len(self.processes):
                slot, packet, _received_at = self._receive()
                if packet[0] != "ready" or slot in ready:
                    raise RGFWorkerError(f"Unexpected RGF startup reply: {packet[0]}")
                self._clear_pending(slot)
                info = packet[1]
                if isinstance(info, dict):
                    ready[slot] = info["pid"]
                    self.worker_info.append((slot, info))
                else:
                    ready[slot] = info
            self.worker_pids = [ready[slot] for slot in range(len(self.processes))]
            self.worker_info = [info for _slot, info in sorted(self.worker_info)]
            for info in self.worker_info:
                log.info(
                    "RGF worker pid=%s device=%s torch_threads=%s/%s",
                    info.get("pid"), info.get("device"),
                    info.get("torch_intra_threads"), info.get("torch_inter_threads"),
                )
                log.debug(
                    "RGF worker pid=%s threadpools=%s",
                    info.get("pid"), info.get("threadpools"),
                )
        except BaseException:
            try:
                self.terminate()
            except RGFWorkerError:
                log.exception("RGF pool cleanup failed while handling startup error")
            raise

    @classmethod
    def from_runner(cls, runner, share_cpu_hs=True):
        """Create workers from cache paths; CPU workers share prepared H/S blocks."""
        from dpnegf.negf.rgf_solver import rgf_solver_options
        prop = runner.deviceprop
        ham = prop.hamiltonian
        spec = _cpu_payload({
            "default_dtype": torch.get_default_dtype(),
            "options": rgf_solver_options(runner),
            "hamiltonian": {
                "cache_path": ham.saved_HS_path or ham.results_path,
                "device_norbs": ham.device_norbs,
                "input_orbital_mapping": getattr(ham, "_input_orbital_mapping", None),
            },
            "device": {name: getattr(prop, name) for name in (
                "results_path", "e_T", "chemiPot", "E_ref")},
            "leads": {
                name: {key: getattr(getattr(prop, name), key) for key in (
                    "voltage", "e_T", "efermi", "E_ref")}
                for name in ("lead_L", "lead_R")
            },
        })
        cpu_workers = getattr(runner, "cpu_workers", 1)
        threads = getattr(runner, "cpu_threads", None)
        worker_threads = threads if threads is not None else (1 if cpu_workers > 1 else None)
        if worker_threads is not None:
            spec["options"]["worker_threads"] = worker_threads
        pool_options = getattr(runner, "rgf_pool_options", {})
        pool = cls(
            runner.rgf_devices, spec,
            startup_timeout=pool_options.get("startup_timeout", 600.0),
            sync_timeout=pool_options.get("sync_timeout", 600.0),
            task_timeout=pool_options.get("task_timeout", 3000.0),
            max_pending_results=pool_options.get("max_pending_results"),
            share_cpu_hs=share_cpu_hs and cpu_workers > 1,
            cpu_threads=worker_threads,
        )
        pool._deviceprop = prop
        return pool

    def _receiver_main(self, slot, connection):
        while True:
            try:
                packet = _recv_packet(connection)
                event = (slot, packet, None, time.monotonic())
            except BaseException as exc:
                event = (slot, None, exc, time.monotonic())
            self._queue_message(event)
            if event[1] is None:
                return
            del packet, event

    def _queue_message(self, event):
        with self._messages_lock:
            if not self.closed:
                self.messages.put(event)

    def _watch_process_exit(self, slot, process):
        exitcode = None
        if all(hasattr(os, name) for name in ("waitid", "P_PID", "WEXITED", "WNOHANG", "WNOWAIT")):
            flags = os.WEXITED | os.WNOHANG | os.WNOWAIT
            while True:
                try:
                    status = os.waitid(os.P_PID, process.pid, flags)
                except ChildProcessError:
                    break
                if status is not None and status.si_pid:
                    exitcode = (
                        status.si_status
                        if status.si_code == getattr(os, "CLD_EXITED", 1)
                        else -status.si_status
                    )
                    break
                time.sleep(0.05)
        else:
            try:
                from multiprocessing.connection import wait as wait_connections
                wait_connections([process.sentinel])
            except (OSError, ValueError):
                return
        self._queue_message((
            slot, ("process_exit", exitcode), None, time.monotonic()))

    def _set_pending(self, slot, phase, started, timeout, metadata):
        deadline = None if timeout is None else started + timeout
        self.pending[slot] = (phase, started, deadline, metadata)

    def _clear_pending(self, slot):
        self.pending.pop(slot, None)

    def _timeout_error(self, slot, phase, started, deadline, metadata, now):
        process = self.processes[slot]
        elapsed = now - started
        limit = "disabled" if deadline is None else f"{deadline - started:.1f}s"
        detail = ""
        if phase == "task":
            detail = (
                f"; task={metadata['index']}, k={metadata['k']}, ik={metadata['ik']}, "
                f"energy=[{metadata['energy_start']:.8g}, {metadata['energy_end']:.8g}], "
                f"offset={metadata['offset']}"
            )
        elif phase == "sync":
            detail = f"; command={metadata.get('kind', 'shared H/S')}"
        message = (
            f"RGF {phase} timeout: pid={process.pid}, device={self.devices[slot]}, "
            f"elapsed={elapsed:.1f}s, limit={limit}{detail}."
        )
        if phase == "task":
            message += (
                "\nThe calculation system may be too large for this timeout. "
                "Increase rgf_options.task_timeout or set it to None to disable "
                "the deadline. Slow I/O or a stalled worker may also cause this error."
            )
        return RGFWorkerError(message)

    def _worker_exit_error(self, slot, exitcode):
        process = self.processes[slot]
        pending = self.pending.get(slot)
        detail = ""
        if pending is not None:
            phase, _started, _deadline, metadata = pending
            detail = f" while waiting for {phase}"
            if phase == "task":
                detail += (
                    f" task={metadata['index']}, k={metadata['k']}, ik={metadata['ik']}, "
                    f"energy=[{metadata['energy_start']:.8g}, "
                    f"{metadata['energy_end']:.8g}], offset={metadata['offset']}"
                )
            elif phase == "sync":
                detail += f" command={metadata.get('kind', 'sync')}"
        return RGFWorkerError(
            f"RGF worker exited: pid={process.pid}, device={self.devices[slot]}, "
            f"exit-code={exitcode}{detail}."
        )

    def _receive(self):
        while True:
            deadlines = [record[2] for record in self.pending.values()
                         if record[2] is not None]
            timeout = None if not deadlines else max(0.0, min(deadlines) - time.monotonic())
            try:
                slot, packet, error, received_at = self.messages.get(timeout=timeout)
            except queue.Empty:
                now = time.monotonic()
                expired = [
                    (slot, record) for slot, record in self.pending.items()
                    if record[2] is not None and now >= record[2]
                ]
                if expired:
                    slot, record = min(expired, key=lambda item: item[1][2])
                    raise self._timeout_error(slot, *record[:3], record[3], now)
                continue

            if error is not None:
                process = self.processes[slot]
                process.join(timeout=0.2)
                if process.exitcode is not None:
                    raise self._worker_exit_error(slot, process.exitcode) from error
                raise RGFWorkerError(
                    f"RGF worker on {self.devices[slot]} disconnected: {error}") from error
            if packet[0] == "process_exit":
                process = self.processes[slot]
                process.join(timeout=0)
                exitcode = process.exitcode if process.exitcode is not None else packet[1]
                raise self._worker_exit_error(slot, exitcode)
            pending = self.pending.get(slot)
            if pending is not None and pending[2] is not None and received_at > pending[2]:
                raise self._timeout_error(slot, *pending[:3], pending[3], received_at)
            if packet[0] == "error":
                _, kind, message, trace = packet
                raise RGFWorkerError(
                    f"RGF worker on {self.devices[slot]} failed ({kind}): {message}\n{trace}")
            return slot, packet, received_at

    def _sync_shared_hs(self, k, bias):
        from dpnegf.utils.rgf_shared import SharedHS, contexts_equal, make_hs_context

        prop = self._deviceprop
        prop.prepare_hamiltonian(k, self.block_tridiagonal, bias)
        context = make_hs_context(prop.kpoint, prop.V, prop.block_tridiagonal)
        if self._shared_hs is not None and contexts_equal(self._shared_hs.context, context):
            return

        previous = self._shared_hs
        version = 1 if previous is None else previous.version + 1
        parts = tuple(getattr(prop, name) for name in ("hd", "sd", "hl", "hu", "sl", "su"))
        shared = SharedHS.publish(parts, context, version)
        self._owner_regions.append(shared)
        shared.descriptor["orbital_mapping"] = copy.deepcopy(
            getattr(prop, "_orbital_mapping_snapshot", None))
        prop.install_shared_hs_owner(shared)
        acknowledgements = set()
        for slot, connection in enumerate(self.connections):
            started = time.monotonic()
            self._set_pending(slot, "sync", started, self.sync_timeout,
                              {"kind": f"shared H/S version {version}"})
            _send_packet(connection, ("shared_hs", shared.descriptor))
        while len(acknowledgements) < len(self.connections):
            slot, packet, _received_at = self._receive()
            if (packet[0] != "shared_hs" or packet[1] != version
                    or packet[2] != shared.descriptor["name"] or slot in acknowledgements):
                raise RGFWorkerError(f"Unexpected shared H/S acknowledgement: {packet[0]}")
            self._clear_pending(slot)
            acknowledgements.add(slot)
        self._shared_hs = shared
        if previous is not None:
            previous.close()
            self._owner_regions.remove(previous)

    @staticmethod
    def _merge_outputs(out, k, outputs):
        output_key = str(k)
        worker_key = str(_cpu_payload(k))
        for name, by_k in outputs.items():
            out.setdefault(name, {}).setdefault(output_key, []).extend(by_k[worker_key])

    def run_energy_loop(self, grid, k, ik, bias, out):
        if self.closed:
            raise RuntimeError("RGF process pool is closed")
        try:
            if len(grid) == 0:
                return
            if self.share_cpu_hs:
                self._sync_shared_hs(k, bias)

            per_device = max(1, len(grid) // len(self.processes))
            for connection in self.connections:
                slot = self.connections.index(connection)
                started = time.monotonic()
                self._set_pending(slot, "sync", started, self.sync_timeout,
                                  {"kind": "batch-size"})
                _send_packet(connection, ("chunk_size", (
                    per_device if not self.share_cpu_hs else 1,
                    None if self.share_cpu_hs else _cpu_payload(k),
                    None if self.share_cpu_hs else _cpu_payload(bias))))
            caps = {}
            while len(caps) < len(self.processes):
                slot, packet, _received_at = self._receive()
                if packet[0] != "chunk_size" or slot in caps:
                    raise RGFWorkerError(f"Unexpected RGF batch-size reply: {packet[0]}")
                self._clear_pending(slot)
                caps[slot] = packet[1]
            chunk = min(caps.values())
            log.info("RGF spawn workers %s on %s; e_batch_size=%s",
                     self.worker_pids, [str(device) for device in self.devices], chunk)
            grid_cpu = grid.detach().to("cpu")
            total_tasks = (len(grid_cpu) + chunk - 1) // chunk
            next_assign = 0
            next_index = 0
            active, completed = {}, {}
            scan_outputs = {}

            def submit_available():
                nonlocal next_assign
                for slot in range(len(self.processes)):
                    if (slot in active or next_assign >= total_tasks
                            or len(active) + len(completed) >= self.max_pending_results):
                        continue
                    index = next_assign
                    offset = index * chunk
                    energies = grid_cpu[offset:min(offset + chunk, len(grid_cpu))].clone()
                    metadata = {
                        "index": index,
                        "k": np.asarray(_cpu_payload(k)).tolist(), "ik": ik,
                        "energy_start": float(energies[0]),
                        "energy_end": float(energies[-1]), "offset": offset,
                    }
                    started = time.monotonic()
                    self._set_pending(slot, "task", started, self.task_timeout, metadata)
                    shared_version = self._shared_hs.version if self.share_cpu_hs else None
                    _send_packet(self.connections[slot], ("solve", (
                        index, energies, _cpu_payload(k), ik,
                        None if self.share_cpu_hs else _cpu_payload(bias), offset,
                        shared_version)))
                    active[slot] = index
                    next_assign += 1

            submit_available()
            while active or next_assign < total_tasks:
                slot, packet, _received_at = self._receive()
                if packet[0] != "result" or active.get(slot) != packet[1]:
                    raise RGFWorkerError(f"Unexpected RGF result: {packet[0]}")
                self._clear_pending(slot)
                del active[slot]
                completed[packet[1]] = (slot, packet[2])
                submit_available()
                while next_index in completed:
                    _slot, outputs = completed.pop(next_index)
                    self._merge_outputs(scan_outputs, k, outputs)
                    next_index += 1
                submit_available()
            self._merge_outputs(out, k, scan_outputs)
        except BaseException:
            try:
                self.terminate()
            except RGFWorkerError:
                log.exception("RGF pool cleanup failed while handling worker error")
            raise

    def _release_shared_hs(self):
        prop = self._deviceprop
        if prop is not None and hasattr(prop, "release_shared_hs_owner"):
            prop.release_shared_hs_owner()
        for region in list(self._owner_regions):
            try:
                region.close()
            except (BufferError, FileNotFoundError, OSError):
                pass
        self._owner_regions.clear()
        self._shared_hs = None

    def terminate(self):
        """Stop all workers, including peers blocked in a solver or IPC send."""
        with self._messages_lock:
            self.closed = True
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=2)
        for process in self.processes:
            if process.is_alive():
                process.kill()
        for process in self.processes:
            process.join(timeout=2)
        for connection in self.connections:
            connection.close()
        threads = self.receiver_threads + self.process_watchers
        for thread in threads:
            thread.join(timeout=1)
        alive_threads = [thread for thread in threads if thread.is_alive()]
        with self._messages_lock:
            while True:
                try:
                    self.messages.get_nowait()
                except queue.Empty:
                    break
        self.pending.clear()
        self._release_shared_hs()
        if alive_threads:
            details = ", ".join(
                f"{thread.name}(ident={thread.ident})" for thread in alive_threads)
            raise RGFWorkerError(
                f"RGF pool cleanup left threads alive: {details}"
            )

    def close(self):
        if self.closed:
            return
        try:
            for connection in self.connections:
                try:
                    _send_packet(connection, ("close", ()))
                except (OSError, EOFError):
                    pass
            for process in self.processes:
                process.join(timeout=2)
        finally:
            self.terminate()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()
        else:
            try:
                self.terminate()
            except RGFWorkerError:
                log.exception("RGF pool cleanup failed while handling calculation error")


class MPIMultiLevel:
    """
    Multilevel MPICommunicator. Currently, two level partition is supported.
    The global ranks are divided into subgroups, with sub comminicator.
    The global ranks own a master comm that commute between subgroups by grouping the submaster ranks.
    The communication method are wrapped and performed in different level.

    There are three level of comm here:
        0. The submaster comm
        1. The subgroup comm
        2. The global comm

    Attributes
    ----------
    __comm: 'mpi4py.MPI.Intracomm' class
        default global mpi communicator
    __rank: int
        id of this process in mpi communicator
    __size: int
        number of processes in mpi communicator
    __subcomm:  'mpi4py.MPI.Intracomm' class
        sub communicator of the subgroup
    __subrank:  int
        the subrank of the process in its subgroup
    __subsize: int
        the size of the subgroup this rank belongs to
    __color: int
        the subgroup number
    """
    def __init__(self, enable_mpi: bool = False,
                 verbose: bool = False) -> None:
        """
        :param enable_mpi: whether to enable parallelization using MPI
        :param verbose: whether to report parallelization details
        :return: None
        """
        # Initialize MPI variables
        if enable_mpi:
            if MPI is not None:
                self.__comm = MPI.COMM_WORLD
                self.__rank = self.__comm.Get_rank()
                self.__size = self.__comm.Get_size()
                self.__subcomm = self.__comm
                self.__subrank = 0
                self.__n_group = 1
                self.__color = 0
                self.__subsize = self.__size
            else:
                raise ImportError("MPI4PY cannot be imported")
        else:
            self.__comm = None
            self.__rank = 0
            self.__size = 1
            self.__color = 0
            self.__n_group = 1
            self.__subcomm = self.__comm
            self.__subrank = 0
            self.__subsize = self.__size
        
        self.__color_all = [0] * self.__size

        # Print simulation details
        if verbose:
            spaces = " " * 2
            self.print("\nParallelization details:")
            if self.mpi_enabled:
                self.print(f"{spaces}{'MPI processes':16s} : {self.__size:<6d}")
            else:
                self.print(f"{spaces}{'MPI disabled':16s}")
            for env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                try:
                    env_value = os.environ[env_name]
                except KeyError:
                    env_value = "n/a"
                self.print(f"{spaces}{env_name:16s} : {env_value:<6s}")
            self.print()

    @staticmethod
    def __get_array_order(array: np.ndarray) -> str:
        """
        Get data order of array.

        NOTE: Order of memory layout is particularly important when using
        MPI. If data_local is returned by FORTRAN subroutines, it should
        in column-major order. Otherwise, it will be in row-major order.
        If mistaken, no errors will be raised, but the results will be weired.

        :param array: incoming numpy array
        :return: whether the array is in C or Fortran order
        :raise ValueError: if array is neither C nor FORTRAN contiguous
        """
        if array.flags.c_contiguous:
            order = "C"
        elif array.flags.f_contiguous:
            order = "F"
        else:
            raise ValueError("Array is neither C nor FORTRAN contiguous")
        return order
    
    def partition(self, n_group):

        if n_group == 1:
            self.__master_comm = None
            return True
        
        color_all = []
        group_count = []
        for i in range(n_group-1):
            group_count += [self.__size // n_group]
            color_all += [i] * group_count[-1]
            
        group_count += [self.__size % n_group]
        color_all += [n_group-1] * group_count[-1]

        if float(self.__size % n_group) / float(self.__size // n_group) < 0.8:
            raise RuntimeWarning("Group partition is ", group_count, " , bad allocation!")

        self.__color_all = color_all
        color = self.__color_all[self.__rank]

        # create sub comm:
        self.__subcomm = self.__comm.split(color=color, key=self.__rank)
        self.__color = color
        # Optional: check if the current process is in the new_comm
        if self.__subcomm != MPI.COMM_NULL:
            new_rank = self.__subcomm.Get_rank()
            new_size = self.__subcomm.Get_size()
            print(f"Global rank {self.__rank} is in the new_comm with local rank {new_rank} and size {new_size}")
            self.__subrank = new_rank
            self.__subsize = new_size
        else:
            print(f"Global rank {self.__rank} is not in the new_comm")

        # create the master comm for subgroup masters
        if self.__subrank == 0:
            self.__master_comm = self.__comm.split(color=0, key=self.__rank)
            self.__master_rank = self.__master_comm.Get_rank()
        self.__n_group = n_group

        return True

    def dist_list(self, raw_list: List[Any],
                  algorithm: str = "range", level=2) -> List[Any]:
        """
        Distribute a list over processes.

        :param raw_list: raw list to distribute
        :param algorithm: distribution algorithm, should be either "remainder"
            or "range"
        :return: sublist assigned to this process
        """
        if level == 0:
            if self.is_submaster:
                return split_list(raw_list, self.__n_group, algorithm)[self.__master_rank]
            else:
                return None
        elif level == 1:
            return split_list(raw_list, self.__subsize, algorithm)[self.__subrank]
        elif level == 2:
            return split_list(raw_list, self.__size, algorithm)[self.__rank]
        else:
            raise RuntimeError("The level only support 0 / 1 / 2 !")

    def dist_range(self, n_max: int, level=2) -> range:
        """
        Distribute range(n_max) over processes.

        :param n_max: upper bound of the range
        :return: subrange assigned to this process
        """
        if level == 0:
            if self.is_submaster:
                return split_range(n_max, num_group=self.__n_group)[self.__master_rank]
            else:
                return None
        elif level == 1:
            return split_range(n_max, num_group=self.__subsize)[self.__subrank]
        elif level == 2:
            return split_range(n_max, num_group=self.__size)[self.__rank]
        else:
            raise RuntimeError("The level only support 0 / 1 / 2 !")

    def dist_bound(self, n_max: int, level=2) -> Tuple[int, int]:
        """
        Same as dist_range, but returns the lower and upper bounds.
        Both of the bounds are close, i.e. [i_min, i_max].

        :param n_max: upper bound of range
        :return: lower and upper bounds of subrange assigned to this process
        """
        i_index = self.dist_range(n_max, level=level)
        if self.is_submaster:
            i_min, i_max = min(i_index), max(i_index)
        else:
            i_min, i_max = None, None

        return i_min, i_max

    def reduce(self, data_local: np.ndarray, level=2) -> np.ndarray:
        """
        Reduce local data to master process.

        :param data_local: local results on each process
        :return: summed data from data_local
        """
        if self.mpi_enabled:
            if level == 2:
                if self.is_master:
                    data = np.zeros(data_local.shape, dtype=data_local.dtype,
                                    order=self.__get_array_order(data_local))
                else:
                    data = None
                self.__comm.Reduce(data_local, data, op=MPI.SUM, root=0)
            elif level == 1:
                if self.is_submaster:
                    data = np.zeros(data_local.shape, dtype=data_local.dtype,
                                    order=self.__get_array_order(data_local))
                else:
                    data = None
                self.__subcomm.Reduce(data_local, data, op=MPI.SUM, root=0)
            elif level == 0:
                if self.is_submaster:
                    if self.is_master:
                        data = np.zeros(data_local.shape, dtype=data_local.dtype,
                                        order=self.__get_array_order(data_local))
                    else:
                        data = None
                    
                    self.__master_comm.Reduce(data_local, data, op=MPI.SUM, root=0)
            else:
                raise RuntimeError("The level only support 0 / 1 / 2 !")

        else:
            data = data_local

        return data

    def all_reduce(self, data_local: np.ndarray, level=2) -> np.ndarray:
        """
        Reduce local data and broadcast to all processes.

        :param data_local: local results on each process
        :return: summed data from data_local
        """
        if self.mpi_enabled:
            if level == 2:
                data = np.zeros(data_local.shape, dtype=data_local.dtype,
                                order=self.__get_array_order(data_local))
                self.__comm.Allreduce(data_local, data, op=MPI.SUM)
            elif level == 1:
                data = np.zeros(data_local.shape, dtype=data_local.dtype,
                                order=self.__get_array_order(data_local))
                self.__subcomm.Allreduce(data_local, data, op=MPI.SUM)
            elif level == 0:
                if self.is_submaster:
                    data = np.zeros(data_local.shape, dtype=data_local.dtype,
                                    order=self.__get_array_order(data_local))
                    self.__master_comm.Allreduce(data_local, data, op=MPI.SUM)
            else:
                raise RuntimeError("The level only support 0 / 1 / 2 !")
        else:
            data = data_local
        return data

    def average(self, data_local: np.ndarray, level=2) -> np.ndarray:
        """
        Average results over random samples and store results to master process.

        :param data_local: local results on each process
        :return: averaged data from data_local
        """
        if self.mpi_enabled:
            if level == 2:
                if self.is_master:
                    data = np.zeros(data_local.shape, dtype=data_local.dtype,
                                    order=self.__get_array_order(data_local))
                else:
                    data = None
                self.__comm.Reduce(data_local, data, op=MPI.SUM, root=0)
                if self.is_master:
                    data /= self.__size
            elif level == 1:
                if self.is_submaster:
                    data = np.zeros(data_local.shape, dtype=data_local.dtype,
                                    order=self.__get_array_order(data_local))
                else:
                    data = None
                self.__subcomm.Reduce(data_local, data, op=MPI.SUM, root=0)
                if self.is_submaster:
                    data /= self.__subsize
            elif level == 0:
                if self.is_submaster:
                    if self.is_master:
                        data = np.zeros(data_local.shape, dtype=data_local.dtype,
                                        order=self.__get_array_order(data_local))
                    else:
                        data = None
                    self.__subcomm.Reduce(data_local, data, op=MPI.SUM, root=0)
                    if self.is_master:
                        data /= self.__n_group
            else:
                raise RuntimeError("The level only support 0 / 1 / 2 !")

        else:
            data = data_local
        return data

    def all_average(self, data_local: np.ndarray, level=2) -> np.ndarray:
        """
        Average results over random samples broadcast to all process.

        :param data_local: local results on each process
        :return: averaged data from data_local
        """
        if self.mpi_enabled:
            if level == 2:
                data = np.zeros(data_local.shape, dtype=data_local.dtype,
                                order=self.__get_array_order(data_local))
                self.__comm.Allreduce(data_local, data, op=MPI.SUM)
                data /= self.__size
            elif level == 1:
                data = np.zeros(data_local.shape, dtype=data_local.dtype,
                                order=self.__get_array_order(data_local))
                self.__subcomm.Allreduce(data_local, data, op=MPI.SUM)
                data /= self.__subsize
            elif level == 0:
                if self.is_submaster:
                    data = np.zeros(data_local.shape, dtype=data_local.dtype,
                                order=self.__get_array_order(data_local))
                    self.__master_comm.Allreduce(data_local, data, op=MPI.SUM)
                    data /= self.__n_group
            else:
                raise RuntimeError("The level only support 0 / 1 / 2 !")
        else:
            data = data_local
        return data

    def bcast(self, data_local: np.ndarray, level=2) -> None:
        """
        Broadcast data from master to other processes.

        :param data_local: local results on each process
        :return: None
        """
        if self.mpi_enabled:
            if level == 2:
                self.__comm.Bcast(data_local, root=0)
            elif level == 1:
                self.__subcomm.Bcast(data_local, root=0)
            elif level == 0:
                if self.is_submaster:
                    self.__master_comm.Bcast(data_local, root=0)
            else:
                raise RuntimeError("The level only support 0 / 1 / 2 !")
        

    def barrier(self, level=2) -> None:
        """Wrapper for self.comm.Barrier."""
        if self.mpi_enabled:
            if level == 2:
                self.__comm.Barrier()
            elif level == 1:
                self.__subcomm.Barrier()
            elif level == 0:
                self.__master_comm.Barrier()
            else:
                raise RuntimeError("The level only support 0 / 1 / 2 !")


    def gather(self, data_local: np.ndarray, level=2) -> np.ndarray:

        if self.mpi_enabled:
            shape = data_local.shape
            if level == 2:
                shape[0] = shape[0] * self.__size
                if self.is_master:
                    data = np.zeros(shape, dtype=data_local.dtype, order=self.__get_array_order(data_local))
                else:
                    data = None
                self.__comm.Gather(data_local, data, root=0)
            elif level == 1:
                shape[0] = shape[0] * self.__subsize
                if self.is_master:
                    data = np.zeros(shape, dtype=data_local.dtype, order=self.__get_array_order(data_local))
                else:
                    data = None
                self.__subcomm.Gather(data_local, data, root=0)
            elif level == 2:
                if self.is_submaster:
                    shape[0] = shape[0] * self.__n_group
                    if self.is_master:
                        data = np.zeros(shape, dtype=data_local.dtype, order=self.__get_array_order(data_local))
                    else:
                        data = None
                    self.__master_comm.Gather(data_local, data, root=0)
        else:
            data = data_local

        return data

    def print(self, text: str = "") -> None:
        """
        Print text on master process.

        NOTE: flush=True is essential for some MPI implementations,
        e.g. MPICH3.

        :param text: text to print
        :return: None
        """
        if self.is_master:
            print(text, flush=True)

    def log(self, event: str = "", fmt: str = "%x %X") -> None:
        """
        Log the date and time of event.

        :param event: notice of the event
        :param fmt: date and time format
        :return: None.
        """
        if self.is_master:
            date_time = get_datetime(fmt=fmt)
            print(f"{event} at {date_time}", flush=True)

    @property
    def mpi_enabled(self) -> bool:
        """Determine whether MPI is enabled."""
        return self.__comm is not None

    @property
    def is_master(self) -> bool:
        """Determine whether this is the master process."""
        return self.__rank == 0
    
    @property
    def is_submaster(self) -> bool:
        return self.__subrank == 0

    @property
    def rank(self) -> int:
        """
        Interface for the '__rank' attribute.

        :return: rank of this MPI process
        """
        return self.__rank

    @property
    def group(self, rank) -> int:
        return self.__group_vec[rank]


    @property
    def size(self) -> int:
        """
        Interface for the '__size' attribute.

        :return: number of MPI processes
        """
        return self.__size


def split_list(raw_list: List[Any],
               num_group: int,
               algorithm: str = "remainder") -> List[List[Any]]:
    """
    Split given list into different groups.

    Two algorithms are implemented: by the remainder of the index of each
    element divided by the number of group, or the range of index. For example,
    if we are to split the list of [0, 1, 2, 3] into two groups, by remainder
    we will get [[0, 2], [1, 3]] while by range we will get [[0, 1], [2, 3]].

    :param raw_list: incoming list to split
    :param num_group: number of groups
    :param algorithm: algorithm for grouping elements, should be either
        "remainder" or "range"
    :return: split list from raw_list
    """
    assert num_group in range(1, len(raw_list)+1)
    assert algorithm in ("remainder", "range")
    num_element = len(raw_list)
    if algorithm == "remainder":
        list_split = [[raw_list[i] for i in range(num_element)
                      if i % num_group == k] for k in range(num_group)]
    else:
        # Get the numbers of items for each group
        num_item = [num_element // num_group for _ in range(num_group)]
        for i in range(num_element % num_group):
            num_item[i] += 1
        # Divide the list according to num_item
        list_split = []
        for i in range(num_group):
            j0 = sum(num_item[:i])
            j1 = j0 + num_item[i]
            list_split.append([raw_list[j] for j in range(j0, j1)])
    return list_split


def split_range(n_max: int, num_group: int = 1) -> List[range]:
    """
    Split range(n_max) into different groups.

    Adapted from split_list with algorithm = "range".

    :param n_max: upperbound of range, starting from 0
    :param num_group: number of groups
    :return: list of ranges split from range(n_max)
    """
    # Get the numbers of items for each group
    num_item = [n_max // num_group for _ in range(num_group)]
    for i in range(n_max % num_group):
        num_item[i] += 1
    range_list = []
    for i in range(num_group):
        j0 = sum(num_item[:i])
        j1 = j0 + num_item[i]
        range_list.append(range(j0, j1))
    return range_list


def get_datetime(fmt: str = "%x %X") -> str:
    """
    Return current date and time.

    :param fmt: date and time format
    :return: current date and time
    """
    now = datetime.datetime.now()
    return now.strftime(fmt)
