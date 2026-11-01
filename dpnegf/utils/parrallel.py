"""Functions and classes for the parallel environment."""

import os
import logging
import multiprocessing as mp
from multiprocessing.connection import wait
import pickle
import traceback
from typing import Tuple, List, Any
import datetime

import numpy as np
import torch

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


def _auto_chunk_size(runner, n_grid):
    """Pick a chunk size from free CUDA memory when the user didn't set
    ``e_batch_size``. Returns the full grid length on CPU / when the
    device geometry isn't probable yet.

    Per-energy peak (post per-slot-release, complex128) approximated as
        bytes_per_E ~= C * K * n_max**2 * 16
    with C bundling the live tensors in the worst backward-sweep slot
    (grd full + grl + gru full + decaying gr_left tail + gU + transients
    + the K-resident H/S diagonal & off-diagonal lists that survive across
    chunks). C=14 with a 0.5x free-memory budget; deliberately conservative
    because without expandable_segments the allocator can't defragment on
    demand, and on real workloads (CNT10/long6) the previous 10x / 0.7
    combination still picked a chunk that OOM'd on a 15.77 GiB V100.
    """
    rgf_dev = runner.rgf_device
    if not (isinstance(rgf_dev, torch.device) and rgf_dev.type == "cuda"):
        return n_grid
    try:
        free_bytes, _total = torch.cuda.mem_get_info(rgf_dev)
        if hasattr(runner.deviceprop, "hd"):
            sizes = [int(b.shape[-1]) for b in runner.deviceprop.hd]
        else:
            # Read geometry without allocating H/S on a GPU before the first batch.
            sizes = runner.deviceprop.hamiltonian.get_hs_device(only_subblocks=True)
            if not runner.block_tridiagonal:
                sizes = [sum(sizes)]
        n_max = max(sizes)
        K = len(sizes)
    except Exception:
        return n_grid
    per_e = 14 * K * (n_max ** 2) * 16
    if per_e <= 0:
        return n_grid
    b = max(1, min(n_grid, int(0.5 * free_bytes) // per_e))
    log.info(
        f"auto e_batch_size={b} (free={free_bytes/2**30:.2f} GiB, "
        f"per_E~={per_e/2**20:.1f} MiB, K={K}, n_max={n_max})"
    )
    return b


def _rgf_chunk_size(runner, n_grid):
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
    """Spawn entry point: build an independent solver, then serve all k points."""
    try:
        from dpnegf.negf.rgf_solver import RGFSolver
        torch.set_default_dtype(spec["default_dtype"])
        torch.set_num_threads(1)
        device = torch.device(device)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        solver = RGFSolver.from_cache(spec, device)
        _send_packet(connection, ("ready", os.getpid()))
        while True:
            command, args = _recv_packet(connection)
            if command == "close":
                break
            if command == "chunk_size":
                _send_packet(connection, ("chunk_size", _rgf_chunk_size(solver, *args)))
            elif command == "solve":
                index, energies, k, ik, bias, offset = args
                outputs = solver.compute_rgf_chunk(energies, k, ik, bias, offset)
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
        connection.close()


class RGFWorkerError(RuntimeError):
    """A spawned solver failed or exited before returning its result."""


class RGFProcessPool:
    """One persistent spawn process per device, with bounded in-flight batches.

    Inputs contain only cache paths and numerical metadata. Replies may arrive
    in any order; failures are observed immediately, even if an earlier batch
    is still running. Results are merged and new work assigned in grid order.
    """

    def __init__(self, devices, spec):
        if not devices:
            raise ValueError("An RGF process pool requires at least one device.")
        self.devices = [torch.device(device) for device in devices]
        self.processes = []
        self.connections = []
        self.worker_pids = []
        self.closed = False
        context = mp.get_context("spawn")
        try:
            for device in self.devices:
                parent, child = context.Pipe()
                process = context.Process(target=_rgf_worker_main,
                                          args=(child, str(device), spec))
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
                self.connections.append(parent)
                self.processes.append(process)
            ready = {}
            while len(ready) < len(self.processes):
                slot, packet = self._receive()
                if packet[0] != "ready" or slot in ready:
                    raise RGFWorkerError(f"Unexpected RGF startup reply: {packet[0]}")
                ready[slot] = packet[1]
            self.worker_pids = [ready[slot] for slot in range(len(self.processes))]
        except BaseException:
            self.terminate()
            raise

    @classmethod
    def from_runner(cls, runner):
        """Pass saved H/S paths and metadata; workers read the files themselves."""
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
        return cls(runner.rgf_devices, spec)

    def _receive(self):
        handles = self.connections + [process.sentinel for process in self.processes]
        readable = wait(handles)
        for slot, connection in enumerate(self.connections):
            if connection in readable:
                try:
                    packet = _recv_packet(connection)
                except (OSError, EOFError) as exc:
                    raise RGFWorkerError(f"RGF worker on {self.devices[slot]} disconnected") from exc
                if packet[0] == "error":
                    _, kind, message, trace = packet
                    raise RGFWorkerError(
                        f"RGF worker on {self.devices[slot]} failed ({kind}): {message}\n{trace}")
                return slot, packet
        for slot, process in enumerate(self.processes):
            if process.sentinel in readable:
                raise RGFWorkerError(
                    f"RGF worker on {self.devices[slot]} exited with code {process.exitcode}")
        raise RGFWorkerError("No RGF worker reply was received")

    def run_energy_loop(self, grid, k, ik, bias, out):
        if self.closed:
            raise RuntimeError("RGF process pool is closed")
        try:
            per_device = max(1, len(grid) // len(self.processes))
            for connection in self.connections:
                _send_packet(connection, ("chunk_size", (per_device,)))
            caps = {}
            while len(caps) < len(self.processes):
                slot, packet = self._receive()
                if packet[0] != "chunk_size" or slot in caps:
                    raise RGFWorkerError(f"Unexpected RGF batch-size reply: {packet[0]}")
                caps[slot] = packet[1]
            chunk = min(caps.values())
            log.info("Multi-GPU RGF spawn workers %s on %s; e_batch_size=%s per GPU",
                     self.worker_pids, [str(device) for device in self.devices], chunk)
            batches = iter(enumerate(torch.split(grid.detach().cpu(), chunk)))
            active, completed = {}, {}

            def submit(slot):
                item = next(batches, None)
                if item is None:
                    return
                index, energies = item
                _send_packet(self.connections[slot], ("solve", (
                    index, energies, _cpu_payload(k), ik, _cpu_payload(bias), index * chunk)))
                active[slot] = index

            for slot in range(len(self.processes)):
                submit(slot)
            next_index = 0
            while active:
                slot, packet = self._receive()
                if packet[0] != "result" or active.get(slot) != packet[1]:
                    raise RGFWorkerError(f"Unexpected RGF result: {packet[0]}")
                completed[packet[1]] = (slot, packet[2])
                while next_index in completed:
                    slot, outputs = completed.pop(next_index)
                    for name, by_k in outputs.items():
                        out.setdefault(name, {}).setdefault(str(k), []).extend(by_k[str(k)])
                    del active[slot]
                    submit(slot)
                    next_index += 1
        except BaseException:
            self.terminate()
            raise

    def terminate(self):
        """Stop all workers, including peers blocked in a solver or IPC send."""
        if self.closed:
            return
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

    def close(self):
        if self.closed:
            return
        try:
            for connection in self.connections:
                _send_packet(connection, ("close", ()))
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
            self.terminate()


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
