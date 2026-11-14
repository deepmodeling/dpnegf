"""Shared-memory transport for the six device H/S block groups used by RGF."""

import gc
import os
import shutil
from multiprocessing import shared_memory

import numpy as np
import torch


HS_NAMES = ("hd", "sd", "hl", "hu", "sl", "su")
_ALIGNMENT = 64


def _aligned(value):
    return (value + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT


def make_hs_context(kpoint, potential, block_tridiagonal):
    """Return an exact, pickleable description of a prepared H/S context."""
    k = torch.as_tensor(kpoint, dtype=torch.float64, device="cpu").reshape(-1)
    if k.numel() != 3 or not torch.isfinite(k).all():
        raise ValueError("RGF k-point must contain exactly three finite coordinates.")
    v = torch.as_tensor(potential, device="cpu").detach().contiguous()
    if not torch.isfinite(v).all():
        raise ValueError("RGF device potential must contain only finite values.")
    arr = v.numpy().copy()
    return {
        "kpoint": k.numpy().copy(),
        "potential": arr,
        "potential_dtype": arr.dtype.str,
        "block_tridiagonal": bool(block_tridiagonal),
    }


def contexts_equal(left, right):
    if left is None or right is None:
        return False
    return (
        left["block_tridiagonal"] == right["block_tridiagonal"]
        and np.array_equal(left["kpoint"], right["kpoint"])
        and left["potential"].shape == right["potential"].shape
        and np.array_equal(left["potential"], right["potential"])
    )


class SharedHS:
    """Owner or attachment for one packed H/S shared-memory region."""

    def __init__(self, shm, descriptor, *, owner, arrays, context=None):
        self.shm = shm
        self.descriptor = descriptor
        self.owner = owner
        self.arrays = arrays
        self.context = context
        self.closed = False

    @classmethod
    def publish(cls, parts, context, version):
        if len(parts) != len(HS_NAMES):
            raise ValueError("Shared H/S publication requires six block groups.")

        offset = 0
        groups = {}
        for name, blocks in zip(HS_NAMES, parts):
            groups[name] = []
            for tensor in blocks:
                source = tensor.detach().to("cpu").contiguous().numpy()
                offset = _aligned(offset)
                groups[name].append((offset, source))
                offset += source.nbytes

        total_size = max(1, offset)
        shm_dir = "/dev/shm"
        if os.path.isdir(shm_dir):
            available = shutil.disk_usage(shm_dir).free
            if total_size > available:
                raise OSError(
                    f"Shared H/S needs {total_size} bytes, but {shm_dir} has "
                    f"only {available} bytes available."
                )

        shm = shared_memory.SharedMemory(create=True, size=total_size)
        arrays = {}
        try:
            for name, blocks in groups.items():
                arrays[name] = []
                for block_offset, source in blocks:
                    target = np.ndarray(
                        source.shape, dtype=source.dtype, buffer=shm.buf,
                        offset=block_offset,
                    )
                    target[...] = source
                    arrays[name].append(target)
            block_descriptors = {
                name: [
                    {"offset": block_offset, "shape": list(source.shape),
                     "dtype": source.dtype.str, "nbytes": source.nbytes}
                    for block_offset, source in blocks
                ]
                for name, blocks in groups.items()
            }
            descriptor = {
                "name": shm.name,
                "size": total_size,
                "version": int(version),
                "blocks": block_descriptors,
                "context": context,
            }
            return cls(shm, descriptor, owner=True, arrays=arrays, context=context)
        except BaseException:
            arrays.clear()
            gc.collect()
            shm.unlink()
            shm.close()
            raise

    @classmethod
    def attach(cls, descriptor):
        shm = shared_memory.SharedMemory(name=descriptor["name"], create=False)
        arrays = {}
        try:
            for name in HS_NAMES:
                arrays[name] = []
                for block in descriptor["blocks"][name]:
                    arr = np.ndarray(
                        tuple(block["shape"]), dtype=np.dtype(block["dtype"]),
                        buffer=shm.buf, offset=block["offset"],
                    )
                    arrays[name].append(arr)
            return cls(shm, descriptor, owner=False, arrays=arrays,
                       context=descriptor["context"])
        except BaseException:
            arrays.clear()
            gc.collect()
            shm.close()
            raise

    @property
    def version(self):
        return self.descriptor["version"]

    def torch_views(self):
        if self.closed:
            raise RuntimeError("Shared H/S mapping is closed.")
        return tuple(
            [torch.from_numpy(block) for block in self.arrays[name]]
            for name in HS_NAMES
        )

    def close(self):
        if self.closed:
            return
        self.arrays.clear()
        gc.collect()
        if self.owner:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass
        self.shm.close()
        self.closed = True


__all__ = ["HS_NAMES", "SharedHS", "contexts_equal", "make_hs_context"]
