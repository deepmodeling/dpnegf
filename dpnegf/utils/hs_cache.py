"""Selective HDF5 readers shared by dense and direct BTD caches."""

import numpy as np
import torch


def find_kpoint_index(kpoints, kpoint):
    """Return the cache index matching ``kpoint`` within the legacy tolerance."""

    matches = np.flatnonzero(
        np.abs(np.asarray(kpoints) - np.asarray(kpoint)).sum(axis=1) < 1e-8
    )
    if not len(matches):
        raise AssertionError(
            f"The requested k-point {np.asarray(kpoint)} is not in the H/S cache."
        )

    return int(matches[0])


def read_complex(group, name, device=None, index=None):
    """Read one real/imaginary dataset pair without dtype down-casting."""

    selection = () if index is None else index
    real = torch.from_numpy(np.asarray(group[f"{name}_real"][selection]))
    imag = torch.from_numpy(np.asarray(group[f"{name}_imag"][selection]))
    value = torch.complex(real, imag).to(torch.complex128)

    return value.to(device) if device is not None else value


def read_optional_metadata(dataset):
    """Decode an optional array stored as an array or the string ``None``."""

    value = dataset[()]
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, bytes) and value.decode() == "None":
        return None
    if isinstance(value, str) and value == "None":
        return None
    raise ValueError(f"Unsupported value {value} for key {dataset.name}")


def read_btd_kpoint(cache, kpoint, voltage, device):
    """Read only one k point, retaining the existing RGF return convention."""


    k_index = find_kpoint_index(cache["kpoints"][()], kpoint)
    blocks = cache["subblocks"][()]
    parts = {}

    for name in ("hd", "sd", "hl", "su", "sl", "hu"):

        count = len(blocks) if name.endswith("d") else len(blocks) - 1
        parts[name] = [
            read_complex(
                cache[name],
                f"{name}_k{k_index}_b{block_index}",
                device=device,
            )
            for block_index in range(count)
        ]
    voltage = torch.as_tensor(
        0 if voltage is None else voltage, device=device
    )
    if voltage.ndim == 0:
        voltage = voltage.expand(int(sum(blocks)))
    if voltage.numel() != sum(blocks):
        raise ValueError(
            "Device voltage must be scalar or have one value per orbital."
        )
    start = 0

    for block_index, size in enumerate(blocks):
        local_voltage = voltage[start:start + size].reshape(-1, 1)
        parts["hd"][block_index] -= (
            local_voltage * parts["sd"][block_index]
        )
        if block_index < len(blocks) - 1:
            parts["hu"][block_index] -= (
                local_voltage * parts["su"][block_index]
            )
        if block_index > 0:
            parts["hl"][block_index - 1] -= (
                local_voltage * parts["sl"][block_index - 1]
            )
        start += size

    return tuple(
        parts[name] for name in ("hd", "sd", "hl", "su", "sl", "hu")
    )