"""Selective HDF5 readers shared by dense and direct BTD caches."""

import copy
import logging
import os

import h5py
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
    """Read one block, assembling complex values on the requested device."""

    selection = () if index is None else index
    real = torch.from_numpy(np.asarray(group[f"{name}_real"][selection])).to(device=device)
    imag = torch.from_numpy(np.asarray(group[f"{name}_imag"][selection])).to(device=device)
    value = torch.complex(real, imag).to(torch.complex128)

    return value


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

log = logging.getLogger(__name__)


class CachedHamiltonian:
    """Read saved device H/S files and orbital provenance from disk.

    H/S tensors are loaded by get_hs_device(), without constructing a model
    or receiving Hamiltonian tensors from another process.
    """

    def __init__(self, cache_path, device_norbs, input_orbital_mapping=None,
                 device="cpu"):
        self.saved_HS_path = os.fspath(cache_path)
        self.results_path = self.saved_HS_path
        self.torch_device = torch.device(device)
        self._device_norbs = list(device_norbs)
        self._input_orbital_mapping = input_orbital_mapping
        self._loaded_orbital_mapping = None

    @property
    def device_norbs(self):
        return self._device_norbs

    def get_hs_device(self, kpoint=[0,0,0], V=None, block_tridiagonal=False, only_subblocks=False, device=None):
        """ get the device Hamiltonian and overlap matrix at a specific kpoint

        In diagonalization mode, the Hamiltonian and overlap matrix are block tridiagonalized,
        and hd,hu,hl refers to the diagnonal, upper and lower blocks of the Hamiltonian, respectively.
        The same rules apply to sd, su, sl.

        Args:
            kpoints: k-points in the Brillouin zone with three coordinates (kx, ky, kz)
            V: voltage bias
            block_tridiagonal:  a boolean flag that shows whether Hamiltonian has been diagonalized or not
            device: output device; defaults to the initialization device.
        Returns:
            if not diagonalized, return the whole Hamiltonian and Overlap HD-V*SD, SD
            if diagonalized, return the block tridiagonalized Hamiltonian and Overlap component hd, hu, hl,
            sd, su, sl.
        """
        if not only_subblocks:
            self._loaded_orbital_mapping = None
        target_device = None if only_subblocks else (
            self.torch_device if device is None else torch.device(device)
        )
        from dpnegf.utils.orbital_mapping import read_orbital_mapping
        if self.saved_HS_path is None:
            self.saved_HS_path = self.results_path

        HS_device_path_pth = os.path.join(self.saved_HS_path, "HS_device.pth")
        HS_device_path_h5 = os.path.join(self.saved_HS_path, "HS_device.h5")

        if os.path.exists(HS_device_path_h5):
            with h5py.File(HS_device_path_h5, "r") as f:
                if (
                    f.attrs.get("layout_version", 1) == 2
                    and not f.attrs.get("complete", False)
                ):
                    raise ValueError(
                        "Incomplete direct BTD cache; initialization did not "
                        "finish."
                    )
                if only_subblocks:
                    if "subblocks" not in f:
                        log.warning(msg=" 'subblocks' might not be saved in the HS_device.h5 for old version.")
                        log.error(msg="The subblocks are not saved in the HS_device.h5.")
                        raise ValueError
                    return f["subblocks"][()]

                if block_tridiagonal:
                    from dpnegf.utils.hs_cache import read_btd_kpoint
                    parts = read_btd_kpoint(f, kpoint, V, target_device)
                    self._capture_orbital_mapping(read_orbital_mapping(f), parts,
                                                  f["subblocks"][()], HS_device_path_h5)
                    return parts
                from dpnegf.utils.hs_cache import find_kpoint_index
                ik = find_kpoint_index(f["kpoints"][()], kpoint)
                HD_k = torch.from_numpy(f["HD"][ik])
                SD_k = torch.from_numpy(f["SD"][ik])
                mapping = read_orbital_mapping(f)
                declared_blocks = f["subblocks"][()] if "subblocks" in f else [HD_k.shape[-1]]
                source = HS_device_path_h5

        elif os.path.exists(HS_device_path_pth):
            # log.info(msg="The HS_device.pth exists in the saved path {}.".format(self.saved_HS_path))
            HS_device = torch.load(HS_device_path_pth, map_location="cpu", weights_only=False)
            if only_subblocks:
                if "subblocks" not in HS_device:
                    log.warning(msg=" 'subblocks' might not be saved in the HS_device.pth for old version.")
                    log.error(msg="The subblocks are not saved in the HS_device.pth.")
                    raise ValueError
                return HS_device["subblocks"]

            ik = self._find_kpoint_index(HS_device["kpoints"], kpoint)
            mapping = read_orbital_mapping(HS_device)
            source = HS_device_path_pth
            if block_tridiagonal:
                hd_k, sd_k, hl_k, su_k, sl_k, hu_k = (
                    HS_device["hd"][ik], HS_device["sd"][ik],
                    HS_device["hl"][ik], HS_device["su"][ik],
                    HS_device["sl"][ik], HS_device["hu"][ik],
                )
            else:
                HD_k, SD_k = HS_device["HD"][ik], HS_device["SD"][ik]
            declared_blocks = HS_device.get(
                "subblocks", [b.shape[0] for b in hd_k] if block_tridiagonal
                else [HD_k.shape[-1]])
        else:
            raise FileNotFoundError(f"Neither HS_device.pth nor HS_device.h5 found in {self.saved_HS_path}. " )



        if block_tridiagonal:
            self._capture_orbital_mapping(mapping, (hd_k, sd_k, hl_k, su_k, sl_k, hu_k),
                                          declared_blocks, source)
            # move blocks onto torch_device so downstream RGF runs on the chosen device.
            hd_k = [b.to(target_device) for b in hd_k]
            sd_k = [b.to(target_device) for b in sd_k]
            hl_k = [b.to(target_device) for b in hl_k]
            su_k = [b.to(target_device) for b in su_k]
            sl_k = [b.to(target_device) for b in sl_k]
            hu_k = [b.to(target_device) for b in hu_k]
            V = torch.as_tensor(0 if V is None else V, device = target_device)

            if V.shape == torch.Size([]):
                allorb = sum([hd_k[i].shape[0] for i in range(len(hd_k))])
                V = V.repeat(allorb)
            # V = torch.diag(V).cdouble()
            counted = 0
            for i in range(len(hd_k)): # TODO: this part may have probelms when V!=0
                l_slice = slice(counted, counted+hd_k[i].shape[0])
                V_sub = V[l_slice].view(-1,1).cdouble()
                hd_k[i] = hd_k[i] - V_sub * sd_k[i]
                if i<len(hd_k)-1:
                    hu_k[i] = hu_k[i] - V_sub * su_k[i]
                if i > 0:
                    hl_k[i-1] = hl_k[i-1] - V_sub * sl_k[i-1]
                counted += hd_k[i].shape[0]

            return hd_k , sd_k, hl_k , su_k, sl_k, hu_k
        else:
            self._capture_orbital_mapping(mapping, (HD_k, SD_k, [], [], [], []),
                                          declared_blocks, source)
            HD_k = HD_k.to(target_device)
            SD_k = SD_k.to(target_device)
            V = torch.as_tensor(0 if V is None else V, device = target_device)
            return HD_k - V*SD_k, SD_k, [], [], [], []

    def _capture_orbital_mapping(self, metadata, parts, subblocks, source):
        from dpnegf.utils.orbital_mapping import validate_hs_layout
        self._loaded_orbital_mapping = None
        hd, sd, *rest = parts
        if torch.is_tensor(hd) and hd.ndim == 2:
            parts = ([hd], [sd], *rest)
        blocks = validate_hs_layout(parts, subblocks)
        self._loaded_orbital_mapping = (copy.deepcopy(metadata), blocks, os.path.abspath(source))
        self.get_orbital_map()

    def get_orbital_map(self):
        """Validate mapping from the actual loaded H/S cache and return a snapshot."""
        from dpnegf.utils.orbital_mapping import validate_orbital_mapping
        loaded = getattr(self, "_loaded_orbital_mapping", None)
        if loaded is None:
            raise RuntimeError("Load device H/S before requesting its orbital mapping")
        metadata, blocks, source = loaded
        return validate_orbital_mapping(metadata, getattr(self, "_input_orbital_mapping", None),
                                        blocks, source)

    @staticmethod
    def _find_kpoint_index(kpoints, kpoint):
        from dpnegf.utils.hs_cache import find_kpoint_index
        return find_kpoint_index(kpoints, kpoint)
