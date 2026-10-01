import logging
import os
import re
import uuid

import h5py
import numpy as np
import torch
from dptb.data import AtomicData, AtomicDataDict
from dptb.utils.constants import anglrMId

from dpnegf.negf.split_btd import compute_blocks

log = logging.getLogger(__name__)





class LocalBlocks:
    """Expand local triangular features without constructing dense H/S."""

    def __init__(self, data, idp, overlap=False):
        soc = data.get(AtomicDataDict.NODE_SOC_SWITCH_KEY, False)
        if torch.as_tensor(soc).any():
            raise ValueError("Direct BTD does not support SOC.")
        idp.get_orbpair_maps()
        edge_key = (
            AtomicDataDict.EDGE_OVERLAP_KEY
            if overlap
            else AtomicDataDict.EDGE_FEATURES_KEY
        )
        node_key = (
            AtomicDataDict.NODE_OVERLAP_KEY
            if overlap
            else AtomicDataDict.NODE_FEATURES_KEY
        )
        edge_features = data[edge_key]
        node_features = data.get(node_key)
        if node_features is None:
            raise ValueError(f"Direct BTD requires DeePTB field {node_key}.")
        atom_types = data[AtomicDataDict.ATOM_TYPE_KEY].flatten()
        self.masks = [idp.mask_to_basis[int(atom_type)] for atom_type in atom_types]
        self.offsets = np.cumsum(
            [0] + [int(mask.sum()) for mask in self.masks]
        )
        self.edges = data[AtomicDataDict.EDGE_INDEX_KEY].cpu().numpy()
        self.shifts = data[AtomicDataDict.EDGE_CELL_SHIFT_KEY]
        self.edge_orders = [
            np.argsort(endpoints, kind="stable") for endpoints in self.edges
        ]
        self.edge_starts = [
            np.concatenate(
                (
                    [0],
                    np.cumsum(
                        np.bincount(endpoints, minlength=len(atom_types))
                    ),
                )
            )
            for endpoints in self.edges
        ]
        full_norb = idp.full_basis_norb
        self.nodes = node_features.new_zeros(
            (len(atom_types), full_norb, full_norb)
        )
        self.hoppings = edge_features.new_zeros(
            (len(edge_features), full_norb, full_norb)
        )
        row_start = 0
        for row_index, row_orbital in enumerate(idp.full_basis):
            row_size = (
                2 * anglrMId[re.findall(r"[a-zA-Z]+", row_orbital)[0]] + 1
            )
            column_start = 0
            for column_index, column_orbital in enumerate(idp.full_basis):
                column_size = (
                    2
                    * anglrMId[
                        re.findall(r"[a-zA-Z]+", column_orbital)[0]
                    ]
                    + 1
                )
                if row_index <= column_index:
                    factor = 0.5 if row_orbital == column_orbital else 1.0
                    pair_slice = idp.orbpair_maps[
                        row_orbital + "-" + column_orbital
                    ]
                    matrix_slice = (
                        slice(row_start, row_start + row_size),
                        slice(column_start, column_start + column_size),
                    )
                    self.nodes[:, matrix_slice[0], matrix_slice[1]] = (
                        factor
                        * node_features[:, pair_slice].reshape(
                            -1, row_size, column_size
                        )
                    )
                    self.hoppings[:, matrix_slice[0], matrix_slice[1]] = (
                        factor
                        * edge_features[:, pair_slice].reshape(
                            -1, row_size, column_size
                        )
                    )
                column_start += column_size
            row_start += row_size

    def contributions(self):
        """Yield unhermitianized onsite/edge blocks in HR2HK order."""
        for atom_index, block in enumerate(self.nodes):
            mask = self.masks[atom_index]
            yield atom_index, atom_index, block[mask][:, mask], None
        for edge_index, (source, target) in enumerate(self.edges.T):
            block = self.hoppings[edge_index][self.masks[source]][
                :, self.masks[target]
            ]
            yield int(source), int(target), block, edge_index

    def phases(self, kpoint):
        kpoint = torch.as_tensor(
            kpoint, dtype=self.nodes.dtype, device=self.nodes.device
        )
        return torch.exp(-2j * torch.pi * (self.shifts @ kpoint))

    def pairs(self, kpoint, atoms=None, phases=None):
        """Yield selected atom rows with only one reduced row resident."""
        complex_dtype = (
            torch.complex64
            if self.nodes.dtype == torch.float32
            else torch.complex128
        )
        if phases is None:
            phases = self.phases(kpoint)
        if atoms is None:
            atoms = range(len(self.nodes))
        if self.nodes.device.type == "cpu" and not torch.is_grad_enabled():
            yield from self._cpu_pairs(atoms, phases)
            return
        for source in atoms:
            mask = self.masks[source]
            onsite = self.nodes[source][mask][:, mask].to(complex_dtype)
            halves = [{source: onsite}, {}]
            for direction in (0, 1):
                start, end = self.edge_starts[direction][source:source + 2]
                for edge_index in self.edge_orders[direction][start:end]:
                    row, column = self.edges[:, edge_index]
                    if direction == 1 and row == column:
                        continue
                    target = int(column if direction == 0 else row)
                    value = self.hoppings[edge_index][self.masks[row]][
                        :, self.masks[column]
                    ]
                    value = value.to(complex_dtype) * phases[edge_index]
                    if target in halves[direction]:
                        halves[direction][target] += value
                    else:
                        halves[direction][target] = value
            for target in sorted(halves[0].keys() | halves[1].keys()):
                value = halves[0].get(target)
                reverse = value if target == source else halves[1].get(target)
                if value is None:
                    value = reverse.mH
                elif reverse is not None:
                    value = value + reverse.mH
                yield source, target, value

    def _cpu_pairs(self, atoms, phases):
        """Use zero-copy CPU views to avoid small Torch allocations."""
        nodes = self.nodes.numpy()
        hoppings = self.hoppings.numpy()
        phases = phases.numpy()
        masks = [mask.numpy() for mask in self.masks]
        for source in atoms:
            mask = masks[source]
            onsite = nodes[source][mask][:, mask].astype(phases.dtype)
            halves = [{source: onsite}, {}]
            for direction in (0, 1):
                start, end = self.edge_starts[direction][source:source + 2]
                for edge_index in self.edge_orders[direction][start:end]:
                    row, column = self.edges[:, edge_index]
                    if direction == 1 and row == column:
                        continue
                    target = int(column if direction == 0 else row)
                    value = hoppings[edge_index][masks[row]][:, masks[column]]
                    value = value.astype(phases.dtype) * phases[edge_index]
                    if target in halves[direction]:
                        halves[direction][target] += value
                    else:
                        halves[direction][target] = value
            for target in sorted(halves[0].keys() | halves[1].keys()):
                value = halves[0].get(target)
                reverse = value if target == source else halves[1].get(target)
                if value is None:
                    value = reverse.T.conj()
                elif reverse is not None:
                    value = value + reverse.T.conj()
                yield source, target, torch.from_numpy(value)


def partition_profiles(edge, reverse_edge, left, right, fixed=False):
    """Partition exact H/S union profiles, preserving fixed boundaries."""
    size = len(edge)
    if min(left, right) <= 0 or max(left, right) > size:
        raise ValueError(
            "Direct BTD boundary sizes must lie inside the device."
        )
    if fixed and left + right > size:
        raise ValueError("Fixed cache boundaries overlap.")
    edge = np.maximum.accumulate(edge)
    reverse_edge = np.maximum.accumulate(reverse_edge)
    blocks = compute_blocks(
        left, right, edge, reverse_edge, use_jit=False
    )
    if fixed and (blocks[0] != left or blocks[-1] != right):
        raise ValueError(
            "Fixed self-energy cache boundaries are incompatible with H/S "
            "connectivity."
        )
    bounds = np.cumsum([0] + blocks)
    for block_index in range(len(blocks)):
        end = bounds[block_index + 1]
        if edge[end - 1] > bounds[min(block_index + 2, len(blocks))]:
            raise ValueError(
                "Direct BTD partition has non-neighbouring H/S connections."
            )
        reverse_end = size - bounds[block_index]
        if (
            reverse_edge[reverse_end - 1]
            > size - bounds[max(block_index - 1, 0)]
        ):
            raise ValueError(
                "Direct BTD partition has non-neighbouring H/S connections."
            )
    return blocks


def _window_add(
    output,
    value,
    row_start,
    column_start,
    row_window,
    column_window,
):

    row_lo = max(row_start, row_window[0])
    row_hi = min(row_start + value.shape[0], row_window[1])
    column_lo = max(column_start, column_window[0])
    column_hi = min(column_start + value.shape[1], column_window[1])

    if row_lo < row_hi and column_lo < column_hi:
        output[
            row_lo - row_window[0]:row_hi - row_window[0],
            column_lo - column_window[0]:column_hi - column_window[0],
        ] += value[
            row_lo - row_start:row_hi - row_start,
            column_lo - column_start:column_hi - column_start,
        ]


def assemble(local, kpoint, device_window, blocks, lead_windows):
    """Assemble one k point, splitting atoms at orbital boundaries."""


    bounds = np.cumsum([device_window[0]] + list(blocks))
    dtype = (
        torch.complex64
        if local.nodes.dtype == torch.float32
        else torch.complex128
    )

    kwargs = dict(dtype=dtype, device=local.nodes.device)
    diagonal = [
        torch.zeros((size, size), **kwargs) for size in blocks
    ]
    upper = [
        torch.zeros((left, right), **kwargs)
        for left, right in zip(blocks[:-1], blocks[1:])
    ]
    lower = [
        torch.zeros((right, left), **kwargs)
        for left, right in zip(blocks[:-1], blocks[1:])
    ]

    contacts = {}
    principal = {}

    for tab, window in lead_windows.items():

        boundary = 0 if tab == "lead_L" else len(blocks) - 1
        contacts[tab] = torch.zeros(
            (blocks[boundary], window[1] - window[0]), **kwargs
        )
        principal[tab] = torch.zeros(
            (window[1] - window[0],) * 2, **kwargs
        )


    for source, target, value in local.pairs(kpoint):

        row_start, row_end = local.offsets[source:source + 2]
        column_start, column_end = local.offsets[target:target + 2]

        if row_start < device_window[1] and row_end > device_window[0]:
            first_row = max(
                0, np.searchsorted(bounds, row_start, side="right") - 1
            )
            last_row = min(
                len(blocks),
                np.searchsorted(bounds, row_end, side="left"),
            )
            first_column = max(
                0, np.searchsorted(bounds, column_start, side="right") - 1
            )
            last_column = min(
                len(blocks),
                np.searchsorted(bounds, column_end, side="left"),
            )
            for row_block in range(first_row, last_row):
                for column_block in range(first_column, last_column):

                    if row_block == column_block:
                        output = diagonal[row_block]

                    elif column_block == row_block + 1:
                        output = upper[row_block]

                    elif row_block == column_block + 1:
                        output = lower[column_block]

                    else:
                        row_slice = slice(
                            max(0, bounds[row_block] - row_start),
                            min(
                                value.shape[0],
                                bounds[row_block + 1] - row_start,
                            ),
                        )
                        column_slice = slice(
                            max(0, bounds[column_block] - column_start),
                            min(
                                value.shape[1],
                                bounds[column_block + 1] - column_start,
                            ),
                        )

                        if torch.any(value[row_slice, column_slice] != 0):
                            raise ValueError(
                                "Non-neighbouring block contribution would "
                                "be lost."
                            )

                        continue

                    _window_add(
                        output,
                        value,
                        row_start,
                        column_start,
                        bounds[row_block:row_block + 2],
                        bounds[column_block:column_block + 2],
                    )

            for tab, window in lead_windows.items():
                boundary = 0 if tab == "lead_L" else len(blocks) - 1
                _window_add(
                    contacts[tab],
                    value,
                    row_start,
                    column_start,
                    bounds[boundary:boundary + 2],
                    window,
                )

        for tab, window in lead_windows.items():
            _window_add(
                principal[tab],
                value,
                row_start,
                column_start,
                window,
                window,
            )

    return diagonal, upper, lower, contacts, principal

def _write_complex(group, name, value):
    array = value.cpu().numpy()
    group.create_dataset(f"{name}_real", data=array.real)
    group.create_dataset(f"{name}_imag", data=array.imag)


def _build_profiles(local_blocks, kpoints, offsets, device_window, lead_windows):
    size = int(device_window[1] - device_window[0])
    edge = np.arange(1, size + 1)
    reverse_edge = edge.copy()

    contact_extents = {"lead_L": 1, "lead_R": 1}


    for local in local_blocks:
        if local is None:
            continue

        for kpoint in kpoints:

            for source, target, value in local.pairs(kpoint):

                rows, columns = np.nonzero(value.cpu().numpy())
                rows = rows + offsets[source]
                columns = columns + offsets[target]

                device_rows = (
                    (rows >= device_window[0])
                    & (rows < device_window[1])
                )

                inside = (
                    device_rows
                    & (columns >= device_window[0])
                    & (columns < device_window[1])
                )

                row_indices = rows[inside] - device_window[0]
                column_indices = columns[inside] - device_window[0]

                np.maximum.at(edge, row_indices, column_indices + 1)
                np.maximum.at(
                    reverse_edge,
                    size - 1 - row_indices,
                    size - column_indices,
                )

                for tab, window in lead_windows.items():

                    selected = (
                        device_rows
                        & (columns >= window[0])
                        & (columns < window[1])
                    )

                    if selected.any():

                        if tab == "lead_L":
                            depth = rows[selected].max() - device_window[0] + 1
                        else:
                            depth = device_window[1] - rows[selected].min()

                        contact_extents[tab] = max(
                            contact_extents[tab], int(depth)
                        )

    return edge, reverse_edge, contact_extents

def write_block_rows(
    cache,
    prefix,
    local,
    kpoint,
    k_index,
    device_window,
    blocks,
    lead_windows,
    factor=1.0,
    orthogonal=False,
):
    """Write one diagonal and at most two neighboring blocks at a time."""

    bounds = np.cumsum([device_window[0]] + list(blocks))

    dtype = (
        torch.complex64
        if local.nodes.dtype == torch.float32
        else torch.complex128
    )


    kwargs = dict(dtype=dtype, device=local.nodes.device)
    phases = None if orthogonal else local.phases(kpoint)
    maximum = max(blocks)
    workspace = torch.empty((3, maximum, maximum), **kwargs)

    contacts = {}
    principal = {}

    for tab, window in lead_windows.items():

        boundary = 0 if tab == "lead_L" else len(blocks) - 1
        width = window[1] - window[0]
        contacts[tab] = torch.zeros(
            (blocks[boundary], width), **kwargs
        )

        principal[tab] = torch.zeros((width, width), **kwargs)
        if orthogonal:
            principal[tab].diagonal().fill_(1)


    for block_index, size in enumerate(blocks):

        row_window = bounds[block_index:block_index + 2]
        neighbours = range(
            max(0, block_index - 1),
            min(len(blocks), block_index + 2),
        )

        outputs = {}

        for slot, column_block in enumerate(neighbours):
            output = workspace[slot, :size, :blocks[column_block]]
            output.zero_()
            outputs[column_block] = output

        if orthogonal:
            outputs[block_index].diagonal().fill_(1)

        else:
            first_atom = (
                np.searchsorted(local.offsets, row_window[0], side="right") - 1
            )
            last_atom = np.searchsorted(
                local.offsets, row_window[1], side="left"
            )
            for source, target, value in local.pairs(
                kpoint, range(first_atom, last_atom), phases=phases
            ):

                row_start = local.offsets[source]
                column_start = local.offsets[target]
                for column_block, output in outputs.items():
                    _window_add(
                        output,
                        value,
                        row_start,
                        column_start,
                        row_window,
                        bounds[column_block:column_block + 2],
                    )
                device_column_lo = max(column_start, device_window[0])
                device_column_hi = min(
                    column_start + value.shape[1], device_window[1]
                )

                allowed_lo = bounds[max(0, block_index - 1)]
                allowed_hi = bounds[min(len(blocks), block_index + 2)]
                row_slice = slice(
                    max(0, row_window[0] - row_start),
                    min(value.shape[0], row_window[1] - row_start),
                )

                for column_lo, column_hi in (
                    (
                        device_column_lo,
                        min(device_column_hi, allowed_lo),
                    ),
                    (
                        max(device_column_lo, allowed_hi),
                        device_column_hi,
                    ),
                ):
                    if (
                        column_lo < column_hi
                        and torch.any(
                            value[
                                row_slice,
                                column_lo - column_start:
                                column_hi - column_start,
                            ]
                            != 0
                        )
                    ):
                        raise ValueError(
                            "Non-neighbouring block contribution would be lost."
                        )


                for tab, window in lead_windows.items():
                    boundary = 0 if tab == "lead_L" else len(blocks) - 1
                    if block_index == boundary:
                        _window_add(
                            contacts[tab],
                            value,
                            row_start,
                            column_start,
                            row_window,
                            window,
                        )

                    else:
                        column_lo = max(column_start, window[0])
                        column_hi = min(
                            column_start + value.shape[1], window[1]
                        )

                        if (
                            column_lo < column_hi
                            and torch.any(
                                value[
                                    row_slice,
                                    column_lo - column_start:
                                    column_hi - column_start,
                                ]
                                != 0
                            )
                        ):
                            raise ValueError(
                                "Device-lead coupling extends beyond its "
                                "boundary block."
                            )

        for column_block, output in outputs.items():
            suffix = (
                "d"
                if column_block == block_index
                else ("u" if column_block > block_index else "l")
            )
            name = prefix + suffix
            cache_index = min(block_index, column_block)
            output.mul_(factor)
            _write_complex(
                cache[name],
                f"{name}_k{k_index}_b{cache_index}",
                output,
            )
        del outputs, output

    if not orthogonal:

        for tab, window in lead_windows.items():
            first_atom = (
                np.searchsorted(local.offsets, window[0], side="right") - 1
            )
            last_atom = np.searchsorted(
                local.offsets, window[1], side="left"
            )

            for source, target, value in local.pairs(
                kpoint, range(first_atom, last_atom), phases=phases
            ):
                _window_add(
                    principal[tab],
                    value,
                    local.offsets[source],
                    local.offsets[target],
                    window,
                    window,
                )
    return contacts, principal


@torch.no_grad()
def initialize_direct(owner, kpoints, structure_leads):
    """Predict local features, partition exact H/S, and stream caches."""
    stages = []

    data = AtomicData.from_ase(owner.structase, **owner.AtomicData_options)
    data = owner.model.idp(
        AtomicData.to_AtomicDataDict(data.to(owner.torch_device))
    )
    data[AtomicDataDict.KPOINT_KEY] = torch.as_tensor(
        kpoints, dtype=owner.model.dtype, device=owner.torch_device
    )
    data = owner.model(data)

    owner.overlap = AtomicDataDict.EDGE_OVERLAP_KEY in data
    owner.remove_bonds_nonpbc(data, owner.pbc_negf, owner.overlap)

    local_h = LocalBlocks(data, owner.model.idp)
    local_s = (
        LocalBlocks(data, owner.model.idp, overlap=True)
        if owner.overlap
        else None
    )
    del data

    offsets = local_h.offsets
    device_window = offsets[owner.device_id]
    device_size = int(device_window[1] - device_window[0])

    lead_windows = {}
    for tab, atom_range in owner.lead_ids.items():
        start, end = offsets[atom_range]
        lead_windows[tab] = (start, start + (end - start) // 2)

    edge, reverse_edge, contact_extents = _build_profiles(
        (local_h, local_s),
        kpoints,
        offsets,
        device_window,
        lead_windows,
    )
    left = contact_extents["lead_L"]
    right = contact_extents["lead_R"]

    fixed_edges = owner._self_energy_cache_edge_sizes
    if fixed_edges:
        if fixed_edges[0] < left or fixed_edges[1] < right:
            raise ValueError(
                "Cached boundary cannot contain the full H/S device-lead "
                "coupling."
            )
        left, right = fixed_edges
        log.warning(
            "Reusing legacy self-energy cache: dimensions do not verify "
            "model/contact/k/E compatibility; caller must ensure identical "
            "physical inputs."
        )

    blocks = partition_profiles(
        edge,
        reverse_edge,
        left,
        right,
        fixed=bool(fixed_edges),
    )

    elements = sum(size**2 for size in blocks) + 2 * sum(
        left_size * right_size
        for left_size, right_size in zip(blocks[:-1], blocks[1:])
    )
    estimated_mib = 2 * 16 * elements / 2**20
    log.info(
        "Direct BTD: orbitals=%s blocks=%s max_block=%s "
        "H+S complex128=%.2f MiB",
        device_size,
        len(blocks),
        max(blocks),
        estimated_mib,
    )

    if (
        owner.direct_btd_max_mib is not None
        and estimated_mib > owner.direct_btd_max_mib
    ):
        raise MemoryError(
            f"BTD H/S requires {estimated_mib:.2f} MiB, exceeding "
            f"direct_btd_max_mib; largest block {max(blocks)}. "
            "No matrices allocated."
        )

    log.info(
        "Direct BTD block-row workspace: %.3f MiB; total resident BTD H/S "
        "at downstream read: %.3f MiB.",
        3
        * max(blocks) ** 2
        * (8 if owner.model.dtype == torch.float32 else 16)
        / 2**20,
        estimated_mib,
    )
    owner.subblocks = blocks

    from dptb.nn.hr2hk import HR2HK

    overlap_transform = (
        HR2HK(
            idp=owner.model.idp,
            overlap=True,
            edge_field=AtomicDataDict.EDGE_OVERLAP_KEY,
            node_field=AtomicDataDict.NODE_OVERLAP_KEY,
            out_field=AtomicDataDict.OVERLAP_KEY,
            dtype=owner.model.dtype,
            device=owner.torch_device,
        )
        if owner.overlap
        else None
    )

    device_path = os.path.join(owner.results_path, "HS_device.h5")
    with h5py.File(device_path, "w") as cache:
        cache.attrs["layout_version"] = 2
        cache.attrs["initializer"] = "direct"
        cache.attrs["cache_id"] = uuid.uuid4().hex
        cache.attrs["complete"] = False

        cache.create_dataset("kpoints", data=kpoints)
        cache.create_dataset("subblocks", data=blocks)
        cache.create_dataset("block_tridiagonal", data=True)

        for name in ("hd", "hu", "hl", "sd", "su", "sl"):
            cache.create_group(name)

        for k_index, kpoint in enumerate(kpoints):
            h_contacts, h_principal = write_block_rows(
                cache,
                "h",
                local_h,
                kpoint,
                k_index,
                device_window,
                blocks,
                lead_windows,
                factor=owner.h_factor,
            )
            s_contacts, s_principal = write_block_rows(
                cache,
                "s",
                local_s if local_s is not None else local_h,
                kpoint,
                k_index,
                device_window,
                blocks,
                lead_windows,
                orthogonal=local_s is None,
            )

            for tab, structure in structure_leads.items():
                lead_data = AtomicData.from_ase(
                    structure, **owner.AtomicData_options
                )
                lead_data = owner.model.idp(
                    AtomicData.to_AtomicDataDict(
                        lead_data.to(owner.torch_device)
                    )
                )
                lead_data[AtomicDataDict.KPOINT_KEY] = torch.as_tensor(
                    np.asarray(kpoint).reshape(1, 3),
                    dtype=owner.model.dtype,
                    device=owner.torch_device,
                )
                lead_data = owner.model(lead_data)

                owner.remove_bonds_nonpbc(
                    lead_data, owner.pbc_negf, owner.overlap
                )

                lead_h = owner.h2k(lead_data)[
                    AtomicDataDict.HAMILTONIAN_KEY
                ][0]
                lead_s = (
                    overlap_transform(lead_data)[
                        AtomicDataDict.OVERLAP_KEY
                    ][0]
                    if owner.overlap
                    else torch.eye(
                        lead_h.shape[0],
                        dtype=owner.model.dtype,
                        device=owner.torch_device,
                    )
                )

                principal_size = lead_h.shape[0] // 2
                principal_h = lead_h[:principal_size, :principal_size]
                principal_s = lead_s[:principal_size, :principal_size]

                for name, actual, expected in (
                    ("H", principal_h, h_principal[tab]),
                    ("S", principal_s, s_principal[tab]),
                ):
                    rmse = (
                        (actual - expected).abs().square().mean()
                    ).sqrt().item()
                    if rmse >= 1e-4:
                        raise ValueError(
                            f"{tab} independent/device {name} "
                            f"principal-layer mismatch: RMSE {rmse}."
                        )

                hopping_h = lead_h[:principal_size, principal_size:].clone()
                principal_h = principal_h.clone()
                principal_h[principal_h.abs() < 1e-6] = 0
                hopping_h[hopping_h.abs() < 1e-6] = 0

                values = {
                    "HL": principal_h.cdouble() * owner.h_factor,
                    "HLL": hopping_h.cdouble() * owner.h_factor,
                    "SL": principal_s,
                    "SLL": lead_s[:principal_size, principal_size:],
                    "HDL": h_contacts[tab].cdouble() * owner.h_factor,
                    "SDL": s_contacts[tab],
                }

                lead_path = os.path.join(
                    owner.results_path, f"HS_{tab}.h5"
                )
                with h5py.File(
                    lead_path, "w" if k_index == 0 else "a"
                ) as lead_cache:
                    if k_index == 0:
                        boundary = 0 if tab == "lead_L" else len(blocks) - 1
                        lead_cache.attrs.update(
                            layout_version=2,
                            contact_reduced=True,
                            lead=tab,
                            cache_id=cache.attrs["cache_id"],
                            device_block=boundary,
                            device_orbital_offset=sum(blocks[:boundary]),
                            device_orbitals=device_size,
                        )

                        lead_cache.create_dataset("kpoints", data=kpoints)
                        lead_cache.create_dataset("useBloch", data=False)

                        for name in ("kpoints_bloch", "bloch_factor"):
                            lead_cache.create_dataset(
                                name,
                                data=np.array(
                                    "None", dtype=h5py.string_dtype()
                                ),
                            )

                        for name, value in values.items():
                            for component in ("real", "imag"):
                                lead_cache.create_dataset(
                                    name + "_" + component,
                                    shape=(len(kpoints),) + tuple(value.shape),
                                    dtype=(
                                        np.float32
                                        if owner.model.dtype == torch.float32
                                        else np.float64
                                    ),
                                )

                    for name, value in values.items():
                        lead_cache[name + "_real"][k_index] = (
                            value.real.cpu().numpy()
                        )
                        lead_cache[name + "_imag"][k_index] = (
                            value.imag.cpu().numpy()
                            if value.is_complex()
                            else 0
                        )

                del lead_data, lead_h, lead_s

            del h_contacts, s_contacts, h_principal, s_principal

        cache.attrs["complete"] = True

    owner.direct_btd_stages = stages