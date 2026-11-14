"""Shared H/S mapping ownership, layout and cleanup regression tests."""

import hashlib
import multiprocessing as mp
import os
from multiprocessing import shared_memory

import pytest
import torch

from dpnegf.negf.device_property import DeviceProperty
from dpnegf.utils.rgf_shared import SharedHS, make_hs_context


class _UncopyableMapping:
    def __deepcopy__(self, memo):
        raise RuntimeError("mapping copy failed")


def _attach_then_exit(descriptor):
    SharedHS.attach(descriptor)
    os._exit(7)


def _parts():
    return (
        [torch.arange(6, dtype=torch.float64).reshape(2, 3).to(torch.complex128)],
        [torch.eye(2, dtype=torch.complex128)],
        [torch.ones((2, 1), dtype=torch.complex128)],
        [torch.full((1, 2), 2, dtype=torch.complex128)],
        [torch.full((1, 2), 3, dtype=torch.complex128)],
        [torch.full((2, 1), 4, dtype=torch.complex128)],
    )


def test_shared_hs_owner_and_attachment_map_all_six_groups():
    parts = _parts()
    before = [[block.clone() for block in group] for group in parts]
    hashes = [[hashlib.sha256(block.numpy().tobytes()).digest() for block in group]
              for group in parts]
    context = make_hs_context([0.25, 0.0, -0.5], torch.tensor([0.1, 0.2]), True)
    owner = SharedHS.publish(parts, context, version=3)
    name = owner.descriptor["name"]
    attachment = SharedHS.attach(owner.descriptor)
    try:
        assert owner.version == attachment.version == 3
        views = attachment.torch_views()
        for expected_group, actual_group in zip(before, views):
            assert len(expected_group) == len(actual_group)
            for expected, actual in zip(expected_group, actual_group):
                torch.testing.assert_close(actual, expected)
        for group in owner.descriptor["blocks"].values():
            assert all(block["offset"] % 64 == 0 for block in group)
        for expected_group, actual_group in zip(before, parts):
            for expected, actual in zip(expected_group, actual_group):
                torch.testing.assert_close(actual, expected)
        after_hashes = [
            [hashlib.sha256(block.numpy().tobytes()).digest() for block in group]
            for group in parts
        ]
        assert after_hashes == hashes
        del views, actual_group, actual
    finally:
        attachment.close()
        owner.close()
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name, create=False)


def test_abrupt_spawn_worker_exit_does_not_unlink_owner_region():
    context = make_hs_context([0.0, 0.0, 0.0], torch.tensor(0.1), True)
    owner = SharedHS.publish(_parts(), context, version=1)
    name = owner.descriptor["name"]
    process = mp.get_context("spawn").Process(
        target=_attach_then_exit, args=(owner.descriptor,))
    process.start()
    process.join(timeout=5)
    assert not process.is_alive()
    assert process.exitcode == 7

    attachment = SharedHS.attach(owner.descriptor)
    attachment.close()
    owner.close()
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name, create=False)


def test_shared_hs_capacity_failure_happens_before_allocation(monkeypatch):
    import dpnegf.utils.rgf_shared as rgf_shared

    monkeypatch.setattr(rgf_shared.os.path, "isdir", lambda _path: True)
    monkeypatch.setattr(rgf_shared.shutil, "disk_usage", lambda _path: type(
        "Usage", (), {"free": 0})())
    context = make_hs_context([0.0, 0.0, 0.0], torch.tensor(0.0), True)
    with pytest.raises(OSError, match="Shared H/S needs"):
        SharedHS.publish(_parts(), context, version=1)


def test_context_matching_is_exact_and_shape_sensitive():
    base = make_hs_context([0.0, 0.0, 0.0], torch.tensor(0.1), True)
    tiny_k = make_hs_context([1e-12, 0.0, 0.0], torch.tensor(0.1), True)
    tiny_v = make_hs_context([0.0, 0.0, 0.0], torch.tensor(0.10000001), True)
    vector_v = make_hs_context([0.0, 0.0, 0.0], torch.tensor([0.1]), True)

    from dpnegf.utils.rgf_shared import contexts_equal

    assert contexts_equal(base, make_hs_context([0.0, 0.0, 0.0], torch.tensor(0.1), True))
    assert not contexts_equal(base, tiny_k)
    assert not contexts_equal(base, tiny_v)
    assert not contexts_equal(base, vector_v)


def test_worker_context_is_float64_and_cannot_reload_unsynchronized_hs():
    parts = _parts()
    context = make_hs_context(
        [0.125, 0.0, 0.0], torch.tensor(0.2, dtype=torch.float32), True)
    owner = SharedHS.publish(parts, context, version=1)
    deviceprop = DeviceProperty.__new__(DeviceProperty)
    deviceprop.greenfuncs = 0
    deviceprop.rgf_device = torch.device("cpu")
    deviceprop.results_path = "/tmp"
    deviceprop.kpoint = None

    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        deviceprop.install_shared_hs(owner.descriptor)
        assert deviceprop.kpoint.dtype == torch.float64
        deviceprop.prepare_hamiltonian(
            [0.125, 0.0, 0.0], True, torch.tensor(0.2))
        for name in ("hd", "sd", "hl", "hu", "sl", "su"):
            delattr(deviceprop, name)
        deviceprop.prepare_hamiltonian(
            [0.125, 0.0, 0.0], True, torch.tensor(0.2))
        assert all(hasattr(deviceprop, name) for name in ("hd", "sd", "hl", "hu", "sl", "su"))
        with pytest.raises(RuntimeError, match="not synchronized"):
            deviceprop.prepare_hamiltonian(
                [0.125000001, 0.0, 0.0], True, torch.tensor(0.2))
    finally:
        torch.set_default_dtype(original_dtype)
        deviceprop.detach_shared_hs()
        owner.close()


class _ReusableHamiltonian:
    def __init__(self, parts):
        self.parts = parts

    def get_hs_device(self, kpoint, potential, block_tridiagonal, device=None):
        return tuple([[block.to(device) for block in group] for group in self.parts])

    def get_orbital_map(self):
        return {"status": "unknown", "orbital_map": None}


def test_owner_pool_release_restores_reusable_device_property(tmp_path):
    parts = _parts()
    deviceprop = DeviceProperty.__new__(DeviceProperty)
    deviceprop.greenfuncs = 0
    deviceprop.hamiltonian = _ReusableHamiltonian(parts)
    deviceprop.rgf_device = torch.device("cpu")
    deviceprop.results_path = str(tmp_path)
    deviceprop.kpoint = None
    deviceprop.prepare_hamiltonian(
        [0.25, 0.0, -0.5], True, torch.tensor([0.1, 0.2]))
    expected = {
        name: [block.clone() for block in getattr(deviceprop, name)]
        for name in ("hd", "sd", "hl", "hu", "sl", "su")
    }
    context = make_hs_context(deviceprop.kpoint, deviceprop.V, True)
    shared = SharedHS.publish(
        tuple(getattr(deviceprop, name) for name in ("hd", "sd", "hl", "hu", "sl", "su")),
        context, version=1,
    )
    shared.descriptor["orbital_mapping"] = deviceprop._orbital_mapping_snapshot
    deviceprop.install_shared_hs_owner(shared)

    deviceprop.release_shared_hs_owner()
    shared.close()
    assert not hasattr(deviceprop, "_shared_hs_owner")
    assert deviceprop.kpoint.dtype == torch.float64
    assert torch.equal(deviceprop.V, torch.tensor([0.1, 0.2]))
    for name, expected_blocks in expected.items():
        actual_blocks = getattr(deviceprop, name)
        assert len(actual_blocks) == len(expected_blocks)
        for actual, expected_block in zip(actual_blocks, expected_blocks):
            torch.testing.assert_close(actual, expected_block)
            assert actual.untyped_storage().data_ptr() != expected_block.untyped_storage().data_ptr()

    deviceprop.prepare_hamiltonian([0.25, 0.0, -0.5], True, torch.tensor([0.1, 0.2]))
    assert all(hasattr(deviceprop, name) for name in expected)


def test_owner_view_install_failure_can_release_shared_mapping():
    context = make_hs_context([0.0, 0.0, 0.0], torch.tensor(0.1), True)
    owner = SharedHS.publish(_parts(), context, version=1)
    name = owner.descriptor["name"]
    owner.descriptor["orbital_mapping"] = _UncopyableMapping()
    deviceprop = DeviceProperty.__new__(DeviceProperty)
    deviceprop.greenfuncs = 0
    deviceprop.rgf_device = torch.device("cpu")

    with pytest.raises(RuntimeError, match="mapping copy failed"):
        deviceprop.install_shared_hs_owner(owner)
    deviceprop.release_shared_hs_owner()
    owner.close()
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name, create=False)
