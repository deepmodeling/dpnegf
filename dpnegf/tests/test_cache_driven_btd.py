"""Tests for cache-driven block-tridiagonal (BTD) edge matching.

The optimized BTD splitter treats the leftmost/rightmost block sizes only as
lower bounds and re-optimizes cut points from the interior sparsity, so two
devices with an identical electrode region but a slightly different scattering
region could end up with different first/last block sizes. Because the lead
self-energy is pinned to those outer block sizes (see
``LeadProperty.HDL_reduced``), a cached self-energy from one run became
dimension-incompatible with the next.

This module tests the two pieces of the fix:

* ``inspect_self_energy_cache`` — infers the required first/last device block
  sizes from a saved self-energy cache (HDF5 preferred, legacy PTH fallback).
* ``dpnegf.negf.split_btd.constrained_subblocks`` /
  ``validate_block_tridiagonal`` — partition the device with the first/last
  block sizes pinned exactly, growing only the interior, and reject layouts
  incompatible with the device sparsity.
"""

import os
import inspect

import numpy as np
import pytest
import torch

from dpnegf.negf.negf_hamiltonian_init import NEGFHamiltonianInit
from dpnegf.negf.lead_property import inspect_self_energy_cache, write_to_hdf5
from dpnegf.negf.split_btd import (
    split_into_subblocks,
    split_into_subblocks_optimized,
    constrained_subblocks,
    validate_block_tridiagonal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_btd_device(blocks, coupling=0.5, onsite=1.0):
    """Build a Hamiltonian/overlap pair that is block-tridiagonal for ``blocks``.

    On-site terms populate the diagonal; nearest-block couplings populate the
    super/sub block bands. The result is genuinely block-tridiagonal for the
    given partition and its refinements.
    """
    D = sum(blocks)
    H = np.eye(D) * onsite
    bnd = np.cumsum([0] + list(blocks))
    for bi in range(len(blocks) - 1):
        r0, r1 = bnd[bi], bnd[bi + 1]
        c0, c1 = bnd[bi + 1], bnd[bi + 2]
        H[r0:r1, c0:c1] = coupling
        H[c0:c1, r0:r1] = coupling
    H = torch.tensor(H, dtype=torch.float64)
    S = torch.eye(D, dtype=torch.float64)
    return H, S


def _bare_ham():
    """A NEGFHamiltonianInit instance without running __init__.

    Used to exercise ``get_block_tridiagonal`` on synthetic Hamiltonians
    without building a model.
    """
    return object.__new__(NEGFHamiltonianInit)


def test_fixed_edge_sizes_are_not_public_constructor_arguments():
    parameters = inspect.signature(NEGFHamiltonianInit.__init__).parameters
    assert "fixed_leftmost_size" not in parameters
    assert "fixed_rightmost_size" not in parameters


def test_fresh_btd_uses_historical_splitter_path():
    ham = _bare_ham()
    ham._self_energy_cache_edge_sizes = None
    ham.results_path = None
    H, S = _make_btd_device([3, 2, 2, 3])
    HK = H.unsqueeze(0)
    SK = S.unsqueeze(0)

    expected = split_into_subblocks_optimized(HK[0], 3, 3)
    if expected[0] < 3 or expected[-1] < 3:
        expected = split_into_subblocks(HK[0], 3, 3)

    *_, actual = ham.get_block_tridiagonal(
        HK, SK, structase=None, leftmost_size=3, rightmost_size=3)
    assert actual == expected


def _write_h5_cache(path, tab, size, ks, es):
    """Write a valid HDF5 self-energy cache with a single invariant size."""
    for k in ks:
        for e in es:
            se = torch.eye(size, dtype=torch.complex128) * (e + 1j)
            write_to_hdf5(path, k, e, se)


# ---------------------------------------------------------------------------
# inspect_self_energy_cache
# ---------------------------------------------------------------------------
def test_inspect_cache_valid_h5(tmp_path):
    d = str(tmp_path)
    ks = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]]
    es = [-1.0, 0.0, 1.0]
    _write_h5_cache(os.path.join(d, "self_energy_leadL.h5"), "lead_L", 4, ks, es)
    _write_h5_cache(os.path.join(d, "self_energy_leadR.h5"), "lead_R", 6, ks, es)

    nL, nR, fmt = inspect_self_energy_cache(d)
    assert (nL, nR, fmt) == (4, 6, "h5")


def test_inspect_cache_valid_pth(tmp_path):
    d = str(tmp_path)
    torch.save(torch.eye(3, dtype=torch.complex128),
               os.path.join(d, "se_lead_L_k0.0000_0.0000_0.0000_E0.000000.pth"))
    torch.save(torch.eye(3, dtype=torch.complex128),
               os.path.join(d, "se_lead_L_k0.0000_0.0000_0.0000_E1.000000.pth"))
    torch.save(torch.eye(5, dtype=torch.complex128),
               os.path.join(d, "se_lead_R_k0.0000_0.0000_0.0000_E0.000000.pth"))

    nL, nR, fmt = inspect_self_energy_cache(d)
    assert (nL, nR, fmt) == (3, 5, "pth")


def test_inspect_cache_h5_precedence(tmp_path):
    d = str(tmp_path)
    ks = [[0.0, 0.0, 0.0]]
    es = [0.0]
    # HDF5 says (4, 6); PTH says (3, 5). HDF5 must win.
    _write_h5_cache(os.path.join(d, "self_energy_leadL.h5"), "lead_L", 4, ks, es)
    _write_h5_cache(os.path.join(d, "self_energy_leadR.h5"), "lead_R", 6, ks, es)
    torch.save(torch.eye(3, dtype=torch.complex128),
               os.path.join(d, "se_lead_L_k0.0000_0.0000_0.0000_E0.000000.pth"))
    torch.save(torch.eye(5, dtype=torch.complex128),
               os.path.join(d, "se_lead_R_k0.0000_0.0000_0.0000_E0.000000.pth"))

    nL, nR, fmt = inspect_self_energy_cache(d)
    assert (nL, nR, fmt) == (4, 6, "h5")


def test_inspect_cache_missing_lead(tmp_path):
    d = str(tmp_path)
    _write_h5_cache(os.path.join(d, "self_energy_leadL.h5"), "lead_L", 4,
                    [[0.0, 0.0, 0.0]], [0.0])
    # Only left lead present.
    with pytest.raises(ValueError, match="[Ii]ncomplete"):
        inspect_self_energy_cache(d)


def test_inspect_cache_absent(tmp_path):
    with pytest.raises(ValueError, match="No self-energy cache"):
        inspect_self_energy_cache(str(tmp_path))


def test_inspect_cache_missing_dir():
    with pytest.raises(ValueError, match="does not exist"):
        inspect_self_energy_cache(os.path.join(os.sep, "no", "such", "dir_xyz"))


def test_inspect_cache_empty_h5(tmp_path):
    d = str(tmp_path)
    import h5py
    # Two present but empty HDF5 files.
    h5py.File(os.path.join(d, "self_energy_leadL.h5"), "w").close()
    h5py.File(os.path.join(d, "self_energy_leadR.h5"), "w").close()
    with pytest.raises(ValueError, match="empty"):
        inspect_self_energy_cache(d)


def test_inspect_cache_non_square(tmp_path):
    d = str(tmp_path)
    import h5py
    with h5py.File(os.path.join(d, "self_energy_leadL.h5"), "w") as f:
        g = f.require_group("E_0.00000000")
        g.create_dataset("k_0.0_0.0_0.0", data=np.zeros((3, 4), dtype=np.complex128))
    _write_h5_cache(os.path.join(d, "self_energy_leadR.h5"), "lead_R", 5,
                    [[0.0, 0.0, 0.0]], [0.0])
    with pytest.raises(ValueError, match="non-square"):
        inspect_self_energy_cache(d)


def test_inspect_cache_mixed_shape(tmp_path):
    d = str(tmp_path)
    import h5py
    with h5py.File(os.path.join(d, "self_energy_leadL.h5"), "w") as f:
        f.require_group("E_0.00000000").create_dataset(
            "k_0.0_0.0_0.0", data=np.eye(4, dtype=np.complex128))
        f.require_group("E_1.00000000").create_dataset(
            "k_0.0_0.0_0.0", data=np.eye(5, dtype=np.complex128))  # size drift
    _write_h5_cache(os.path.join(d, "self_energy_leadR.h5"), "lead_R", 6,
                    [[0.0, 0.0, 0.0]], [0.0])
    with pytest.raises(ValueError, match="mixed matrix sizes"):
        inspect_self_energy_cache(d)


# ---------------------------------------------------------------------------
# constrained_subblocks / validate_block_tridiagonal
# ---------------------------------------------------------------------------
def test_constrained_endpoints_preserved():
    H, S = _make_btd_device([3, 2, 2, 3])
    sub = constrained_subblocks(H, S, 3, 3)
    assert sub[0] == 3 and sub[-1] == 3
    assert sum(sub) == H.shape[0]


def test_constrained_endpoints_stable_under_interior_change():
    """Same electrodes (nL=nR=3), different scattering interiors, same endpoints.

    This is the whole point of the fix: cached self-energy edge sizes stay
    reusable across scattering-region variants.
    """
    for interior in ([2, 2], [5], [2, 4], [3, 3, 2]):
        H, S = _make_btd_device([3] + interior + [3])
        sub = constrained_subblocks(H, S, 3, 3)
        assert sub[0] == 3, (interior, sub)
        assert sub[-1] == 3, (interior, sub)
        assert sum(sub) == H.shape[0]
        assert validate_block_tridiagonal(
            (H.abs() + S.abs()) != 0, sub)


def test_constrained_full_coverage_and_tridiagonal():
    H, S = _make_btd_device([4, 3, 3, 4])
    sub = constrained_subblocks(H, S, 4, 4)
    assert sum(sub) == H.shape[0]
    mask = (H.abs() + S.abs()) != 0
    assert validate_block_tridiagonal(mask, sub)


def test_constrained_two_block_exact_tiling():
    H, S = _make_btd_device([5, 5])
    sub = constrained_subblocks(H, S, 5, 5)
    assert sub == [5, 5]


def test_validate_rejects_non_tridiagonal_partition():
    # [3,2,2,3] device: partition [2,2,2,2,2] splits the first natural block and
    # makes block 0 couple block 2 -> not block-tridiagonal.
    H, S = _make_btd_device([3, 2, 2, 3])
    mask = (H.abs() + S.abs()) != 0
    assert not validate_block_tridiagonal(mask, [2, 2, 2, 2, 2])
    assert validate_block_tridiagonal(mask, [3, 2, 2, 3])


# ---------------------------------------------------------------------------
# Expected failures
# ---------------------------------------------------------------------------
def test_constrained_edge_exceeds_device():
    H, S = _make_btd_device([3, 2, 3])
    with pytest.raises(ValueError, match="exceed"):
        constrained_subblocks(H, S, H.shape[0] + 1, 3)


def test_constrained_non_positive_edge():
    H, S = _make_btd_device([3, 2, 3])
    with pytest.raises(ValueError, match="positive"):
        constrained_subblocks(H, S, 0, 3)


def test_constrained_no_interior_no_tiling():
    H, S = _make_btd_device([3, 2, 3])  # D = 8
    # 6 + 6 > 8 but 6 + 6 != 8 -> cannot tile.
    with pytest.raises(ValueError, match="no room|tile"):
        constrained_subblocks(H, S, 6, 6)


def test_constrained_edge_too_small_for_coupling():
    """A [5,5] device cannot honour nL=3: the coupling forces a larger first block.

    The greedy splitter would grow the edge to 7, which the constrained path
    must reject because the cached (3x3) self-energy would not fit.
    """
    H, S = _make_btd_device([5, 5])
    with pytest.raises(ValueError, match="incompatible"):
        constrained_subblocks(H, S, 3, 3)


# ---------------------------------------------------------------------------
# End-to-end integration: run once to populate the cache, then reuse it with
# cache-driven BTD and confirm the endpoint blocks match the cached self-energy
# and no self-energy is recomputed.
# ---------------------------------------------------------------------------
@pytest.fixture(scope='session')
def root_directory(request):
    return str(request.config.rootdir)


def _read_subblocks(results_path):
    import h5py
    with h5py.File(os.path.join(results_path, "HS_device.h5"), "r") as f:
        return [int(x) for x in np.array(f["subblocks"])]


def test_cache_driven_btd_end_to_end(root_directory, tmp_path):
    import json
    from dpnegf.entrypoints.run import run

    data = os.path.join(root_directory, "dpnegf/tests/data/test_negf/test_negf_run")
    ckpt = os.path.join(data, "nnsk_C_new.json")
    stru = os.path.join(data, "chain.vasp")
    base = json.load(open(os.path.join(data, "negf_chain_new.json")))

    # Block-tridiagonal, tiny grid, single SE worker for determinism.
    base["task_options"]["block_tridiagonal"] = True
    base["task_options"]["emin"] = -0.05
    base["task_options"]["emax"] = 0.05
    base["task_options"]["espacing"] = 0.05
    base["task_options"]["n_cpus"] = 1

    # --- run 1: fresh, populate the self-energy cache ---
    in1 = str(tmp_path / "run1.json"); json.dump(base, open(in1, "w"))
    out1 = str(tmp_path / "out1")
    run(INPUT=in1, init_model=ckpt, structure=stru, output=out1,
        log_level=20, log_path=os.path.join(out1, "log.txt"))

    res1 = os.path.join(out1, "results")
    se_dir = os.path.join(res1, "self_energy")
    assert os.path.isfile(os.path.join(se_dir, "self_energy_leadL.h5"))
    assert os.path.isfile(os.path.join(se_dir, "self_energy_leadR.h5"))

    nL, nR, fmt = inspect_self_energy_cache(se_dir)
    assert fmt == "h5"
    sub1 = _read_subblocks(res1)
    assert sub1[0] == nL and sub1[-1] == nR

    # --- run 2: reuse the cache; endpoints must match the cached SE dims ---
    base2 = json.loads(json.dumps(base))
    base2["task_options"]["use_saved_se"] = True
    base2["task_options"]["self_energy_save_path"] = se_dir
    in2 = str(tmp_path / "run2.json"); json.dump(base2, open(in2, "w"))
    out2 = str(tmp_path / "out2")
    run(INPUT=in2, init_model=ckpt, structure=stru, output=out2,
        log_level=20, log_path=os.path.join(out2, "log.txt"))

    res2 = os.path.join(out2, "results")
    sub2 = _read_subblocks(res2)
    # The core guarantee: cache-driven endpoints equal the cached self-energy size.
    assert sub2[0] == nL and sub2[-1] == nR
    assert os.path.exists(os.path.join(res2, "negf.out.pth"))

    log2 = open(os.path.join(out2, "log.txt")).read()
    assert "Cache-driven BTD active" in log2
    assert "Using saved self-energy" in log2

