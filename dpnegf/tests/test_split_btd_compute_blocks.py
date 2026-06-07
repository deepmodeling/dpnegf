"""Equivalence test for the refactored block-tridiagonalization helpers in
`dpnegf/negf/split_btd.py`.

The previous implementations were pure-Python recursive (`compute_blocks`) and
a Python loop with per-iteration NumPy round-trips (`find_optimal_cut`). The
refactor converts `compute_blocks` to iteration, hoists invariants in
`find_optimal_cut`, and optionally `@njit`-decorates both. This file pins
exact equality between the new and legacy implementations on a battery of
random sparse matrices, the docstring examples, and an NEGF-like Hamiltonian
sparsity pattern.
"""

import math

import numpy as np
import pytest
from itertools import product

from dpnegf.negf.split_btd import (
    compute_blocks,
    compute_blocks_optimized,
    compute_edge,
    find_optimal_cut,
    split_into_subblocks,
    split_into_subblocks_optimized,
)


# ---------------------------------------------------------------------------
# Legacy oracles — verbatim copies of the pre-refactor implementations.
# Keep these byte-for-byte equivalent to the original so failures are easy to
# attribute. Do not "clean up".
# ---------------------------------------------------------------------------


def _compute_blocks_legacy(left_block, right_block, edge, edge1):
    size = len(edge)
    left_block = max(1, left_block)
    right_block = max(1, right_block)

    if left_block + right_block < size:
        new_left_block = edge[left_block - 1] - left_block
        new_right_block = edge1[right_block - 1] - right_block

        if left_block + new_left_block <= size - right_block and \
                size - right_block - new_right_block >= left_block:
            blocks = _compute_blocks_legacy(
                new_left_block,
                new_right_block,
                edge[left_block:-right_block] - left_block,
                edge1[right_block:-left_block] - right_block,
            )
            return [left_block] + blocks + [right_block]
        else:
            if new_left_block > new_right_block:
                return [left_block] + [size - left_block]
            else:
                return [size - right_block] + [right_block]
    elif left_block + right_block == size:
        return [left_block] + [right_block]
    else:
        return [size]


def _find_optimal_cut_legacy(edge, edge1, left, right):
    unique_indices = np.arange(left, len(edge) - right + 1)
    blocks_all = []
    seps = []
    sizes = []
    metric = []
    size = len(edge)

    for j1, item1 in enumerate(unique_indices):
        seps.append(item1)
        item2 = size - item1

        edge_1 = edge[:item1]
        edge_2 = (edge1 - np.arange(len(edge1)))[item2:] + np.arange(item1)

        edge_3 = edge1[:item2]
        edge_4 = (edge - np.arange(len(edge)))[item1:] + np.arange(item2)

        block1 = _compute_blocks_legacy(
            left, (edge1 - np.arange(len(edge)))[item2], edge_1, edge_2
        )
        block2 = _compute_blocks_legacy(
            right, (edge - np.arange(len(edge1)))[item1], edge_3, edge_4
        )

        block = block1 + block2[::-1]
        blocks_all.append(block)
        metric.append(np.sum(np.array(block) ** 3))
        sizes.append((block1[-1], block2[-1]))

    if len(metric) == 0:
        return [left, right], np.nan, 0, 0
    else:
        best = int(np.argmin(np.array(metric)))
        blocks = [item for item in blocks_all[best] if item != 0]
        sep = seps[best]
        right_block, left_block = sizes[best]
        return blocks, sep, right_block, left_block


def _compute_blocks_optimized_legacy(edge, edge1, left=1, right=1):
    blocks, sep, right_block, left_block = _find_optimal_cut_legacy(
        edge, edge1, left=left, right=right
    )
    flag = False

    if not math.isnan(sep):
        if left + right_block < sep:
            edge_1 = edge[:sep]
            edge_2 = (edge1 - np.arange(len(edge1)))[-sep:] + np.arange(sep)
            blocks1 = _compute_blocks_optimized_legacy(
                edge_1, edge_2, left=left, right=right_block
            )
        elif left + right_block == sep:
            blocks1 = [left, right_block]
        else:
            flag = True

        if right + left_block < len(edge) - sep:
            edge_3 = (edge - np.arange(len(edge)))[sep:] + np.arange(len(edge) - sep)
            edge_4 = edge1[:-sep]
            blocks2 = _compute_blocks_optimized_legacy(
                edge_3, edge_4, left=left_block, right=right
            )
        elif right + left_block == len(edge) - sep:
            blocks2 = [left_block, right]
        else:
            flag = True

        if flag:
            return blocks
        else:
            return blocks1 + blocks2


def _split_into_subblocks_legacy(h_0, h_l, h_r):
    """Pre-refactor `split_into_subblocks`. Only the integer-arg branch is
    exercised here — the ndarray branch goes through `find_nonzero_lines`
    which is not under test."""
    if isinstance(h_l, int) and isinstance(h_r, int):
        left_block = h_l
        right_block = h_r
    else:
        raise TypeError
    edge, edge1 = compute_edge(h_0)
    return _compute_blocks_legacy(left_block, right_block, edge, edge1)


# ---------------------------------------------------------------------------
# Test inputs
# ---------------------------------------------------------------------------


def _random_matrix_with_full_diagonal(n, density, seed):
    """Square binary matrix with a populated diagonal so every row has a nonzero.

    This matches the tight-binding Hamiltonian use case (on-site terms always
    present) and gives deterministic, non-trivial sparsity patterns.
    """
    rng = np.random.default_rng(seed)
    mat = (rng.random((n, n)) < density).astype(np.float64)
    mat[np.arange(n), np.arange(n)] = 1.0
    return mat


def _cnt_like_matrix(n, coupling_density, seed):
    """Tridiagonal Hamiltonian + symmetric random long-range couplings.

    Mirrors the CNT NEGF use case. This is the input class that exposed the
    `_compute_blocks_into` `lo_e` / `lo_e1` asymmetry bug during Tier B
    development — the random `_random_matrix_with_full_diagonal` inputs
    rarely produce sequences of peels where `left_peel != right_peel`, but
    sparse long-range couplings on top of a tridiagonal do.
    """
    rng = np.random.default_rng(seed)
    mat = np.zeros((n, n))
    idx = np.arange(n)
    mat[idx, idx] = 1.0
    mat[idx[:-1], idx[1:]] = 1.0
    mat[idx[1:], idx[:-1]] = 1.0
    n_couplings = max(1, int(coupling_density * n))
    for _ in range(n_couplings):
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        mat[i, j] = 1.0
        mat[j, i] = 1.0
    return mat


def _banded_matrix(n, bandwidth):
    """Symmetric matrix with nonzeros only within ±bandwidth of the diagonal.

    Banded patterns produce deterministic block lists and exercise the
    candidate-sweep in `find_optimal_cut` across a clean range.
    """
    mat = np.zeros((n, n))
    for offset in range(-bandwidth, bandwidth + 1):
        if offset >= 0:
            i = np.arange(n - offset)
            j = i + offset
        else:
            j = np.arange(n + offset)
            i = j - offset
        mat[i, j] = 1.0
    return mat


SIZES = [8, 64, 512]
DENSITIES = [0.01, 0.1, 0.5]
SEEDS = [0, 1, 2]


# ---------------------------------------------------------------------------
# compute_blocks equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,density,seed", list(product(SIZES, DENSITIES, SEEDS)))
@pytest.mark.parametrize("left,right", [(1, 1), (2, 2)])
def test_compute_blocks_matches_legacy(n, density, seed, left, right):
    mat = _random_matrix_with_full_diagonal(n, density, seed)
    edge, edge1 = compute_edge(mat)
    new = compute_blocks(left, right, edge, edge1)
    old = _compute_blocks_legacy(left, right, edge, edge1)
    assert new == old


def test_compute_blocks_docstring_example_1():
    mat = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]])
    edge, edge1 = compute_edge(mat)
    assert compute_blocks(1, 1, edge, edge1) == [1, 1, 1, 1]


def test_compute_blocks_docstring_example_2():
    mat = np.array([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 1], [0, 0, 1, 1]])
    edge, edge1 = compute_edge(mat)
    assert compute_blocks(1, 1, edge, edge1) == [1, 2, 1]
    assert compute_blocks(2, 2, edge, edge1) == [2, 2]


# ---------------------------------------------------------------------------
# find_optimal_cut equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,density,seed", list(product(SIZES, DENSITIES, SEEDS)))
def test_find_optimal_cut_matches_legacy(n, density, seed):
    mat = _random_matrix_with_full_diagonal(n, density, seed)
    edge, edge1 = compute_edge(mat)

    blocks_new, sep_new, rb_new, lb_new = find_optimal_cut(edge, edge1, 1, 1)
    blocks_old, sep_old, rb_old, lb_old = _find_optimal_cut_legacy(edge, edge1, 1, 1)

    assert blocks_new == blocks_old
    # sep can be NaN when the candidate set is empty; compare both branches.
    if isinstance(sep_old, float) and math.isnan(sep_old):
        assert isinstance(sep_new, float) and math.isnan(sep_new)
    else:
        assert sep_new == sep_old
    assert rb_new == rb_old
    assert lb_new == lb_old


# ---------------------------------------------------------------------------
# compute_blocks_optimized equivalence — the public path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,density,seed", list(product(SIZES, DENSITIES, SEEDS)))
def test_compute_blocks_optimized_matches_legacy(n, density, seed):
    mat = _random_matrix_with_full_diagonal(n, density, seed)
    edge, edge1 = compute_edge(mat)
    new = compute_blocks_optimized(edge, edge1, left=1, right=1)
    old = _compute_blocks_optimized_legacy(edge, edge1, left=1, right=1)
    assert new == old


@pytest.mark.parametrize("n,density,seed", list(product(SIZES, DENSITIES, SEEDS)))
def test_split_into_subblocks_optimized_sums_to_n(n, density, seed):
    """The block list must always tile the matrix exactly."""
    mat = _random_matrix_with_full_diagonal(n, density, seed)
    blocks = split_into_subblocks_optimized(mat, left=1, right=1)
    assert sum(blocks) == n


# ---------------------------------------------------------------------------
# NEGF-like sparsity pattern — tridiagonal + a couple of long-range couplings.
# Mirrors what compute_blocks_optimized actually sees during the NEGF run.
# ---------------------------------------------------------------------------


def test_compute_blocks_optimized_negf_like_pattern():
    n = 32
    mat = np.zeros((n, n))
    idx = np.arange(n)
    mat[idx, idx] = 1.0
    mat[idx[:-1], idx[1:]] = 1.0
    mat[idx[1:], idx[:-1]] = 1.0
    mat[3, 17] = 1.0
    mat[17, 3] = 1.0
    mat[8, 25] = 1.0
    mat[25, 8] = 1.0

    edge, edge1 = compute_edge(mat)
    new = compute_blocks_optimized(edge, edge1, left=1, right=1)
    old = _compute_blocks_optimized_legacy(edge, edge1, left=1, right=1)
    assert new == old


def test_compute_blocks_optimized_pure_tridiagonal():
    n = 64
    mat = np.zeros((n, n))
    idx = np.arange(n)
    mat[idx, idx] = 1.0
    mat[idx[:-1], idx[1:]] = 1.0
    mat[idx[1:], idx[:-1]] = 1.0

    edge, edge1 = compute_edge(mat)
    new = compute_blocks_optimized(edge, edge1, left=1, right=1)
    old = _compute_blocks_optimized_legacy(edge, edge1, left=1, right=1)
    assert new == old
    assert sum(new) == n


# ---------------------------------------------------------------------------
# Asymmetric-peel coverage — CNT-like inputs.
#
# These trigger consecutive recursion levels where `new_left_block` differs
# from `new_right_block`. A Tier B JIT bug (single cumulative offset shared
# between `edge` and `edge1`, instead of the legacy's asymmetric `edge[L:-R]`
# vs `edge1[R:-L]` slicing) slipped through every existing random sweep but
# fires reliably here. Keep these tests load-bearing.
# ---------------------------------------------------------------------------


CNT_SIZES = [256, 1024]
CNT_COUPLING_DENSITIES = [0.005, 0.02, 0.05]


@pytest.mark.parametrize(
    "n,coupling_density,seed",
    list(product(CNT_SIZES, CNT_COUPLING_DENSITIES, SEEDS)),
)
def test_compute_blocks_optimized_cnt_like(n, coupling_density, seed):
    mat = _cnt_like_matrix(n, coupling_density, seed)
    edge, edge1 = compute_edge(mat)
    new = compute_blocks_optimized(edge, edge1, left=1, right=1)
    old = _compute_blocks_optimized_legacy(edge, edge1, left=1, right=1)
    assert new == old
    assert sum(new) == n


@pytest.mark.parametrize("seed", SEEDS)
def test_compute_blocks_cnt_like_direct(seed):
    """Direct test on `compute_blocks` (the JIT core's most exposed surface),
    bypassing `find_optimal_cut`'s candidate sweep so any asymmetric-peel
    failure points straight at `_compute_blocks_into`. One size is enough —
    the optimized sweep above covers the broader matrix."""
    mat = _cnt_like_matrix(512, coupling_density=0.02, seed=seed)
    edge, edge1 = compute_edge(mat)
    new = compute_blocks(1, 1, edge, edge1)
    old = _compute_blocks_legacy(1, 1, edge, edge1)
    assert new == old


# ---------------------------------------------------------------------------
# Non-trivial (left, right) constraints — exercises the asymmetric-constraint
# path that `split_into_subblocks_optimized` reaches when `find_nonzero_lines`
# yields large left/right block requirements.
# ---------------------------------------------------------------------------


LEFT_RIGHT_COMBOS = [(1, 1), (1, 3), (3, 1), (2, 5), (5, 2), (4, 4)]


@pytest.mark.parametrize(
    "n,density,seed", list(product([64, 256], [0.05, 0.2], SEEDS))
)
@pytest.mark.parametrize("left,right", LEFT_RIGHT_COMBOS)
def test_compute_blocks_optimized_left_right_combos(n, density, seed, left, right):
    mat = _random_matrix_with_full_diagonal(n, density, seed)
    edge, edge1 = compute_edge(mat)
    if left + right > n:
        pytest.skip("constraints larger than matrix")
    new = compute_blocks_optimized(edge, edge1, left=left, right=right)
    old = _compute_blocks_optimized_legacy(edge, edge1, left=left, right=right)
    assert new == old


# ---------------------------------------------------------------------------
# Banded matrices — fully-deterministic block-list structure, exercises the
# candidate sweep in find_optimal_cut across a clean range.
# ---------------------------------------------------------------------------


BANDED_SIZES = [64, 256]
BANDWIDTHS = [1, 2, 4, 8, 16]


@pytest.mark.parametrize("n,bandwidth", list(product(BANDED_SIZES, BANDWIDTHS)))
def test_compute_blocks_optimized_banded(n, bandwidth):
    mat = _banded_matrix(n, bandwidth)
    edge, edge1 = compute_edge(mat)
    new = compute_blocks_optimized(edge, edge1, left=1, right=1)
    old = _compute_blocks_optimized_legacy(edge, edge1, left=1, right=1)
    assert new == old
    assert sum(new) == n
    assert all(b >= 1 for b in new)


@pytest.mark.parametrize("n,bandwidth", list(product(BANDED_SIZES, BANDWIDTHS)))
def test_compute_blocks_banded(n, bandwidth):
    mat = _banded_matrix(n, bandwidth)
    edge, edge1 = compute_edge(mat)
    new = compute_blocks(1, 1, edge, edge1)
    old = _compute_blocks_legacy(1, 1, edge, edge1)
    assert new == old


# ---------------------------------------------------------------------------
# `split_into_subblocks` (non-optimized variant) equivalence.
#
# Reachable via `negf_hamiltonian_init.py` as a fallback when the optimized
# variant is disabled. Was previously uncovered by this file.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n,density,seed", list(product([64, 256], DENSITIES, SEEDS))
)
@pytest.mark.parametrize("left,right", [(1, 1), (2, 3), (3, 2)])
def test_split_into_subblocks_matches_legacy(n, density, seed, left, right):
    mat = _random_matrix_with_full_diagonal(n, density, seed)
    if left + right > n:
        pytest.skip("constraints larger than matrix")
    new = split_into_subblocks(mat, left, right)
    old = _split_into_subblocks_legacy(mat, left, right)
    assert new == old


@pytest.mark.parametrize("n,bandwidth", [(64, 2), (256, 4), (256, 8)])
def test_split_into_subblocks_banded(n, bandwidth):
    mat = _banded_matrix(n, bandwidth)
    new = split_into_subblocks(mat, 1, 1)
    old = _split_into_subblocks_legacy(mat, 1, 1)
    assert new == old


@pytest.mark.parametrize("seed", SEEDS)
def test_split_into_subblocks_cnt_like(seed):
    mat = _cnt_like_matrix(512, 0.02, seed)
    new = split_into_subblocks(mat, 1, 1)
    old = _split_into_subblocks_legacy(mat, 1, 1)
    assert new == old


# ---------------------------------------------------------------------------
# Stress test — one larger random matrix to catch deep-recursion bugs that
# only show up past n ≈ 1000.
# ---------------------------------------------------------------------------


def test_compute_blocks_optimized_stress_large():
    n = 2000
    mat = _random_matrix_with_full_diagonal(n, density=0.005, seed=0)
    edge, edge1 = compute_edge(mat)
    new = compute_blocks_optimized(edge, edge1, left=1, right=1)
    old = _compute_blocks_optimized_legacy(edge, edge1, left=1, right=1)
    assert new == old
    assert sum(new) == n


# ---------------------------------------------------------------------------
# Trivial sizes — the JIT core has separate branches for `size == 1`,
# `L + R == size`, and `L + R > size`. These were not explicitly tested.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n,left,right",
    [
        (1, 1, 1),
        (2, 1, 1),
        (2, 1, 2),
        (2, 2, 1),
        (2, 2, 2),
        (3, 1, 1),
        (3, 1, 2),
        (3, 2, 1),
        (3, 2, 2),
        (3, 3, 3),
    ],
)
def test_compute_blocks_trivial_sizes(n, left, right):
    """Identity-matrix at trivial sizes — exercises terminal branches."""
    mat = np.eye(n)
    edge, edge1 = compute_edge(mat)
    new = compute_blocks(left, right, edge, edge1)
    old = _compute_blocks_legacy(left, right, edge, edge1)
    assert new == old
    assert sum(new) == n


@pytest.mark.parametrize("n", [1, 2, 3])
def test_compute_blocks_optimized_trivial_sizes(n):
    mat = np.eye(n)
    edge, edge1 = compute_edge(mat)
    # find_optimal_cut returns sep=NaN for these — compute_blocks_optimized
    # then returns None (legacy behavior). Verify new matches old, including
    # the None case.
    new = compute_blocks_optimized(edge, edge1, left=1, right=1)
    old = _compute_blocks_optimized_legacy(edge, edge1, left=1, right=1)
    assert new == old
