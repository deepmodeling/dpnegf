"""Equivalence test for the vectorized `compute_edge` in `dpnegf/negf/split_btd.py`.

The previous implementation used `accum()` (a Matlab-style accumarray with a
nested Python loop) inside `compute_edge`. The vectorized rewrite uses
`np.maximum.at` + `np.maximum.accumulate`. This file checks that the two
implementations produce identical `(edge, edge1)` outputs on a battery of
random sparse matrices and the docstring examples.
"""

import numpy as np
import pytest
import scipy.sparse
from itertools import product

from dpnegf.negf.split_btd import accum, compute_edge


def _compute_edge_legacy(mat):
    """The pre-refactor implementation, kept here as the oracle."""
    if isinstance(mat, scipy.sparse.lil_matrix):
        row, col = mat.nonzero()
    else:
        row, col = np.where(mat != 0.0)

    edge = accum(row, col, np.max) + 1
    edge[0] = max(0, edge[0])
    edge = np.maximum.accumulate(edge)

    edge1 = accum(np.max(row) - row[::-1], np.max(row) - col[::-1], np.max) + 1
    edge1[0] = max(0, edge1[0])
    edge1 = np.maximum.accumulate(edge1)

    return edge, edge1


def _random_matrix_with_full_diagonal(n, density, seed):
    """Square binary matrix with a populated diagonal so every row has a nonzero.

    This matches the Hamiltonian use case (tight-binding on-site terms always
    present) and aligns the legacy oracle's output size (`max(row)+1 == n`)
    with the new implementation's (`mat.shape[0] == n`).
    """
    rng = np.random.default_rng(seed)
    mat = (rng.random((n, n)) < density).astype(np.float64)
    mat[np.arange(n), np.arange(n)] = 1.0
    return mat


SIZES = [8, 64, 512]
DENSITIES = [0.01, 0.1, 0.5]
SEEDS = [0, 1, 2]


@pytest.mark.parametrize("n,density,seed", list(product(SIZES, DENSITIES, SEEDS)))
def test_compute_edge_dense_matches_legacy(n, density, seed):
    mat = _random_matrix_with_full_diagonal(n, density, seed)
    e1_new, e2_new = compute_edge(mat)
    e1_old, e2_old = _compute_edge_legacy(mat)
    assert np.array_equal(e1_new, e1_old)
    assert np.array_equal(e2_new, e2_old)


@pytest.mark.parametrize("n,density,seed", list(product(SIZES, DENSITIES, SEEDS)))
def test_compute_edge_sparse_lil_matches_legacy(n, density, seed):
    mat = _random_matrix_with_full_diagonal(n, density, seed)
    lil = scipy.sparse.lil_matrix(mat)
    e1_new, e2_new = compute_edge(lil)
    e1_old, e2_old = _compute_edge_legacy(lil)
    assert np.array_equal(e1_new, e1_old)
    assert np.array_equal(e2_new, e2_old)


def test_compute_edge_docstring_example_1():
    mat = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]])
    e1, e2 = compute_edge(mat)
    assert np.array_equal(e1, np.array([2, 3, 4, 4]))
    assert np.array_equal(e2, np.array([2, 3, 4, 4]))


def test_compute_edge_docstring_example_2():
    mat = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]])
    e1, e2 = compute_edge(mat)
    assert np.array_equal(e1, np.array([1, 3, 4, 4]))
    assert np.array_equal(e2, np.array([2, 3, 3, 4]))


def test_compute_edge_negf_like_pattern():
    """Tridiagonal Hamiltonian with a couple of long-range couplings — the
    shape that compute_edge actually sees during the NEGF run."""
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

    e1_new, e2_new = compute_edge(mat)
    e1_old, e2_old = _compute_edge_legacy(mat)
    assert np.array_equal(e1_new, e1_old)
    assert np.array_equal(e2_new, e2_old)
