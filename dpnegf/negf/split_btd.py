"""This module contains a set of functions facilitating computations of
the block-tridiagonal structure of a band matrix.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from itertools import product
import math
import scipy

from numba import njit


def accum(accmap, input, func=None, size=None, fill_value=0, dtype=None):
    """An accumulation function similar to Matlab's `accumarray` function.

    Parameters
    ----------
    accmap : ndarray
        This is the "accumulation map".  It maps input (i.e. indices into
        `a`) to their destination in the output array.  The first `a.ndim`
        dimensions of `accmap` must be the same as `a.shape`.  That is,
        `accmap.shape[:a.ndim]` must equal `a.shape`.  For example, if `a`
        has shape (15,4), then `accmap.shape[:2]` must equal (15,4).  In this
        case `accmap[i,j]` gives the index into the output array where
        element (i,j) of `a` is to be accumulated.  If the output is, say,
        a 2D, then `accmap` must have shape (15,4,2).  The value in the
        last dimension give indices into the output array. If the output is
        1D, then the shape of `accmap` can be either (15,4) or (15,4,1)
    input : ndarray
        The input data to be accumulated.
    func : callable or None
        The accumulation function.  The function will be passed a list
        of values from `a` to be accumulated.
        If None, numpy.sum is assumed. (Default value = None)
    size : ndarray or None
        The size of the output array.  If None, the size will be determined
        from `accmap`. (Default value = None)
    fill_value : scalar
        The default value for elements of the output array.
    dtype : numpy data type, or None
        The data type of the output array.  If None, the data type of
        `a` is used. (Default value = None)

    Returns
    -------


    """

    # Check for bad arguments and handle the defaults.
    if accmap.shape[:input.ndim] != input.shape:
        raise ValueError("The initial dimensions of accmap must be the same as a.shape")
    if func is None:
        func = np.sum
    if dtype is None:
        dtype = input.dtype
    if accmap.shape == input.shape:
        accmap = np.expand_dims(accmap, -1)
    adims = tuple(range(input.ndim))
    if size is None:
        size = 1 + np.squeeze(np.apply_over_axes(np.max, accmap, axes=adims))
    size = np.atleast_1d(size)

    # Create an array of python lists of values.
    vals = np.empty(size, dtype='O')
    for s in product(*[range(k) for k in size]):
        vals[s] = []
    for s in product(*[range(k) for k in input.shape]):
        indx = tuple(accmap[s])
        val = input[s]
        vals[indx].append(val)

    # Create the output array.
    out = np.empty(size, dtype=dtype)
    for s in product(*[range(k) for k in size]):
        if vals[s] == []:
            out[s] = fill_value
        else:
            out[s] = func(vals[s])

    return out


@njit(cache=True)
def _compute_blocks_into(left_block, right_block, edge, edge1, out, write):
    """Iterative core of `compute_blocks`, JIT-compiled.

    Writes the block sequence into `out[write:]` and returns the new write
    cursor. The original recursion peels one block off each end per level and
    narrows the working edges; this version walks the same recurrence with
    explicit `lo`/`hi` indices instead of slicing.

    Note: the legacy code slices `edge[L:-R]` but `edge1[R:-L]` — the offsets
    are swapped — so we track the cumulative-L peel (`lo_e`) and cumulative-R
    peel (`lo_e1`) separately. `size = N - lo_e - lo_e1` is consistent for
    both.
    """
    if left_block < 1:
        left_block = 1
    if right_block < 1:
        right_block = 1

    N = edge.shape[0]
    lo_e = 0      # cumulative left-side peel — indexes into `edge`
    lo_e1 = 0     # cumulative right-side peel — indexes into `edge1`
    front = write
    back = out.shape[0]

    while True:
        size = N - lo_e - lo_e1
        if left_block + right_block < size:
            new_left_block = edge[lo_e + left_block - 1] - left_block - lo_e
            new_right_block = edge1[lo_e1 + right_block - 1] - right_block - lo_e1

            cond_a = left_block + new_left_block <= size - right_block
            cond_b = size - right_block - new_right_block >= left_block

            if cond_a and cond_b:
                out[front] = left_block
                front += 1
                back -= 1
                out[back] = right_block
                lo_e += left_block
                lo_e1 += right_block
                if new_left_block < 1:
                    left_block = 1
                else:
                    left_block = new_left_block
                if new_right_block < 1:
                    right_block = 1
                else:
                    right_block = new_right_block
                continue

            if new_left_block > new_right_block:
                out[front] = left_block
                out[front + 1] = size - left_block
                front += 2
            else:
                out[front] = size - right_block
                out[front + 1] = right_block
                front += 2
        elif left_block + right_block == size:
            out[front] = left_block
            out[front + 1] = right_block
            front += 2
        else:
            out[front] = size
            front += 1

        # The back tail at out[back:end] is already in the right order
        # (innermost first, outermost last) because we wrote at out[--back]
        # in outer-to-inner peel order. Copy forward.
        n_back = out.shape[0] - back
        for i in range(n_back):
            out[front + i] = out[back + i]
        return front + n_back


@njit(cache=True)
def _find_optimal_cut_core(edge, edge1, left, right):
    """JIT core for `find_optimal_cut`: sweeps candidate split points, runs
    `_compute_blocks_into` for each, picks the cube-sum-minimizing split.

    Returns (best_blocks, n_blocks, best_sep, right_block, left_block,
    found). `found` is 0 when the candidate set is empty (legacy caller maps
    that to sep=NaN).
    """
    size = edge.shape[0]
    n_candidates = size - right + 1 - left
    if n_candidates <= 0:
        return np.empty(0, dtype=np.int64), 0, 0, 0, 0, 0

    arange_n = np.arange(size, dtype=np.int64)
    edge_m_idx = edge - arange_n
    edge1_m_idx = edge1 - arange_n

    # An upper bound on the block-list length for the join of `block1` and
    # reversed `block2`. Each call returns at most `size + 2` entries; two
    # of them give a comfortable upper bound.
    buf_cap = 2 * (size + 4)
    buf1 = np.empty(buf_cap, dtype=np.int64)
    buf2 = np.empty(buf_cap, dtype=np.int64)
    best_buf = np.empty(buf_cap, dtype=np.int64)

    best_metric = np.int64(-1)
    best_sep = np.int64(0)
    best_right_block = np.int64(0)
    best_left_block = np.int64(0)
    best_n = np.int64(0)

    for item1 in range(left, size - right + 1):
        item2 = size - item1

        # Reconstruct edge_2 = (edge1 - arange_n)[item2:] + arange(item1).
        edge_2 = np.empty(item1, dtype=np.int64)
        for k in range(item1):
            edge_2[k] = edge1_m_idx[item2 + k] + k
        edge_4 = np.empty(item2, dtype=np.int64)
        for k in range(item2):
            edge_4[k] = edge_m_idx[item1 + k] + k

        # block1 = compute_blocks(left, edge1_m_idx[item2], edge[:item1], edge_2)
        n1 = _compute_blocks_into(left, edge1_m_idx[item2], edge[:item1], edge_2, buf1, 0)
        # block2 = compute_blocks(right, edge_m_idx[item1], edge1[:item2], edge_4)
        n2 = _compute_blocks_into(right, edge_m_idx[item1], edge1[:item2], edge_4, buf2, 0)

        # metric = sum( (block1 + reversed(block2))^3 )
        metric = np.int64(0)
        for k in range(n1):
            v = buf1[k]
            metric += v * v * v
        for k in range(n2):
            v = buf2[k]
            metric += v * v * v

        if best_metric < 0 or metric < best_metric:
            best_metric = metric
            best_sep = item1
            best_right_block = buf1[n1 - 1]
            best_left_block = buf2[n2 - 1]
            # Stash block1 + reversed(block2) into best_buf.
            for k in range(n1):
                best_buf[k] = buf1[k]
            for k in range(n2):
                best_buf[n1 + k] = buf2[n2 - 1 - k]
            best_n = n1 + n2

    # Filter zeros — legacy `[item for item in blocks if item != 0]`.
    out = np.empty(best_n, dtype=np.int64)
    w = 0
    for k in range(best_n):
        if best_buf[k] != 0:
            out[w] = best_buf[k]
            w += 1
    return out[:w], w, best_sep, best_right_block, best_left_block, 1


def cut_in_blocks(h_0, blocks):
    """Cut a matrix into diagonal, upper-diagonal and lower-diagonal blocks
    if sizes of the diagonal blocks are specified.

    Parameters
    ----------
    h_0 : ndarray
        Input matrix
    blocks : ndarray(dtype=int)
        Sizes of diagonal blocks

    Returns
    -------
    h_0_s, h_l_s, h_r_s : ndarray
        List of diagonal matrices,
        list of lower-diagonal matrices and
        list of upper-diagonal matrices.
        Note that if the size of the list h_0_s is N,
        the sizes of h_l_s, h_r_s are N-1.

    Examples
    --------
    >>> import numpy as np
    >>> from nanonet.tb.block_tridiagonalization import cut_in_blocks
    >>> a = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]])
    >>> a
    array([[1, 1, 0, 0],
           [1, 1, 1, 0],
           [0, 1, 1, 1],
           [0, 0, 1, 1]])
    >>> # Sum the diagonals.
    >>> blocks = [2, 2]
    >>> blocks
    [2, 2]
    >>> h0, h1, h2 = cut_in_blocks(a, blocks)
    >>> h0
    [array([[1, 1],
           [1, 1]]), array([[1, 1],
           [1, 1]])]
    >>> h1
    [array([[0, 1],
           [0, 0]])]
    >>> h2
    [array([[0, 0],
           [1, 0]])]
    """

    j1 = 0

    h_0_s = []
    h_l_s = []
    h_r_s = []

    for j, block in enumerate(blocks):
        h_0_s.append(h_0[j1:block + j1, j1:block + j1])
        if j < len(blocks) - 1:
            h_l_s.append(h_0[block + j1:block + j1 + blocks[j + 1], j1:block + j1])
            h_r_s.append(h_0[j1:block + j1, j1 + block:j1 + block + blocks[j + 1]])
        j1 += block

    return h_0_s, h_l_s, h_r_s


def find_optimal_cut(edge, edge1, left, right):
    """Computes the index corresponding to the optimal cut such that applying
    the function compute_blocks() to the sub-blocks defined by the cut reduces
    the cost function comparing to the case when the function compute_blocks() is
    applied to the whole matrix. If cutting point can not be find, the algorithm returns
    the result from the function compute_blocks().

    Parameters
    ----------
    edge : ndarray
        sparsity pattern profile of the matrix
    edge1 : ndarray
        conjugated sparsity pattern profile of the matrix
    left : int
        size of the leftmost diagonal block
    right : int
        size of the rightmost diagonal block

    Returns
    -------


    """

    edge_arr = np.ascontiguousarray(edge, dtype=np.int64)
    edge1_arr = np.ascontiguousarray(edge1, dtype=np.int64)
    out, _, sep, right_block, left_block, found = _find_optimal_cut_core(
        edge_arr, edge1_arr, int(left), int(right)
    )
    if not found:
        return [left, right], np.nan, 0, 0
    return out.tolist(), int(sep), int(right_block), int(left_block)


def compute_blocks_optimized(edge, edge1, left=1, right=1):
    """Computes optimal sizes of diagonal blocks of a matrix whose
    sparsity pattern is defined by the sparsity pattern profiles edge and edge1.
    This function is based on the algorithm which uses defined above function
    find_optimal_cut() to subdivide the problem into sub-problems in a optimal way
    according to some cost function.

    Parameters
    ----------
    edge : ndarray
        sparsity pattern profile of the matrix
    edge1 : ndarray
        conjugated sparsity pattern profile of the matrix
    left : int
        size of the leftmost diagonal block (constrained) (Default value = 1)
    right : int
        size of the rightmost diagonal block (constrained) (Default value = 1)

    Returns
    -------


    """

    blocks, sep, right_block, left_block = find_optimal_cut(edge, edge1, left=left, right=right)
    flag = False

    if not math.isnan(sep):

        # print(left, right_block, sep)

        if left + right_block < sep:

            edge_1 = edge[:sep]
            # edge_1[edge_1 > sep] = sep
            edge_2 = (edge1 - np.arange(len(edge1)))[-sep:] + np.arange(sep)

            blocks1 = compute_blocks_optimized(edge_1, edge_2, left=left, right=right_block)

        elif left + right_block == sep:

            blocks1 = [left, right_block]
        else:

            flag = True

        # print(left_block, right, len(edge) - sep)

        if right + left_block < len(edge) - sep:

            edge_3 = (edge - np.arange(len(edge)))[sep:] + np.arange(len(edge) - sep)
            edge_4 = edge1[:-sep]
            # edge_4[edge_4 > len(edge) - sep] = len(edge) - sep

            blocks2 = compute_blocks_optimized(edge_3, edge_4, left=left_block, right=right)

        elif right + left_block == len(edge) - sep:
            blocks2 = [left_block, right]
        else:
            flag = True

        if flag:
            return blocks
        else:
            blocks = blocks1 + blocks2

            return blocks


def find_nonzero_lines(mat, order):
    """

    Parameters
    ----------
    mat :

    order :


    Returns
    -------

    """

    if scipy.sparse.issparse(mat):
        lines = _find_nonzero_lines_sparse(mat, order)
    else:
        lines = _find_nonzero_lines(mat, order)

    if lines == max(mat.shape[0], mat.shape[1]) - 1:
        lines = 1
    if lines == 0:
        lines = 1

    return lines


def _find_nonzero_lines(mat, order):
    """

    Parameters
    ----------
    mat :

    order :


    Returns
    -------

    """
    if order == 'top':
        line = mat.shape[0]
        while line > 0:
            if np.count_nonzero(mat[line - 1, :]) == 0:
                line -= 1
            else:
                break
    elif order == 'bottom':
        line = -1
        while line < mat.shape[0] - 1:
            if np.count_nonzero(mat[line + 1, :]) == 0:
                line += 1
            else:
                line = mat.shape[0] - (line + 1)
                break
    elif order == 'left':
        line = mat.shape[1]
        while line > 0:
            if np.count_nonzero(mat[:, line - 1]) == 0:
                line -= 1
            else:
                break
    elif order == 'right':
        line = -1
        while line < mat.shape[1] - 1:
            if np.count_nonzero(mat[:, line + 1]) == 0:
                line += 1
            else:
                line = mat.shape[1] - (line + 1)
                break
    else:
        raise ValueError('Wrong value of the parameter order')

    return line


def _find_nonzero_lines_sparse(mat, order):
    """

    Parameters
    ----------
    mat :

    order :


    Returns
    -------

    """
    if order == 'top':
        line = mat.shape[0]
        while line > 0:
            if np.count_nonzero(mat[line - 1, :].todense()) == 0:
                line -= 1
            else:
                break
    elif order == 'bottom':
        line = -1
        while line < mat.shape[0] - 1:
            if np.count_nonzero(mat[line + 1, :].todense()) == 0:
                line += 1
            else:
                line = mat.shape[0] - (line + 1)
                break
    elif order == 'left':
        line = mat.shape[1]
        while line > 0:
            if np.count_nonzero(mat[:, line - 1].todense()) == 0:
                line -= 1
            else:
                break
    elif order == 'right':
        line = -1
        while line < mat.shape[1] - 1:
            if np.count_nonzero(mat[:, line + 1].todense()) == 0:
                line += 1
            else:
                line = mat.shape[1] - (line + 1)
                break
    else:
        raise ValueError('Wrong value of the parameter order')

    return line


def split_into_subblocks_optimized(h_0, left=1, right=1):
    """

    Parameters
    ----------
    h_0 :
        param left:
    right :
        return: (Default value = 1)
    left :
         (Default value = 1)

    Returns
    -------

    """

    if not (isinstance(left, int) and isinstance(right, int)):
        h_r_h = find_nonzero_lines(right, 'bottom')
        h_r_v = find_nonzero_lines(right[-h_r_h:, :], 'left')
        h_l_h = find_nonzero_lines(left, 'top')
        h_l_v = find_nonzero_lines(left[:h_l_h, :], 'right')
        left = max(h_l_h, h_r_v)
        right = max(h_r_h, h_l_v)

    if left + right > h_0.shape[0]:
        return [h_0.shape[0]]
    else:
        edge, edge1 = compute_edge(h_0)
        return compute_blocks_optimized(edge, edge1, left=left, right=right)


def split_into_subblocks(h_0, h_l, h_r):
    """Split Hamiltonian matrix and coupling matrices into subblocks

    Parameters
    ----------
    h_0 :
        Hamiltonian matrix
    h_l :
        left inter-cell coupling matrices
    h_r :
        right inter-cell coupling matrices
        :return h_0_s, h_l_s, h_r_s:    lists of subblocks

    Returns
    -------

    """

    if isinstance(h_l, np.ndarray) and isinstance(h_r, np.ndarray):
        h_r_h = find_nonzero_lines(h_r, 'bottom')
        h_r_v = find_nonzero_lines(h_r[-h_r_h:, :], 'left')
        h_l_h = find_nonzero_lines(h_l, 'top')
        h_l_v = find_nonzero_lines(h_l[:h_l_h, :], 'right')
        left_block = max(h_l_h, h_r_v)
        right_block = max(h_r_h, h_l_v)
    elif isinstance(h_l, int) and isinstance(h_r, int):
        left_block = h_l
        right_block = h_r
    else:
        raise TypeError

    edge, edge1 = compute_edge(h_0)

    blocks = compute_blocks(left_block, right_block, edge, edge1)

    return blocks


def compute_edge(mat):
    """Computes edges of the sparsity pattern of a matrix.

    Parameters
    ----------
    mat : ndarray
        Input matrix

    Returns
    -------
    edge : ndarray
        edge of the sparsity pattern
    edge1 : ndarray
        conjugate edge of the sparsity pattern

    Examples
    --------
    >>> import numpy as np
    >>> from nanonet.tb.block_tridiagonalization import compute_edge
    >>> input_matrix = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]])
    >>> input_matrix
    array([[1, 1, 0, 0],
           [1, 1, 1, 0],
           [0, 1, 1, 1],
           [0, 0, 1, 1]])
    >>> e1, e2 = compute_edge(input_matrix)
    >>> e1
    array([2, 3, 4, 4])
    >>> e2
    array([2, 3, 4, 4])
    >>> input_matrix = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]])
    >>> input_matrix
    array([[1, 0, 0, 0],
           [0, 1, 1, 0],
           [0, 1, 1, 1],
           [0, 0, 1, 1]])
    >>> e1, e2 = compute_edge(input_matrix)
    >>> e1
    array([1, 3, 4, 4])
    >>> e2
    array([2, 3, 3, 4])
    """

    if isinstance(mat, scipy.sparse.lil_matrix):
        row, col = mat.nonzero()
    else:
        row, col = np.where(mat != 0.0)

    n = mat.shape[0]
    edge = np.zeros(n, dtype=np.int64)
    np.maximum.at(edge, row, col + 1)
    edge = np.maximum.accumulate(edge)

    # Rotate 180°: row i, col j → row n-1-i, col n-1-j; max col + 1 = n - j.
    edge1 = np.zeros(n, dtype=np.int64)
    np.maximum.at(edge1, (n - 1) - row, n - col)
    edge1 = np.maximum.accumulate(edge1)

    return edge, edge1


def compute_blocks(left_block, right_block, edge, edge1):
    """This is an implementation of the greedy algorithm for
     computing block-tridiagonal representation of a matrix.
     The information regarding the input matrix is represented
     by the sparsity patters edges, `edge` and `edge1`.

    Parameters
    ----------
    left_block : int
        a predefined size of the leftmost block
    right_block : int
        a predefined size of the rightmost block
    edge : ndarray
        edge of sparsity pattern
    edge1 : ndarray
        conjugate edge of sparsity pattern

    Returns
    -------
    ans : list


    Examples
    --------
    >>> import numpy as np
    >>> from nanonet.tb.block_tridiagonalization import compute_edge
    >>> input_matrix = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]])
    >>> input_matrix
    array([[1, 1, 0, 0],
           [1, 1, 1, 0],
           [0, 1, 1, 1],
           [0, 0, 1, 1]])
    >>> e1, e2 = compute_edge(input_matrix)
    >>> compute_blocks(1, 1, e1, e2)
    [1, 1, 1, 1]
    >>> input_matrix = np.array([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 1], [0, 0, 1, 1]])
    >>> input_matrix
    array([[1, 1, 1, 0],
           [1, 1, 1, 0],
           [1, 1, 1, 1],
           [0, 0, 1, 1]])
    >>> e1, e2 = compute_edge(input_matrix)
    >>> compute_blocks(1, 1, e1, e2)
    [1, 2, 1]
    >>> e1, e2 = compute_edge(input_matrix)
    >>> compute_blocks(2, 2, e1, e2)
    [2, 2]
    """

    left_block = max(1, left_block)
    right_block = max(1, right_block)

    edge_arr = np.ascontiguousarray(edge, dtype=np.int64)
    edge1_arr = np.ascontiguousarray(edge1, dtype=np.int64)

    # Upper bound on the block-list length: each peel writes 2 entries and
    # the terminating step writes at most 2 more. Allocate `size + 2` and
    # let the JIT core return the actual length.
    size = edge_arr.shape[0]
    out = np.empty(size + 2, dtype=np.int64)
    n = _compute_blocks_into(left_block, right_block, edge_arr, edge1_arr, out, 0)
    return out[:n].tolist()


def constrained_subblocks(HK0, SK0, leftmost_size:int, rightmost_size:int):
    """Partition the device into block-tridiagonal subblocks with fixed edges.

    Cache-driven BTD requires the first and last subblocks to keep exactly
    ``leftmost_size`` / ``rightmost_size`` so the reused lead self-energy stays
    dimension-compatible (see :meth:`LeadProperty.HDL_reduced`). Only the
    interior is grown freely.

    The greedy splitter :func:`compute_blocks` is run on the combined
    ``|H| + |S|`` sparsity with the requested edge sizes as the leftmost /
    rightmost blocks. Unlike the optimized splitter, it honours the requested
    outer sizes *exactly* whenever they are compatible with the sparsity: it
    pins ``blocks[0] == nL`` and, walking from the right, ``blocks[-1] == nR``.
    When the requested size is too small to contain the block's coupling, the
    greedy algorithm grows that edge block instead — which we detect and reject,
    because the cached self-energy would then be dimension-incompatible.

    Parameters
    ----------
    HK0, SK0 : torch.Tensor
        Gamma-point Hamiltonian / overlap of the device, shape (D, D).
    leftmost_size, rightmost_size : int
        The exact required first / last block sizes.

    Returns
    -------
    list[int]
        Subblock sizes with ``subblocks[0] == leftmost_size`` and
        ``subblocks[-1] == rightmost_size`` summing to ``D``.

    Raises
    ------
    ValueError
        If the requested edge sizes cannot form a valid block-tridiagonal
        layout (too large for the device, no room for the interior, the greedy
        splitter could not honour an edge size, or the couplings extend beyond
        neighbouring blocks).
    """
    D = int(HK0.shape[0])
    nL = int(leftmost_size)
    nR = int(rightmost_size)

    if nL <= 0 or nR <= 0:
        raise ValueError(f"Cache-driven BTD edge sizes must be positive, got leftmost={nL}, rightmost={nR}.")
    if nL > D or nR > D:
        raise ValueError(f"Cache-driven BTD edge sizes (leftmost={nL}, rightmost={nR}) exceed the device basis size {D}.")

    # Build the combined sparsity mask once; used for the greedy split and the
    # final block-tridiagonality validation.
    mask = (HK0.abs() + SK0.abs()) != 0

    if nL + nR >= D:
        # No interior room: the device is (at most) the two edge blocks. Valid
        # only if they exactly tile the device (BTD is trivial for <=2 blocks).
        if nL + nR != D:
            raise ValueError(
                f"Cache-driven BTD edge sizes (leftmost={nL}, rightmost={nR}) leave "
                f"no room for the interior and do not tile the device basis {D}.")
        subblocks = [nL, nR]
    else:
        edge, edge1 = compute_edge(mask.detach().cpu().numpy())
        subblocks = [int(b) for b in compute_blocks(nL, nR, edge, edge1)]
        # The greedy splitter grows an edge block when the requested size cannot
        # contain its coupling. A grown edge means the cached self-energy shape
        # would not match the first/last Hamiltonian block, so reject it.
        if subblocks[0] != nL:
            raise ValueError(
                f"Cache-driven leftmost block size {nL} is incompatible with the "
                f"device sparsity: the block-tridiagonal splitter requires at least "
                f"{subblocks[0]} orbitals in the first block. The cached self-energy "
                f"cannot be reused for this device.")
        if subblocks[-1] != nR:
            raise ValueError(
                f"Cache-driven rightmost block size {nR} is incompatible with the "
                f"device sparsity: the block-tridiagonal splitter requires at least "
                f"{subblocks[-1]} orbitals in the last block. The cached self-energy "
                f"cannot be reused for this device.")

    subblocks = [int(b) for b in subblocks]
    if sum(subblocks) != D:
        raise ValueError(
            f"Cache-driven BTD partition {subblocks} sums to {sum(subblocks)} but the "
            f"device basis size is {D}.")

    if not validate_block_tridiagonal(mask, subblocks):
        raise ValueError(
            f"Cache-driven BTD partition with fixed edges (leftmost={nL}, "
            f"rightmost={nR}) is not block-tridiagonal: the Hamiltonian/overlap "
            f"couples non-neighbouring blocks. The cached self-energy edge sizes "
            f"are incompatible with the current device sparsity.")

    return subblocks


def validate_block_tridiagonal(mask, subblocks) -> bool:
    """Check that ``mask`` is block-tridiagonal under the given partition.

    A partition is block-tridiagonal iff no nonzero entry couples blocks that
    are more than one apart. Using the per-row sparsity profile (last nonzero
    column per row, an O(D) quantity, cf. :func:`compute_edge`), a
    row belonging to block ``i`` must not reach past the end of block ``i+1``,
    and by symmetry of the coupling structure this also bounds the lower part.

    Parameters
    ----------
    mask : torch.Tensor
        Boolean / 0-1 matrix marking nonzero entries of ``|H| + |S|``.
    subblocks : list[int]
        Candidate diagonal block sizes.

    Returns
    -------
    bool
        True if the layout is block-tridiagonal.
    """
    mat = mask.detach().cpu().numpy()
    edge, edge1 = compute_edge(mat)  # edge[i]: 1 + last nonzero col in row i
    bounds = np.cumsum([0] + list(subblocks))  # block boundaries
    for bi in range(len(subblocks)):
        r0, r1 = bounds[bi], bounds[bi + 1]
        # Rightmost column any row in this block may reach: end of next block.
        allowed = bounds[min(bi + 2, len(subblocks))]
        if r1 > r0 and int(edge[r0:r1].max()) > allowed:
            return False
        # Lower part: leftmost column reachable is start of previous block.
        # edge1 is the 180-degree-rotated profile; reuse the same bound from
        # the other side to catch couplings below the sub-diagonal blocks.
        allowed_lo = bounds[len(subblocks)] - bounds[max(bi - 1, 0)]
        rr0 = bounds[len(subblocks)] - r1
        rr1 = bounds[len(subblocks)] - r0
        if rr1 > rr0 and int(edge1[rr0:rr1].max()) > allowed_lo:
            return False
    return True


def show_blocks(subblocks, input_mat, results_path):
    """This is a script for visualizing the sparsity pattern and
     a block-tridiagonal structure of a matrix.

    Parameters
    ----------
    subblocks :

    input_mat :


    Returns
    -------


    """

    cumsum = np.cumsum(np.array(subblocks))[:-1]
    cumsum = np.insert(cumsum, 0, 0)

    fig, ax = plt.subplots(1)
    plt.spy(input_mat, markersize=0.9, c='k')
    # plt.plot(edge)

    for jj in range(2):
        cumsum = cumsum + jj * input_mat.shape[0]

        if jj == 1:
            rect = Rectangle((input_mat.shape[0] - subblocks[-1] - 0.5, input_mat.shape[1] - 0.5),
                             subblocks[-1], subblocks[0],
                             linestyle='--',
                             linewidth=1.3,
                             edgecolor='b',
                             facecolor='none', zorder=200)
            ax.add_patch(rect)
            rect = Rectangle((input_mat.shape[0] - 0.5, input_mat.shape[1] - subblocks[-1] - 0.5),
                             subblocks[0], subblocks[-1],
                             linestyle='--',
                             linewidth=1.3,
                             edgecolor='g',
                             facecolor='none', zorder=200)
            ax.add_patch(rect)

        for j, item in enumerate(cumsum):
            if j < len(cumsum) - 1:
                rect = Rectangle((item - 0.5, cumsum[j + 1] - 0.5), subblocks[j], subblocks[j + 1],
                                 linewidth=1.3,
                                 edgecolor='b',
                                 facecolor='none', zorder=200)
                ax.add_patch(rect)
                rect = Rectangle((cumsum[j + 1] - 0.5, item - 0.5), subblocks[j + 1], subblocks[j],
                                 linewidth=1.3,
                                 edgecolor='g',
                                 facecolor='none', zorder=200)
                ax.add_patch(rect)
            rect = Rectangle((item - 0.5, item - 0.5), subblocks[j], subblocks[j],
                             linewidth=1.3,
                             edgecolor='r',
                             facecolor='none', zorder=200)
            ax.add_patch(rect)

    plt.xlim(input_mat.shape[0] - 0.5, -1.0)
    plt.ylim(-1.0, input_mat.shape[0] - 0.5)
    plt.axis('off')
    plt.savefig(results_path +'/subblocks_HK0.png', dpi=300)
    plt.close()

# if __name__ == "__main__":
#     import doctest

#     doctest.testmod()
