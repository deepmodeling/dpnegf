"""Correctness + GPU smoke tests for the batched recursive Green's function kernel.

The kernel in ``dpnegf/negf/recursive_green_cal.py`` was rewritten to accept
a leading energy-batch dim. These tests pin the contract:

1. Batched call over B energies == stack of B scalar calls (allclose 1e-10).
2. Same inputs moved to CUDA produce the same result (allclose 1e-8).

The CUDA test is skipped on machines without a GPU.
"""

import pytest
import torch

from dpnegf.negf.recursive_green_cal import recursive_gf


def _make_btd_inputs(B, block_sizes, seed=0, device="cpu", dtype=torch.complex128):
    """Build a synthetic block-tridiagonal Hamiltonian + overlap and lead self-
    energies for B energies. Returns the lists in the shape ``recursive_gf``
    expects from a batched caller (k-blocks 2-D, energy-dep tensors leading-B)."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    def _randc(*shape):
        return (torch.randn(*shape, generator=g, dtype=torch.float64) +
                1j * torch.randn(*shape, generator=g, dtype=torch.float64)).to(dtype).to(device)

    N = len(block_sizes)

    hd, sd = [], []
    for n in block_sizes:
        h = _randc(n, n)
        h = 0.5 * (h + h.mH)  # hermitian diagonal block
        hd.append(h)
        s = torch.eye(n, dtype=dtype, device=device)
        sd.append(s)

    hl, hu, sl, su = [], [], [], []
    for q in range(N - 1):
        n_q, n_qp = block_sizes[q], block_sizes[q + 1]
        u = _randc(n_q, n_qp)
        hu.append(u)
        hl.append(u.mH)            # H_l = H_u^H so the full H is hermitian
        su.append(torch.zeros(n_q, n_qp, dtype=dtype, device=device))
        sl.append(torch.zeros(n_qp, n_q, dtype=dtype, device=device))

    n0, nN = block_sizes[0], block_sizes[-1]
    left_se = _randc(B, n0, n0)
    right_se = _randc(B, nN, nN)
    # batched energy grid; small imaginary offset stays inside the eta envelope
    energies = (torch.linspace(-1.0, 1.0, B, dtype=torch.float64, device=device)
                + 1j * 1e-4).to(dtype)

    return hd, sd, hl, hu, sl, su, left_se, right_se, energies


def _stack_scalar(hd, sd, hl, hu, sl, su, left_se, right_se, energies,
                  need_lesser=False, s_in_batched=None):
    """Drive the scalar path B times and stack the results along a new dim 0."""
    B = energies.shape[0]
    per_e = []
    for b in range(B):
        s_in_b = 0
        if need_lesser:
            s_in_b = [s_in_batched[q][b] for q in range(len(s_in_batched))]
        ans = recursive_gf(
            energy=energies[b],
            hl=hl, hd=hd, hu=hu, sd=sd, su=su, sl=sl,
            left_se=left_se[b], right_se=right_se[b],
            s_in=s_in_b,
            eta=1e-5,
            need_lesser=need_lesser,
            need_gr_lc=True,
        )
        per_e.append(ans)

    def _stack_field(idx):
        first = per_e[0][idx]
        if first is None:
            return None
        if torch.is_tensor(first):
            return torch.stack([per_e[b][idx] for b in range(B)], dim=0)
        # list of per-block tensors
        return [torch.stack([per_e[b][idx][q] for b in range(B)], dim=0)
                for q in range(len(first))]

    return tuple(_stack_field(i) for i in range(len(per_e[0])))


def _assert_ans_close(stacked, batched, atol):
    for s, b in zip(stacked, batched):
        if s is None:
            assert b is None
            continue
        if torch.is_tensor(s):
            assert torch.allclose(s, b, atol=atol), \
                f"tensor mismatch: max abs diff {(s - b).abs().max().item():.3e}"
        else:
            for sq, bq in zip(s, b):
                assert torch.allclose(sq, bq, atol=atol), \
                    f"list-elem mismatch: max abs diff {(sq - bq).abs().max().item():.3e}"


def test_batched_matches_scalar_retarded():
    B = 16
    block_sizes = [8, 6, 8]
    hd, sd, hl, hu, sl, su, left_se, right_se, energies = _make_btd_inputs(B, block_sizes)

    stacked = _stack_scalar(hd, sd, hl, hu, sl, su, left_se, right_se, energies,
                            need_lesser=False)

    batched = recursive_gf(
        energy=energies,
        hl=hl, hd=hd, hu=hu, sd=sd, su=su, sl=sl,
        left_se=left_se, right_se=right_se,
        s_in=0, eta=1e-5,
        need_lesser=False, need_gr_lc=True,
    )

    _assert_ans_close(stacked, batched, atol=1e-10)


def test_batched_matches_scalar_lesser():
    B = 8
    block_sizes = [6, 6, 6]
    hd, sd, hl, hu, sl, su, left_se, right_se, energies = _make_btd_inputs(
        B, block_sizes, seed=1)

    rng = torch.Generator(device="cpu").manual_seed(2)
    s_in_batched = []
    for n in block_sizes:
        x = (torch.randn(B, n, n, generator=rng, dtype=torch.float64) +
             1j * torch.randn(B, n, n, generator=rng, dtype=torch.float64)).to(torch.complex128)
        # Hermitize so s_in resembles a physical -i*(Sigma^< - Sigma^<^dag);
        # the kernel doesn't enforce this but it keeps test inputs sane.
        s_in_batched.append(0.5 * (x + x.mH))

    stacked = _stack_scalar(hd, sd, hl, hu, sl, su, left_se, right_se, energies,
                            need_lesser=True, s_in_batched=s_in_batched)

    batched = recursive_gf(
        energy=energies,
        hl=hl, hd=hd, hu=hu, sd=sd, su=su, sl=sl,
        left_se=left_se, right_se=right_se,
        s_in=s_in_batched, eta=1e-5,
        need_lesser=True, need_gr_lc=True,
    )

    _assert_ans_close(stacked, batched, atol=1e-10)


def test_batched_matches_scalar_retarded_uniform():
    # Uniform block sizes exercise the [K, B, n, n] stacked fast path.
    B = 16
    block_sizes = [8, 8, 8, 8, 8]
    hd, sd, hl, hu, sl, su, left_se, right_se, energies = _make_btd_inputs(
        B, block_sizes, seed=3)

    stacked = _stack_scalar(hd, sd, hl, hu, sl, su, left_se, right_se, energies,
                            need_lesser=False)

    batched = recursive_gf(
        energy=energies,
        hl=hl, hd=hd, hu=hu, sd=sd, su=su, sl=sl,
        left_se=left_se, right_se=right_se,
        s_in=0, eta=1e-5,
        need_lesser=False, need_gr_lc=True,
    )

    _assert_ans_close(stacked, batched, atol=1e-10)


def test_batched_matches_scalar_lesser_uniform():
    B = 8
    block_sizes = [6, 6, 6, 6]
    hd, sd, hl, hu, sl, su, left_se, right_se, energies = _make_btd_inputs(
        B, block_sizes, seed=4)

    rng = torch.Generator(device="cpu").manual_seed(5)
    s_in_batched = []
    for n in block_sizes:
        x = (torch.randn(B, n, n, generator=rng, dtype=torch.float64) +
             1j * torch.randn(B, n, n, generator=rng, dtype=torch.float64)).to(torch.complex128)
        s_in_batched.append(0.5 * (x + x.mH))

    stacked = _stack_scalar(hd, sd, hl, hu, sl, su, left_se, right_se, energies,
                            need_lesser=True, s_in_batched=s_in_batched)

    batched = recursive_gf(
        energy=energies,
        hl=hl, hd=hd, hu=hu, sd=sd, su=su, sl=sl,
        left_se=left_se, right_se=right_se,
        s_in=s_in_batched, eta=1e-5,
        need_lesser=True, need_gr_lc=True,
    )

    _assert_ans_close(stacked, batched, atol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_batched_cuda_smoke():
    B = 16
    block_sizes = [8, 6, 8]
    cpu_inputs = _make_btd_inputs(B, block_sizes, device="cpu")
    cuda_inputs = _make_btd_inputs(B, block_sizes, device="cuda")

    cpu_ans = recursive_gf(
        energy=cpu_inputs[-1],
        hl=cpu_inputs[2], hd=cpu_inputs[0], hu=cpu_inputs[3],
        sd=cpu_inputs[1], su=cpu_inputs[5], sl=cpu_inputs[4],
        left_se=cpu_inputs[6], right_se=cpu_inputs[7],
        s_in=0, eta=1e-5, need_lesser=False, need_gr_lc=True,
    )
    cuda_ans = recursive_gf(
        energy=cuda_inputs[-1],
        hl=cuda_inputs[2], hd=cuda_inputs[0], hu=cuda_inputs[3],
        sd=cuda_inputs[1], su=cuda_inputs[5], sl=cuda_inputs[4],
        left_se=cuda_inputs[6], right_se=cuda_inputs[7],
        s_in=0, eta=1e-5, need_lesser=False, need_gr_lc=True,
    )

    for c, g in zip(cpu_ans, cuda_ans):
        if c is None:
            assert g is None
            continue
        if torch.is_tensor(c):
            assert g.device.type == "cuda"
            assert torch.allclose(c, g.cpu(), atol=1e-8)
        else:
            for cq, gq in zip(c, g):
                assert gq.device.type == "cuda"
                assert torch.allclose(cq, gq.cpu(), atol=1e-8)


def _ans_equal_skipping_gr_left(a, b):
    """Compare two recursive_gf outputs slot-by-slot. Position 5 (gr_left) is
    expected to differ when one call uses keep_gr_left=False — skip it.
    Everything else must be bit-identical."""
    for i, (x, y) in enumerate(zip(a, b)):
        if i == 5:                          # gr_left slot
            continue
        if x is None:
            assert y is None
            continue
        if torch.is_tensor(x):
            assert torch.equal(x, y), f"tuple slot {i} mismatch"
        else:
            for q, (xq, yq) in enumerate(zip(x, y)):
                if xq is None:
                    assert yq is None
                    continue
                assert torch.equal(xq, yq), f"tuple slot {i}, block {q} mismatch"


def test_keep_gr_left_false_outputs_match_true():
    """Lever 1 + 2 release path: with need_lesser=False, need_greater=False, the
    keep_gr_left=False call must produce the same tensors (modulo gr_left) as
    the keep_gr_left=True call. Runs on CPU; correctness, not memory."""
    B = 8
    block_sizes = [8, 6, 8]
    hd, sd, hl, hu, sl, su, left_se, right_se, energies = _make_btd_inputs(
        B, block_sizes, seed=11)

    common = dict(energy=energies, hl=hl, hd=hd, hu=hu, sd=sd, su=su, sl=sl,
                  left_se=left_se, right_se=right_se,
                  s_in=0, eta=1e-5,
                  need_lesser=False, need_greater=False, need_gr_lc=False)
    ans_keep = recursive_gf(**common, keep_gr_left=True)
    ans_drop = recursive_gf(**common, keep_gr_left=False)

    _ans_equal_skipping_gr_left(ans_keep, ans_drop)
    assert ans_drop[5] is None                   # gr_left dropped
    assert isinstance(ans_keep[5], list)         # gr_left populated


def test_keep_gr_left_false_outputs_match_true_uniform():
    """Same equivalence check but on the uniform [K,B,n,n] fast path (Lever 1b)."""
    B = 8
    block_sizes = [8, 8, 8, 8]
    hd, sd, hl, hu, sl, su, left_se, right_se, energies = _make_btd_inputs(
        B, block_sizes, seed=12)

    common = dict(energy=energies, hl=hl, hd=hd, hu=hu, sd=sd, su=su, sl=sl,
                  left_se=left_se, right_se=right_se,
                  s_in=0, eta=1e-5,
                  need_lesser=False, need_greater=False, need_gr_lc=False)
    ans_keep = recursive_gf(**common, keep_gr_left=True)
    ans_drop = recursive_gf(**common, keep_gr_left=False)

    _ans_equal_skipping_gr_left(ans_keep, ans_drop)
    assert ans_drop[5] is None
    assert isinstance(ans_keep[5], list)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_peak_memory_decreases_with_keep_gr_left_false():
    """Lever 1 (per-slot release) + Lever 2 (keep_gr_left wiring) regression.
    On a synthetic batched problem, keep_gr_left=False must use strictly less
    CUDA peak memory than keep_gr_left=True, and outputs (excluding gr_left)
    must be bit-identical."""
    B, block_sizes = 8, [16, 16, 16, 16, 16, 16, 16, 16]
    hd, sd, hl, hu, sl, su, left_se, right_se, energies = _make_btd_inputs(
        B, block_sizes, seed=13, device="cuda")

    common = dict(energy=energies, hl=hl, hd=hd, hu=hu, sd=sd, su=su, sl=sl,
                  left_se=left_se, right_se=right_se,
                  s_in=0, eta=1e-5,
                  need_lesser=False, need_greater=False, need_gr_lc=False)

    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    ans_keep = recursive_gf(**common, keep_gr_left=True)
    peak_keep = torch.cuda.max_memory_allocated()

    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    ans_drop = recursive_gf(**common, keep_gr_left=False)
    peak_drop = torch.cuda.max_memory_allocated()

    _ans_equal_skipping_gr_left(ans_keep, ans_drop)
    assert peak_drop < peak_keep, (
        f"keep_gr_left=False peak {peak_drop} >= keep_gr_left=True peak {peak_keep}; "
        "per-slot release did not reduce memory."
    )
