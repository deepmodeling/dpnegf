"""Equivalence and micro-benchmark for `_surface_green_numba_core`.

The numba core is compiled with ``fastmath=True`` in production (see
`dpnegf/negf/surface_green.py`), which lets numba reassociate the pointwise
complex ops and use FMA where available. Reassociation can change the last
bit or two of the answer, so we pin numerical equivalence against a sibling
compiled without ``fastmath`` on a battery of synthetic lead problems.

We also time both variants at a couple of realistic principal-layer sizes and
print the ratio, so a reviewer can see the acceleration. The timings are
noisy under CI load and are not asserted.
"""

import time

import numpy as np
import pytest

from dpnegf.negf.surface_green import (
    _numba_available,
    _surface_green_numba_core,
    _surface_green_numba_core_nofastmath,
)


pytestmark = pytest.mark.skipif(not _numba_available, reason="numba is not available")


def _make_lead_problem(N, seed):
    """Random hermitian on-site + small off-diagonal hopping. Not physical but
    passes Lopez-Sancho convergence over the tested energies, which is all the
    core cares about."""
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    H = ((H + H.conj().T) / 2).astype(np.complex128)
    H = np.ascontiguousarray(H)

    h01 = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) * 0.05
    h01 = np.ascontiguousarray(h01.astype(np.complex128))

    S = np.eye(N, dtype=np.complex128)
    s01 = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) * 0.005
    s01 = np.ascontiguousarray(s01.astype(np.complex128))

    h10 = np.ascontiguousarray(np.conj(h01.T))
    s10 = np.ascontiguousarray(np.conj(s01.T))
    return H, h01, S, s01, h10, s10


@pytest.mark.parametrize("N,seed", [(32, 0), (32, 1), (64, 2), (64, 3)])
@pytest.mark.parametrize("ee", [
    np.complex128(-1.5 + 1e-5j),
    np.complex128(0.25 + 1e-5j),
    np.complex128(2.0 + 1e-5j),
])
def test_fastmath_matches_nofastmath(N, seed, ee):
    """Fastmath must not perturb the surface Green's function beyond
    reassociation noise. 1e-10 leaves 2 orders of margin over the outer
    Lopez warning threshold (1e-8) but is strict enough to catch a genuine
    numerical divergence."""
    H, h01, S, s01, h10, s10 = _make_lead_problem(N, seed)

    gs_fast, _, _, _ = _surface_green_numba_core(H, h01, S, s01, h10, s10, ee)
    gs_slow, _, _, _ = _surface_green_numba_core_nofastmath(H, h01, S, s01, h10, s10, ee)

    max_abs = np.max(np.abs(gs_fast - gs_slow))
    assert max_abs < 1e-10, f"fastmath deviated by {max_abs:g} at N={N}, seed={seed}, ee={ee}"


def _time_core(core, args, n_calls):
    # warm-up: absorb JIT compile
    core(*args)
    t0 = time.perf_counter()
    for _ in range(n_calls):
        core(*args)
    return (time.perf_counter() - t0) / n_calls


# @pytest.mark.parametrize("N,n_calls", [(256, 5), (512, 3)])
# def test_fastmath_speedup(N, n_calls, capsys):
#     """Not asserted — just prints so ``pytest -s`` shows the speedup."""
#     H, h01, S, s01, h10, s10 = _make_lead_problem(N, seed=42)
#     ee = np.complex128(0.25 + 1e-5j)
#     args = (H, h01, S, s01, h10, s10, ee)

#     t_fast = _time_core(_surface_green_numba_core, args, n_calls)
#     t_slow = _time_core(_surface_green_numba_core_nofastmath, args, n_calls)

#     with capsys.disabled():
#         print(
#             f"\n[N={N:4d}] fastmath={t_fast * 1e3:7.1f} ms  "
#             f"nofastmath={t_slow * 1e3:7.1f} ms  "
#             f"speedup={t_slow / t_fast:.2f}x"
#         )
