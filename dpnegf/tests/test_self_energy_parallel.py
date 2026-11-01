"""Regression tests for actual-concurrency BLAS budgets and bounded probe cost."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import h5py
import numpy as np
import pytest
import torch

from dpnegf.negf import lead_property as lp


@pytest.fixture
def scheduler(monkeypatch, tmp_path):
    """Exercise real scheduling/tuning, recording dispatch instead of spawning."""
    batches = []
    lead_L = SimpleNamespace(results_path=str(tmp_path), dim=256)
    lead_R = SimpleNamespace(results_path=str(tmp_path), dim=256)

    def precompute(lead, kpoints):
        return {"kdata": {
            lp._k_key(k): {"HLk": np.empty((lead.dim, lead.dim))}
            for k in kpoints
        }}

    class RecordingParallel:
        def __init__(self, n_jobs, backend):
            assert backend == "loky"
            self.n_jobs = n_jobs

        def __call__(self, tasks):
            tasks = list(tasks)
            batches.append((self.n_jobs, [args[-1] for _, args, _ in tasks]))
            return [fn(*args, **kwargs) for fn, args, kwargs in tasks]

    probe = Mock()
    worker = Mock()
    merge = Mock()
    tune = Mock(wraps=lp._autotune_blas_threads)
    precompute_mock = Mock(side_effect=precompute)
    monkeypatch.setattr(lp, "_precompute_lead_kdata", precompute_mock)
    monkeypatch.setattr(lp, "_estimate_worker_memory", lambda *a, **kw: 1024)
    monkeypatch.setattr(lp.psutil, "virtual_memory", lambda: SimpleNamespace(available=10**12))
    monkeypatch.setattr(lp, "Parallel", RecordingParallel)
    monkeypatch.setattr(lp, "_self_energy_worker_blas", worker)
    monkeypatch.setattr(lp, "_compute_self_energy_from_pack", probe)
    monkeypatch.setattr(lp, "_autotune_blas_threads", tune)
    monkeypatch.setattr(lp, "merge_hdf5_files", merge)

    def run(task_count, **options):
        options.setdefault("cpu_budget", 20)
        kpoints = options.pop("kpoints_grid", [[0., 0., 0.]])
        lp.compute_all_self_energy(
            1e-5, lead_L, lead_R, kpoints, np.arange(task_count, dtype=float),
            se_numba_jit=False, **options,
        )

    return SimpleNamespace(run=run, batches=batches, probe=probe, tune=tune,
                           worker=worker, merge=merge, precompute=precompute_mock,
                           lead_L=lead_L, lead_R=lead_R)


@pytest.mark.parametrize("requested,expected", [(4, 4), (10, 4), (1, 1)])
def test_explicit_threads_use_actual_workers_without_probes(scheduler, caplog, requested, expected):
    with caplog.at_level(logging.INFO, logger=lp.__name__):
        scheduler.run(5, blas_threads=requested)
    assert scheduler.batches == [(5, [expected] * 5)]
    scheduler.probe.assert_not_called()
    assert scheduler.worker.call_count == 5
    assert scheduler.merge.call_count == 2
    assert f"tasks=5, active_workers=5, blas_threads={expected} per worker, cpu_budget=20" in caplog.text
    assert "autotune=skipped (user-requested)" in caplog.text


@pytest.mark.parametrize("left_dim,right_dim,expected", [
    (255, 255, 1), (256, 32, 4), (32, 256, 4), (512, 512, 4),
])
def test_short_batch_checks_both_electrodes_without_probes(scheduler, left_dim, right_dim, expected):
    scheduler.lead_L.dim, scheduler.lead_R.dim = left_dim, right_dim
    scheduler.run(5)
    assert scheduler.batches == [(5, [expected] * 5)]
    scheduler.tune.assert_not_called()
    scheduler.probe.assert_not_called()


@pytest.mark.parametrize("cpu_budget,expected", [(32, 4), (3, 3), (1, 1)])
def test_single_point_uses_budgeted_short_batch_strategy(scheduler, cpu_budget, expected):
    scheduler.run(1, cpu_budget=cpu_budget)
    assert scheduler.batches == [(1, [expected])]
    scheduler.probe.assert_not_called()


def test_explicit_threads_override_short_batch_heuristic(scheduler):
    scheduler.lead_L.dim = scheduler.lead_R.dim = 8
    scheduler.run(1, cpu_budget=32, blas_threads=16)
    assert scheduler.batches == [(1, [16])]
    scheduler.probe.assert_not_called()


def test_batch_size_limits_concurrency_and_reuses_choice(scheduler):
    scheduler.run(24, cpu_budget=32, ek_batch_size=8, blas_threads=4)
    assert scheduler.batches == [(8, [4] * 8)] * 3
    assert scheduler.tune.call_count == 1
    scheduler.probe.assert_not_called()


def test_concurrency_counts_the_full_k_energy_product(scheduler):
    scheduler.run(3, kpoints_grid=[[0., 0., 0.], [0.5, 0., 0.]], blas_threads=4)
    assert scheduler.batches == [(6, [3] * 6)]
    scheduler.probe.assert_not_called()


@pytest.mark.parametrize("requested", [None, 4])
def test_tail_batch_gets_new_budget_without_probes(scheduler, requested, caplog):
    with caplog.at_level(logging.INFO, logger=lp.__name__):
        scheduler.run(205, blas_threads=requested)
    assert scheduler.batches == [(20, [1] * 200), (5, [4] * 5)]
    assert scheduler.worker.call_count == 205
    scheduler.probe.assert_not_called()
    assert "tasks=5, active_workers=5, blas_threads=4 per worker, cpu_budget=20" in caplog.text
    assert "autotune=skipped" in caplog.text


def test_short_batch_includes_two_full_waves(scheduler):
    scheduler.run(8, n_workers=4)
    assert scheduler.batches == [(4, [4] * 8)]
    scheduler.tune.assert_not_called()
    scheduler.probe.assert_not_called()


def test_long_batches_tune_once_per_worker_count_and_call(scheduler):
    scheduler.run(83, n_workers=4, ek_batch_size=40)
    assert [n for n, _ in scheduler.batches] == [4, 4, 3]
    assert scheduler.batches[0][1][0] == scheduler.batches[1][1][0]
    assert scheduler.batches[2] == (3, [4] * 3)
    for n_workers, threads in scheduler.batches:
        assert all(1 <= t <= 20 // n_workers for t in threads)
    assert scheduler.tune.call_count == 1
    assert scheduler.tune.call_args.args[4:6] == (4, 20)
    # Budget=5: candidates 1, 2, 4, 5; one warm-up and timed call each.
    assert scheduler.probe.call_count == 8
    scheduler.run(40, n_workers=4)
    assert scheduler.tune.call_count == 2  # cache does not cross sweep calls
    assert scheduler.probe.call_count == 16


def test_long_batch_with_small_sample_skips_probes(scheduler):
    scheduler.lead_L.dim = scheduler.lead_R.dim = 32
    scheduler.run(20, n_workers=4)
    assert scheduler.batches == [(4, [1] * 20)]
    scheduler.probe.assert_not_called()


def test_memory_cap_is_retained_when_assigning_threads(scheduler, monkeypatch):
    # 0.9 * 2500 // 1024 allows only two workers, despite a 20-CPU budget.
    monkeypatch.setattr(lp.psutil, "virtual_memory", lambda: SimpleNamespace(available=2500))
    scheduler.run(4, blas_threads=10)
    assert scheduler.batches == [(2, [10] * 4)]
    scheduler.probe.assert_not_called()


@pytest.mark.parametrize("detected", [None, 0, -1])
def test_cpu_detection_fallback_is_shared(scheduler, monkeypatch, detected):
    cpu_count = Mock(return_value=detected)
    monkeypatch.setattr(lp.os, "cpu_count", cpu_count)
    scheduler.run(3, cpu_budget=None, blas_threads=4)
    assert scheduler.batches == [(1, [1] * 3)]
    cpu_count.assert_called_once_with()
    scheduler.probe.assert_not_called()


@pytest.mark.parametrize("n_workers,expected", [(None, 8), (-1, 8), (-2, 7), (2, 2)])
def test_worker_input_semantics(scheduler, n_workers, expected):
    scheduler.run(8, cpu_budget=8, n_workers=n_workers, blas_threads=1)
    assert scheduler.batches == [(expected, [1] * 8)]


@pytest.mark.parametrize("name,value", [
    ("cpu_budget", 0), ("cpu_budget", -1), ("cpu_budget", 1.5), ("cpu_budget", True),
    ("n_workers", 0), ("n_workers", 2.5), ("n_workers", "2"), ("n_workers", False),
    ("blas_threads", 0), ("blas_threads", -1), ("blas_threads", 1.5), ("blas_threads", True),
    ("ek_batch_size", 0), ("ek_batch_size", -1), ("ek_batch_size", 1.5),
    ("ek_batch_size", True), ("ek_batch_size", None),
])
def test_invalid_inputs_fail_before_work(scheduler, name, value):
    with pytest.raises(ValueError, match=rf"self_energy_options\.parallel\.{name}"):
        scheduler.run(5, **{name: value})
    scheduler.precompute.assert_not_called()
    scheduler.probe.assert_not_called()
    scheduler.worker.assert_not_called()
    scheduler.merge.assert_not_called()


@pytest.mark.parametrize("empty_kpoints", [False, True])
def test_empty_grid_does_not_probe_dispatch_or_merge(scheduler, empty_kpoints):
    scheduler.run(5 if empty_kpoints else 0,
                  kpoints_grid=[] if empty_kpoints else [[0., 0., 0.]])
    assert scheduler.batches == []
    scheduler.precompute.assert_not_called()
    scheduler.probe.assert_not_called()
    scheduler.merge.assert_not_called()


class _ChainHamiltonian:
    """One-orbital nearest-neighbor lead with an analytic self-energy."""

    def get_hs_device(self, kpoint, only_subblocks):
        return [1]

    def get_hs_lead(self, kpoint, tab, v):
        zero = torch.zeros((1, 1), dtype=torch.complex128)
        one = torch.ones((1, 1), dtype=torch.complex128)
        return zero, -one, -one, one, zero, zero


def test_real_parallel_batches_match_serial_and_analytic_self_energy(tmp_path):
    """Run loky, BLAS limits, SGF and HDF5 merging on both sides of a tail."""
    energies = np.array([0.25, 0.75, 1.25, 1.75, 2.25])
    eta = 1e-3
    outputs = []
    for name, workers in (("serial", 1), ("parallel", 2)):
        path = tmp_path / name
        path.mkdir()
        leads = [lp.LeadProperty(tab, _ChainHamiltonian(), None, str(path), 0.)
                 for tab in ("lead_L", "lead_R")]
        lp.compute_all_self_energy(
            eta, *leads, [[0., 0., 0.]], energies,
            self_energy_save_path=str(path), cpu_budget=4, n_workers=workers,
            blas_threads=4, ek_batch_size=4, se_numba_jit=False,
        )
        sides = []
        for side in ("L", "R"):
            with h5py.File(path / f"self_energy_lead{side}.h5") as handle:
                assert len(handle) == len(energies)
                sides.append(np.array([
                    handle[f"E_{e:.8f}"]["k_0.0_0.0_0.0"][0, 0] for e in energies
                ]))
        assert not list(path.glob("tmp_*.h5"))
        outputs.append(sides)
    z = energies + 1j * eta
    expected = (z - np.sqrt(z * z - 4)) / 2
    np.testing.assert_allclose(outputs[0], [expected, expected], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(outputs[1], outputs[0], rtol=1e-12, atol=1e-12)
