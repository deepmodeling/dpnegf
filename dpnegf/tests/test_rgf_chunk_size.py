"""CUDA auto-batch sizing must fail conservatively without hiding the cause."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from dpnegf.utils.parrallel import _auto_chunk_size, _rgf_chunk_size


def _runner(device="cuda:0", sizes=(64, 128), batch=None):
    return SimpleNamespace(
        rgf_device=torch.device(device),
        deviceprop=SimpleNamespace(
            hd=[SimpleNamespace(shape=(size, size)) for size in sizes],
            prepare_hamiltonian=Mock()),
        e_batch_size=batch,
        block_tridiagonal=True,
        need_trans_eigs=False,
        out_trans_eigchannel=False,
        trans_eigs_options={},
    )


def test_memory_query_failure_uses_single_energy_and_logs_cause(monkeypatch, caplog):
    error = RuntimeError("CUDA memory query failed")
    query = Mock(side_effect=error)
    monkeypatch.setattr(torch.cuda, "mem_get_info", query)
    with caplog.at_level(logging.ERROR, logger="dpnegf.utils.parrallel"):
        assert _rgf_chunk_size(_runner(), 100, [0, 0, 0]) == 1

    query.assert_called_once_with(torch.device("cuda:0"))
    record, = caplog.records
    assert record.exc_info[1] is error
    assert "cuda:0" in record.message
    assert "e_batch_size=1" in record.message
    assert "RuntimeError: CUDA memory query failed" in record.message


def test_hamiltonian_load_failure_propagates_before_memory_query(monkeypatch):
    query = Mock(return_value=(2**30, 2**31))
    monkeypatch.setattr(torch.cuda, "mem_get_info", query)
    error = OSError("Hamiltonian geometry unavailable")
    runner = _runner()
    runner.deviceprop.prepare_hamiltonian.side_effect = error

    with pytest.raises(OSError, match="Hamiltonian geometry unavailable") as caught:
        _rgf_chunk_size(runner, 100, [0, 0, 0])
    assert caught.value is error
    query.assert_not_called()


@pytest.mark.parametrize("sizes", [(), (0,)])
def test_invalid_geometry_uses_single_energy(monkeypatch, caplog, sizes):
    monkeypatch.setattr(torch.cuda, "mem_get_info", Mock(return_value=(2**30, 2**31)))
    assert _auto_chunk_size(_runner(sizes=sizes), 100) == 1
    assert "falling back to e_batch_size=1" in caplog.text
    assert caplog.records[0].exc_info is not None


def test_cpu_uses_full_grid_without_querying_cuda(monkeypatch):
    query = Mock(side_effect=AssertionError("CPU must not query CUDA"))
    monkeypatch.setattr(torch.cuda, "mem_get_info", query)
    assert _auto_chunk_size(_runner(device="cpu"), 100) == 100
    query.assert_not_called()


@pytest.mark.parametrize("free_bytes,expected", [(0, 1), (64 * 2**20, 4), (2**31, 100)])
def test_successful_memory_estimate(monkeypatch, free_bytes, expected):
    monkeypatch.setattr(torch.cuda, "mem_get_info", Mock(return_value=(free_bytes, 2**31)))
    assert _auto_chunk_size(_runner(), 100) == expected


def test_first_batch_queries_memory_after_loading_hamiltonian(monkeypatch):
    events = []
    runner = _runner()
    del runner.deviceprop.hd
    k, bias = [0.5, 0, 0], torch.tensor(0.2)

    def prepare(*args):
        assert not torch.is_grad_enabled()
        events.append("load_hs")
        runner.deviceprop.hd = [SimpleNamespace(shape=(128, 128))] * 2

    def query(device):
        assert events == ["load_hs"]
        events.append("query_memory")
        return 64 * 2**20, 2**31

    runner.deviceprop.prepare_hamiltonian.side_effect = prepare
    monkeypatch.setattr(torch.cuda, "mem_get_info", query)
    assert _rgf_chunk_size(runner, 100, k, bias) == 4
    runner.deviceprop.prepare_hamiltonian.assert_called_once_with(k, True, bias)
    assert events == ["load_hs", "query_memory"]


def test_unloaded_geometry_never_uses_preload_memory_budget(monkeypatch, caplog):
    query = Mock(return_value=(2**31, 2**31))
    monkeypatch.setattr(torch.cuda, "mem_get_info", query)
    runner = _runner()
    del runner.deviceprop.hd
    assert _auto_chunk_size(runner, 100) == 1
    query.assert_not_called()
    assert "AttributeError" in caplog.text


def test_eigen_outputs_have_separate_memory_budgets(monkeypatch, caplog):
    monkeypatch.setattr(torch.cuda, "mem_get_info", Mock(return_value=(116 * 2**20, 2**31)))
    runner = _runner(sizes=(128,) * 4)
    assert _auto_chunk_size(runner, 100) == 4
    runner.need_trans_eigs = True
    assert _auto_chunk_size(runner, 100) == 3
    runner.out_trans_eigchannel = True
    with caplog.at_level(logging.INFO, logger="dpnegf.utils.parrallel"):
        assert _auto_chunk_size(runner, 100) == 2
    assert "eig_workspace~=" in caplog.text


@pytest.mark.parametrize("selection", [{"n_channels": 1}, {"channel_indices": [0]}])
def test_selected_channels_reduce_coefficient_workspace(monkeypatch, selection):
    monkeypatch.setattr(torch.cuda, "mem_get_info", Mock(return_value=(116 * 2**20, 2**31)))
    runner = _runner(sizes=(128,) * 4)
    runner.need_trans_eigs = runner.out_trans_eigchannel = True
    assert _auto_chunk_size(runner, 100) == 2
    runner.trans_eigs_options = selection
    assert _auto_chunk_size(runner, 100) == 3


def test_original_order_reserves_extra_coefficient_copies(monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", Mock(return_value=(120 * 2**20, 2**31)))
    runner = _runner(sizes=(128,) * 4)
    runner.need_trans_eigs = runner.out_trans_eigchannel = True
    assert _auto_chunk_size(runner, 100) == 3
    runner.trans_eigs_options = {"return_format": "original"}
    assert _auto_chunk_size(runner, 100) == 2


def test_workspace_exceeding_budget_still_uses_single_energy(monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", Mock(return_value=(2**20, 2**31)))
    runner = _runner(sizes=(128,) * 4)
    runner.need_trans_eigs = runner.out_trans_eigchannel = True
    assert _auto_chunk_size(runner, 100) == 1



def test_explicit_batch_size_is_preserved_on_query_failure(monkeypatch, caplog):
    monkeypatch.setattr(torch.cuda, "mem_get_info", Mock(side_effect=RuntimeError("query failed")))
    assert _rgf_chunk_size(_runner(batch=3), 100, [0, 0, 0]) == 3
    assert "query failed" in caplog.text
    assert "user e_batch_size=3 exceeds CUDA auto-cap=1" in caplog.text
