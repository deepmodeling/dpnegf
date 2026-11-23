"""Tests for the standalone finite/periodic band-gap task."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
from scipy import sparse

from dpnegf.runner.cal_bandgap import (
    BandGapCalculator,
    _checked_superlu_negative_count,
    _solve_shift_invert,
)
from dpnegf.utils.argcheck import normalize_run


ROOT = Path(__file__).resolve().parents[2]
HBN_CHECKPOINT = ROOT / "examples/hBN/train/train_out/checkpoint/nnsk.ep3000.pth"
HBN_STRUCTURE = ROOT / "examples/hBN/train/data/POSCAR"


def _finite_options(method="dense"):
    return {
        "structure_options": {"mode": "finite"},
        "occupation_options": {"nel_atom": {"B": 3, "N": 5}},
        "solver_options": {"method": method, "edge_tol": 1e-6,
                           "energy_bounds": [-30.0, 10.0]},
        "output_options": {"plot": False},
    }


def test_bandgap_schema_defaults_and_rejects_invalid_periodicity():
    normalized = normalize_run({
        "structure": "structure.xyz",
        "task_options": {"task": "cal_bandgap"},
    })
    options = normalized["task_options"]
    assert options["solver_options"]["method"] == "auto"
    assert options["k_options"]["gap_tol"] == pytest.approx(0.01)
    assert options["cache_options"]["use_saved"] is True

    with pytest.raises(ValueError, match="one-dimensional"):
        normalize_run({
            "structure": "structure.xyz", "pbc": [True, True, False],
            "task_options": {"task": "cal_bandgap"},
        })
    with pytest.raises(ValueError, match="conflicts"):
        normalize_run({
            "structure": "structure.xyz", "pbc": [False, False, True],
            "task_options": {"task": "cal_bandgap",
                             "structure_options": {"mode": "finite"}},
        })


def test_checked_superlu_inertia_real_complex_and_unsafe_pivot():
    assert _checked_superlu_negative_count(sparse.diags([-2.0, 1.0, 3.0])) == 1
    hermitian = sparse.csr_matrix(np.array([[2.0, 1j], [-1j, -3.0]]))
    assert _checked_superlu_negative_count(hermitian) == 1
    with pytest.raises(RuntimeError, match="permutations"):
        _checked_superlu_negative_count(sparse.csr_matrix([[0.0, 1.0], [1.0, 0.0]]))


def test_checked_shift_invert_recovers_global_band_indices():
    expected = np.linspace(-5.0, 5.0, 64)
    hamiltonian = sparse.diags(expected, dtype=np.complex128, format="csr")
    overlap = sparse.eye(64, dtype=np.complex128, format="csr")
    ranks = list(range(30, 36))
    spectrum, allowance, probes, solver = _solve_shift_invert(
        hamiltonian, overlap, ranks,
        {"energy_bounds": [-10.0, 10.0], "edge_tol": 1e-6,
         "energy_shift": None, "dense_limit": 20},
    )
    assert solver == "shift-invert"
    assert probes > 0
    assert allowance <= 1e-7
    for rank in ranks:
        assert spectrum[rank][0] == pytest.approx(expected[rank - 1], abs=1e-8)


def test_hbn_finite_gap_and_model_free_hs_cache_reuse(tmp_path):
    cache = tmp_path / "validated-hs.npz"
    common = {
        "structure": str(HBN_STRUCTURE),
        "checkpoint": str(HBN_CHECKPOINT),
        "pbc": [False, False, False],
        "results_path": None,
    }
    first = BandGapCalculator(
        **common,
        task_options={**_finite_options("dense"), "cache_options": {"path": str(cache)}},
    ).compute()
    assert first["gap_type"] == "finite_homo_lumo_gap"
    assert first["gap_status"] == "positive_gap"
    assert first["converged"] is True
    assert first["n_atoms"] == 2
    assert first["n_orbitals"] == 18
    assert first["overlap_included"] is True
    assert first["cache"]["hit"] is False
    assert len(first["k_spectra"]) == 1
    assert len(first["k_spectra"][0]["eigenvalues_eV"]) == len(first["band_indices"])
    assert cache.is_file()

    # The cached real-space blocks are self-contained; do not construct a model
    # for the second call, even though no result-output directory was supplied.
    with patch("dptb.nn.build.build_model", side_effect=AssertionError("cache miss")):
        second = BandGapCalculator(
            **common,
            task_options={**_finite_options("inertia"), "cache_options": {"path": str(cache)}},
        ).compute()
    assert second["cache"]["hit"] is True
    assert second["solver"] == ["inertia"]
    assert second["vbm_eV"] == pytest.approx(first["vbm_eV"], abs=2e-5)
    assert second["cbm_eV"] == pytest.approx(first["cbm_eV"], abs=2e-5)
    with patch("dptb.nn.build.build_model", side_effect=AssertionError("cache miss")):
        third = BandGapCalculator(
            **common,
            task_options={**_finite_options("inertia"), "cache_options": {"path": str(cache)}},
        ).compute()
    assert third["cache"]["spectrum_hits"] == 1
    assert third["cache"]["spectrum_misses"] == 0


def test_realspace_blocks_match_native_hr2hk_for_complex_kpoints():
    pytest.importorskip("dptb")
    import torch
    from ase.io import read
    from dptb.data import AtomicData, AtomicDataDict
    from dptb.data.interfaces.ham_to_feature import feature_to_block
    from dptb.nn.build import build_model
    from dptb.nn.hr2hk import HR2HK
    from dptb.utils.argcheck import get_cutoffs_from_model_options

    model = build_model(checkpoint=str(HBN_CHECKPOINT), common_options={"device": "cpu"})
    atoms = read(str(HBN_STRUCTURE))
    cell = np.asarray(atoms.cell).copy()
    cell[2] = [0.0, 0.0, 4.0]
    atoms.set_cell(cell, scale_atoms=False)
    atoms.set_pbc([False, False, True])
    cutoffs = dict(zip(("r_max", "er_max", "oer_max"),
                       get_cutoffs_from_model_options(model.model_options)))
    data = AtomicData.to_AtomicDataDict(AtomicData.from_ase(atoms, **cutoffs).to("cpu"))
    data = model.idp(data)
    kpoints = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.2], [0.0, 0.0, 0.5]])
    data[AtomicDataDict.KPOINT_KEY] = torch.nested.as_nested_tensor(
        [torch.as_tensor(kpoints, dtype=model.dtype, device="cpu")]
    )
    with torch.no_grad():
        predicted = model(data)

    h_blocks = feature_to_block(predicted, model.idp)
    s_blocks = feature_to_block(predicted, model.idp, overlap=True)
    h_native = HR2HK(idp=model.idp, dtype=model.dtype)(predicted)[
        AtomicDataDict.HAMILTONIAN_KEY
    ].detach().cpu().numpy()
    s_native = HR2HK(
        idp=model.idp, edge_field=AtomicDataDict.EDGE_OVERLAP_KEY,
        node_field=AtomicDataDict.NODE_OVERLAP_KEY,
        out_field=AtomicDataDict.OVERLAP_KEY, overlap=True,
        dtype=model.dtype,
    )(predicted)[AtomicDataDict.OVERLAP_KEY].detach().cpu().numpy()
    atom_norbs = [
        int(model.idp.atom_norb[model.idp.chemical_symbol_to_type[symbol]])
        for symbol in atoms.get_chemical_symbols()
    ]
    calculator = BandGapCalculator(
        structure=atoms,
        checkpoint=str(HBN_CHECKPOINT),
        pbc=[False, False, True],
        task_options={"task": "cal_bandgap"},
    )
    calculator._atoms = atoms
    model_metadata = {
        "basis": model.idp.basis,
        "idp_method": model.idp.method,
        "AtomicData_options": cutoffs,
        "dtype": str(model.dtype).split(".")[-1],
    }
    assert calculator._prepare_hr2hk(h_blocks, s_blocks, model_metadata)

    for index, kpoint in enumerate(kpoints):
        h_built = calculator._hr2hk_fourier(kpoint).toarray()
        s_built = calculator._hr2hk_fourier(kpoint, overlap=True).toarray()
        np.testing.assert_allclose(h_built, h_native[index], rtol=1e-6, atol=2e-6)
        np.testing.assert_allclose(s_built, s_native[index], rtol=1e-6, atol=2e-6)
        h_sparse = BandGapCalculator._assemble_sparse_blocks(h_blocks, atom_norbs, kpoint).toarray()
        s_sparse = BandGapCalculator._assemble_sparse_blocks(s_blocks, atom_norbs, kpoint).toarray()
        np.testing.assert_allclose(h_sparse, h_native[index], rtol=1e-6, atol=2e-6)
        np.testing.assert_allclose(s_sparse, s_native[index], rtol=1e-6, atol=2e-6)


def test_bandgap_cli_task_writes_reviewable_outputs(tmp_path):
    pytest.importorskip("dptb")
    from dpnegf.entrypoints.run import run

    config_path = tmp_path / "gap.json"
    config_path.write_text(json.dumps({
        "pbc": [False, False, False],
        "task_options": {
            "task": "cal_bandgap",
            "structure_options": {"mode": "finite"},
            "occupation_options": {"nel_atom": {"B": 3, "N": 5}},
            "solver_options": {"method": "dense"},
            "output_options": {"plot": True},
        }
    }))
    result = run(
        INPUT=str(config_path), init_model=str(HBN_CHECKPOINT),
        structure=str(HBN_STRUCTURE), output=str(tmp_path / "run"),
        log_path=None, log_level=40,
    )
    result_dir = tmp_path / "run" / "results"
    assert result["gap_type"] == "finite_homo_lumo_gap"
    assert (result_dir / "gap.json").is_file()
    assert (result_dir / "bands.npz").is_file()
    assert (result_dir / "bands.csv").is_file()
    assert (result_dir / "bands.png").is_file()
    assert (result_dir / "gap.png").is_file()
    assert json.loads((result_dir / "gap.json").read_text())["signed_gap_eV"] == pytest.approx(result["signed_gap_eV"])


def test_run_import_defers_transport_runners():
    code = (
        "import builtins; original=builtins.__import__; "
        "guard=lambda name,*a,**k: (_ for _ in ()).throw(AssertionError(name)) "
        "if name in ('dpnegf.runner.NEGF','dpnegf.runner.tbtrans_init') "
        "else original(name,*a,**k); builtins.__import__=guard; "
        "import dpnegf.entrypoints.run"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
