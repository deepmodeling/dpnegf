import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import torch
import pytest
from ase.io import read
from dptb.data import AtomicData, AtomicDataDict
from dptb.data.transforms import OrbitalMapper
from dptb.nn.build import build_model
from dptb.nn.hr2hk import HR2HK

from dpnegf.negf.negf_hamiltonian_init import NEGFHamiltonianInit
from dpnegf.negf.lead_property import LeadProperty
from dpnegf.negf.split_btd import compute_edge
from dpnegf.utils.direct_btd import (
    LocalBlocks,
    assemble,
    partition_profiles,
)

ROOT = Path(__file__).resolve().parents[2]

@pytest.fixture(autouse=True)
def restore_torch_settings():
    original_dtype = torch.get_default_dtype()
    original_threads = torch.get_num_threads()
    yield
    torch.set_default_dtype(original_dtype)
    torch.set_num_threads(original_threads)


def dense_from_parts(parts):
    sizes = [block.shape[0] for block in parts[0]]
    bounds = np.cumsum([0] + sizes)
    matrix = torch.zeros(
        (sum(sizes), sum(sizes)),
        dtype=parts[0][0].dtype,
    )
    for index in range(len(sizes)):
        section = slice(bounds[index], bounds[index + 1])
        matrix[section, section] = parts[0][index]
        if index < len(sizes) - 1:
            neighbour = slice(bounds[index + 1], bounds[index + 2])
            matrix[section, neighbour] = parts[1][index]
            matrix[neighbour, section] = parts[2][index]
    return matrix


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_local_matches_hr2hk_mixed_masks_phases_duplicates(overlap, dtype):
    mapper = OrbitalMapper(
        {"C": ["2s", "2p"], "H": ["1s"]},
        method="e3tb",
    )
    mapper.get_orbpair_maps()
    generator = torch.Generator().manual_seed(19)
    data = {
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor(
            [
                [mapper.chemical_symbol_to_type[symbol]]
                for symbol in ("C", "H", "C")
            ]
        ),
        AtomicDataDict.EDGE_INDEX_KEY: torch.tensor(
            [[0, 1, 1, 2, 0, 0], [1, 0, 2, 1, 1, 0]]
        ),
        AtomicDataDict.EDGE_CELL_SHIFT_KEY: torch.tensor(
            [
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [1, 0, 0],
                [1, 0, 0],
            ],
            dtype=dtype,
        ),
        AtomicDataDict.KPOINT_KEY: torch.tensor(
            [[0, 0, 0], [0.17, -0.23, 0]],
            dtype=dtype,
        ),
    }
    edge_key = (
        AtomicDataDict.EDGE_OVERLAP_KEY
        if overlap
        else AtomicDataDict.EDGE_FEATURES_KEY
    )
    node_key = (
        AtomicDataDict.NODE_OVERLAP_KEY
        if overlap
        else AtomicDataDict.NODE_FEATURES_KEY
    )
    data[edge_key] = torch.randn(
        (6, mapper.reduced_matrix_element),
        generator=generator,
        dtype=dtype,
    )
    data[node_key] = torch.randn(
        (3, mapper.reduced_matrix_element),
        generator=generator,
        dtype=dtype,
    )

    local = LocalBlocks(data, mapper, overlap=overlap)
    reference = HR2HK(
        idp=mapper,
        edge_field=edge_key,
        node_field=node_key,
        overlap=overlap,
        dtype=dtype,
    )(dict(data))[AtomicDataDict.HAMILTONIAN_KEY]

    tolerance = 1e-6 if dtype == torch.float32 else 1e-14
    for k_index, kpoint in enumerate(data[AtomicDataDict.KPOINT_KEY]):
        result = assemble(local, kpoint, (0, 9), [3, 3, 3], {})
        torch.testing.assert_close(
            dense_from_parts(result),
            reference[k_index],
            atol=tolerance,
            rtol=tolerance,
        )

    data[AtomicDataDict.NODE_SOC_SWITCH_KEY] = torch.tensor([True, False])
    with pytest.raises(ValueError, match="SOC"):
        LocalBlocks(data, mapper)


def test_profiles_fixed_boundaries_and_overlap_only():
    matrix = np.eye(12)
    matrix[0, 5] = matrix[5, 0] = 1
    matrix[5, 8] = matrix[8, 5] = 1

    edge, reverse = compute_edge(matrix)
    blocks = partition_profiles(edge, reverse, 3, 2, fixed=True)

    assert blocks[0] == 3
    assert blocks[-1] == 2
    assert sum(blocks) == 12

    matrix[0, 11] = matrix[11, 0] = 1
    edge, reverse = compute_edge(matrix)
    with pytest.raises(ValueError, match="incompatible"):
        partition_profiles(edge, reverse, 3, 2, fixed=True)

    with pytest.raises(ValueError, match="overlap"):
        partition_profiles(edge, reverse, 8, 8, fixed=True)


def test_reduced_contacts_explicit_side():
    h_contact = torch.zeros((3, 4))
    s_contact = torch.ones_like(h_contact)
    actual = LeadProperty.HDL_reduced(
        h_contact,
        s_contact,
        [2, 6, 3],
        tab="lead_R",
        reduced=True,
    )

    assert actual[0] is h_contact
    assert actual[1] is s_contact

    with pytest.raises(ValueError, match="height"):
        LeadProperty.HDL_reduced(
            h_contact,
            s_contact,
            [2, 6, 4],
            tab="lead_R",
            reduced=True,
        )


def cnt_initializer(path, mode, dtype="float64"):
    path.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(getattr(torch, dtype))
    options = json.loads(
        (ROOT / "examples/CNT/input.json").read_text()
    )["task_options"]["stru_options"]

    model = build_model(
        checkpoint=str(ROOT / "examples/CNT/nnsk_dftb.json"),
        common_options={"dtype": dtype, "device": "cpu"},
    )
    return NEGFHamiltonianInit(
        model=model,
        AtomicData_options={"r_max": 4.6},
        structure=read(ROOT / "examples/CNT/cnt7_0.xyz"),
        block_tridiagonal=True,
        pbc_negf=options["pbc"],
        stru_options=copy.deepcopy(options),
        unit="eV",
        results_path=str(path),
        btd_initialization=mode,
    )

@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cnt_cache_equivalence_and_no_global_hr2hk(
    tmp_path, monkeypatch, dtype
):
    torch.set_num_threads(1)
    reference = cnt_initializer(tmp_path / "dense", "dense", dtype)
    kpoints = [[0, 0, 0], [0.17, 0.23, 0]]
    with torch.no_grad():
        reference.initialize(kpoints, block_tridiagnal=True)
    direct = cnt_initializer(tmp_path / "direct", "direct", dtype)
    direct._self_energy_cache_edge_sizes = (
        reference.subblocks[0],
        reference.subblocks[-1],
    )
    forward = HR2HK.forward

    def guarded_forward(transform, data):
        assert len(data[AtomicDataDict.ATOM_TYPE_KEY]) <= 56
        return forward(transform, data)

    monkeypatch.setattr(HR2HK, "forward", guarded_forward)
    original_zeros = torch.zeros
    original_eye = torch.eye
    forbidden_sizes = {
        sum(direct.atom_norbs),
        sum(direct.device_norbs),
    }

    def guarded_zeros(*shape, **kwargs):
        dimensions = (
            shape[0]
            if len(shape) == 1
            and isinstance(shape[0], (tuple, list, torch.Size))
            else shape
        )
        assert not (
            len(dimensions) >= 2
            and dimensions[-1] == dimensions[-2]
            and dimensions[-1] in forbidden_sizes
        )
        return original_zeros(*shape, **kwargs)

    def guarded_eye(size, *args, **kwargs):
        assert size not in forbidden_sizes
        return original_eye(size, *args, **kwargs)

    with monkeypatch.context() as allocation_guard:
        def forbidden_resident_assembly(*args, **kwargs):
            raise AssertionError(
                "Production initialization must stream BTD blocks."
            )

        allocation_guard.setattr(
            "dpnegf.utils.direct_btd.assemble",
            forbidden_resident_assembly,
        )
        allocation_guard.setattr(torch, "zeros", guarded_zeros)
        allocation_guard.setattr(torch, "eye", guarded_eye)
        direct.initialize(kpoints, block_tridiagnal=True)
    assert direct.subblocks[0] == reference.subblocks[0]
    assert direct.subblocks[-1] == reference.subblocks[-1]
    tolerance = 2e-6 if dtype == "float32" else 1e-13
    for kpoint in kpoints:
        reference_parts = reference.get_hs_device(
            kpoint, V=0, block_tridiagonal=True
        )
        direct_parts = direct.get_hs_device(
            kpoint, V=0, block_tridiagonal=True
        )
        for indices in ((0, 5, 2), (1, 3, 4)):
            expected = dense_from_parts(
                [reference_parts[index] for index in indices]
            )
            actual = dense_from_parts(
                [direct_parts[index] for index in indices]
            )
            torch.testing.assert_close(
                actual, expected, atol=tolerance, rtol=tolerance
            )
        for tab in ("lead_L", "lead_R"):
            expected = reference.get_hs_lead(kpoint, tab, 0)
            actual = direct.get_hs_lead(kpoint, tab, 0)
            reduced_h, reduced_s = LeadProperty.HDL_reduced(
                expected[2],
                expected[5],
                reference.subblocks,
                tab=tab,
            )
            for index in (0, 1, 3, 4):
                torch.testing.assert_close(
                    actual[index],
                    expected[index],
                    atol=tolerance,
                    rtol=tolerance,
                )
            torch.testing.assert_close(
                actual[2], reduced_h, atol=tolerance, rtol=tolerance
            )
            torch.testing.assert_close(
                actual[5], reduced_s, atol=tolerance, rtol=tolerance
            )
    if dtype == "float64":
        expected = direct.get_hs_lead(kpoints[0], "lead_L", 0)
        torch.set_default_dtype(torch.float32)
        actual = direct.get_hs_lead(kpoints[0], "lead_L", 0)
        torch.set_default_dtype(torch.float64)
        for expected_matrix, actual_matrix in zip(expected, actual):
            torch.testing.assert_close(
                actual_matrix, expected_matrix, atol=0, rtol=0
            )
    with h5py.File(tmp_path / "direct/HS_device.h5", "r") as cache:
        assert cache.attrs["layout_version"] == 2
        assert cache.attrs["complete"]
        assert "Hall" not in cache
    with pytest.raises(ValueError, match="Bloch"):
        direct.initialize(
            kpoints, block_tridiagnal=True, useBloch=True
        )
    with pytest.raises(ValueError, match="plot_blocks"):
        direct.initialize(
            kpoints, block_tridiagnal=True, plot_blocks=True
        )
    with h5py.File(tmp_path / "direct/HS_lead_R.h5", "a") as cache:
        cache.attrs["cache_id"] = "different-physical-cache"
    with pytest.raises(ValueError, match="identities"):
        direct.get_hs_lead(kpoints[0], "lead_R", 0)


def test_orthogonal_chain_and_budget(tmp_path):
    fixture = ROOT / "dpnegf/tests/data/test_negf/test_negf_run"
    options = json.loads(
        (fixture / "negf_chain_new.json").read_text()
    )["task_options"]["stru_options"]
    matrices = []
    for mode in ("dense", "direct"):
        path = tmp_path / mode
        path.mkdir()
        torch.set_default_dtype(torch.float64)
        model = build_model(
            checkpoint=str(fixture / "nnsk_C_new.json"),
            common_options={"dtype": "float64", "device": "cpu"},
        )
        initializer = NEGFHamiltonianInit(
            model,
            {"r_max": 2.0},
            str(fixture / "chain.vasp"),
            True,
            options["pbc"],
            options,
            "Hartree",
            results_path=str(path),
            btd_initialization=mode,
        )
        with torch.no_grad():
            initializer.initialize(
                [[0, 0, 0]], block_tridiagnal=True
            )
        parts = initializer.get_hs_device(
            V=0, block_tridiagonal=True
        )
        matrices.append(
            dense_from_parts([parts[index] for index in (0, 5, 2)])
        )
        torch.testing.assert_close(
            dense_from_parts([parts[index] for index in (1, 3, 4)]),
            torch.eye(
                sum(initializer.subblocks), dtype=torch.complex128
            ),
        )
    torch.testing.assert_close(*matrices, atol=1e-12, rtol=1e-12)
    initializer.direct_btd_max_mib = 1e-12
    with pytest.raises(MemoryError, match="No matrices allocated"):
        initializer.initialize([[0, 0, 0]], block_tridiagnal=True)


def test_cnt_runner_transport_and_cache_reuse(tmp_path):
    results = []
    for mode in ("dense", "direct", "reuse_se", "reuse_hs"):
        config = json.loads((ROOT / "examples/CNT/input.json").read_text())
        config["dtype"] = "float64"
        task = config["task_options"]
        task["btd_initialization"] = (
            "dense" if mode == "dense" else "direct"
        )
        for key in ("emin", "emax", "espacing"):
            task.pop(key)
        task["energy_grid"] = {
            "method": "uniform",
            "emin": -1.0,
            "emax": 1.0,
            "num_points": 31,
        }
        task["e_fermi"] = 0.0
        task["output_options"] = {"tc": True}
        task["self_energy_options"]["numba_jit"] = False
        task["self_energy_options"]["parallel"] = {
            "cpu_budget": 1,
            "n_workers": 1,
            "blas_threads": 1,
        }
        if mode == "reuse_se":
            task["self_energy_options"]["cache"] = {
                "use_saved": True,
                "save_path": str(tmp_path / "dense/results/self_energy"),
            }
        if mode == "reuse_hs":
            task["hs_cache"] = {
                "use_saved": True,
                "save_path": str(tmp_path / "direct/results"),
            }
        task["rgf_options"] = {"device": "cpu", "e_batch_size": 3}
        config_path = tmp_path / f"{mode}.json"
        config_path.write_text(json.dumps(config))
        output = tmp_path / mode
        command = [
            sys.executable,
            "-c",
            (
                "from dpnegf.entrypoints.run import run; "
                f"run(INPUT={str(config_path)!r}, "
                f"init_model={str(ROOT / 'examples/CNT/nnsk_dftb.json')!r}, "
                f"structure={str(ROOT / 'examples/CNT/cnt7_0.xyz')!r}, "
                f"output={str(output)!r}, log_level=20, "
                f"log_path={str(output / 'run.log')!r})"
            ),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=dict(os.environ, MPLCONFIGDIR=str(tmp_path / "matplotlib")),
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert completed.returncode == 0, completed.stderr
        results.append(
            torch.load(
                output / "results/negf.out.pth", weights_only=False
            )["T_avg"]
        )
    for actual in results[1:]:
        torch.testing.assert_close(
            actual, results[0], atol=1e-10, rtol=1e-10
        )