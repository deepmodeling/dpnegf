import json
from pathlib import Path

import numpy as np
import pytest
import torch
from dptb.data import AtomicData, AtomicDataDict
from dptb.nn.hr2hk import HR2HK

from dpnegf.tests.test_direct_btd import dense_from_parts

ROOT = Path(__file__).parent.parent.parent

def example_structure(material, repeat):
    """Extend the device while preserving both original lead contacts."""
    import numpy as np
    from ase.io import read

    structure = read(ROOT / f"examples/{material}/stru_negf.xyz")
    options = json.loads(
        (ROOT / f"examples/{material}/negf.json").read_text()
    )["task_options"]["stru_options"]
    original_device = structure[32:64]
    displacement = (
        structure.positions[64:80] - structure.positions[32:48]
    ).mean(axis=0)
    assert np.allclose(
        structure.positions[64:80] - structure.positions[32:48],
        displacement,
        atol=1e-6,
    )
    extended = structure[:32]
    for image_index in range(repeat):
        image = original_device.copy()
        image.positions += image_index * displacement
        extended += image
    right = structure[64:].copy()
    right.positions += (repeat - 1) * displacement
    extended += right
    cell = structure.cell.copy()
    cell[2] += (repeat - 1) * displacement
    extended.set_cell(cell)
    extended.set_pbc(structure.pbc)
    options["device"]["id"] = f"32-{32 + 32 * repeat}"
    options["lead_R"]["id"] = (
        f"{32 + 32 * repeat}-{64 + 32 * repeat}"
    )
    return extended, options


def prepare_example(material, mode, repeat, output, dtype="float64"):
    import torch
    from ase.io import write
    from dptb.nn.build import build_model

    from dpnegf.negf.negf_hamiltonian_init import NEGFHamiltonianInit
    from dpnegf.utils.argcheck import get_cutoffs_from_model_options

    torch.set_default_dtype(getattr(torch, dtype))
    structure, options = example_structure(material, repeat)
    output.mkdir(parents=True, exist_ok=True)
    write(output / "structure.xyz", structure)
    model = build_model(
        checkpoint=str(
            ROOT
            / f"examples/{material}/train/train_out/checkpoint/nnsk.ep3000.pth"
        ),
        common_options={"dtype": dtype, "device": "cpu"},
    )
    cutoffs = dict(
        zip(
            ("r_max", "er_max", "oer_max"),
            get_cutoffs_from_model_options(model.model_options),
        )
    )
    return NEGFHamiltonianInit(
        model,
        cutoffs,
        structure,
        True,
        options["pbc"],
        options,
        "eV",
        results_path=str(output),
        btd_initialization=mode,
    )


def initialize_example(
    material, mode, repeat, output, kpoints, dtype="float64"
):
    import torch

    initializer = prepare_example(
        material, mode, repeat, output, dtype
    )
    with torch.no_grad():
        initializer.initialize(kpoints, block_tridiagnal=True)
    return initializer

@pytest.mark.parametrize("material", ["graphene", "hBN"])
def test_2d_example_dense_equivalence(material, tmp_path, monkeypatch):
    original_dtype = torch.get_default_dtype()
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        kpoints = [[0.0, 0.0, 0.0], [0.0, 0.23, 0.0]]
        dense = initialize_example(
            material, "dense", 1, tmp_path / "dense", kpoints
        )
        forward = HR2HK.forward

        def lead_only(transform, data):
            assert len(data[AtomicDataDict.ATOM_TYPE_KEY]) <= 32
            return forward(transform, data)

        monkeypatch.setattr(HR2HK, "forward", lead_only)
        direct = initialize_example(
            material, "direct", 1, tmp_path / "direct", kpoints
        )
        full_size = sum(direct.subblocks)
        for kpoint in kpoints:
            old_parts = dense.get_hs_device(
                kpoint, V=0, block_tridiagonal=True
            )
            new_parts = direct.get_hs_device(
                kpoint, V=0, block_tridiagonal=True
            )
            for indices in ((0, 5, 2), (1, 3, 4)):
                expected = dense_from_parts(
                    [old_parts[index] for index in indices]
                )
                actual = dense_from_parts(
                    [new_parts[index] for index in indices]
                )
                torch.testing.assert_close(
                    actual, expected, atol=1e-12, rtol=1e-12
                )
            for tab in ("lead_L", "lead_R"):
                expected = dense.get_hs_lead(kpoint, tab, 0)
                actual = direct.get_hs_lead(kpoint, tab, 0)
                for index in (0, 1, 3, 4):
                    torch.testing.assert_close(
                        actual[index],
                        expected[index],
                        atol=1e-12,
                        rtol=1e-12,
                    )
                for index in (2, 5):
                    padded = torch.zeros(
                        (full_size, actual[index].shape[1]),
                        dtype=torch.complex128,
                    )
                    section = (
                        slice(0, actual[index].shape[0])
                        if tab == "lead_L"
                        else slice(
                            full_size - actual[index].shape[0],
                            full_size,
                        )
                    )
                    padded[section] = actual[index]
                    torch.testing.assert_close(
                        padded,
                        expected[index],
                        atol=1e-12,
                        rtol=1e-12,
                    )
        gamma = direct.get_hs_device(
            kpoints[0], V=0, block_tridiagonal=True
        )[0]
        non_gamma = direct.get_hs_device(
            kpoints[1], V=0, block_tridiagonal=True
        )[0]
        assert max(
            float((left - right).abs().max())
            for left, right in zip(gamma, non_gamma)
        ) > 1e-3
    finally:
        torch.set_default_dtype(original_dtype)
        torch.set_num_threads(original_threads)


@pytest.mark.parametrize("material", ["graphene", "hBN"])
def test_extended_sheet_contacts_and_periodicity(material):
    original, _ = example_structure(material, 1)
    extended, options = example_structure(material, 10)
    assert len(extended) == len(original) + 9 * 32
    assert options["pbc"] == [False, True, False]
    np.testing.assert_allclose(extended.cell[:2], original.cell[:2])
    np.testing.assert_allclose(
        extended.positions[:32], original.positions[:32]
    )
    shift = extended.positions[-32:] - original.positions[-32:]
    np.testing.assert_allclose(
        shift, np.broadcast_to(shift[0], shift.shape), atol=1e-7
    )
    assert (
        extended.get_chemical_symbols()[-32:]
        == original.get_chemical_symbols()[-32:]
    )


@pytest.mark.parametrize("material", ["graphene", "hBN"])
def test_2d_runner_transmission(material, tmp_path):
    from dpnegf.entrypoints.run import run

    original_dtype = torch.get_default_dtype()
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    outputs = []
    try:
        for mode in ("dense", "direct"):
            config = json.loads(
                (ROOT / f"examples/{material}/negf.json").read_text()
            )
            config["dtype"] = "float64"
            task = config["task_options"]
            for key in ("emin", "emax", "espacing"):
                task.pop(key, None)
            task["energy_grid"] = {
                "method": "uniform",
                "emin": -8.0,
                "emax": 2.0,
                "num_points": 11,
            }
            task["e_fermi"] = 0.0
            task["stru_options"]["compute_band_edges"] = False
            task["stru_options"]["kmesh"] = [1, 3, 1]
            task["output_options"] = {"tc": True}
            task["self_energy_options"] = {
                "solver": "Sancho-Rubio",
                "numba_jit": False,
                "parallel": {
                    "n_workers": 1,
                    "cpu_budget": 1,
                    "blas_threads": 1,
                },
            }
            task["rgf_options"] = {
                "device": "cpu",
                "e_batch_size": 4,
            }
            task["btd_initialization"] = mode
            config_path = tmp_path / f"{mode}.json"
            config_path.write_text(json.dumps(config))
            output = tmp_path / mode
            torch.set_default_dtype(torch.float64)
            run(
                INPUT=str(config_path),
                init_model=str(
                    ROOT
                    / f"examples/{material}/train/train_out/checkpoint/nnsk.ep3000.pth"
                ),
                structure=str(ROOT / f"examples/{material}/stru_negf.xyz"),
                output=str(output),
                log_level=20,
                log_path=str(output / "run.log"),
            )
            outputs.append(
                torch.load(
                    output / "results/negf.out.pth",
                    weights_only=False,
                )
            )
        torch.testing.assert_close(
            outputs[0]["T_avg"],
            outputs[1]["T_avg"],
            atol=1e-9,
            rtol=1e-9,
        )
        assert len(outputs[0]["T_k"]) > 1
        for key in outputs[0]["T_k"]:
            torch.testing.assert_close(
                outputs[0]["T_k"][key],
                outputs[1]["T_k"][key],
                atol=1e-9,
                rtol=1e-9,
            )
    finally:
        torch.set_default_dtype(original_dtype)
        torch.set_num_threads(original_threads)