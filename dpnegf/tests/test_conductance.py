import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.special import expit

from dpnegf.utils.argcheck import normalize_run
from dpnegf.utils.energy_grid import (
    build_energy_grid,
    clenshaw_curtis_nodes,
    thermal_energy,
    uniform_nodes,
)
from dpnegf.utils.conductance import (
    conductance_clenshaw_curtis,
    conductance_uniform,
    integrate_conductance,
    G0_S,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "dpnegf/tests/data/test_negf/test_negf_run"


@pytest.mark.parametrize("temperature", [10, 300, 1000])
@pytest.mark.parametrize("n", [17, 33, 65])
def test_cc_mass_symmetry_nesting(temperature, n):
    offsets, weights = clenshaw_curtis_nodes(n, temperature)

    assert np.isfinite(offsets).all()
    np.testing.assert_array_equal(offsets, -offsets[::-1])
    assert np.all(weights > 0)

    mass = np.tanh(0.4 / (2 * thermal_energy(temperature)))
    assert weights.sum() == pytest.approx(mass, abs=1e-14)

    refined, _ = clenshaw_curtis_nodes(2 * n - 1, temperature)
    np.testing.assert_allclose(
        offsets,
        refined[::2],
        atol=1e-15,
        rtol=0,
    )

    grid = build_energy_grid(
        {
            "method": "clenshaw_curtis",
            "num_points": n,
        },
        temperature=temperature,
    )
    result = conductance_clenshaw_curtis(np.ones(n), grid)

    assert result["G_over_G0"] == pytest.approx(mass)
    assert result["G_S"] == pytest.approx(G0_S * mass)


@pytest.mark.parametrize("shape", ["smooth", "narrow_peak", "step"])
def test_physical_spectra_against_quad(shape):
    kbt = thermal_energy(300)

    if shape == "smooth":
        spectrum = lambda e: 1 + 0.5 * np.cos(e / 0.03)
    elif shape == "narrow_peak":
        spectrum = lambda e: (
            0.002**2 / ((e - 0.02)**2 + 0.002**2)
        )
    else:
        spectrum = lambda e: np.asarray(e >= 0.017, dtype=float)

    kernel = lambda e: (
        expit(-abs(e / kbt))
        * (1 - expit(-abs(e / kbt)))
        / kbt
    )

    reference = quad(
        lambda e: spectrum(e) * kernel(e),
        -0.4,
        0.4,
        points=[0.017, 0.02],
        epsabs=1e-11,
    )[0]

    energies = np.linspace(-0.4, 0.4, 16001)
    result = conductance_uniform(energies, spectrum(energies))

    assert result["G_over_G0"] == pytest.approx(
        reference,
        rel=1e-3,
    )

    grid = build_energy_grid(
        {
            "method": "clenshaw_curtis",
            "num_points": 1025,
        }
    )
    cc = conductance_clenshaw_curtis(
        spectrum(grid.energies),
        grid,
    )

    assert cc["G_over_G0"] == pytest.approx(
        reference,
        rel=0.01,
    )


def test_multi_target_and_k_weighting():
    options = {"method": "clenshaw_curtis"}
    grid = build_energy_grid(options, [-0.1, 0.1, -0.1])

    assert len(grid.energies) == 66

    spectrum = np.stack(
        [
            1 + np.cos(grid.energies * 10),
            0.5 + grid.energies**2,
        ]
    )
    k_weights = np.array([0.3, 0.7])

    results = integrate_conductance(spectrum, grid)
    averaged = integrate_conductance(k_weights @ spectrum, grid)

    for result, avg in zip(results, averaged):
        assert (
            k_weights @ result["G_over_G0"]
        ) == pytest.approx(avg["G_over_G0"])

    assert averaged[0] == averaged[2]

    shifted = build_energy_grid(options, [0.2, 0.4, 0.2])
    values = np.stack(
        [
            1 + np.cos((shifted.energies - 0.3) * 10),
            0.5 + (shifted.energies - 0.3)**2,
        ]
    )

    for a, b in zip(
        results,
        integrate_conductance(values, shifted),
    ):
        np.testing.assert_allclose(
            a["G_over_G0"],
            b["G_over_G0"],
        )

    spin = integrate_conductance(
        spectrum,
        grid,
        spin_degeneracy=1,
    )
    np.testing.assert_allclose(
        spin[0]["G_S"],
        np.array(results[0]["G_S"]) / 2,
    )


def config():
    return json.loads(
        (DATA / "negf_chain_new.json").read_text()
    )


def test_legacy_grid_unchanged_and_new_precedence():
    old = config()
    task = old["task_options"]

    expected = torch.linspace(
        task["emin"],
        task["emax"],
        int(
            (task["emax"] - task["emin"])
            / task["espacing"]
        ),
    )

    normalized = normalize_run(copy.deepcopy(old))
    actual = uniform_nodes(
        normalized["task_options"]["energy_grid"]
    )

    np.testing.assert_array_equal(actual, expected.numpy())
    assert actual.dtype == expected.numpy().dtype

    old["task_options"]["energy_grid"] = {
        "method": "clenshaw_curtis"
    }
    old["task_options"]["out_current_nscf"] = False

    task = normalize_run(old)["task_options"]

    assert task["energy_grid"] == {
        "method": "clenshaw_curtis",
        "num_points": 33,
        "half_width": 0.4,
    }
    assert "emin" not in task


@pytest.mark.parametrize(
    "change",
    [
        {"ele_T": 0},
        {"unit": "Hartree"},
        {"scf": True},
        {"conductance_options": {"mu": []}},
        {"conductance_options": {"mu": ["bad"]}},
        {
            "energy_grid": {
                "method": "clenshaw_curtis",
                "num_points": 32,
            }
        },
        {"output_options": {"density": True}},
    ],
)
def test_invalid_configuration(change):
    raw = config()
    raw["task_options"]["energy_grid"] = {
        "method": "clenshaw_curtis"
    }
    raw["task_options"].update(change)

    with pytest.raises(ValueError):
        normalize_run(raw)


def test_bad_quadrature_inputs():
    with pytest.raises(ValueError, match="cache precision"):
        build_energy_grid(
            {"method": "clenshaw_curtis"},
            [0.0, 1e-10],
        )

    with pytest.raises(ValueError):
        conductance_uniform(
            [0, 1, 1.1],
            [1, 1, 1],
        )

    with pytest.raises(ValueError):
        conductance_clenshaw_curtis(np.ones(32))

    with pytest.raises(ValueError):
        conductance_clenshaw_curtis(np.full(33, np.nan))


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA unavailable",
            ),
        ),
    ],
)
def test_chain_runner_cache_and_batching(
    tmp_path,
    monkeypatch,
    device,
):
    from dpnegf.entrypoints.run import run
    import importlib

    runner = importlib.import_module("dpnegf.runner.NEGF")
    raw = config()
    task = raw["task_options"]

    task["energy_grid"] = {"method": "clenshaw_curtis"}
    task["conductance_options"] = {
        "mu": [0.0, 0.05, 0.0]
    }
    task["output_options"] = {"conductance": True}

    for key in list(task):
        if key.startswith("out_"):
            del task[key]

    task["rgf_options"] = {
        "device": device,
        "e_batch_size": 8,
    }

    cache = tmp_path / "self_energy"
    task["self_energy_options"] = {
        "cache": {
            "save_path": str(cache),
        },
        "numba_jit": False,
        "parallel": {
            "n_workers": 1,
            "blas_threads": 1,
        },
    }

    def execute(name):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(raw))
        out = tmp_path / name

        run(
            str(path),
            str(DATA / "nnsk_C_new.json"),
            str(DATA / "chain.vasp"),
            str(out),
            None,
        )

        return torch.load(
            out / "results/negf.out.pth",
            weights_only=False,
        )

    first = execute("cold")

    assert len(first["energy_grid"]) == 66
    assert first["conductance"][0]["G_over_G0"] == pytest.approx(
        1,
        rel=1e-4,
    )
    assert first["conductance"][0] == first["conductance"][2]

    task["self_energy_options"]["cache"]["use_saved"] = True
    task["rgf_options"]["e_batch_size"] = 1

    def no_precompute(*args, **kwargs):
        raise AssertionError(
            "cached run must not precompute self energy"
        )

    monkeypatch.setattr(
        runner,
        "compute_all_self_energy",
        no_precompute,
    )

    second = execute("warm")

    torch.testing.assert_close(
        first["T_avg"],
        second["T_avg"],
        rtol=1e-9,
        atol=1e-10,
    )

    grid = build_energy_grid(
        {"method": "clenshaw_curtis"},
        [0, 0.05, 0],
    )
    post = integrate_conductance(second["T_avg"], grid)

    for a, b in zip(post, second["conductance"]):
        assert a["G_S"] == pytest.approx(b["G_S"])