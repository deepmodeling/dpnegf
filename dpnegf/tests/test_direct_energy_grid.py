import numpy as np
import pytest
import torch
from dargs.dargs import ArgumentKeyError

from dpnegf.utils.argcheck import normalize_run, validate_energy_options
from dpnegf.utils.energy_grid import build_energy_grid, direct_nodes, fermi_window
from dpnegf.utils.conductance import integrate_conductance


def config(key, energies):
    return {"task_options": {
        "task": "negf",
        "ele_T": 300,
        "unit": "eV",
        "energy_grid": {"method": "direct", key: energies},
        "stru_options": {
            "device": {"id": "4-8"},
            "lead_L": {"id": "0-4", "voltage": 0.0},
            "lead_R": {"id": "8-12", "voltage": 0.0},
        },
        "output_options": {"tc": True},
    }}


@pytest.mark.parametrize("key", ["energies", "enerigies"])
@pytest.mark.parametrize("energies", [[0.125], [-0.4, -0.03, 0, 0.27], [0.2, -0.1, 0.2]])
def test_direct_input_generates_exact_runner_grid(key, energies):
    from dpnegf.runner.NEGF import NEGF

    task = normalize_run(config(key, energies))["task_options"]
    assert task["energy_grid"] == {"method": "direct", "energies": energies}
    negf = NEGF.__new__(NEGF)
    negf.scf = False
    for name in ("density", "potential", "current", "lcurrent", "dos", "current_nscf", "ldos", "conductance"):
        setattr(negf, "out_" + name, False)
    negf.out_tc = True
    negf.energy_grid_options = task["energy_grid"]
    negf.generate_energy_grid()
    assert torch.equal(negf.uni_grid, torch.tensor(energies, dtype=torch.float64))
    assert negf.energy_grid is negf.uni_grid
    # The Python API also accepts either spelling without JSON normalization.
    np.testing.assert_array_equal(direct_nodes({key: energies}), energies)


@pytest.mark.parametrize("energies", [[], [float("nan")], [float("inf")], [True], ["0.1"], [[0.1]], [None]])
def test_invalid_direct_energies(energies):
    with pytest.raises(ValueError, match="nonempty list of finite numbers"):
        normalize_run(config("energies", energies))


def test_missing_direct_energies():
    raw = config("energies", [])
    del raw["task_options"]["energy_grid"]["energies"]
    with pytest.raises(ArgumentKeyError, match="energies"):
        normalize_run(raw)


@pytest.mark.parametrize("energies", [[0.1], [0.2, -0.1], [-0.1, -0.1, 0.2]])
def test_direct_conductance_requires_increasing_grid(energies):
    task = config("energies", energies)["task_options"]
    task["output_options"]["conductance"] = True
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_energy_options(task)


def test_direct_conductance_on_nonuniform_nodes():
    energies = np.array([-0.3, -0.1, -0.02, 0.0, 0.04, 0.15, 0.4])
    grid = build_energy_grid({"method": "direct", "energies": energies.tolist()}, mu=[0.0, 0.05])
    np.testing.assert_array_equal(grid.energies, energies)
    transmission = 1 + energies**2
    results = integrate_conductance(transmission, grid)
    for mu, result in zip([0.0, 0.05], results):
        expected = np.trapz(transmission * fermi_window(energies, mu, 300), energies)
        assert result["G_over_G0"] == pytest.approx(expected)
    assert grid.metadata()["method"] == "direct"
