"""
Finite-temperature, zero-bias conductance calculation.
"""

import numpy as np
from scipy.constants import elementary_charge, h

from dpnegf.utils.energy_grid import (
    EnergyGrid, build_energy_grid, fermi_window, thermal_energy, trapezoid_weights,
)
from scipy.special import expit

G0_S = 2 * elementary_charge ** 2 / h  

def _array(value):
    """Convert input data to a finite, real-valued float64 NumPy array."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if np.iscomplexobj(value) or not np.isfinite(value).all():
        raise ValueError("transmission must be real and finite")
    return value.astype(np.float64, copy=False)


def _result(transmission, weights, spin_degeneracy, metadata): 
    if spin_degeneracy not in (1, 2): 
        raise ValueError("spin_degeneracy must be 1 or 2") 
    values = _array(transmission) 
    if values.ndim == 0 or values.shape[-1] != len(weights): 
        raise ValueError("last transmission dimension must match the energy nodes") 
    integral = values @ np.asarray(weights) 
    reduced = integral * (spin_degeneracy / 2) 
    return dict(metadata, G_over_G0=reduced.tolist(), G_S=(G0_S * reduced).tolist(), 
                G0_S=G0_S, spin_degeneracy=spin_degeneracy)


def conductance_uniform(energies, transmission, mu=0.0, temperature=300.0, 
                        spin_degeneracy=2):
    """Trapezoidal integral of T(E)*(-df/dE) on a uniform energy grid.

    The integral is restricted to the supplied window; tails are not
    extrapolated or renormalized. Refinement is required to assess quadrature
    error. The omitted kernel mass alone is not a conductance error bound.
    """
    raw_energies = (
        energies.detach().cpu().numpy() if hasattr(energies, "detach") else np.asarray(energies)
    )
    precision = ( np.finfo(raw_energies.dtype).eps if np.issubdtype(raw_energies.dtype, np.floating) else 0.0)

    energies = _array(raw_energies)
    weights = trapezoid_weights(energies)
    delta = np.diff(energies)

    spacing_atol = max(1e-10, 4 * precision * np.max(np.abs(energies)),)

    if not np.allclose( delta, delta.mean(), rtol=1e-4, atol=spacing_atol):
        raise ValueError("conductance_uniform requires uniformly spaced energies")

    if not np.isfinite(mu):
        raise ValueError("mu must be finite")

    kbt = thermal_energy(temperature)
    tail = float(
        expit((energies[0] - mu) / kbt)
        + expit((mu - energies[-1]) / kbt)
    )

    return _result( transmission, weights * fermi_window(energies, mu, temperature), spin_degeneracy,
        {
            "method": "uniform",
            "mu_eV": float(mu),
            "temperature_K": float(temperature),
            "energy_unit": "eV",
            "window_eV": [
                float(energies[0]),
                float(energies[-1]),
            ],
            "kernel_mass": 1 - tail,
            "omitted_kernel_mass": tail,
        },
    )

def conductance_clenshaw_curtis(
            transmission, grid=None, *, mu=0.0,
            temperature=300.0, half_width=0.4,
            num_points=33,spin_degeneracy=2):
    """Integrate transmission evaluated at CC nodes, never interpolated.

    With no grid, transmission must follow the single-target fixed rule
    specified by the keyword arguments. Returns one record for a single target,
    or a list for a multi-target grid.
    """
    if grid is None:
        grid = build_energy_grid(
            {
                "method": "clenshaw_curtis",
                "num_points": num_points,
                "half_width": half_width,
            }, [mu], temperature,
        )

    if grid.method != "clenshaw_curtis":
        raise ValueError("conductance_clenshaw_curtis requires a CC energy grid")

    results = integrate_conductance(transmission,grid,spin_degeneracy)

    return results[0] if len(results) == 1 else results



def integrate_conductance(transmission,grid,spin_degeneracy=2):
    """Integrate all targets on an EnergyGrid without resampling transmission."""
    if not isinstance(grid, EnergyGrid):
        raise TypeError("grid must be an EnergyGrid returned by build_energy_grid")
    values = _array(transmission)

    if values.ndim == 0 or values.shape[-1] != len(grid.energies):
        raise ValueError("transmission must match the combined energy grid")

    return [
        _result(
            values[..., target["indices"]],
            target["weights"],
            spin_degeneracy,
            dict(
                target,
                method=grid.method,
                temperature_K=grid.temperature,
                energy_unit="eV",
            ),
        )
        for target in grid.targets
    ]