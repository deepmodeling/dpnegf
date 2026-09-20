"""
Fixed transmission grids and Landauer quadrature weights (energies in eV)
"""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.special import expit

from dpnegf.utils.constants import Boltzmann, eV2J



def thermal_energy(temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError( "temperature must be finite and greater than zero Kelvin")
    return Boltzmann * float(temperature) / eV2J


def fermi_window(energies, mu, temperature):
    kbt = thermal_energy(temperature)
    # expit(-abs(x)) avoids cancellation on the positive-energy tail.
    p = expit(-np.abs((np.asarray(energies) - mu) / kbt))

    return p * (1 - p) / kbt


def trapezoid_weights(energies):
    energies = np.asarray(energies, dtype=float)

    if (
        energies.ndim != 1
        or len(energies) < 2
        or not np.isfinite(energies).all()
        or np.any(np.diff(energies) <= 0)
    ):
        raise ValueError("energies must be finite, strictly increasing, and contain at least two points")

    delta = np.diff(energies)
    return np.r_[
        delta[0],
        delta[:-1] + delta[1:],
        delta[-1],
    ] / 2


def uniform_nodes(options):
    lo, hi = options["emin"], options["emax"]

    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError("uniform energy grid requires finite emin < emax")

    n, spacing = options.get("num_points"), options.get("espacing")

    if (n is None) == (spacing is None):
        raise ValueError( "uniform grid requires exactly one of num_points and espacing" )

    if spacing is not None:
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("espacing must be finite and positive")

        n = int((hi - lo) / spacing)

        if n < 2:
            raise ValueError("espacing must produce at least two energy points")

        # Preserve both the node count and torch default dtype of legacy inputs.
        return torch.linspace(lo, hi, steps=n).numpy()

    if (isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 2):
        raise ValueError("num_points must be an integer at least two")

    return np.linspace(lo, hi, n, dtype=np.float64)


def clenshaw_curtis_nodes(
    num_points=33,
    temperature=300.0,
    half_width=0.4,
):
    """Return finite energy offsets and probability-space integration weights.

    The finite interval is [-half_width, half_width]. Weights sum to its
    Fermi-window mass, not one. No transmission extrapolation is performed.
    Odd orders include zero; orders 2**m + 1 are nested.
    """
    n = num_points

    if (
        isinstance(n, bool)
        or not isinstance(n, (int, np.integer))
        or n < 3
        or n % 2 == 0
    ):
        raise ValueError(
            "Clenshaw-Curtis requires an odd num_points >= 3"
        )

    kbt = thermal_energy(temperature)

    if not np.isfinite(half_width) or half_width <= 0:
        raise ValueError("half_width must be finite and positive")

    order = n - 1
    theta = np.pi * np.arange(n) / order

    weights = np.empty(n)
    weights[0] = weights[-1] = 1 / (order * order - 1)

    interior = np.ones(order - 1)

    for k in range(1, order // 2):
        interior -= (2 * np.cos(2 * k * theta[1:-1])/ (4 * k * k - 1))

    interior -= (np.cos(order * theta[1:-1])/ (order * order - 1))
    weights[1:-1] = 2 * interior / order
    p0 = expit(-half_width / kbt)
    mass = np.tanh(half_width / (2 * kbt))

    # Compute the negative half and reflect to avoid logit(1) at low T.
    p = (p0 + mass * np.sin(theta[: n // 2 + 1] / 2)**2)

    offsets = np.empty(n)
    offsets[0] = -half_width
    offsets[1 : n // 2] = kbt * (np.log(p[1:-1]) - np.log1p(-p[1:-1]))
    offsets[n // 2] = 0.0
    offsets[n // 2 + 1 :] = -offsets[: n // 2][::-1]

    return offsets, weights * (mass / 2)


@dataclass
class EnergyGrid:
    """Sorted unique nodes and per-target indices/weights into that array."""

    energies: np.ndarray
    method: str
    temperature: float
    targets: list

    def metadata(self):
        return {
            "method": self.method,
            "temperature_K": self.temperature,
            "energy_unit": "eV",
            "targets": self.targets,
        }


def _uniform(options, mu, temperature):
    energies = uniform_nodes(options)

    return (
        energies,
        trapezoid_weights(energies)
        * fermi_window(energies, mu, temperature),
    )


def _clenshaw_curtis(options, mu, temperature):
    offsets, weights = clenshaw_curtis_nodes(
        options.get("num_points", 33),
        temperature,
        options.get("half_width", 0.4),
    )

    return mu + offsets, weights


GRID_METHODS = {
    "uniform": _uniform,
    "clenshaw_curtis": _clenshaw_curtis,
}


def build_energy_grid(
    options,
    mu=(0.0,),
    temperature=300.0,
    labels=None,
):
    """Build fixed nodes for all chemical potentials, relative to one E_ref.

    mu is a sequence of numeric energies in eV. Band-edge labels must be
    resolved by the caller. Repeated targets are retained but evaluated once.
    """
    method = options.get("method", "uniform")

    if method not in GRID_METHODS:
        raise ValueError(f"Unknown energy-grid method: {method}")

    thermal_energy(temperature)
    mu = np.asarray(mu, dtype=float)

    if (
        mu.ndim != 1
        or len(mu) == 0
        or not np.isfinite(mu).all()
    ):
        raise ValueError(
            "mu must be a nonempty sequence of finite energies"
        )

    labels = list(mu) if labels is None else list(labels)

    if len(labels) != len(mu):
        raise ValueError("labels and mu must have equal lengths")

    rules = [
        GRID_METHODS[method](options, center, temperature)
        for center in mu
    ]

    energies = np.unique(
        np.concatenate([e for e, w in rules])
    )

    # Existing SE cache uses E_{energy:.8f}; do not silently merge collisions.
    if len({f"{e:.8f}" for e in energies}) != len(energies):
        raise ValueError(
            "Distinct energy nodes collide at the self-energy "
            "cache precision (8 decimals)"
        )

    targets = []
    kbt = thermal_energy(temperature)

    for center, label, (nodes, weights) in zip(mu, labels, rules):
        tail = float(
            expit((nodes[0] - center) / kbt)
            + expit((center - nodes[-1]) / kbt)
        )

        targets.append(
            {
                "label": label,
                "mu_eV": float(center),
                "indices": np.searchsorted(energies, nodes).tolist(),
                "weights": weights.tolist(),
                "window_eV": [
                    float(nodes[0]),
                    float(nodes[-1]),
                ],
                "kernel_mass": 1 - tail,
                "omitted_kernel_mass": tail,
            }
        )

    return EnergyGrid(
        energies,
        method,
        float(temperature),
        targets,
    )