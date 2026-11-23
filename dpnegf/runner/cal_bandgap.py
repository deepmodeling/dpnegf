"""Standalone single-particle band-gap calculations from DeePTB checkpoints.

The model is evaluated once for a fixed geometry.  Its real-space Hamiltonian
and overlap blocks are then Fourier transformed for each requested k point.
The module deliberately keeps DeePTB imports inside the model/cache-miss path
so a validated H/S cache can be used without constructing a checkpoint model.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy import linalg, sparse
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import ArpackNoConvergence, eigsh, splu

log = logging.getLogger(__name__)

_CACHE_VERSION = 2
_ALGORITHM_VERSION = "realspace-blocks-v2"


def _finite_number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number.")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}.")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_fingerprint(value) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _real_symmetric_for_lu(matrix: sparse.spmatrix) -> Tuple[sparse.csc_matrix, int]:
    """Return a real symmetric matrix suitable for the checked SuperLU path.

    Complex Hermitian matrices are represented by their standard realification;
    every eigenvalue is duplicated, so the returned divisor is two.
    """
    matrix = sparse.csc_matrix(matrix)
    scale = max(1.0, float(np.max(np.abs(matrix.data))) if matrix.nnz else 0.0)
    if np.iscomplexobj(matrix.data) and matrix.nnz:
        imag_max = float(np.max(np.abs(matrix.data.imag)))
        if imag_max <= 1e-14 * scale:
            return sparse.csc_matrix(matrix.real), 1
        real = sparse.csc_matrix(matrix.real)
        imag = sparse.csc_matrix(matrix.imag)
        return sparse.bmat([[real, -imag], [imag, real]], format="csc"), 2
    return sparse.csc_matrix(matrix.real), 1


def _checked_superlu_negative_count(matrix: sparse.spmatrix) -> int:
    """Count negative eigenvalues only when SuperLU behaves as symmetric LDL."""
    matrix, divisor = _real_symmetric_for_lu(matrix)
    matrix = matrix.tocsc(copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.data = np.ascontiguousarray(matrix.data)
    matrix.indices = np.ascontiguousarray(matrix.indices)
    matrix.indptr = np.ascontiguousarray(matrix.indptr)
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("Inertia count requires a nonempty square matrix.")

    scale = max(1.0, float(np.max(np.abs(matrix.data))) if matrix.nnz else 0.0)
    asymmetry = matrix - matrix.T
    if asymmetry.nnz and float(np.max(np.abs(asymmetry.data))) > 1e-9 * scale:
        raise ValueError("Inertia count received a matrix that is not symmetric.")
    if asymmetry.nnz:
        matrix = (0.5 * (matrix + matrix.T)).tocsc()
        matrix.data = np.ascontiguousarray(matrix.data)
        matrix.indices = np.ascontiguousarray(matrix.indices)
        matrix.indptr = np.ascontiguousarray(matrix.indptr)

    try:
        factor = splu(
            matrix,
            permc_spec="MMD_AT_PLUS_A",
            diag_pivot_thresh=0.0,
            options={"SymmetricMode": True, "Equil": False},
        )
    except Exception as exc:
        raise RuntimeError(f"SuperLU could not factor the Hermitian matrix reliably: {exc}") from exc

    if not np.array_equal(factor.perm_r, factor.perm_c):
        raise RuntimeError("SuperLU used different row/column permutations; inertia is not reliable.")

    diagonal = np.asarray(factor.U.diagonal())
    d_scale = max(1.0, float(np.max(np.abs(diagonal))) if diagonal.size else 0.0)
    if np.iscomplexobj(diagonal) and float(np.max(np.abs(diagonal.imag))) > 1e-8 * d_scale:
        raise RuntimeError("SuperLU produced a materially complex diagonal; inertia is not reliable.")
    diagonal = diagonal.real
    if not diagonal.size or float(np.min(np.abs(diagonal))) < 1e-12 * scale:
        raise RuntimeError("SuperLU encountered a near-zero pivot; inertia is not reliable.")

    l_factor = factor.L.tocsc()
    u_factor = factor.U.tocsc()
    ldl_residual = u_factor - sparse.diags(diagonal, format="csc") @ l_factor.T
    ldl_max = float(np.max(np.abs(ldl_residual.data))) if ldl_residual.nnz else 0.0
    u_scale = max(1.0, float(np.max(np.abs(u_factor.data))) if u_factor.nnz else 0.0)
    if ldl_max > 1e-8 * u_scale:
        raise RuntimeError("SuperLU factors do not satisfy the checked LDL structure.")
    if u_scale > 1e8 * scale:
        raise RuntimeError("SuperLU factor growth is too large for a reliable inertia count.")

    n = matrix.shape[0]
    indices = np.arange(n, dtype=np.float64)
    rhs = np.column_stack((1.0 + (indices % 7) / 7.0, np.where(indices % 2, -1.0, 1.0)))
    solution = factor.solve(rhs)
    residual = matrix @ solution - rhs
    a_inf = float(np.max(np.asarray(np.abs(matrix).sum(axis=1))))
    x_inf = float(np.max(np.abs(solution)))
    b_inf = float(np.max(np.abs(rhs)))
    residual_inf = float(np.max(np.abs(residual)))
    backward_error = residual_inf / max(a_inf * x_inf + b_inf, np.finfo(float).tiny)
    if not math.isfinite(backward_error) or backward_error > 1e-9:
        raise RuntimeError(f"SuperLU backward error {backward_error:.3e} exceeds 1e-9.")

    negative = int(np.count_nonzero(diagonal < 0))
    if divisor == 2:
        if negative % 2:
            raise RuntimeError("Complex-Hermitian realification returned an unpaired inertia count.")
        negative //= 2
    return negative


def _inertia_count_below(hamiltonian, overlap, energy: float) -> int:
    shifted = hamiltonian - float(energy) * overlap
    return _checked_superlu_negative_count(shifted)


def _bisect_rank_intervals(hamiltonian, overlap, ranks, energy_bounds, edge_tol):
    """Locate several one-based eigenvalue ranks with shared inertia probes."""
    ranks = sorted(set(int(rank) for rank in ranks))
    n = hamiltonian.shape[0]
    if not ranks or ranks[0] < 1 or ranks[-1] > n:
        raise ValueError("Requested band rank is outside the model spectrum.")
    counts = {}

    def count(energy):
        key = float(energy)
        if key not in counts:
            counts[key] = _inertia_count_below(hamiltonian, overlap, key)
        return counts[key]

    lo, hi = map(float, energy_bounds)
    for _ in range(32):
        try:
            count_lo, count_hi = count(lo), count(hi)
        except (RuntimeError, ValueError):
            width = hi - lo
            lo -= width * 0.013
            hi += width * 0.017
            continue
        if count_lo < ranks[0] and count_hi >= ranks[-1]:
            break
        width = hi - lo
        if count_lo >= ranks[0]:
            lo -= width
        if count_hi < ranks[-1]:
            hi += width
    else:
        raise RuntimeError("Could not bracket requested bands by expanding energy_bounds.")

    intervals = {}

    def bisect(left, right, pending, depth=0):
        if not pending:
            return
        if right - left <= edge_tol:
            for rank in pending:
                intervals[rank] = (0.5 * (left + right), 0.5 * (right - left))
            return
        if depth > 128:
            raise RuntimeError("Inertia bisection exceeded 128 levels.")
        width = right - left
        probe = None
        below = None
        last_error = None
        for fraction in (0.5, 0.493, 0.517, 0.461, 0.539):
            candidate = left + fraction * width
            try:
                candidate_count = count(candidate)
            except (RuntimeError, ValueError) as exc:
                last_error = exc
                continue
            if left < candidate < right:
                probe, below = candidate, candidate_count
                break
        if probe is None:
            raise RuntimeError(f"All inertia probes failed inside [{left}, {right}].") from last_error
        lower_ranks = [rank for rank in pending if rank <= below]
        upper_ranks = [rank for rank in pending if rank > below]
        bisect(left, probe, lower_ranks, depth + 1)
        bisect(probe, right, upper_ranks, depth + 1)

    bisect(lo, hi, ranks)
    return intervals, len(counts)


def _solve_dense(hamiltonian, overlap, ranks):
    if not ranks:
        return {}, 0.0
    first, last = min(ranks) - 1, max(ranks) - 1
    try:
        values = linalg.eigh(
            hamiltonian.toarray(), overlap.toarray(),
            eigvals_only=True, subset_by_index=[first, last], driver="gvx",
            check_finite=False,
        )
    except linalg.LinAlgError as exc:
        raise ValueError(f"Dense generalized eigensolver failed; H/S may be non-Hermitian or S not positive definite: {exc}") from exc
    selected = {rank: float(values[rank - min(ranks)]) for rank in ranks}
    return {rank: (value, 0.0) for rank, value in selected.items()}, 0.0


def _solve_inertia(hamiltonian, overlap, ranks, solver_options):
    try:
        negative = _checked_superlu_negative_count(overlap)
    except Exception as exc:
        raise ValueError(f"Could not verify overlap positive definiteness with SuperLU: {exc}") from exc
    if negative:
        raise ValueError(f"Overlap matrix is not positive definite ({negative} negative modes).")
    intervals, probes = _bisect_rank_intervals(
        hamiltonian, overlap, ranks,
        solver_options["energy_bounds"], solver_options["edge_tol"],
    )
    return intervals, max((item[1] for item in intervals.values()), default=0.0), probes


def _solve_shift_invert(hamiltonian, overlap, ranks, solver_options):
    """Use inertia to verify the band indices returned by generalized eigsh."""
    reference, _, probes = _solve_inertia(hamiltonian, overlap, ranks, solver_options)
    safe_shift = 0.5 * (reference[ranks[len(ranks) // 2]][0] + reference[ranks[(len(ranks) - 1) // 2]][0])
    if abs(safe_shift - reference[ranks[0]][0]) < 4 * solver_options["edge_tol"]:
        safe_shift -= max(1e-4, 10 * solver_options["edge_tol"])
    shifts = [solver_options["energy_shift"], safe_shift]
    shifts = [float(shift) for i, shift in enumerate(shifts) if shift is not None and shift not in shifts[:i]]
    n = hamiltonian.shape[0]
    if n <= 2:
        values, allowance = _solve_dense(hamiltonian, overlap, ranks)
        return values, allowance, probes, "dense-fallback"

    last_error = None
    for shift in shifts:
        max_eigs = n - 2  # complex Hermitian eigsh delegates to eigs, which needs k < N - 1.
        n_eigs = min(max_eigs, max(len(ranks) + 4, 8))
        while n_eigs < n:
            try:
                values, vectors = eigsh(
                    hamiltonian, k=n_eigs, M=overlap, sigma=shift, which="LM",
                    tol=1e-10, maxiter=max(1000, 20 * n),
                )
            except ArpackNoConvergence as exc:
                last_error = RuntimeError(f"ARPACK did not converge near shift {shift:g} eV.")
                n_eigs = min(max_eigs, max(n_eigs + 4, n_eigs * 2))
                if n_eigs == max_eigs and n_eigs + 4 >= n:
                    last_error = RuntimeError("ARPACK failed at the maximum sparse Ritz window.")
                    break
                continue
            order = np.argsort(values)
            values, vectors = np.asarray(values[order]), np.asarray(vectors[:, order])

            # Rayleigh-Ritz removes loss of S-orthogonality from the ARPACK subspace.
            projected_h = vectors.conj().T @ (hamiltonian @ vectors)
            projected_s = vectors.conj().T @ (overlap @ vectors)
            try:
                ritz, rotation = linalg.eigh(projected_h, projected_s, check_finite=False)
            except linalg.LinAlgError as exc:
                last_error = RuntimeError(f"Rayleigh-Ritz projection failed near shift {shift:g} eV: {exc}")
                break
            coefficients = vectors @ rotation
            hc = hamiltonian @ coefficients
            sc = overlap @ coefficients
            residual_vectors = hc - sc * ritz[np.newaxis, :]
            denominator = np.linalg.norm(hc, axis=0) + np.abs(ritz) * np.linalg.norm(sc, axis=0)
            residuals = np.linalg.norm(residual_vectors, axis=0) / np.maximum(denominator, np.finfo(float).tiny)
            orthogonality = coefficients.conj().T @ sc - np.eye(len(ritz))
            orth_error = float(np.max(np.abs(orthogonality)))
            if float(np.max(residuals)) > 1e-8 or orth_error > 1e-7:
                last_error = RuntimeError(
                    "Shift-invert Ritz checks failed: max residual="
                    f"{np.max(residuals):.3e}, S-orthogonality error={orth_error:.3e}."
                )
                break

            margin = max(1e-6, float(np.max(np.abs(ritz))) * 1e-9)
            complete = False
            for _ in range(12):
                try:
                    low_count = _inertia_count_below(hamiltonian, overlap, float(ritz[0] - margin))
                    high_count = _inertia_count_below(hamiltonian, overlap, float(ritz[-1] + margin))
                except (RuntimeError, ValueError) as exc:
                    last_error = exc
                    margin *= 3
                    continue
                if (
                    high_count - low_count == len(ritz)
                    and low_count < min(ranks)
                    and high_count >= max(ranks)
                ):
                    complete = True
                    break
                if high_count - low_count > len(ritz):
                    break
                margin *= 3
            if complete:
                results = {}
                for rank in ranks:
                    index = rank - low_count - 1
                    if not 0 <= index < len(ritz):
                        complete = False
                        break
                    allowance = max(1e-8, float(residuals[index]) * max(1.0, abs(float(ritz[index]))))
                    results[rank] = (float(ritz[index]), allowance)
                if complete:
                    return results, max(v[1] for v in results.values()), probes, "shift-invert"
            last_error = RuntimeError(
                "Shift-invert spectrum was incomplete or did not cover the requested global band indices."
            )
            if n_eigs == max_eigs:
                break
            n_eigs = min(max_eigs, max(n_eigs + 4, n_eigs * 2))

    if n <= solver_options["dense_limit"]:
        dense_values, dense_allowance = _solve_dense(hamiltonian, overlap, ranks)
        return dense_values, dense_allowance, probes, "dense-fallback"
    raise RuntimeError(f"Checked shift-invert solve failed: {last_error}")


class BandGapCalculator:
    """Calculate a finite HOMO-LUMO gap or one-dimensional periodic gap.

    Parameters follow the `cal_bandgap` task schema. The checkpoint identifies
    the model and is loaded only when the H/S block cache cannot be used.
    """

    def __init__(
        self,
        structure,
        checkpoint: str,
        device: str = "cpu",
        dtype: Optional[str] = None,
        task_options: Optional[dict] = None,
        results_path: Optional[str] = None,
        pbc=None,
        AtomicData_options: Optional[dict] = None,
    ):
        if device is not None and str(device) != "cpu":
            raise ValueError("cal_bandgap currently supports CPU model inference and eigensolving only.")
        if dtype not in (None, "float32", "float64"):
            raise ValueError("dtype must be None, 'float32', or 'float64'.")
        self.structure = structure
        if checkpoint is None:
            raise ValueError("A model checkpoint is required to validate the H/S cache.")
        self.checkpoint = Path(checkpoint).expanduser()
        self._model = None
        self._hr2hk_data = None
        self._hr2hk_h = None
        self._hr2hk_s = None
        self.dtype = dtype
        supplied_options = dict(task_options or {})
        supplied_options.pop("task", None)
        supplied_options.pop("use_gui", None)
        # Keep the Python API under the same strict schema and cross-field
        # validation as `dpnegf run`; the calculator remains usable without the
        # CLI and receives normalized defaults either way.
        from dpnegf.utils.argcheck import normalize_run

        normalized = normalize_run({
            "pbc": pbc,
            "task_options": {"task": "cal_bandgap", **supplied_options},
        })
        self.task_options = normalized["task_options"]
        self.pbc_override = normalized.get("pbc")
        self.results_path = Path(results_path) if results_path is not None else None
        self.atomic_data_options = AtomicData_options
        self.structure_options = dict(self.task_options.get("structure_options") or {})
        self.occupation_options = dict(self.task_options.get("occupation_options") or {})
        self.solver_options = dict(self.task_options.get("solver_options") or {})
        self.k_options = dict(self.task_options.get("k_options") or {})
        self.output_options = dict(self.task_options.get("output_options") or {})
        self.cache_options = dict(self.task_options.get("cache_options") or {})
        self._set_defaults()

    def _set_defaults(self):
        self.structure_options.setdefault("mode", "auto")
        self.structure_options.setdefault("axis", 2)
        self.structure_options.setdefault("periodic_length", None)
        self.occupation_options.setdefault("nel_atom", None)
        self.occupation_options.setdefault("charge", 0.0)
        for key, value in {
            "method": "auto", "backend": "superlu", "dense_limit": 1000,
            "edge_tol": 1e-5, "energy_bounds": [-10.0, 10.0], "energy_shift": None,
        }.items():
            self.solver_options.setdefault(key, value)
        for key, value in {"initial_nk": 3, "max_nk": 9, "gap_tol": 0.01, "local_refine": False}.items():
            self.k_options.setdefault(key, value)
        for key, value in {"bands_per_side": 4, "all_bands": False, "plot": True,
                           "plot_nk": 0, "plot_window": None}.items():
            self.output_options.setdefault(key, value)
        self.cache_options.setdefault("use_saved", True)
        self.cache_options.setdefault("save", True)
        self.cache_options.setdefault("path", None)
        if self.solver_options["backend"] != "superlu":
            raise ValueError("Only solver_options.backend='superlu' is supported.")

    def _read_structure(self):
        from ase import Atoms
        from ase.io import read

        if isinstance(self.structure, Atoms):
            atoms = self.structure.copy()
        elif isinstance(self.structure, (str, os.PathLike)):
            atoms = read(os.fspath(self.structure))
        else:
            raise TypeError("structure must be an ASE Atoms object or a structure file path.")
        if len(atoms) == 0:
            raise ValueError("The input structure contains no atoms.")
        if self.pbc_override is not None:
            pbc = self.pbc_override
            if isinstance(pbc, bool):
                pbc = [pbc] * 3
            if len(pbc) != 3 or not all(isinstance(value, (bool, np.bool_)) for value in pbc):
                raise ValueError("pbc must be a bool or a list of three booleans.")
            atoms.set_pbc(pbc)

        mode = self.structure_options["mode"]
        axis = self.structure_options["axis"]
        if mode not in ("auto", "finite", "periodic") or axis not in (0, 1, 2):
            raise ValueError("Invalid structure mode or periodic axis.")
        active_axes = np.flatnonzero(np.asarray(atoms.pbc, dtype=bool)).tolist()
        if mode == "auto":
            if len(active_axes) > 1:
                raise ValueError("Only finite or one-dimensional periodic structures are supported.")
            mode = "periodic" if active_axes else "finite"
            if active_axes:
                axis = active_axes[0]
        if mode == "finite":
            if active_axes or self.structure_options["periodic_length"] is not None:
                raise ValueError("Finite mode conflicts with a periodic boundary or periodic_length.")
            atoms.set_pbc([False, False, False])
        else:
            if active_axes and active_axes != [axis]:
                raise ValueError("Periodic mode requires exactly one periodic axis matching structure_options.axis.")
            periodic_length = self.structure_options["periodic_length"]
            cell = np.asarray(atoms.cell, dtype=float).copy()
            vector = cell[axis]
            vector_length = float(np.linalg.norm(vector))
            if periodic_length is not None:
                periodic_length = _finite_number(periodic_length, "periodic_length", positive=True)
                if vector_length <= 1e-12:
                    vector = np.eye(3)[axis] * periodic_length
                else:
                    vector = vector * (periodic_length / vector_length)
                cell[axis] = vector
                atoms.set_cell(cell, scale_atoms=False)
            elif vector_length <= 1e-12:
                raise ValueError("Periodic mode needs a nonzero cell vector or periodic_length.")
            atoms.set_pbc([i == axis for i in range(3)])
        if not np.all(np.isfinite(atoms.positions)) or not np.all(np.isfinite(np.asarray(atoms.cell))):
            raise ValueError("Structure positions and cell must be finite.")
        return atoms, mode, axis

    def _source_fingerprint(self):
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Model checkpoint does not exist: {self.checkpoint}")
        return "checkpoint:" + _sha256_file(self.checkpoint)

    def _cache_path(self):
        path = self.cache_options.get("path")
        if path is not None:
            return Path(path).expanduser()
        return self.results_path / "hs_blocks.npz" if self.results_path is not None else None

    def _spectrum_cache_path(self, identity):
        hs_path = self._cache_path()
        if hs_path is None:
            return None
        cache_dir = hs_path.parent / (hs_path.stem + "_spectra")
        return cache_dir / f"{identity}.npz"

    def _spectrum_identity(self, kpoint, ranks, occupied):
        return _json_fingerprint({
            "cache_version": _CACHE_VERSION,
            "algorithm_version": _ALGORITHM_VERSION,
            "hs_cache_key": self._hs_cache_key,
            "kpoint_fractional": np.asarray(kpoint, dtype=float).round(12).tolist(),
            "occupied_bands": int(occupied),
            "band_indices": list(map(int, ranks)),
            "solver_options": self.solver_options,
        })

    def _load_cached_spectrum(self, path, identity, ranks):
        if path is None or not self.cache_options["use_saved"] or not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"].item()))
                if metadata.get("identity") != identity:
                    return None
                cached_ranks = np.asarray(archive["band_indices"], dtype=int).tolist()
                values = np.asarray(archive["eigenvalues_eV"], dtype=float)
                allowances = np.asarray(archive["band_allowances_eV"], dtype=float)
                if (
                    cached_ranks != list(map(int, ranks))
                    or values.shape != (len(ranks),)
                    or allowances.shape != (len(ranks),)
                ):
                    return None
                spectrum = {
                    rank: (float(value), float(allowance))
                    for rank, value, allowance in zip(ranks, values, allowances)
                }
                return spectrum, float(metadata["allowance_eV"]), metadata["solver"], int(metadata["inertia_probes"])
        except Exception as exc:
            log.warning("Ignoring unreadable per-k spectrum cache %s: %s", path, exc)
            return None

    def _save_cached_spectrum(self, path, identity, spectrum, ranks, allowance, solver, probes):
        if path is None or not self.cache_options["save"]:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "identity": identity,
            "allowance_eV": float(allowance),
            "solver": str(solver),
            "inertia_probes": int(probes),
        }
        temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(
                    stream,
                    metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                    band_indices=np.asarray(ranks, dtype=int),
                    eigenvalues_eV=np.asarray([spectrum[rank][0] for rank in ranks], dtype=float),
                    band_allowances_eV=np.asarray([spectrum[rank][1] for rank in ranks], dtype=float),
                )
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _cache_key(self, atoms, mode, axis, source_fingerprint):
        import torch

        explicit_cutoffs = self.atomic_data_options
        payload = {
            "cache_version": _CACHE_VERSION,
            "algorithm_version": _ALGORITHM_VERSION,
            "dpnegf_version": _package_version("dpnegf"),
            "dptb_version": _package_version("dptb"),
            "source": source_fingerprint,
            "symbols": atoms.get_chemical_symbols(),
            "positions": np.asarray(atoms.positions, dtype=np.float64).tolist(),
            "cell": np.asarray(atoms.cell, dtype=np.float64).tolist(),
            "pbc": np.asarray(atoms.pbc, dtype=bool).tolist(),
            "mode": mode,
            "axis": axis,
            "dtype": self.dtype,
            "graph_default_dtype": str(torch.get_default_dtype()),
            "AtomicData_options": explicit_cutoffs,
        }
        return _json_fingerprint(payload)

    def _load_cached_blocks(self, cache_path, cache_key):
        if cache_path is None or not self.cache_options["use_saved"] or not cache_path.is_file():
            return None
        try:
            with np.load(cache_path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"].item()))
                if metadata.get("cache_key") != cache_key or metadata.get("cache_version") != _CACHE_VERSION:
                    log.info("Ignoring H/S cache with mismatched structure, model, cutoff, or format fingerprint.")
                    return None
                h_blocks = {
                    key: np.array(archive[f"h_{i}"], copy=True)
                    for i, key in enumerate(metadata["h_keys"])
                }
                s_blocks = None
                if metadata["overlap_present"]:
                    s_blocks = {
                        key: np.array(archive[f"s_{i}"], copy=True)
                        for i, key in enumerate(metadata["s_keys"])
                    }
                norbs = [int(value) for value in metadata["atom_norbs"]]
                model_metadata = metadata.get("model_metadata", {})
            return h_blocks, s_blocks, norbs, model_metadata
        except Exception as exc:
            log.warning("Ignoring unreadable H/S cache %s: %s", cache_path, exc)
            return None

    @staticmethod
    def _write_cached_blocks(cache_path, cache_key, h_blocks, s_blocks, norbs, model_metadata):
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        h_keys = sorted(h_blocks)
        s_keys = sorted(s_blocks) if s_blocks is not None else []
        metadata = {
            "cache_version": _CACHE_VERSION,
            "cache_key": cache_key,
            "algorithm_version": _ALGORITHM_VERSION,
            "atom_norbs": list(map(int, norbs)),
            "overlap_present": s_blocks is not None,
            "h_keys": h_keys,
            "s_keys": s_keys,
            "model_metadata": model_metadata,
        }
        arrays = {f"h_{i}": np.asarray(h_blocks[key]) for i, key in enumerate(h_keys)}
        arrays.update({f"s_{i}": np.asarray(s_blocks[key]) for i, key in enumerate(s_keys)})
        arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True, default=str))
        temporary = cache_path.with_name(cache_path.name + f".{os.getpid()}.tmp")
        try:
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            os.replace(temporary, cache_path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _load_model(self):
        if self._model is None:
            from dptb.nn.build import build_model

            common_options = {"device": "cpu"}
            if self.dtype is not None:
                common_options["dtype"] = self.dtype
            self._model = build_model(checkpoint=str(self.checkpoint), common_options=common_options)
        model_device = getattr(self._model, "device", None)
        if model_device is None:
            try:
                model_device = next(self._model.parameters()).device
            except (AttributeError, StopIteration):
                model_device = "cpu"
        if str(model_device).split(":")[0] != "cpu":
            raise ValueError(
                f"The checkpoint model is on {model_device}; cal_bandgap requires CPU model inference."
            )
        if self.dtype is not None and str(getattr(self._model, "dtype", "")).split(".")[-1] != self.dtype:
            raise ValueError("The checkpoint model dtype does not match the requested dtype.")
        self._model.eval()
        return self._model

    def _effective_atomic_data_options(self, model):
        from dptb.utils.argcheck import get_cutoffs_from_model_options

        expected = dict(zip(("r_max", "er_max", "oer_max"), get_cutoffs_from_model_options(model.model_options)))
        if self.atomic_data_options is not None:
            for key, value in self.atomic_data_options.items():
                if value is not None and value != expected.get(key):
                    raise ValueError(
                        f"AtomicData_options.{key}={value!r} differs from the model cutoff "
                        f"{expected.get(key)!r}; changing the graph cutoff changes the prediction."
                    )
        return expected

    @staticmethod
    def _to_numpy_blocks(blocks):
        converted = {}
        for key, value in blocks.items():
            array = value.detach().to(device="cpu").numpy() if hasattr(value, "detach") else np.asarray(value)
            if not np.all(np.isfinite(array)):
                raise ValueError(f"Predicted real-space block {key} contains non-finite values.")
            converted[key] = np.asarray(array, dtype=np.float64)
        return converted

    def _predict_blocks(self, atoms):
        import torch
        from dptb.data import AtomicData, AtomicDataDict
        from dptb.data.interfaces.ham_to_feature import feature_to_block

        model = self._load_model()
        options = self._effective_atomic_data_options(model)
        data = AtomicData.from_ase(atoms, **options)
        data = AtomicData.to_AtomicDataDict(data.to(device="cpu"))
        for key in (AtomicDataDict.POSITIONS_KEY, AtomicDataDict.CELL_KEY):
            if key in data:
                data[key] = data[key].to(dtype=model.dtype)
        data = model.idp(data)
        data[AtomicDataDict.KPOINT_KEY] = torch.nested.as_nested_tensor(
            [torch.zeros((1, 3), dtype=model.dtype, device="cpu")]
        )
        with torch.no_grad():
            predicted = model(data)

        soc_key = AtomicDataDict.NODE_SOC_SWITCH_KEY
        soc = predicted.get(soc_key, False)
        if hasattr(soc, "numel") and soc.numel():
            soc = bool(soc.bool().all().item())
        if soc:
            raise ValueError("SOC/spinor Hamiltonians are outside the initial cal_bandgap scope.")

        h_blocks = self._to_numpy_blocks(feature_to_block(predicted, model.idp))
        overlap_present = AtomicDataDict.NODE_OVERLAP_KEY in predicted
        s_blocks = self._to_numpy_blocks(feature_to_block(predicted, model.idp, overlap=True)) if overlap_present else None
        atom_norbs = []
        for symbol in atoms.get_chemical_symbols():
            try:
                type_index = model.idp.chemical_symbol_to_type[symbol]
            except KeyError as exc:
                raise ValueError(f"The model basis does not include element {symbol}.") from exc
            atom_norbs.append(int(model.idp.atom_norb[type_index]))
        if not h_blocks:
            raise ValueError("DeePTB did not produce any real-space Hamiltonian blocks.")
        model_metadata = {
            "basis": model.idp.basis,
            "idp_method": model.idp.method,
            "AtomicData_options": options,
            "dtype": str(model.dtype).split(".")[-1],
        }
        return h_blocks, s_blocks, atom_norbs, model_metadata

    def _prepare_hr2hk(self, h_blocks, s_blocks, model_metadata):
        """Rebuild DeePTB feature data from cached real-space blocks for HR2HK."""
        self._hr2hk_data = None
        self._hr2hk_h = None
        self._hr2hk_s = None
        if model_metadata.get("idp_method", "e3tb") != "e3tb":
            return False

        import torch
        from dptb.data import AtomicData, AtomicDataDict
        from dptb.data.interfaces.ham_to_feature import block_to_feature
        from dptb.data.transforms import OrbitalMapper
        from dptb.nn.hr2hk import HR2HK

        basis = model_metadata.get("basis")
        atomic_options = model_metadata.get("AtomicData_options")
        if not basis or not atomic_options:
            raise ValueError("The H/S cache lacks the DeePTB basis or graph cutoffs needed by HR2HK.")

        idp = OrbitalMapper(basis=basis, method="e3tb", device="cpu")
        graph = AtomicData.from_ase(self._atoms, **atomic_options)
        if graph[AtomicDataDict.EDGE_INDEX_KEY].shape[1] == 0:
            return False
        data = AtomicData.to_AtomicDataDict(graph.to(device="cpu"))
        data = idp(data)
        block_to_feature(
            data,
            idp,
            blocks=h_blocks,
            overlap_blocks=s_blocks if s_blocks is not None else False,
        )
        dtype = getattr(torch, model_metadata.get("dtype", "float32"))
        for key in (
            AtomicDataDict.NODE_FEATURES_KEY,
            AtomicDataDict.EDGE_FEATURES_KEY,
            AtomicDataDict.NODE_OVERLAP_KEY,
            AtomicDataDict.EDGE_OVERLAP_KEY,
        ):
            if key in data:
                data[key] = data[key].to(dtype=dtype, device="cpu")

        self._hr2hk_data = data
        self._hr2hk_h = HR2HK(idp=idp, dtype=dtype, device="cpu")
        self._hr2hk_s = None
        if s_blocks is not None:
            self._hr2hk_s = HR2HK(
                idp=idp,
                edge_field=AtomicDataDict.EDGE_OVERLAP_KEY,
                node_field=AtomicDataDict.NODE_OVERLAP_KEY,
                out_field=AtomicDataDict.OVERLAP_KEY,
                overlap=True,
                dtype=dtype,
                device="cpu",
            )
        return True

    def _hr2hk_fourier(self, kpoint, *, overlap=False):
        import torch
        from dptb.data import AtomicDataDict

        transform = self._hr2hk_s if overlap else self._hr2hk_h
        if transform is None:
            raise ValueError("The checkpoint does not provide overlap blocks.")
        dtype = transform.dtype
        data = dict(self._hr2hk_data)
        point = torch.as_tensor(np.asarray(kpoint).reshape(1, 3), dtype=dtype, device="cpu")
        data[AtomicDataDict.KPOINT_KEY] = torch.nested.as_nested_tensor([point])
        output_key = AtomicDataDict.OVERLAP_KEY if overlap else AtomicDataDict.HAMILTONIAN_KEY
        matrix = transform(data)[output_key][0].detach().cpu().numpy()
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"DeePTB HR2HK produced non-finite {'S' if overlap else 'H'}(k).")
        return sparse.csr_matrix(np.asarray(matrix, dtype=np.complex128))

    @staticmethod
    def _assemble_sparse_blocks(blocks, atom_norbs, kpoint):
        """Sparse Bloch assembly for spectra too large for dense HR2HK output."""
        offsets = np.concatenate(([0], np.cumsum(atom_norbs))).astype(np.int64)
        dimension = int(offsets[-1])
        rows, columns, values = [], [], []
        for key, block in blocks.items():
            parts = list(map(int, key.split("_")))
            atom_i, atom_j, shift = parts[0], parts[1], np.asarray(parts[2:5], dtype=float)
            if not (0 <= atom_i < len(atom_norbs) and 0 <= atom_j < len(atom_norbs)):
                raise ValueError(f"Invalid atom index in DeePTB real-space block {key}.")
            block = np.asarray(block)
            if block.shape != (atom_norbs[atom_i], atom_norbs[atom_j]):
                raise ValueError(f"Unexpected orbital block shape {block.shape} for {key}.")
            row = offsets[atom_i] + np.arange(atom_norbs[atom_i])
            col = offsets[atom_j] + np.arange(atom_norbs[atom_j])
            rr, cc = np.meshgrid(row, col, indexing="ij")
            phase = np.exp(-2j * np.pi * float(np.dot(kpoint, shift)))
            if atom_i == atom_j and not np.any(shift):
                # feature_to_block already materializes the complete onsite block.
                rows.append(rr.ravel())
                columns.append(cc.ravel())
                values.append(block.ravel())
                continue
            phased = block * phase
            rows.extend((rr.ravel(), cc.T.ravel()))
            columns.extend((cc.ravel(), rr.T.ravel()))
            values.extend((phased.ravel(), phased.conj().T.ravel()))

        if values:
            matrix = sparse.coo_matrix(
                (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
                shape=(dimension, dimension), dtype=np.complex128,
            ).tocsr()
        else:
            matrix = sparse.csr_matrix((dimension, dimension), dtype=np.complex128)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        return matrix

    def _make_solver(self, h_blocks, s_blocks, atom_norbs, model_metadata):
        if self.output_options["all_bands"] and self.solver_options["method"] not in ("auto", "dense"):
            raise ValueError("all_bands=True requires the dense solver.")
        dimension = sum(atom_norbs)
        if dimension < 2:
            raise ValueError("At least two spatial orbitals are required for an occupied/unoccupied gap.")
        nel_atom = self.occupation_options["nel_atom"]
        if nel_atom is None:
            from dpnegf.utils.constants import valence_electron
            nel_atom = valence_electron
        electrons = 0.0
        for symbol in self._atoms.get_chemical_symbols():
            if symbol not in nel_atom:
                raise ValueError(f"No valence electron count was provided for element {symbol}.")
            electrons += float(nel_atom[symbol])
        electrons -= _finite_number(self.occupation_options["charge"], "charge")
        if electrons <= 0:
            raise ValueError("The electron count must be positive after applying charge.")
        occupied_float = electrons / 2.0
        occupied = int(round(occupied_float))
        if not math.isclose(occupied_float, occupied, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("The initial calculator supports nonmagnetic even-electron integer filling only.")
        if not 1 <= occupied < dimension:
            raise ValueError(f"Occupied band index {occupied} is outside the {dimension}-orbital spectrum.")
        self._electron_count = float(electrons)
        if self._model is not None and getattr(self._model, "soc_param", None) is not None:
            raise ValueError("SOC/spinor Hamiltonians are outside the initial cal_bandgap scope.")

        if self.output_options["all_bands"]:
            ranks = list(range(1, dimension + 1))
        else:
            side = int(self.output_options["bands_per_side"])
            ranks = list(range(max(1, occupied - side + 1), min(dimension, occupied + side) + 1))
        method = self.solver_options["method"]
        if method == "auto":
            method = "dense" if dimension <= self.solver_options["dense_limit"] else "inertia"
        if self.output_options["all_bands"] and method != "dense":
            raise ValueError("The selected automatic solver is sparse; all_bands requires dimension <= dense_limit or method='dense'.")
        use_hr2hk = method == "dense" and self._prepare_hr2hk(h_blocks, s_blocks, model_metadata)

        def solve(kpoint):
            if use_hr2hk:
                h = self._hr2hk_fourier(kpoint)
            else:
                h = self._assemble_sparse_blocks(h_blocks, atom_norbs, kpoint)
            if s_blocks is None:
                s = sparse.eye(dimension, dtype=np.complex128, format="csr")
            elif use_hr2hk:
                s = self._hr2hk_fourier(kpoint, overlap=True)
            else:
                s = self._assemble_sparse_blocks(s_blocks, atom_norbs, kpoint)
            self._check_hermitian(h, "H")
            self._check_hermitian(s, "S")
            if method == "dense":
                spectrum, allowance = _solve_dense(h, s, ranks)
                return spectrum, allowance, "dense", 0
            if method == "inertia":
                spectrum, allowance, probes = _solve_inertia(h, s, ranks, self.solver_options)
                return spectrum, allowance, "inertia", probes
            if method == "shift-invert":
                spectrum, allowance, probes, solver_name = _solve_shift_invert(h, s, ranks, self.solver_options)
                return spectrum, allowance, solver_name, probes
            raise ValueError(f"Unknown spectrum solver {method!r}.")

        return occupied, ranks, solve

    @staticmethod
    def _check_hermitian(matrix, name):
        difference = matrix - matrix.getH()
        max_difference = float(np.max(np.abs(difference.data))) if difference.nnz else 0.0
        scale = max(1.0, float(np.max(np.abs(matrix.data))) if matrix.nnz else 0.0)
        if max_difference > 1e-9 * scale:
            raise ValueError(f"{name}(k) is not Hermitian: residual {max_difference:.3e} (scale {scale:.3e}).")
        if matrix.nnz and not np.all(np.isfinite(matrix.data)):
            raise ValueError(f"{name}(k) contains non-finite values.")

    @staticmethod
    def _band_edges(evaluations, occupied):
        ordered = sorted(evaluations.items(), key=lambda item: item[0])
        valence = [(key, value["spectrum"][occupied][0]) for key, value in ordered]
        conduction = [(key, value["spectrum"][occupied + 1][0]) for key, value in ordered]
        vbm_key, vbm = max(valence, key=lambda item: item[1])
        cbm_key, cbm = min(conduction, key=lambda item: item[1])
        direct_gap = min(value["spectrum"][occupied + 1][0] - value["spectrum"][occupied][0]
                         for _, value in ordered)
        uncertainty = max((value["allowance"] for _, value in ordered), default=0.0)
        return {
            "vbm_eV": float(vbm), "vbm_k_fractional": [float(vbm_key)],
            "cbm_eV": float(cbm), "cbm_k_fractional": [float(cbm_key)],
            "signed_gap_eV": float(cbm - vbm),
            "band_gap_eV": float(max(0.0, cbm - vbm)),
            "minimum_direct_gap_eV": float(direct_gap),
            "numerical_allowance_eV": float(uncertainty),
        }

    def _refine_extrema(self, evaluations, evaluate, occupied):
        points = sorted(evaluations)
        if len(points) < 2:
            return
        v = [evaluations[x]["spectrum"][occupied][0] for x in points]
        c = [evaluations[x]["spectrum"][occupied + 1][0] for x in points]
        intervals = set()
        for i in range(len(points)):
            is_vmax = (i == 0 or v[i] >= v[i - 1]) and (i == len(points) - 1 or v[i] >= v[i + 1])
            is_cmin = (i == 0 or c[i] <= c[i - 1]) and (i == len(points) - 1 or c[i] <= c[i + 1])
            if is_vmax or is_cmin:
                if i > 0:
                    intervals.add((points[i - 1], points[i]))
                if i + 1 < len(points):
                    intervals.add((points[i], points[i + 1]))
        for left, right in sorted(intervals):
            for band_rank, sign in ((occupied, -1.0), (occupied + 1, 1.0)):
                result = minimize_scalar(
                    lambda x: sign * evaluate(float(x))["spectrum"][band_rank][0],
                    bounds=(left, right), method="bounded",
                    options={"xatol": 1e-5, "maxiter": 32},
                )
                evaluate(float(result.x))

    def _run_periodic(self, evaluate, occupied, axis):
        n_k = int(self.k_options["initial_nk"])
        max_nk = int(self.k_options["max_nk"])
        previous = None
        convergence = []
        converged = False
        while True:
            for coordinate in np.linspace(0.0, 0.5, n_k):
                evaluate(float(coordinate))
            edges = self._band_edges(self._evaluations, occupied)
            delta = None
            if previous is not None:
                delta = max(abs(edges["vbm_eV"] - previous["vbm_eV"]),
                            abs(edges["cbm_eV"] - previous["cbm_eV"]))
                converged = delta + 2.0 * edges["numerical_allowance_eV"] <= self.k_options["gap_tol"]
            convergence.append({
                "nk": n_k, "vbm_eV": edges["vbm_eV"], "cbm_eV": edges["cbm_eV"],
                "max_edge_change_eV": None if delta is None else float(delta),
                "numerical_allowance_eV": edges["numerical_allowance_eV"],
                "converged": bool(converged),
            })
            if converged:
                break
            next_n = 2 * (n_k - 1) + 1
            if next_n > max_nk:
                break
            previous, n_k = edges, next_n

        if self.k_options["local_refine"]:
            old_edges = self._band_edges(self._evaluations, occupied)
            self._refine_extrema(self._evaluations, evaluate, occupied)
            new_edges = self._band_edges(self._evaluations, occupied)
            if convergence:
                convergence[-1]["local_refinement_change_eV"] = float(max(
                    abs(new_edges["vbm_eV"] - old_edges["vbm_eV"]),
                    abs(new_edges["cbm_eV"] - old_edges["cbm_eV"]),
                ))
                if convergence[-1]["local_refinement_change_eV"] + 2 * new_edges["numerical_allowance_eV"] > self.k_options["gap_tol"]:
                    converged = False
                    convergence[-1]["converged"] = False

        return converged, convergence

    def _save_outputs(self, result):
        if self.results_path is None:
            return
        self.results_path.mkdir(parents=True, exist_ok=True)
        with (self.results_path / "gap.json").open("w", encoding="utf-8") as stream:
            metadata = {key: value for key, value in result.items() if key != "k_spectra"}
            json.dump(metadata, stream, indent=2, sort_keys=True)

        k_values = sorted(self._evaluations)
        ranks = result["band_indices"]
        eigenvalues = np.asarray([
            [self._evaluations[key]["spectrum"][rank][0] for rank in ranks]
            for key in k_values
        ], dtype=float)
        eigenvalue_allowances = np.asarray([
            [self._evaluations[key]["spectrum"][rank][1] for rank in ranks]
            for key in k_values
        ], dtype=float)
        kpoints = np.zeros((len(k_values), 3), dtype=float)
        if result["structure_mode"] == "periodic":
            kpoints[:, result["periodic_axis"]] = k_values
        np.savez_compressed(
            self.results_path / "bands.npz", kpoints_fractional=kpoints,
            band_indices=np.asarray(ranks, dtype=int), eigenvalues_eV=eigenvalues,
            eigenvalue_allowances_eV=eigenvalue_allowances,
        )
        with (self.results_path / "bands.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["kx_fractional", "ky_fractional", "kz_fractional", "band_index", "energy_eV", "numeric_allowance_eV"])
            for kpoint, row, allowances in zip(kpoints, eigenvalues, eigenvalue_allowances):
                for rank, energy, allowance in zip(ranks, row, allowances):
                    writer.writerow([*kpoint.tolist(), rank, f"{energy:.12g}", f"{allowance:.12g}"])

        with (self.results_path / "band_edges.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["k_fractional", "vbm_eV", "cbm_eV", "direct_gap_eV", "numerical_allowance_eV"])
            for key in k_values:
                item = self._evaluations[key]
                writer.writerow([
                    key, item["spectrum"][result["occupied_bands"]][0],
                    item["spectrum"][result["occupied_bands"] + 1][0],
                    item["spectrum"][result["occupied_bands"] + 1][0] - item["spectrum"][result["occupied_bands"]][0],
                    item["allowance"],
                ])
        if self.output_options["plot"]:
            self._plot_outputs(result, k_values, ranks, eigenvalues)

    def _plot_outputs(self, result, k_values, ranks, eigenvalues):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        reference = 0.5 * (result["vbm_eV"] + result["cbm_eV"])
        relative = eigenvalues - reference
        window = self.output_options["plot_window"]
        fig = Figure(figsize=(8, 5), dpi=150)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        x = np.asarray(k_values) if result["structure_mode"] == "periodic" else np.zeros(len(k_values))
        for column, _rank in enumerate(ranks):
            energies = relative[:, column]
            if window is not None:
                visible = (energies >= window[0]) & (energies <= window[1])
                energies = np.where(visible, energies, np.nan)
            if result["structure_mode"] == "finite":
                if np.isfinite(energies[0]):
                    ax.hlines(energies[0], -0.3, 0.3, color="#245b91", linewidth=0.8)
            else:
                ax.plot(x, energies, color="#245b91", linewidth=0.8)
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.6)
        ax.set_ylabel("Energy relative to mid-gap (eV)")
        ax.set_xlabel("Fractional k along periodic axis" if result["structure_mode"] == "periodic" else "Finite structure")
        ax.set_title("Calculated near-gap spectrum")
        if result["structure_mode"] == "finite":
            ax.set_xlim(-0.5, 0.5)
            ax.set_xticks([0], ["finite"])
        fig.tight_layout()
        fig.savefig(self.results_path / "bands.png")

        gap_fig = Figure(figsize=(8, 4), dpi=150)
        FigureCanvasAgg(gap_fig)
        gap_ax = gap_fig.add_subplot(111)
        vbm = np.asarray([self._evaluations[key]["spectrum"][result["occupied_bands"]][0] for key in k_values])
        cbm = np.asarray([self._evaluations[key]["spectrum"][result["occupied_bands"] + 1][0] for key in k_values])
        gap_ax.plot(x, vbm - reference, "o-", label="VBM")
        gap_ax.plot(x, cbm - reference, "o-", label="CBM")
        gap_ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.6)
        gap_ax.set_ylabel("Energy relative to mid-gap (eV)")
        gap_ax.set_xlabel("Fractional k along periodic axis" if result["structure_mode"] == "periodic" else "Finite structure")
        gap_ax.legend()
        gap_fig.tight_layout()
        gap_fig.savefig(self.results_path / "gap.png")

    def compute(self):
        """Compute and return a JSON-compatible result dictionary."""
        from threadpoolctl import threadpool_limits

        self._atoms, mode, axis = self._read_structure()
        source_fingerprint = self._source_fingerprint()
        cache_path = self._cache_path()
        cache_key = self._cache_key(self._atoms, mode, axis, source_fingerprint)
        self._hs_cache_key = cache_key
        cached = self._load_cached_blocks(cache_path, cache_key)
        cache_hit = cached is not None
        if cache_hit:
            h_blocks, s_blocks, atom_norbs, model_metadata = cached
        else:
            h_blocks, s_blocks, atom_norbs, model_metadata = self._predict_blocks(self._atoms)
            if self.cache_options["save"]:
                self._write_cached_blocks(
                    cache_path, cache_key, h_blocks, s_blocks, atom_norbs, model_metadata,
                )

        self._evaluations = {}
        occupied, ranks, solve = self._make_solver(
            h_blocks, s_blocks, atom_norbs, model_metadata,
        )
        solver_names = set()
        spectrum_cache_counts = {"hits": 0, "misses": 0}

        def evaluate(coordinate):
            coordinate = float(coordinate)
            if mode == "periodic":
                if not -1e-12 <= coordinate <= 0.5 + 1e-12:
                    raise ValueError("One-dimensional k coordinates must lie in [0, 0.5].")
                kpoint = np.zeros(3, dtype=float)
                kpoint[axis] = coordinate
                key = float(round(coordinate, 12))
            else:
                kpoint = np.zeros(3, dtype=float)
                key = 0.0
            if key not in self._evaluations:
                kpoint = np.asarray(kpoint, dtype=float).round(12)
                identity = self._spectrum_identity(kpoint, ranks, occupied)
                spectrum_path = self._spectrum_cache_path(identity)
                cached_spectrum = self._load_cached_spectrum(spectrum_path, identity, ranks)
                if cached_spectrum is None:
                    spectrum, allowance, solver_name, probes = solve(kpoint)
                    spectrum_cache_counts["misses"] += 1
                    self._save_cached_spectrum(
                        spectrum_path, identity, spectrum, ranks, allowance, solver_name, probes,
                    )
                else:
                    spectrum, allowance, solver_name, probes = cached_spectrum
                    spectrum_cache_counts["hits"] += 1
                self._evaluations[key] = {
                    "spectrum": spectrum, "allowance": float(allowance),
                    "solver": solver_name, "inertia_probes": int(probes),
                }
                solver_names.add(solver_name)
            return self._evaluations[key]

        with threadpool_limits(limits=1):
            if mode == "finite":
                evaluate(0.0)
                converged, convergence = True, []
            else:
                converged, convergence = self._run_periodic(evaluate, occupied, axis)
            base_edges = self._band_edges(self._evaluations, occupied)
            if mode == "periodic" and self.output_options["plot_nk"]:
                for coordinate in np.linspace(0.0, 0.5, int(self.output_options["plot_nk"])):
                    evaluate(float(coordinate))
                edges_with_plot = self._band_edges(self._evaluations, occupied)
                plot_change = max(
                    abs(edges_with_plot["vbm_eV"] - base_edges["vbm_eV"]),
                    abs(edges_with_plot["cbm_eV"] - base_edges["cbm_eV"]),
                )
                if convergence:
                    convergence[-1]["plot_added_change_eV"] = float(plot_change)
                if plot_change + 2 * edges_with_plot["numerical_allowance_eV"] > self.k_options["gap_tol"]:
                    converged = False
                    if convergence:
                        convergence[-1]["converged"] = False
                base_edges = edges_with_plot

        signed_gap = base_edges["signed_gap_eV"]
        uncertainty = 2 * base_edges["numerical_allowance_eV"]
        if signed_gap > uncertainty:
            gap_status = "positive_gap"
        elif signed_gap < -uncertainty:
            gap_status = "band_overlap"
        else:
            gap_status = "unresolved_zero_gap"
        for edge_key in ("vbm_k_fractional", "cbm_k_fractional"):
            coordinate = base_edges[edge_key][0]
            base_edges[edge_key] = (
                [float(coordinate) if i == axis else 0.0 for i in range(3)]
                if mode == "periodic" else [0.0, 0.0, 0.0]
            )
        result = {
            "algorithm_version": _ALGORITHM_VERSION,
            "structure_mode": mode,
            "gap_type": "periodic_band_gap" if mode == "periodic" else "finite_homo_lumo_gap",
            "periodic_axis": int(axis) if mode == "periodic" else None,
            "formula": self._atoms.get_chemical_formula(),
            "n_atoms": len(self._atoms),
            "n_orbitals": int(sum(atom_norbs)),
            "checkpoint": str(self.checkpoint) if self.checkpoint is not None else None,
            "cell_angstrom": np.asarray(self._atoms.cell, dtype=float).tolist(),
            "pbc": np.asarray(self._atoms.pbc, dtype=bool).tolist(),
            "basis": model_metadata.get("basis"),
            "AtomicData_options": model_metadata.get("AtomicData_options"),
            "model_dtype": model_metadata.get("dtype"),
            "n_electrons": self._electron_count,
            "occupied_bands": int(occupied),
            "band_indices": [int(rank) for rank in ranks],
            "solver": sorted(solver_names),
            "overlap_included": s_blocks is not None,
            **base_edges,
            "numerical_allowance_type": sorted({
                "dense": "not_estimated_dense",
                "dense-fallback": "not_estimated_dense",
                "inertia": "bisection_half_width",
                "shift-invert": "residual_based_estimate",
            }[name] for name in solver_names),
            "gap_status": gap_status,
            "requested_resolution_eV": float(self.k_options["gap_tol"]),
            "below_requested_resolution": bool(abs(signed_gap) < self.k_options["gap_tol"]),
            "converged": bool(converged),
            "k_convergence": convergence,
            "kpoints_fractional": [
                ([float(key) if i == axis else 0.0 for i in range(3)]
                 if mode == "periodic" else [0.0, 0.0, 0.0])
                for key in sorted(self._evaluations)
            ],
            "k_spectra": [
                {
                    "k_fractional": ([float(key) if i == axis else 0.0 for i in range(3)]
                                     if mode == "periodic" else [0.0, 0.0, 0.0]),
                    "band_indices": [int(rank) for rank in ranks],
                    "eigenvalues_eV": [
                        float(self._evaluations[key]["spectrum"][rank][0]) for rank in ranks
                    ],
                    "band_allowances_eV": [
                        float(self._evaluations[key]["spectrum"][rank][1]) for rank in ranks
                    ],
                    "numerical_allowance_eV": float(self._evaluations[key]["allowance"]),
                    "numerical_allowance_type": {
                        "dense": "not_estimated_dense",
                        "dense-fallback": "not_estimated_dense",
                        "inertia": "bisection_half_width",
                        "shift-invert": "residual_based_estimate",
                    }[self._evaluations[key]["solver"]],
                    "solver": self._evaluations[key]["solver"],
                    "inertia_probes": self._evaluations[key]["inertia_probes"],
                }
                for key in sorted(self._evaluations)
            ],
            "cache": {
                "hit": bool(cache_hit), "path": str(cache_path) if cache_path else None,
                "hs_fingerprint": cache_key,
                "source_fingerprint": source_fingerprint,
                "spectrum_hits": int(spectrum_cache_counts["hits"]),
                "spectrum_misses": int(spectrum_cache_counts["misses"]),
            },
            "warnings": [
                "k-grid stability is not a mathematical proof of a global Brillouin-zone extremum."
            ] if mode == "periodic" else [],
        }
        if any(name.startswith("dense") for name in solver_names):
            result["warnings"].append(
                "Dense solving reports zero interval allowance because no bisection interval is used; this is not a zero floating-point error bound."
            )
        if "shift-invert" in solver_names:
            result["warnings"].append(
                "Shift-invert residual allowances are numerical estimates, not certified eigenvalue brackets."
            )
        self._save_outputs(result)
        return result

