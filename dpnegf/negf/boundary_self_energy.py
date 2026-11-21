"""Boundary-Schur reduction for lead self-energies.

The implementation returns only the contact-projected self-energy.  The
reduced Green function is deliberately kept internal because it is not the
surface Green function of the complete principal layer.
"""

from dataclasses import dataclass
import logging

import numpy as np
import torch

from dpnegf.negf.surface_green import surface_green


log = logging.getLogger(__name__)


class BoundarySchurNotApplicable(Exception):
    """Raised when the lead structure cannot benefit from Boundary-Schur."""


@dataclass(frozen=True)
class BoundarySchurConfig:
    size: int
    boundary: np.ndarray
    internal: np.ndarray
    contacts: np.ndarray
    core: np.ndarray
    rest: np.ndarray

    @property
    def has_two_endpoints(self):
        return self.core.size > 0 and self.rest.size > 0


def _as_complex_array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.complex128)


def _as_scalar(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    return np.asarray(value).item()


def prepare_boundary_schur_plan(h01, s01, hDL, sDL):
    """Cache the exact structural partition for one k point and contact."""
    h01 = _as_complex_array(h01)
    s01 = _as_complex_array(s01)
    hDL = _as_complex_array(hDL)
    sDL = _as_complex_array(sDL)
    if h01.ndim != 2 or h01.shape[0] != h01.shape[1]:
        raise ValueError("Inter-layer Hamiltonian must be square.")
    if s01.shape != h01.shape:
        raise ValueError("Inter-layer overlap shape must match the Hamiltonian.")
    if hDL.ndim != 2 or sDL.shape != hDL.shape:
        raise ValueError("Lead-device H/S contacts must be matching matrices.")
    n = h01.shape[0]
    if hDL.shape[1] > n:
        raise ValueError("Lead-device contact columns exceed the principal layer.")

    # Structure is the H/S support union; energy-dependent cancellation must
    # not change the principal-layer partition.
    rows, cols = np.nonzero((h01 != 0) | (s01 != 0))
    active_rows = np.unique(rows)
    active_cols = np.unique(cols)
    contact_mask = np.any(hDL != 0, axis=0) | np.any(sDL != 0, axis=0)
    contacts = np.flatnonzero(contact_mask).astype(np.intp, copy=False)
    boundary = np.union1d(np.union1d(active_rows, active_cols), contacts)
    internal = np.setdiff1d(np.arange(n, dtype=np.intp), boundary, assume_unique=True)

    positions = {int(index): pos for pos, index in enumerate(boundary)}
    core = np.asarray([positions[int(index)] for index in active_cols if int(index) in positions],
                      dtype=np.intp)
    rest = np.asarray([positions[int(index)] for index in active_rows if int(index) in positions],
                      dtype=np.intp)
    is_complete_bipartition = (
        core.size + rest.size == boundary.size
        and not np.intersect1d(core, rest, assume_unique=True).size
    )
    if not is_complete_bipartition:
        core = np.empty(0, dtype=np.intp)
        rest = np.empty(0, dtype=np.intp)

    return BoundarySchurConfig(
        size=n,
        boundary=np.asarray(boundary, dtype=np.intp),
        internal=np.asarray(internal, dtype=np.intp),
        contacts=contacts,
        core=core,
        rest=rest,
    )


def _max_abs(matrix):
    return float(np.max(np.abs(matrix))) if matrix.size else 0.0


def _decimate(d_bulk, d_surface, alpha, beta):
    """Sancho doubling with independent bulk and terminated-surface blocks."""
    d_bulk = np.array(d_bulk, dtype=np.complex128, copy=True)
    d_surface = np.array(d_surface, dtype=np.complex128, copy=True)
    alpha = np.array(alpha, dtype=np.complex128, copy=True)
    beta = np.array(beta, dtype=np.complex128, copy=True)
    size = d_bulk.shape[0]

    for iteration in range(1, 102):
        rhs = np.concatenate((alpha, beta), axis=1)
        solved = np.linalg.solve(d_bulk, rhs)
        alpha_products = alpha @ solved
        alpha_next = alpha_products[:, :size].copy()
        d_bulk_next = d_bulk - alpha_products[:, size:]
        del alpha_products

        beta_products = beta @ solved
        beta_next = beta_products[:, size:].copy()
        beta_x_alpha = beta_products[:, :size]
        d_bulk_next -= beta_x_alpha
        d_surface_next = d_surface - beta_x_alpha
        del beta_products, beta_x_alpha, solved
        d_bulk, d_surface = d_bulk_next, d_surface_next
        alpha, beta = alpha_next, beta_next

        if max(_max_abs(alpha), _max_abs(beta)) < 1.0e-40:
            green = np.linalg.solve(d_surface, np.eye(size, dtype=np.complex128))
            inverse_residual = _max_abs(d_surface @ green - np.eye(size))
            if not np.isfinite(inverse_residual) or inverse_residual > 3.0e-5:
                raise ArithmeticError("Boundary-Schur terminal inverse residual is too large.")
            return green, iteration, inverse_residual

    raise ArithmeticError("Boundary-Schur Sancho decimation did not converge after 101 iterations.")


def _fixed_point_residual(a, green, u=None, lower=None, rest=None, sigma_rr=None,
                          chunk_size=128):
    """Check the original boundary equation without a second dense residual."""
    size = a.shape[0]
    worst = 0.0
    rest_positions = {} if rest is None else {int(row): pos for pos, row in enumerate(rest)}
    rest_green = None if rest is None else green[rest, :]
    for start in range(0, size, chunk_size):
        stop = min(start + chunk_size, size)
        if rest is None:
            rows = (a[start:stop] - u[start:stop] @ green @ lower) @ green
        else:
            rows = a[start:stop] @ green
            for local_row, row in enumerate(range(start, stop)):
                rest_row = rest_positions.get(row)
                if rest_row is not None:
                    rows[local_row] -= sigma_rr[rest_row, :] @ rest_green
        for local_row, row in enumerate(range(start, stop)):
            rows[local_row, row] -= 1.0
        worst = max(worst, _max_abs(rows))
    return worst


def _run_sancho(hL, hLL, sL, sLL, e0, eta, hDL, sDL, contacts,
                h10, s10, numba_jit):
    z = e0 + 1j * eta
    green = surface_green(
        hL, hLL, sL, sLL, z, method="Lopez-Sancho", numba_jit=numba_jit,
        h10=h10, s10=s10,
    )
    if isinstance(green, torch.Tensor):
        green = green.detach().cpu().numpy()
    if contacts.size:
        c_db = e0 * sDL[:, contacts] - hDL[:, contacts]
        c_bd = e0 * sDL[:, contacts].conj().T - hDL[:, contacts].conj().T
        sigma = c_db @ green[np.ix_(contacts, contacts)] @ c_bd
    else:
        sigma = np.zeros((hDL.shape[0], hDL.shape[0]), dtype=np.complex128)
    return torch.as_tensor(sigma, dtype=torch.complex128), green


def _solve_boundary_schur(hL, hLL, sL, sLL, hDL, sDL, e0, eta,
                          h10, s10, plan, contacts):
    b = plan.boundary
    i = plan.internal
    z = e0 + 1j * eta
    a_bb = z * sL[np.ix_(b, b)] - hL[np.ix_(b, b)]
    if i.size:
        a_bi = z * sL[np.ix_(b, i)] - hL[np.ix_(b, i)]
        a_ib = z * sL[np.ix_(i, b)] - hL[np.ix_(i, b)]
        a_ii = z * sL[np.ix_(i, i)] - hL[np.ix_(i, i)]
        x = np.linalg.solve(a_ii, a_ib)
        denom = (np.linalg.norm(a_ii, ord=np.inf) * _max_abs(x)
                 + _max_abs(a_ib))
        internal_residual = _max_abs(a_ii @ x - a_ib) / max(denom, np.finfo(float).tiny)
        if not np.isfinite(internal_residual):
            raise ArithmeticError("Non-finite internal Schur solve residual.")
        a_eff = a_bb - a_bi @ x
        del a_bi, a_ib, a_ii, x
    else:
        internal_residual = 0.0
        a_eff = a_bb
    del a_bb

    boundary_size = b.size
    use_two_endpoints = plan.has_two_endpoints

    if use_two_endpoints:
        c, r = plan.core, plan.rest
        boundary_core = b[c]
        boundary_rest = b[r]
        t = z * sLL[np.ix_(boundary_rest, boundary_core)] - hLL[
            np.ix_(boundary_rest, boundary_core)
        ]
        t_reverse = z * s10[np.ix_(boundary_core, boundary_rest)] - h10[
            np.ix_(boundary_core, boundary_rest)
        ]
        del hL, hLL, sL, sLL, h10, s10, boundary_core, boundary_rest

        acc = a_eff[np.ix_(c, c)]
        acr = a_eff[np.ix_(c, r)]
        arc = a_eff[np.ix_(r, c)]
        arr = a_eff[np.ix_(r, r)]
        endpoint_rhs = np.concatenate((arc, t), axis=1)
        endpoint_solve = np.linalg.solve(arr, endpoint_rhs)
        x_rc = endpoint_solve[:, :c.size]
        x_t = endpoint_solve[:, c.size:]

        d_bulk = acc - acr @ x_rc - t_reverse @ x_t
        d_surface = acc - acr @ x_rc
        u_eff = -acr @ x_t
        lower_eff = -t_reverse @ x_rc
        del acc, acr, arc, arr, endpoint_rhs, endpoint_solve, x_rc, x_t
        g_core, iterations, inverse_residual = _decimate(
            d_bulk, d_surface, -lower_eff, -u_eff,
        )
        del d_bulk, d_surface, u_eff, lower_eff

        sigma_rr = t @ g_core @ t_reverse
        surface_matrix = a_eff.copy()
        surface_matrix[np.ix_(r, r)] -= sigma_rr
        g_boundary = np.linalg.solve(
            surface_matrix, np.eye(boundary_size, dtype=np.complex128),
        )
        del surface_matrix, sigma_rr, g_core

        # Re-evaluate the endpoint self-energy from the reconstructed Green
        # function so this check is independent of the reduced recursion.
        sigma_rr_check = t @ g_boundary[np.ix_(c, c)] @ t_reverse
        mode = "boundary-schur-two-endpoint"
        core_size = int(c.size)
    else:
        u = z * sLL[np.ix_(b, b)] - hLL[np.ix_(b, b)]
        lower = z * s10[np.ix_(b, b)] - h10[np.ix_(b, b)]
        del hL, hLL, sL, sLL, h10, s10
        g_boundary, iterations, inverse_residual = _decimate(
            a_eff, a_eff, -lower, -u,
        )
        sigma_rr_check = None
        mode = "boundary-schur-boundary"
        core_size = int(boundary_size)

    if use_two_endpoints:
        fixed_point_residual = _fixed_point_residual(
            a_eff, g_boundary, rest=r, sigma_rr=sigma_rr_check,
        )
        del c, r, t, t_reverse, sigma_rr_check
    else:
        fixed_point_residual = _fixed_point_residual(
            a_eff, g_boundary, u=u, lower=lower,
        )
        del u, lower
    del a_eff
    if not np.isfinite(fixed_point_residual) or fixed_point_residual >= 3.0e-5:
        raise ArithmeticError(
            f"Boundary-Schur fixed-point residual {fixed_point_residual:.3e} exceeds tolerance."
        )

    contact_positions = {int(index): pos for pos, index in enumerate(b)}
    contact_local = np.asarray([contact_positions[int(index)] for index in contacts], dtype=np.intp)
    if contacts.size:
        c_db = e0 * sDL[:, contacts] - hDL[:, contacts]
        c_bd = e0 * sDL[:, contacts].conj().T - hDL[:, contacts].conj().T
        sigma = c_db @ g_boundary[np.ix_(contact_local, contact_local)] @ c_bd
    else:
        sigma = np.zeros((hDL.shape[0], hDL.shape[0]), dtype=np.complex128)
    if not np.isfinite(sigma).all():
        raise ArithmeticError("Boundary-Schur produced a non-finite self-energy.")

    return torch.as_tensor(sigma, dtype=torch.complex128), {
        "solver": mode,
        "boundary_size": int(boundary_size),
        "internal_size": int(i.size),
        "core_size": core_size,
        "iterations": int(iterations),
        "internal_residual": float(internal_residual),
        "inverse_residual": float(inverse_residual),
        "fixed_point_residual": float(fixed_point_residual),
    }


def boundary_self_energy(hL, hLL, sL, sLL, hDL, sDL, energy, E_ref=0.0,
                         etaLead=1e-5, numba_jit=None, h10=None, s10=None,
                         plan=None):
    """Compute a contact self-energy with exact boundary Schur reduction.

    Returns ``(sigma, diagnostics)``.  Known numerical failures and structures
    without a smaller iteration space use the existing Sancho implementation.
    """
    hL = _as_complex_array(hL)
    hLL = _as_complex_array(hLL)
    sL = _as_complex_array(sL)
    sLL = _as_complex_array(sLL)
    hDL = _as_complex_array(hDL)
    sDL = _as_complex_array(sDL)
    h10 = np.ascontiguousarray(hLL.conj().T) if h10 is None else _as_complex_array(h10)
    s10 = np.ascontiguousarray(sLL.conj().T) if s10 is None else _as_complex_array(s10)
    e0 = _as_scalar(energy) + _as_scalar(E_ref)
    eta = _as_scalar(etaLead)
    if not np.isfinite(e0) or not np.isfinite(eta) or eta < 0:
        raise ValueError("Boundary-Schur requires finite energy and nonnegative etaLead.")
    if hL.ndim != 2 or hL.shape[0] != hL.shape[1]:
        raise ValueError("Principal-layer Hamiltonian must be square.")
    if sL.shape != hL.shape or hLL.shape != hL.shape or sLL.shape != hL.shape:
        raise ValueError("Principal-layer H/S matrices must have matching shapes.")
    if hDL.ndim != 2 or sDL.shape != hDL.shape:
        raise ValueError("Lead-device H/S contacts must be matching matrices.")

    if plan is None:
        plan = prepare_boundary_schur_plan(hLL, sLL, hDL, sDL)
    if plan.size != hL.shape[0]:
        raise ValueError("Boundary-Schur plan does not match the principal-layer size.")
    contacts = plan.contacts

    def fallback(reason):
        log.debug("Boundary-Schur fallback to Sancho-Rubio: %s", reason)
        sigma, _ = _run_sancho(
            hL, hLL, sL, sLL, e0, eta, hDL, sDL, contacts,
            h10, s10, numba_jit,
        )
        return sigma, {
            "solver": "Sancho-Rubio",
            "fallback_reason": str(reason),
            "boundary_size": int(plan.boundary.size),
            "internal_size": int(plan.internal.size),
        }

    if plan.boundary.size == 0 or (
        plan.boundary.size >= plan.size and not plan.has_two_endpoints
    ):
        return fallback(BoundarySchurNotApplicable("no effective dimension reduction"))

    try:
        return _solve_boundary_schur(
            hL, hLL, sL, sLL, hDL, sDL, e0, eta, h10, s10, plan, contacts,
        )
    except (BoundarySchurNotApplicable, np.linalg.LinAlgError, ArithmeticError) as exc:
        reason = str(exc)
        exc.__traceback__ = None
        return fallback(reason)
