"""Numerical ODE solvers for deterministic SIR experiments."""

from __future__ import annotations

from typing import Callable, Iterable
import warnings

import numpy as np
from scipy.integrate import solve_ivp

from config import NEGATIVE_TOL, POPULATION_TOL

RHS = Callable[..., tuple[float, float, float] | np.ndarray]


def _as_args(args: Iterable | None) -> tuple:
    """Convert optional solver arguments to a tuple."""
    return tuple() if args is None else tuple(args)


def check_population_conservation(
    y: np.ndarray,
    tol: float = POPULATION_TOL,
    negative_tol: float = NEGATIVE_TOL,
    context: str = "solution",
) -> dict:
    """Check s+i+r≈1 and non-negativity, issuing warnings when needed."""
    population = np.sum(y, axis=1)
    max_mass_error = float(np.max(np.abs(population - 1.0)))
    min_value = float(np.min(y))

    if max_mass_error > tol:
        warnings.warn(
            f"[{context}] max |s+i+r-1| = {max_mass_error:.3e}, larger than {tol:.1e}.",
            RuntimeWarning,
            stacklevel=2,
        )
    if min_value < -negative_tol:
        warnings.warn(
            f"[{context}] minimum state value = {min_value:.3e}; step size may be too large.",
            RuntimeWarning,
            stacklevel=2,
        )
    return {"max_mass_error": max_mass_error, "min_value": min_value}


def rk4_fixed_step(
    rhs: RHS,
    y0: np.ndarray,
    t_span: tuple[float, float],
    dt: float,
    args: Iterable | None = None,
    context: str = "rk4",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Solve an ODE system with classical fixed-step RK4."""
    t0, t1 = map(float, t_span)
    if t1 <= t0:
        raise ValueError("t_span must satisfy t1 > t0.")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    n_steps = int(np.ceil((t1 - t0) / dt))
    t = t0 + np.arange(n_steps + 1, dtype=float) * dt
    t[-1] = t1

    y0_arr = np.asarray(y0, dtype=float)
    y = np.empty((n_steps + 1, len(y0_arr)), dtype=float)
    y[0] = y0_arr
    args_tuple = _as_args(args)

    for n in range(n_steps):
        h = t[n + 1] - t[n]
        tn = t[n]
        yn = y[n]
        k1 = np.asarray(rhs(tn, yn, *args_tuple), dtype=float)
        k2 = np.asarray(rhs(tn + 0.5 * h, yn + 0.5 * h * k1, *args_tuple), dtype=float)
        k3 = np.asarray(rhs(tn + 0.5 * h, yn + 0.5 * h * k2, *args_tuple), dtype=float)
        k4 = np.asarray(rhs(tn + h, yn + h * k3, *args_tuple), dtype=float)
        y[n + 1] = yn + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    diagnostics = check_population_conservation(y, context=context)
    return t, y, diagnostics


def scipy_solve_ivp(
    rhs: RHS,
    y0: np.ndarray,
    t_span: tuple[float, float],
    dt: float,
    args: Iterable | None = None,
    context: str = "solve_ivp",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Solve an ODE system with scipy.integrate.solve_ivp on a fixed output grid."""
    t0, t1 = map(float, t_span)
    t_eval = np.arange(t0, t1 + 0.5 * dt, dt, dtype=float)
    if t_eval[-1] > t1:
        t_eval[-1] = t1
    elif t_eval[-1] < t1:
        t_eval = np.append(t_eval, t1)

    args_tuple = _as_args(args)
    sol = solve_ivp(
        fun=lambda t, y: rhs(t, y, *args_tuple),
        t_span=(t0, t1),
        y0=np.asarray(y0, dtype=float),
        t_eval=t_eval,
        method="DOP853",
        rtol=1.0e-9,
        atol=1.0e-12,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed in {context}: {sol.message}")

    y = sol.y.T
    diagnostics = check_population_conservation(y, context=context)
    return sol.t, y, diagnostics


def solve_ode(
    rhs: RHS,
    y0: np.ndarray,
    t_span: tuple[float, float],
    dt: float,
    args: Iterable | None = None,
    method: str = "rk4",
    context: str = "ode",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Solve an ODE with RK4 or scipy solve_ivp."""
    method_lower = method.lower()
    if method_lower == "rk4":
        return rk4_fixed_step(rhs, y0, t_span, dt, args=args, context=context)
    if method_lower in {"ivp", "solve_ivp", "scipy"}:
        return scipy_solve_ivp(rhs, y0, t_span, dt, args=args, context=context)
    raise ValueError(f"Unknown solver method: {method!r}.")
