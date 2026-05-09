"""Right-hand sides for deterministic SIR-type epidemic models."""

from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray | tuple[float, float, float]


def beta_from_R0_basic(R0: float, gamma: float) -> float:
    """Return beta for the basic SIR model."""
    return float(R0) * float(gamma)


def beta0_from_R0_vital(R0: float, gamma: float, mu: float) -> float:
    """Return beta0 for SIR models with births and deaths."""
    return float(R0) * (float(gamma) + float(mu))


def seasonal_beta(
    t: float | np.ndarray,
    beta0: float,
    alpha: float | np.ndarray,
    phase: float | np.ndarray = 0.0,
) -> float | np.ndarray:
    """Compute beta(t)=beta0*(1+alpha*cos(2*pi*t+phase))."""
    return float(beta0) * (1.0 + alpha * np.cos(2.0 * np.pi * t + phase))


def sir_basic_rhs(t: float, y: ArrayLike, beta: float, gamma: float) -> tuple[float, float, float]:
    """Right-hand side of the basic SIR model with proportion variables."""
    del t
    s, i, r = y
    ds = -beta * s * i
    di = beta * s * i - gamma * i
    dr = gamma * i
    return ds, di, dr


def sir_vital_rhs(t: float, y: ArrayLike, beta0: float, gamma: float, mu: float) -> tuple[float, float, float]:
    """Right-hand side of the SIR model with balanced births and deaths."""
    del t
    s, i, r = y
    ds = mu - beta0 * s * i - mu * s
    di = beta0 * s * i - gamma * i - mu * i
    dr = gamma * i - mu * r
    return ds, di, dr


def sir_seasonal_vital_rhs(
    t: float,
    y: ArrayLike,
    beta0: float,
    gamma: float,
    mu: float,
    alpha: float,
    phase: float = 0.0,
) -> tuple[float, float, float]:
    """Right-hand side of the seasonal SIR model with births and deaths."""
    beta_t = seasonal_beta(t, beta0=beta0, alpha=alpha, phase=phase)
    s, i, r = y
    ds = mu - beta_t * s * i - mu * s
    di = beta_t * s * i - gamma * i - mu * i
    dr = gamma * i - mu * r
    return ds, di, dr
