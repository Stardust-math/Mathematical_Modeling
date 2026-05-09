"""Gillespie simulations for integer-population stochastic SIR models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GillespieResult:
    """Container for one stochastic SIR simulation."""

    extinct: bool
    extinction_time: float | None
    t_final: float
    s_final: int
    i_final: int
    r_final: int
    n_events: int
    peak_I: int
    t_peak: float
    event_times: np.ndarray | None = None
    event_states: np.ndarray | None = None


def beta_t(t: float, beta0: float, alpha: float = 0.0, phase: float = 0.0) -> float:
    """Return beta(t)=beta0*(1+alpha*cos(2*pi*t+phase))."""
    beta = float(beta0) * (1.0 + float(alpha) * np.cos(2.0 * np.pi * float(t) + float(phase)))
    if beta < 0.0:
        raise ValueError("beta(t) became negative; alpha should not exceed 1 in this project.")
    return beta


def initial_counts(N: int, I0: int, R_init: int = 0) -> tuple[int, int, int]:
    """Create integer S, I, R initial counts."""
    N = int(N)
    I0 = int(I0)
    R_init = int(R_init)
    if N <= 0:
        raise ValueError("N must be positive.")
    if I0 < 0 or R_init < 0:
        raise ValueError("I0 and R_init must be non-negative.")
    if I0 + R_init > N:
        raise ValueError("I0 + R_init cannot exceed N.")
    return N - I0 - R_init, I0, R_init


def _check_counts(S: int, I: int, R: int, N: int) -> None:
    """Validate integer non-negative counts and total population."""
    if min(S, I, R) < 0:
        raise RuntimeError(f"Negative population count detected: S={S}, I={I}, R={R}.")
    if S + I + R != N:
        raise RuntimeError(f"Population is not conserved: S+I+R={S+I+R}, N={N}.")


def simulate_gillespie_sir(
    N: int,
    I0: int,
    beta0: float,
    gamma: float,
    mu: float,
    t_max: float,
    rng: np.random.Generator,
    alpha: float = 0.0,
    phase: float = 0.0,
    R_init: int = 0,
    record_trajectory: bool = False,
    max_events: int | None = None,
    stop_if_peak_at_least: int | None = None,
) -> GillespieResult:
    """Simulate a stochastic demographic SIR model with Gillespie's method.

    Events are infection, recovery, infected death with susceptible replacement,
    and removed death with susceptible replacement. Death and immediate
    replacement of a susceptible individual is a no-op, so it is omitted.
    """
    S, I, R = initial_counts(N=N, I0=I0, R_init=R_init)
    N = int(N)
    t = 0.0
    n_events = 0
    extinction_time = 0.0 if I == 0 else None
    peak_I = int(I)
    t_peak = 0.0
    if max_events is None:
        max_events = 10_000_000

    if record_trajectory:
        times: list[float] = [0.0]
        states: list[tuple[int, int, int]] = [(S, I, R)]
    else:
        times = []
        states = []

    while t < t_max and I > 0 and n_events < max_events:
        current_beta = beta_t(t, beta0=beta0, alpha=alpha, phase=phase)
        infection_rate = current_beta * S * I / float(N)
        recovery_rate = gamma * I
        infected_death_rate = mu * I
        removed_death_rate = mu * R
        total_rate = infection_rate + recovery_rate + infected_death_rate + removed_death_rate

        if total_rate <= 0.0 or not np.isfinite(total_rate):
            break

        tau = rng.exponential(1.0 / total_rate)
        next_t = t + tau
        if next_t > t_max:
            t = float(t_max)
            break

        threshold = rng.random() * total_rate
        if threshold < infection_rate:
            S -= 1
            I += 1
        elif threshold < infection_rate + recovery_rate:
            I -= 1
            R += 1
        elif threshold < infection_rate + recovery_rate + infected_death_rate:
            I -= 1
            S += 1
        else:
            R -= 1
            S += 1

        _check_counts(S, I, R, N)
        t = float(next_t)
        n_events += 1

        if I > peak_I:
            peak_I = int(I)
            t_peak = float(t)

        if stop_if_peak_at_least is not None and peak_I >= int(stop_if_peak_at_least):
            if record_trajectory:
                times.append(t)
                states.append((S, I, R))
            break

        if record_trajectory:
            times.append(t)
            states.append((S, I, R))

        if I == 0:
            extinction_time = t
            break

    extinct = I == 0
    return GillespieResult(
        extinct=bool(extinct),
        extinction_time=None if extinction_time is None else float(extinction_time),
        t_final=float(t),
        s_final=int(S),
        i_final=int(I),
        r_final=int(R),
        n_events=int(n_events),
        peak_I=int(peak_I),
        t_peak=float(t_peak),
        event_times=np.asarray(times, dtype=float) if record_trajectory else None,
        event_states=np.asarray(states, dtype=int) if record_trajectory else None,
    )


def sample_event_trajectory(event_times: np.ndarray, event_states: np.ndarray, sample_times: np.ndarray) -> np.ndarray:
    """Sample a piecewise-constant Gillespie trajectory at fixed times."""
    event_times = np.asarray(event_times, dtype=float)
    event_states = np.asarray(event_states, dtype=int)
    sample_times = np.asarray(sample_times, dtype=float)
    if event_times.ndim != 1 or event_states.ndim != 2:
        raise ValueError("event_times must be 1D and event_states must be 2D.")
    if len(event_times) != len(event_states):
        raise ValueError("event_times and event_states must have the same length.")
    indices = np.searchsorted(event_times, sample_times, side="right") - 1
    indices = np.clip(indices, 0, len(event_times) - 1)
    return event_states[indices]
