"""Stochastic Gillespie experiments for the SIR project."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import config as cfg
from gillespie import sample_event_trajectory, simulate_gillespie_sir
from plotting import (
    plot_alpha_extinction_probability,
    plot_extinction_probability_heatmap,
    plot_extinction_time_distribution,
    plot_phase_sensitivity,
    plot_stochastic_trajectories,
)
from sir_models import beta0_from_R0_vital, sir_vital_rhs
from solvers import solve_ode


@dataclass(frozen=True)
class StochasticSettings:
    """Runtime settings for stochastic experiments."""

    mode: str
    trajectory_R0: float
    trajectory_N: int
    trajectory_I0: int
    trajectory_repeats: int
    trajectory_t_max: float
    trajectory_sample_dt: float
    extinction_R0: float
    critical_s_multiplier: float
    outbreak_threshold: float
    N_values: tuple[int, ...]
    I0_values: tuple[int, ...]
    extinction_repeats: int
    extinction_t_max: float
    alpha_values: tuple[float, ...]
    alpha_N: int
    alpha_I0: int
    alpha_repeats: int
    alpha_t_max: float
    phase_alpha_values: tuple[float, ...]
    phase_values: tuple[float, ...]
    phase_repeats: int


def get_stochastic_settings(mode: str = "full") -> StochasticSettings:
    """Return quick or full stochastic settings."""
    mode = mode.lower().strip()
    if mode == "quick":
        return StochasticSettings(
            mode="quick",
            trajectory_R0=cfg.STOCHASTIC_TRAJECTORY_R0,
            trajectory_N=cfg.STOCHASTIC_TRAJECTORY_N,
            trajectory_I0=cfg.STOCHASTIC_TRAJECTORY_I0,
            trajectory_repeats=cfg.STOCHASTIC_TRAJECTORY_REPEATS_QUICK,
            trajectory_t_max=cfg.STOCHASTIC_TRAJECTORY_T_MAX_QUICK,
            trajectory_sample_dt=cfg.STOCHASTIC_TRAJECTORY_SAMPLE_DT,
            extinction_R0=cfg.STOCHASTIC_EXTINCTION_R0,
            critical_s_multiplier=cfg.STOCHASTIC_CRITICAL_S_MULTIPLIER,
            outbreak_threshold=cfg.STOCHASTIC_OUTBREAK_THRESHOLD,
            N_values=tuple(cfg.STOCHASTIC_N_VALUES_QUICK),
            I0_values=tuple(cfg.STOCHASTIC_I0_VALUES_QUICK),
            extinction_repeats=cfg.STOCHASTIC_REPEATS_QUICK,
            extinction_t_max=cfg.STOCHASTIC_T_MAX_QUICK,
            alpha_values=tuple(cfg.STOCHASTIC_ALPHA_VALUES_QUICK),
            alpha_N=cfg.STOCHASTIC_ALPHA_N,
            alpha_I0=cfg.STOCHASTIC_ALPHA_I0,
            alpha_repeats=cfg.STOCHASTIC_REPEATS_QUICK,
            alpha_t_max=cfg.STOCHASTIC_T_MAX_QUICK,
            phase_alpha_values=tuple(cfg.STOCHASTIC_PHASE_ALPHA_VALUES),
            phase_values=tuple(cfg.STOCHASTIC_PHASE_VALUES),
            phase_repeats=cfg.STOCHASTIC_PHASE_REPEATS_QUICK,
        )
    if mode == "full":
        return StochasticSettings(
            mode="full",
            trajectory_R0=cfg.STOCHASTIC_TRAJECTORY_R0,
            trajectory_N=cfg.STOCHASTIC_TRAJECTORY_N,
            trajectory_I0=cfg.STOCHASTIC_TRAJECTORY_I0,
            trajectory_repeats=cfg.STOCHASTIC_TRAJECTORY_REPEATS_FULL,
            trajectory_t_max=cfg.STOCHASTIC_TRAJECTORY_T_MAX_FULL,
            trajectory_sample_dt=cfg.STOCHASTIC_TRAJECTORY_SAMPLE_DT,
            extinction_R0=cfg.STOCHASTIC_EXTINCTION_R0,
            critical_s_multiplier=cfg.STOCHASTIC_CRITICAL_S_MULTIPLIER,
            outbreak_threshold=cfg.STOCHASTIC_OUTBREAK_THRESHOLD,
            N_values=tuple(cfg.STOCHASTIC_N_VALUES_FULL),
            I0_values=tuple(cfg.STOCHASTIC_I0_VALUES_FULL),
            extinction_repeats=cfg.STOCHASTIC_REPEATS_FULL,
            extinction_t_max=cfg.STOCHASTIC_T_MAX_FULL,
            alpha_values=tuple(cfg.STOCHASTIC_ALPHA_VALUES_FULL),
            alpha_N=cfg.STOCHASTIC_ALPHA_N,
            alpha_I0=cfg.STOCHASTIC_ALPHA_I0,
            alpha_repeats=cfg.STOCHASTIC_REPEATS_FULL,
            alpha_t_max=cfg.STOCHASTIC_T_MAX_FULL,
            phase_alpha_values=tuple(cfg.STOCHASTIC_PHASE_ALPHA_VALUES),
            phase_values=tuple(cfg.STOCHASTIC_PHASE_VALUES),
            phase_repeats=cfg.STOCHASTIC_PHASE_REPEATS_FULL,
        )
    raise ValueError("mode must be 'quick' or 'full'.")


def ensure_output_dirs() -> None:
    """Create output directories."""
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg.RESULT_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(df: pd.DataFrame, filename: str) -> None:
    """Write a DataFrame into results/."""
    cfg.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.RESULT_DIR / filename, index=False)


def _child_rng(master_rng: np.random.Generator) -> np.random.Generator:
    """Create a reproducible child generator."""
    seed = int(master_rng.integers(0, np.iinfo(np.uint32).max))
    return np.random.default_rng(seed)


def near_threshold_initial_counts(N: int, I0: int, R0_basic: float, multiplier: float) -> tuple[int, int, int]:
    """Initialize S0 near the epidemic threshold and place the rest in R_init."""
    N = int(N)
    I0 = int(I0)
    target_S0 = int(np.ceil(float(multiplier) * N / float(R0_basic)))
    S0 = max(I0 + 1, min(N - I0, target_S0))
    R_init = N - S0 - I0
    if R_init < 0:
        raise ValueError(f"Invalid near-threshold initialization for N={N}, I0={I0}.")
    return S0, I0, R_init


def _phase_label(phase: float) -> str:
    """Return a compact label for common seasonal phases."""
    phase = float(phase)
    if np.isclose(phase, 0.0):
        return "phase 0"
    if np.isclose(phase, 0.5 * np.pi):
        return r"phase $\pi/2$"
    if np.isclose(phase, np.pi):
        return r"phase $\pi$"
    return f"phase {phase:.2f}"


def _summary_from_detail(detail_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Summarize early fade-out and major-outbreak probabilities by group."""
    rows = []
    for keys, g in detail_df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        extinct = g["extinct"].astype(bool)
        early_fadeout = g["early_fadeout"].astype(bool)
        major = g["major_outbreak"].astype(bool)
        extinct_times = g.loc[extinct, "extinction_time"].dropna().to_numpy(dtype=float)
        fadeout_times = g.loc[early_fadeout, "extinction_time"].dropna().to_numpy(dtype=float)
        p_fadeout = float(early_fadeout.mean())
        p_major = float(major.mean())
        n = int(len(g))

        row.update(
            {
                "R0_basic": float(g["R0_basic"].iloc[0]),
                "S0": int(g["S0"].iloc[0]),
                "R_init": int(g["R_init"].iloc[0]),
                "alpha": float(g["alpha"].iloc[0]) if "alpha" not in group_cols else row.get("alpha", float(g["alpha"].iloc[0])),
                "phase": float(g["phase"].iloc[0]) if "phase" not in group_cols else row.get("phase", float(g["phase"].iloc[0])),
                "phase_label": g["phase_label"].iloc[0],
                "outbreak_threshold": float(g["outbreak_threshold"].iloc[0]),
                "outbreak_threshold_count": int(g["outbreak_threshold_count"].iloc[0]),
                "n_repeats": n,
                "n_early_fadeout": int(early_fadeout.sum()),
                "early_fadeout_probability": p_fadeout,
                "early_fadeout_se": float(np.sqrt(p_fadeout * (1.0 - p_fadeout) / n)) if n > 0 else np.nan,
                "n_major_outbreak": int(major.sum()),
                "major_outbreak_probability": p_major,
                "major_outbreak_se": float(np.sqrt(p_major * (1.0 - p_major) / n)) if n > 0 else np.nan,
                "n_fadeout_extinct_before_threshold": int(extinct.sum()),
                "fadeout_extinction_probability_before_threshold": float(extinct.mean()),
                "median_extinction_time": float(np.median(extinct_times)) if extinct_times.size else np.nan,
                "median_fadeout_extinction_time": float(np.median(fadeout_times)) if fadeout_times.size else np.nan,
                "mean_peak_I": float(g["peak_I"].mean()),
                "mean_peak_i": float(g["peak_i"].mean()),
                "mean_n_events": float(g["n_events"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _simulate_establishment_replicates(
    N: int,
    I0: int,
    R0_basic: float,
    beta0: float,
    gamma: float,
    mu: float,
    alpha: float,
    t_max: float,
    repeats: int,
    master_rng: np.random.Generator,
    critical_s_multiplier: float,
    outbreak_threshold: float,
    phase: float = 0.0,
    extra_metadata: dict | None = None,
) -> pd.DataFrame:
    """Run near-threshold Gillespie replicates and classify establishment outcomes."""
    metadata = {} if extra_metadata is None else dict(extra_metadata)
    S0, _, R_init = near_threshold_initial_counts(N=N, I0=I0, R0_basic=R0_basic, multiplier=critical_s_multiplier)
    threshold_count = max(int(np.ceil(float(outbreak_threshold) * int(N))), int(I0) + 1)
    phase_label = _phase_label(phase)
    rows = []
    for rep in range(1, int(repeats) + 1):
        result = simulate_gillespie_sir(
            N=N,
            I0=I0,
            beta0=beta0,
            gamma=gamma,
            mu=mu,
            alpha=alpha,
            phase=phase,
            R_init=R_init,
            t_max=t_max,
            rng=_child_rng(master_rng),
            record_trajectory=False,
            max_events=20_000,
            stop_if_peak_at_least=threshold_count,
        )
        major_outbreak = bool(result.peak_I >= threshold_count)
        early_fadeout = bool(not major_outbreak)
        rows.append(
            {
                **metadata,
                "replicate": rep,
                "N": int(N),
                "I0": int(I0),
                "R0_basic": float(R0_basic),
                "S0": int(S0),
                "R_init": int(R_init),
                "alpha": float(alpha),
                "phase": float(phase),
                "phase_label": phase_label,
                "t_max": float(t_max),
                "outbreak_threshold": float(outbreak_threshold),
                "outbreak_threshold_count": int(threshold_count),
                "extinct": bool(result.extinct),
                "extinction_time": result.extinction_time,
                "early_fadeout": early_fadeout,
                "major_outbreak": major_outbreak,
                "establishment_time": result.t_peak if major_outbreak else np.nan,
                "t_final": result.t_final,
                "final_S": result.s_final,
                "final_I": result.i_final,
                "final_R": result.r_final,
                "peak_I": result.peak_I,
                "peak_i": result.peak_I / int(N),
                "t_peak": result.t_peak,
                "n_events": result.n_events,
            }
        )
    return pd.DataFrame(rows)


def run_stochastic_trajectory_comparison(settings: StochasticSettings, master_rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """Figure 11: stochastic trajectories versus deterministic reference."""
    beta0 = beta0_from_R0_vital(settings.trajectory_R0, cfg.GAMMA, cfg.MU)
    sample_times = np.arange(0.0, settings.trajectory_t_max + 0.5 * settings.trajectory_sample_dt, settings.trajectory_sample_dt)
    stochastic_frames, summary_rows = [], []

    for rep in tqdm(range(1, settings.trajectory_repeats + 1), desc="stochastic trajectories", leave=False):
        result = simulate_gillespie_sir(
            N=settings.trajectory_N,
            I0=settings.trajectory_I0,
            beta0=beta0,
            gamma=cfg.GAMMA,
            mu=cfg.MU,
            alpha=0.0,
            phase=0.0,
            t_max=settings.trajectory_t_max,
            rng=_child_rng(master_rng),
            record_trajectory=True,
        )
        states = sample_event_trajectory(result.event_times, result.event_states, sample_times)
        stochastic_frames.append(
            pd.DataFrame(
                {
                    "replicate": rep,
                    "t": sample_times,
                    "S": states[:, 0],
                    "I": states[:, 1],
                    "R": states[:, 2],
                    "s_prop": states[:, 0] / settings.trajectory_N,
                    "i_prop": states[:, 1] / settings.trajectory_N,
                    "r_prop": states[:, 2] / settings.trajectory_N,
                    "N": settings.trajectory_N,
                    "I0": settings.trajectory_I0,
                    "R0_basic": settings.trajectory_R0,
                    "alpha": 0.0,
                    "phase": 0.0,
                }
            )
        )
        summary_rows.append(
            {
                "replicate": rep,
                "N": settings.trajectory_N,
                "I0": settings.trajectory_I0,
                "R0_basic": settings.trajectory_R0,
                "alpha": 0.0,
                "phase": 0.0,
                "extinct": result.extinct,
                "extinction_time": result.extinction_time,
                "peak_I": result.peak_I,
                "peak_i": result.peak_I / settings.trajectory_N,
                "t_peak": result.t_peak,
                "final_I": result.i_final,
                "n_events": result.n_events,
            }
        )

    stochastic_df = pd.concat(stochastic_frames, ignore_index=True)
    trajectory_summary = pd.DataFrame(summary_rows)

    y0 = np.array([1.0 - settings.trajectory_I0 / settings.trajectory_N, settings.trajectory_I0 / settings.trajectory_N, 0.0], dtype=float)
    t_det, y_det, _ = solve_ode(sir_vital_rhs, y0, (0.0, settings.trajectory_t_max), cfg.DEFAULT_DT, args=(beta0, cfg.GAMMA, cfg.MU), method="rk4", context="stochastic_deterministic_reference")
    deterministic_df = pd.DataFrame({"t": t_det, "s": y_det[:, 0], "i": y_det[:, 1], "r": y_det[:, 2], "N": settings.trajectory_N, "I0": settings.trajectory_I0, "alpha": 0.0, "phase": 0.0, "R0_basic": settings.trajectory_R0})

    write_csv(stochastic_df, "stochastic_trajectory_timeseries.csv")
    write_csv(trajectory_summary, "stochastic_trajectory_summary.csv")
    write_csv(deterministic_df, "stochastic_deterministic_reference.csv")
    plot_stochastic_trajectories(stochastic_df, deterministic_df, filename_stem=cfg.FIG11_NAME)
    return {"stochastic": stochastic_df, "deterministic": deterministic_df, "summary": trajectory_summary}


def run_extinction_grid(settings: StochasticSettings, master_rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """Figures 12--13: early fade-out probability and fade-out extinction-time distribution."""
    beta0 = beta0_from_R0_vital(settings.extinction_R0, cfg.GAMMA, cfg.MU)
    detail_frames = []
    pairs = [(N, I0) for N in settings.N_values for I0 in settings.I0_values]
    for N, I0 in tqdm(pairs, desc="fade-out grid", leave=False):
        detail_frames.append(
            _simulate_establishment_replicates(
                N=N,
                I0=I0,
                R0_basic=settings.extinction_R0,
                beta0=beta0,
                gamma=cfg.GAMMA,
                mu=cfg.MU,
                alpha=0.0,
                phase=0.0,
                t_max=settings.extinction_t_max,
                repeats=settings.extinction_repeats,
                master_rng=master_rng,
                critical_s_multiplier=settings.critical_s_multiplier,
                outbreak_threshold=settings.outbreak_threshold,
                extra_metadata={"experiment": "near_threshold_N_I0_grid"},
            )
        )
    detail_df = pd.concat(detail_frames, ignore_index=True)
    summary_df = _summary_from_detail(detail_df, group_cols=["N", "I0"])
    write_csv(detail_df, "stochastic_extinction_detail.csv")
    write_csv(summary_df, "stochastic_extinction_summary.csv")
    plot_extinction_probability_heatmap(summary_df, filename_stem=cfg.FIG12_NAME)
    plot_extinction_time_distribution(detail_df, filename_stem=cfg.FIG13_NAME)
    return {"detail": detail_df, "summary": summary_df}


def run_alpha_extinction_experiment(settings: StochasticSettings, master_rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """Figure 14: early fade-out risk under seasonal forcing strengths at phase 0."""
    beta0 = beta0_from_R0_vital(settings.extinction_R0, cfg.GAMMA, cfg.MU)
    detail_frames = []
    for alpha in tqdm(settings.alpha_values, desc="alpha fade-out", leave=False):
        detail_frames.append(
            _simulate_establishment_replicates(
                N=settings.alpha_N,
                I0=settings.alpha_I0,
                R0_basic=settings.extinction_R0,
                beta0=beta0,
                gamma=cfg.GAMMA,
                mu=cfg.MU,
                alpha=float(alpha),
                phase=0.0,
                t_max=settings.alpha_t_max,
                repeats=settings.alpha_repeats,
                master_rng=master_rng,
                critical_s_multiplier=settings.critical_s_multiplier,
                outbreak_threshold=settings.outbreak_threshold,
                extra_metadata={"experiment": "seasonal_near_threshold_alpha"},
            )
        )
    detail_df = pd.concat(detail_frames, ignore_index=True)
    summary_df = _summary_from_detail(detail_df, group_cols=["alpha"])
    summary_df["N"] = settings.alpha_N
    summary_df["I0"] = settings.alpha_I0
    write_csv(detail_df, "stochastic_alpha_extinction_detail.csv")
    write_csv(summary_df, "stochastic_alpha_extinction.csv")
    plot_alpha_extinction_probability(summary_df, filename_stem=cfg.FIG14_NAME)
    return {"detail": detail_df, "summary": summary_df}


def run_phase_sensitivity_experiment(settings: StochasticSettings, master_rng: np.random.Generator) -> pd.DataFrame:
    """Figure 14b: compare phase choices in seasonal stochastic establishment."""
    beta0 = beta0_from_R0_vital(settings.extinction_R0, cfg.GAMMA, cfg.MU)
    detail_frames = []
    pairs = [(alpha, phase) for alpha in settings.phase_alpha_values for phase in settings.phase_values]
    for alpha, phase in tqdm(pairs, desc="phase sensitivity", leave=False):
        detail_frames.append(
            _simulate_establishment_replicates(
                N=settings.alpha_N,
                I0=settings.alpha_I0,
                R0_basic=settings.extinction_R0,
                beta0=beta0,
                gamma=cfg.GAMMA,
                mu=cfg.MU,
                alpha=float(alpha),
                phase=float(phase),
                t_max=settings.alpha_t_max,
                repeats=settings.phase_repeats,
                master_rng=master_rng,
                critical_s_multiplier=settings.critical_s_multiplier,
                outbreak_threshold=settings.outbreak_threshold,
                extra_metadata={"experiment": "seasonal_phase_sensitivity"},
            )
        )
    detail_df = pd.concat(detail_frames, ignore_index=True)
    summary_df = _summary_from_detail(detail_df, group_cols=["alpha", "phase"])
    summary_df["N"] = settings.alpha_N
    summary_df["I0"] = settings.alpha_I0
    write_csv(summary_df, "stochastic_phase_sensitivity.csv")
    plot_phase_sensitivity(summary_df, filename_stem=cfg.FIG14B_NAME)
    return summary_df


def run_all_stochastic_experiments(mode: str = "full") -> dict[str, object]:
    """Run all stochastic experiments and generate Figures 11--14b."""
    ensure_output_dirs()
    cfg.set_random_seed(cfg.RANDOM_SEED)
    settings = get_stochastic_settings(mode)
    master_rng = np.random.default_rng(cfg.RANDOM_SEED + 911)
    outputs: dict[str, object] = {"settings": settings}
    outputs["trajectory_comparison"] = run_stochastic_trajectory_comparison(settings, master_rng)
    outputs["extinction_grid"] = run_extinction_grid(settings, master_rng)
    outputs["alpha_extinction"] = run_alpha_extinction_experiment(settings, master_rng)
    outputs["phase_sensitivity"] = run_phase_sensitivity_experiment(settings, master_rng)
    return outputs
