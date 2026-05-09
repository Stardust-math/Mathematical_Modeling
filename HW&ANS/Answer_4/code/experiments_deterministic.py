"""Deterministic SIR experiments including seasonal alpha scans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import config as cfg
from metrics import (
    classify_seasonal_pattern,
    crop_time_window,
    detect_infection_peaks,
    downsample_by_dt,
    make_timeseries_dataframe,
    outbreak_threshold_label,
    summarize_peak_statistics,
    summarize_solution,
)
from plotting import (
    plot_R0_comparison,
    plot_alpha_bifurcation,
    plot_basic_sir_triplet,
    plot_basic_vs_demographic,
    plot_beta_infection_phase,
    plot_demographic_R0_comparison,
    plot_demographic_sir,
    plot_gamma_comparison,
    plot_pattern_classification,
    plot_peak_statistics,
    plot_seasonal_alpha_curves,
    plot_threshold_s0,
)
from sir_models import (
    beta0_from_R0_vital,
    beta_from_R0_basic,
    seasonal_beta,
    sir_basic_rhs,
    sir_vital_rhs,
)
from solvers import solve_ode


@dataclass(frozen=True)
class AlphaScanSettings:
    """Settings for seasonal alpha scan."""

    alpha_grid: np.ndarray
    t_end: float
    transient: float
    sample_start: float
    sample_end: float
    dt: float


def get_alpha_scan_settings(mode: str = "full") -> AlphaScanSettings:
    """Return quick or full alpha-scan settings."""
    mode = mode.lower().strip()
    if mode == "quick":
        return AlphaScanSettings(
            alpha_grid=np.linspace(cfg.ALPHA_SCAN_MIN, cfg.ALPHA_SCAN_MAX, cfg.ALPHA_SCAN_N_QUICK),
            t_end=cfg.ALPHA_SCAN_T_END_QUICK,
            transient=cfg.ALPHA_SCAN_TRANSIENT_QUICK,
            sample_start=cfg.ALPHA_SCAN_SAMPLE_START_QUICK,
            sample_end=cfg.ALPHA_SCAN_SAMPLE_END_QUICK,
            dt=cfg.ALPHA_SCAN_DT,
        )
    if mode == "full":
        return AlphaScanSettings(
            alpha_grid=np.linspace(cfg.ALPHA_SCAN_MIN, cfg.ALPHA_SCAN_MAX, cfg.ALPHA_SCAN_N_FULL),
            t_end=cfg.ALPHA_SCAN_T_END_FULL,
            transient=cfg.ALPHA_SCAN_TRANSIENT_FULL,
            sample_start=cfg.ALPHA_SCAN_SAMPLE_START_FULL,
            sample_end=cfg.ALPHA_SCAN_SAMPLE_END_FULL,
            dt=cfg.ALPHA_SCAN_DT,
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


def infection_width_summary(t: np.ndarray, i: np.ndarray, threshold_fraction: float = 0.5) -> dict:
    """Summarize infection-curve width in days at a fraction of peak height."""
    t = np.asarray(t, dtype=float)
    i = np.asarray(i, dtype=float)
    if i.size == 0:
        return {"width_days_at_fractional_peak": np.nan, "width_fraction": threshold_fraction}
    peak = float(np.max(i))
    if peak <= 0.0:
        return {"width_days_at_fractional_peak": 0.0, "width_fraction": threshold_fraction}
    mask = i >= float(threshold_fraction) * peak
    if not np.any(mask):
        return {"width_days_at_fractional_peak": 0.0, "width_fraction": threshold_fraction}
    t_selected = t[mask]
    width_days = (float(t_selected[-1]) - float(t_selected[0])) * cfg.DAYS_PER_YEAR
    return {"width_days_at_fractional_peak": width_days, "width_fraction": threshold_fraction}


def summarize_late_peaks(t: np.ndarray, i: np.ndarray, model: str, start: float, **metadata) -> dict:
    """Summarize recurrent peaks after the initial outbreak has passed."""
    mask = np.asarray(t) >= float(start)
    peak_df = detect_infection_peaks(np.asarray(t)[mask], np.asarray(i)[mask], alpha=0.0, min_distance_years=0.25, prominence_fraction=0.01, min_prominence=1.0e-12)
    row = {"model": model, **metadata, "late_start": float(start), "n_late_peaks": int(len(peak_df))}
    if peak_df.empty:
        row.update({"first_late_peak_time": np.nan, "first_late_peak_i": np.nan, "mean_late_peak_i": np.nan, "mean_late_peak_interval": np.nan})
    else:
        intervals = np.diff(peak_df["t_peak"].to_numpy(dtype=float))
        row.update(
            {
                "first_late_peak_time": float(peak_df["t_peak"].iloc[0]),
                "first_late_peak_i": float(peak_df["i_peak"].iloc[0]),
                "mean_late_peak_i": float(peak_df["i_peak"].mean()),
                "mean_late_peak_interval": float(np.mean(intervals)) if intervals.size else np.nan,
            }
        )
    return row


def solve_seasonal_grid(
    alpha_grid: np.ndarray,
    beta0: float,
    t_end: float,
    dt: float,
    store_start: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve seasonal SIR for all alpha values simultaneously with vectorized RK4."""
    alpha_grid = np.asarray(alpha_grid, dtype=float)
    if np.any(alpha_grid < 0.0) or np.any(alpha_grid > 1.0):
        raise ValueError("alpha values must stay in [0, 1] so beta(t) is non-negative.")

    n_alpha = len(alpha_grid)
    n_steps = int(np.ceil(float(t_end) / float(dt)))
    t_full = np.arange(n_steps + 1, dtype=float) * float(dt)
    t_full[-1] = float(t_end)

    s = np.full(n_alpha, cfg.DEFAULT_Y0[0], dtype=float)
    i = np.full(n_alpha, cfg.DEFAULT_Y0[1], dtype=float)
    r = np.full(n_alpha, cfg.DEFAULT_Y0[2], dtype=float)

    store_start_idx = int(np.searchsorted(t_full, float(store_start), side="left"))
    t_store = t_full[store_start_idx:]
    s_store = np.empty((n_alpha, len(t_store)), dtype=float)
    i_store = np.empty((n_alpha, len(t_store)), dtype=float)
    r_store = np.empty((n_alpha, len(t_store)), dtype=float)
    store_idx = 0
    if store_start_idx == 0:
        s_store[:, store_idx] = s
        i_store[:, store_idx] = i
        r_store[:, store_idx] = r
        store_idx += 1

    two_pi = 2.0 * np.pi

    def rhs(t_now: float, s_now: np.ndarray, i_now: np.ndarray, r_now: np.ndarray):
        beta_now = beta0 * (1.0 + alpha_grid * np.cos(two_pi * t_now))
        if np.any(beta_now < -1.0e-12):
            raise RuntimeError("Negative beta(t) detected.")
        ds = cfg.MU - beta_now * s_now * i_now - cfg.MU * s_now
        di = beta_now * s_now * i_now - cfg.GAMMA * i_now - cfg.MU * i_now
        dr = cfg.GAMMA * i_now - cfg.MU * r_now
        return ds, di, dr

    for step in range(n_steps):
        h = t_full[step + 1] - t_full[step]
        t_now = t_full[step]
        k1s, k1i, k1r = rhs(t_now, s, i, r)
        k2s, k2i, k2r = rhs(t_now + 0.5 * h, s + 0.5 * h * k1s, i + 0.5 * h * k1i, r + 0.5 * h * k1r)
        k3s, k3i, k3r = rhs(t_now + 0.5 * h, s + 0.5 * h * k2s, i + 0.5 * h * k2i, r + 0.5 * h * k2r)
        k4s, k4i, k4r = rhs(t_now + h, s + h * k3s, i + h * k3i, r + h * k3r)
        s = s + (h / 6.0) * (k1s + 2.0 * k2s + 2.0 * k3s + k4s)
        i = i + (h / 6.0) * (k1i + 2.0 * k2i + 2.0 * k3i + k4i)
        r = r + (h / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)

        for arr in (s, i, r):
            tiny_neg = (arr < 0.0) & (arr > -1.0e-14)
            arr[tiny_neg] = 0.0

        if np.min(i) < -cfg.NEGATIVE_TOL:
            raise RuntimeError("Negative infection proportion detected in seasonal grid integration.")

        if step + 1 >= store_start_idx:
            s_store[:, store_idx] = s
            i_store[:, store_idx] = i
            r_store[:, store_idx] = r
            store_idx += 1

    return t_store, s_store, i_store, r_store


def run_basic_single_outbreak(method: str = "rk4") -> dict:
    """Figure 1: basic SIR single-outbreak experiment."""
    R0 = cfg.BASIC_SINGLE_R0
    beta = beta_from_R0_basic(R0, cfg.GAMMA)
    t, y, diagnostics = solve_ode(sir_basic_rhs, cfg.DEFAULT_Y0, (0.0, cfg.BASIC_T_END), cfg.DEFAULT_DT, args=(beta, cfg.GAMMA), method=method, context="basic_sir")
    t_save, y_save = downsample_by_dt(t, y, cfg.SAVE_DT_DAILY)
    df = make_timeseries_dataframe(t_save, y_save, model="basic_sir", R0=R0, beta=beta, gamma=cfg.GAMMA)
    summary = summarize_solution(t, y, model="basic_sir", R0=R0, beta=beta, gamma=cfg.GAMMA)
    summary.update(diagnostics)
    summary["final_infection_near_zero"] = bool(summary["i_final"] < 1.0e-6)
    write_csv(df, "basic_sir_timeseries.csv")
    write_csv(pd.DataFrame([summary]), "basic_sir_summary.csv")
    plot_basic_sir_triplet(df, title=rf"Basic SIR dynamics ($R_0={R0:g}$)", filename_stem=cfg.FIG01_NAME)
    return summary


def run_R0_comparison(method: str = "rk4") -> pd.DataFrame:
    """Figure 2: compare infection curves under different R0 values."""
    frames, summaries = [], []
    for R0 in cfg.R0_VALUES:
        beta = beta_from_R0_basic(R0, cfg.GAMMA)
        t, y, diagnostics = solve_ode(sir_basic_rhs, cfg.DEFAULT_Y0, (0.0, cfg.BASIC_COMPARE_T_END), cfg.DEFAULT_DT, args=(beta, cfg.GAMMA), method=method, context=f"basic_R0_{R0:g}")
        t_save, y_save = downsample_by_dt(t, y, cfg.SAVE_DT_DAILY)
        frames.append(make_timeseries_dataframe(t_save, y_save, model="basic_sir", R0=R0, beta=beta, gamma=cfg.GAMMA))
        summary = summarize_solution(t, y, model="basic_sir", R0=R0, beta=beta, gamma=cfg.GAMMA)
        summary.update(diagnostics)
        summaries.append(summary)
    df = pd.concat(frames, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    write_csv(df, "R0_comparison_timeseries.csv")
    write_csv(summary_df, "R0_comparison_summary.csv")
    plot_R0_comparison(df, title="Basic SIR: infection curves under different $R_0$", filename_stem=cfg.FIG02_NAME)
    return summary_df


def run_gamma_comparison(method: str = "rk4") -> pd.DataFrame:
    """Figure 2b: compare basic SIR dynamics under different recovery rates."""
    R0 = cfg.GAMMA_COMPARISON_R0
    frames, summaries = [], []
    for gamma in cfg.GAMMA_COMPARISON_VALUES:
        beta = beta_from_R0_basic(R0, gamma)
        t, y, diagnostics = solve_ode(
            sir_basic_rhs,
            cfg.DEFAULT_Y0,
            (0.0, cfg.GAMMA_COMPARISON_T_END),
            cfg.DEFAULT_DT,
            args=(beta, gamma),
            method=method,
            context=f"gamma_comparison_{gamma:g}",
        )
        t_save, y_save = downsample_by_dt(t, y, cfg.SAVE_DT_DAILY)
        frames.append(make_timeseries_dataframe(t_save, y_save, model="basic_sir_gamma_comparison", R0=R0, beta=beta, gamma=gamma))
        summary = summarize_solution(t, y, model="basic_sir_gamma_comparison", R0=R0, beta=beta, gamma=gamma)
        summary.update(diagnostics)
        summary.update(infection_width_summary(t, y[:, 1], threshold_fraction=0.5))
        summary["mean_infectious_period_days"] = cfg.DAYS_PER_YEAR / float(gamma)
        summary["t_peak_days"] = float(summary["t_peak"]) * cfg.DAYS_PER_YEAR
        summaries.append(summary)
    df = pd.concat(frames, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    write_csv(df, "gamma_comparison_timeseries.csv")
    write_csv(summary_df, "gamma_comparison_summary.csv")
    plot_gamma_comparison(df, title=rf"Basic SIR: effect of recovery rate at fixed $R_0={R0:g}$", filename_stem=cfg.FIG02B_NAME)
    return summary_df


def run_threshold_experiment(method: str = "rk4") -> pd.DataFrame:
    """Figure 3: verify outbreak threshold s0 > 1/R0 by varying s0.

    The basic SIR model has the invariant
    i+s-(1/R0)log(s)=constant, so the peak infection can be computed
    directly. This avoids a slow loop of many nearly identical ODE solves.
    """
    del method
    R0 = cfg.THRESHOLD_R0
    beta = beta_from_R0_basic(R0, cfg.GAMMA)
    threshold = 1.0 / R0
    rows = []
    for s0 in cfg.THRESHOLD_S0_VALUES:
        i0 = min(cfg.I0, max(1.0e-8, 0.5 * (1.0 - float(s0))))
        r0 = 1.0 - float(s0) - i0
        if r0 < 0.0:
            continue
        if float(s0) > threshold:
            peak_i = i0 + float(s0) - threshold + threshold * np.log(threshold / float(s0))
            peak_i = max(float(peak_i), float(i0))
        else:
            peak_i = float(i0)
        rows.append(
            {
                "model": "basic_sir_threshold",
                "R0": R0,
                "beta": beta,
                "gamma": cfg.GAMMA,
                "s0": float(s0),
                "i0": float(i0),
                "r0": float(r0),
                "threshold_1_over_R0": threshold,
                "threshold_label": outbreak_threshold_label(R0, s0),
                "peak_i": float(peak_i),
                "peak_increase_over_i0": float(peak_i - i0),
                "mass_error_max": 0.0,
                "min_state_value": float(min(s0, i0, r0)),
            }
        )
    df = pd.DataFrame(rows)
    write_csv(df, "threshold_s0_summary.csv")
    plot_threshold_s0(df, R0=R0, filename_stem=cfg.FIG03_NAME)
    return df

def run_demographic_sir(method: str = "rk4", mode: str = "full") -> dict:
    """Figure 4: long-term demographic SIR with recurrent damped peaks."""
    R0 = cfg.VITAL_R0
    beta0 = beta0_from_R0_vital(R0, cfg.GAMMA, cfg.MU)
    t_end = 20.0 if mode == "quick" else cfg.VITAL_T_END
    t, y, diagnostics = solve_ode(sir_vital_rhs, cfg.DEFAULT_Y0, (0.0, t_end), cfg.DEFAULT_DT, args=(beta0, cfg.GAMMA, cfg.MU), method=method, context="demographic_sir")
    t_save, y_save = downsample_by_dt(t, y, cfg.SAVE_DT_DAILY)
    df = make_timeseries_dataframe(t_save, y_save, model="demographic_sir", R0=R0, beta0=beta0, gamma=cfg.GAMMA, mu=cfg.MU)
    summary = summarize_solution(t, y, model="demographic_sir", R0=R0, beta0=beta0, gamma=cfg.GAMMA, mu=cfg.MU)
    summary.update(diagnostics)
    late_summary = summarize_late_peaks(t, y[:, 1], model="demographic_sir", start=cfg.DEMOGRAPHIC_LATE_START, R0=R0, beta0=beta0, gamma=cfg.GAMMA, mu=cfg.MU)
    summary["has_recurrent_peaks"] = bool(late_summary["n_late_peaks"] >= 1)
    summary["n_late_peaks"] = late_summary["n_late_peaks"]
    write_csv(df, "demographic_timeseries.csv")
    write_csv(pd.DataFrame([summary]), "demographic_summary.csv")
    write_csv(pd.DataFrame([late_summary]), "demographic_late_peak_summary.csv")
    plot_demographic_sir(df, title=rf"Demographic SIR dynamics ($R_0={R0:g}$)", filename_stem=cfg.FIG04_NAME)
    return summary


def run_basic_vs_demographic(method: str = "rk4", mode: str = "full") -> pd.DataFrame:
    """Figure 5: compare basic and demographic SIR infection curves."""
    R0 = cfg.VITAL_R0
    beta_basic = beta_from_R0_basic(R0, cfg.GAMMA)
    beta0_demo = beta0_from_R0_vital(R0, cfg.GAMMA, cfg.MU)
    t_end = 5.0 if mode == "quick" else cfg.BASIC_VS_VITAL_T_END
    t_basic, y_basic, diag_basic = solve_ode(sir_basic_rhs, cfg.DEFAULT_Y0, (0.0, t_end), cfg.DEFAULT_DT, args=(beta_basic, cfg.GAMMA), method=method, context="basic_vs_demo_basic")
    t_demo, y_demo, diag_demo = solve_ode(sir_vital_rhs, cfg.DEFAULT_Y0, (0.0, t_end), cfg.DEFAULT_DT, args=(beta0_demo, cfg.GAMMA, cfg.MU), method=method, context="basic_vs_demo_demographic")
    t_basic_save, y_basic_save = downsample_by_dt(t_basic, y_basic, cfg.SAVE_DT_DAILY)
    t_demo_save, y_demo_save = downsample_by_dt(t_demo, y_demo, cfg.SAVE_DT_DAILY)
    df = pd.concat(
        [
            make_timeseries_dataframe(t_basic_save, y_basic_save, model="basic_sir", R0=R0, beta=beta_basic, gamma=cfg.GAMMA),
            make_timeseries_dataframe(t_demo_save, y_demo_save, model="demographic_sir", R0=R0, beta0=beta0_demo, gamma=cfg.GAMMA, mu=cfg.MU),
        ],
        ignore_index=True,
    )
    s1 = summarize_solution(t_basic, y_basic, model="basic_sir", R0=R0, beta=beta_basic, gamma=cfg.GAMMA)
    s1.update(diag_basic)
    s2 = summarize_solution(t_demo, y_demo, model="demographic_sir", R0=R0, beta0=beta0_demo, gamma=cfg.GAMMA, mu=cfg.MU)
    s2.update(diag_demo)
    summary_df = pd.DataFrame([s1, s2])
    write_csv(df, "basic_vs_demographic_timeseries.csv")
    write_csv(summary_df, "basic_vs_demographic_summary.csv")
    plot_basic_vs_demographic(df, title="Basic SIR vs demographic SIR", filename_stem=cfg.FIG05_NAME)
    return summary_df


def run_demographic_R0_comparison(method: str = "rk4", mode: str = "full") -> pd.DataFrame:
    """Figure 4b: compare recurrent peaks in demographic SIR under different R0 values."""
    frames, summaries = [], []
    t_end = cfg.DEMOGRAPHIC_R0_T_END_QUICK if mode == "quick" else cfg.DEMOGRAPHIC_R0_T_END_FULL
    for R0 in cfg.DEMOGRAPHIC_R0_VALUES:
        beta0 = beta0_from_R0_vital(R0, cfg.GAMMA, cfg.MU)
        t, y, diagnostics = solve_ode(sir_vital_rhs, cfg.DEFAULT_Y0, (0.0, t_end), cfg.DEFAULT_DT, args=(beta0, cfg.GAMMA, cfg.MU), method=method, context=f"demographic_R0_{R0:g}")
        t_save, y_save = downsample_by_dt(t, y, cfg.SAVE_DT_WEEKLY)
        frames.append(make_timeseries_dataframe(t_save, y_save, model="demographic_sir_R0_comparison", R0=R0, beta0=beta0, gamma=cfg.GAMMA, mu=cfg.MU))
        summary = summarize_solution(t, y, model="demographic_sir_R0_comparison", R0=R0, beta0=beta0, gamma=cfg.GAMMA, mu=cfg.MU)
        summary.update(diagnostics)
        summary.update(summarize_late_peaks(t, y[:, 1], model="demographic_sir_R0_comparison", start=cfg.DEMOGRAPHIC_LATE_START, R0=R0, beta0=beta0, gamma=cfg.GAMMA, mu=cfg.MU))
        summaries.append(summary)
    df = pd.concat(frames, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    write_csv(df, "demographic_R0_comparison_timeseries.csv")
    write_csv(summary_df, "demographic_R0_comparison_summary.csv")
    plot_demographic_R0_comparison(df, title="Demographic SIR: late recurrent peaks under different $R_0$", filename_stem=cfg.FIG04B_NAME)
    return summary_df


def run_seasonal_alpha_curves(mode: str = "full") -> pd.DataFrame:
    """Figure 6: seasonal SIR long-run infection curves for selected alpha values."""
    R0 = cfg.SEASONAL_R0
    beta0 = beta0_from_R0_vital(R0, cfg.GAMMA, cfg.MU)
    alpha_grid = np.asarray(cfg.SEASONAL_ALPHA_VALUES, dtype=float)
    t_end = 20.0 if mode == "quick" else cfg.SEASONAL_T_END
    plot_start = 10.0 if mode == "quick" else cfg.SEASONAL_PLOT_START
    plot_end = 15.0 if mode == "quick" else cfg.SEASONAL_PLOT_END
    t_late, s_grid, i_grid, r_grid = solve_seasonal_grid(alpha_grid, beta0, t_end=t_end, dt=cfg.LONG_DT, store_start=plot_start)
    frames, summary_rows = [], []
    for idx, alpha in enumerate(alpha_grid):
        y = np.column_stack([s_grid[idx], i_grid[idx], r_grid[idx]])
        t_save, y_save = downsample_by_dt(t_late, y, cfg.SAVE_DT_WEEKLY)
        frames.append(make_timeseries_dataframe(t_save, y_save, model="seasonal_vital_sir", R0=R0, beta0=beta0, gamma=cfg.GAMMA, mu=cfg.MU, alpha=float(alpha)))
        summary_rows.append(summarize_solution(t_late, y, model="seasonal_vital_sir_late", R0=R0, beta0=beta0, gamma=cfg.GAMMA, mu=cfg.MU, alpha=float(alpha)))
    long_df = pd.concat(frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    write_csv(long_df, "seasonal_alpha_curves_timeseries.csv")
    write_csv(summary_df, "seasonal_alpha_curves_summary.csv")
    fig_df = long_df[(long_df["t"] >= plot_start) & (long_df["t"] <= plot_end)].copy()
    plot_seasonal_alpha_curves(fig_df, title="Seasonal SIR: long-run infection curves after transient removal", filename_stem=cfg.FIG06_NAME)
    return summary_df


def run_beta_infection_phase(mode: str = "full") -> pd.DataFrame:
    """Figure 7: beta(t) and i(t) for one seasonal forcing strength."""
    R0 = cfg.SEASONAL_R0
    alpha = cfg.FIG07_ALPHA
    beta0 = beta0_from_R0_vital(R0, cfg.GAMMA, cfg.MU)
    t_start = 12.0 if mode == "quick" else cfg.FIG07_T_START
    t_end = 15.0 if mode == "quick" else cfg.FIG07_T_END
    t_win, _, i_grid, _ = solve_seasonal_grid(np.asarray([alpha]), beta0, t_end=t_end, dt=cfg.LONG_DT, store_start=t_start)
    i_t = i_grid[0]
    beta_values = seasonal_beta(t_win, beta0=beta0, alpha=alpha)
    phase_df = pd.DataFrame({"t": t_win, "beta_t": beta_values, "i_t": i_t, "alpha": alpha, "R0": R0, "beta0": beta0})
    write_csv(phase_df, "beta_infection_phase_timeseries.csv")
    plot_beta_infection_phase(t=t_win, beta_t=beta_values, i_t=i_t, alpha=alpha, filename_stem=cfg.FIG07_NAME)
    return phase_df


def run_seasonal_parameter_scan(mode: str = "full") -> dict[str, pd.DataFrame]:
    """Figures 8--10: alpha bifurcation, peak statistics, and pattern classification."""
    settings = get_alpha_scan_settings(mode)
    beta0 = beta0_from_R0_vital(cfg.SEASONAL_R0, cfg.GAMMA, cfg.MU)
    t_late, _, i_grid, _ = solve_seasonal_grid(settings.alpha_grid, beta0, t_end=settings.t_end, dt=settings.dt, store_start=settings.transient)

    poincare_frames, peak_event_frames, peak_summary_rows, classification_rows = [], [], [], []
    sample_times = np.arange(settings.sample_start, settings.sample_end + 0.5, 1.0)

    for idx, alpha in enumerate(tqdm(settings.alpha_grid, desc="alpha scan", leave=False)):
        i_late = i_grid[idx]
        i_samples = np.interp(sample_times, t_late, i_late)
        poincare_df = pd.DataFrame(
            {"alpha": float(alpha), "sample_index": np.arange(1, len(sample_times) + 1, dtype=int), "t_sample": sample_times, "i_sample": i_samples}
        )
        poincare_frames.append(poincare_df)
        peak_df = detect_infection_peaks(t_late, i_late, alpha=float(alpha), min_distance_years=cfg.PEAK_MIN_DISTANCE_YEARS)
        if not peak_df.empty:
            peak_event_frames.append(peak_df)
        peak_summary = summarize_peak_statistics(float(alpha), peak_df)
        peak_summary_rows.append(peak_summary)
        classification_rows.append(classify_seasonal_pattern(float(alpha), poincare_df["i_sample"].to_numpy(dtype=float), i_late, peak_summary))

    poincare_all = pd.concat(poincare_frames, ignore_index=True)
    peak_events = pd.concat(peak_event_frames, ignore_index=True) if peak_event_frames else pd.DataFrame(columns=["alpha", "peak_index", "t_peak", "i_peak", "prominence"])
    peak_summary_df = pd.DataFrame(peak_summary_rows)
    classification_df = pd.DataFrame(classification_rows)
    write_csv(poincare_all, "alpha_scan_poincare.csv")
    write_csv(peak_events, "seasonal_peak_events.csv")
    write_csv(peak_summary_df, "seasonal_peak_summary.csv")
    write_csv(classification_df, "seasonal_pattern_classification.csv")
    plot_alpha_bifurcation(poincare_all, filename_stem=cfg.FIG08_NAME)
    plot_peak_statistics(peak_summary_df, filename_stem=cfg.FIG09_NAME)
    plot_pattern_classification(classification_df, filename_stem=cfg.FIG10_NAME)
    return {"poincare": poincare_all, "peak_events": peak_events, "peak_summary": peak_summary_df, "classification": classification_df}


def run_all_deterministic_experiments(method: str = "rk4", mode: str = "full") -> dict[str, object]:
    """Run all deterministic experiments and generate Figures 1--10."""
    ensure_output_dirs()
    cfg.set_random_seed(cfg.RANDOM_SEED)
    outputs: dict[str, object] = {}
    outputs["fig01_basic_sir"] = run_basic_single_outbreak(method=method)
    outputs["fig02_R0_comparison"] = run_R0_comparison(method=method)
    outputs["fig02b_gamma_comparison"] = run_gamma_comparison(method=method)
    outputs["fig03_threshold_s0"] = run_threshold_experiment(method=method)
    outputs["fig04_demographic_sir"] = run_demographic_sir(method=method, mode=mode)
    outputs["fig04b_demographic_R0_comparison"] = run_demographic_R0_comparison(method=method, mode=mode)
    outputs["fig05_basic_vs_demographic"] = run_basic_vs_demographic(method=method, mode=mode)
    outputs["fig06_seasonal_alpha_curves"] = run_seasonal_alpha_curves(mode=mode)
    outputs["fig07_beta_infection_phase"] = run_beta_infection_phase(mode=mode)
    outputs["fig08_to_fig10_seasonal_scan"] = run_seasonal_parameter_scan(mode=mode)
    return outputs
