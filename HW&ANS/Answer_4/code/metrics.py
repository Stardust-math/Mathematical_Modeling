"""Metrics, peak detection, and classification utilities for SIR experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

STATE_COLUMNS = ["s", "i", "r"]


def make_timeseries_dataframe(t: np.ndarray, y: np.ndarray, model: str, **metadata) -> pd.DataFrame:
    """Convert time and state arrays into a tidy DataFrame."""
    df = pd.DataFrame({"t": t, "s": y[:, 0], "i": y[:, 1], "r": y[:, 2]})
    df.insert(1, "model", model)
    for key, value in metadata.items():
        df[key] = value
    return df


def downsample_by_dt(t: np.ndarray, y: np.ndarray, save_dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Downsample a fixed-grid solution to a coarser output interval."""
    if len(t) <= 2:
        return t, y
    base_dt = float(np.median(np.diff(t)))
    stride = max(1, int(round(save_dt / base_dt)))
    indices = np.arange(0, len(t), stride, dtype=int)
    if indices[-1] != len(t) - 1:
        indices = np.append(indices, len(t) - 1)
    return t[indices], y[indices]


def crop_time_window(
    t: np.ndarray,
    y: np.ndarray,
    t_min: float | None = None,
    t_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a time-series segment within a specified time window."""
    mask = np.ones_like(t, dtype=bool)
    if t_min is not None:
        mask &= t >= float(t_min)
    if t_max is not None:
        mask &= t <= float(t_max)
    return t[mask], y[mask]


def peak_infection(t: np.ndarray, y: np.ndarray) -> dict:
    """Return the maximum infection level and occurrence time."""
    i = y[:, 1]
    idx = int(np.argmax(i))
    return {"peak_i": float(i[idx]), "t_peak": float(t[idx])}


def final_state(y: np.ndarray) -> dict:
    """Return final s, i, r values."""
    return {"s_final": float(y[-1, 0]), "i_final": float(y[-1, 1]), "r_final": float(y[-1, 2])}


def local_peak_count(t: np.ndarray, y: np.ndarray, min_prominence: float = 1.0e-7) -> dict:
    """Count local infection peaks with a prominence rule."""
    del t
    i = y[:, 1]
    if len(i) < 3:
        return {"n_local_peaks": 0, "largest_local_peak": float(np.max(i))}
    peak_indices, _ = find_peaks(i, prominence=min_prominence)
    largest = float(np.max(i[peak_indices])) if peak_indices.size else float(np.max(i))
    return {"n_local_peaks": int(peak_indices.size), "largest_local_peak": largest}


def summarize_solution(t: np.ndarray, y: np.ndarray, model: str, **metadata) -> dict:
    """Create a compact summary dictionary for one deterministic simulation."""
    summary = {"model": model, **metadata}
    summary.update(peak_infection(t, y))
    summary.update(final_state(y))
    summary.update(local_peak_count(t, y))
    summary["mass_error_max"] = float(np.max(np.abs(np.sum(y, axis=1) - 1.0)))
    summary["min_state_value"] = float(np.min(y))
    return summary


def outbreak_threshold_label(R0: float, s0: float) -> str:
    """Label whether s0 is above or below the SIR threshold 1/R0."""
    return "above_threshold" if float(s0) > 1.0 / float(R0) else "below_threshold"


def annual_infection_samples(
    t: np.ndarray,
    y: np.ndarray,
    start: float,
    end: float | None = None,
    period: float = 1.0,
) -> pd.DataFrame:
    """Sample I(t) at the same phase of each year."""
    end_time = float(t[-1]) if end is None else float(end)
    sample_times = np.arange(float(start), end_time + 0.5 * period, float(period))
    sample_times = sample_times[sample_times <= t[-1]]
    sampled_i = np.interp(sample_times, t, y[:, 1])
    return pd.DataFrame({"t": sample_times, "i_sample": sampled_i})


def detect_infection_peaks(
    t: np.ndarray,
    i: np.ndarray,
    alpha: float,
    min_distance_years: float = 0.25,
    prominence_fraction: float = 0.02,
    min_prominence: float = 1.0e-10,
) -> pd.DataFrame:
    """Detect local infection peaks in a long-run infection curve."""
    t = np.asarray(t, dtype=float)
    i = np.asarray(i, dtype=float)
    columns = ["alpha", "peak_index", "t_peak", "i_peak", "prominence"]
    if len(t) < 3 or len(i) < 3:
        return pd.DataFrame(columns=columns)

    i_range = float(np.max(i) - np.min(i))
    if i_range < min_prominence:
        return pd.DataFrame(columns=columns)

    dt = float(np.median(np.diff(t)))
    distance = max(1, int(round(float(min_distance_years) / dt)))
    prominence = max(float(min_prominence), float(prominence_fraction) * i_range)
    peak_indices, properties = find_peaks(i, distance=distance, prominence=prominence)
    if peak_indices.size == 0:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(
        {
            "alpha": float(alpha),
            "peak_index": np.arange(1, peak_indices.size + 1, dtype=int),
            "t_peak": t[peak_indices],
            "i_peak": i[peak_indices],
            "prominence": properties.get("prominences", np.full(peak_indices.size, np.nan)),
        }
    )


def summarize_peak_statistics(alpha: float, peak_df: pd.DataFrame) -> dict:
    """Summarize peak heights and peak intervals for one alpha."""
    row = {
        "alpha": float(alpha),
        "n_peaks": 0,
        "mean_peak_i": np.nan,
        "max_peak_i": np.nan,
        "mean_peak_interval": np.nan,
        "std_peak_interval": np.nan,
        "peak_height_cv": np.nan,
        "peak_interval_cv": np.nan,
    }
    if peak_df.empty:
        return row

    heights = peak_df["i_peak"].to_numpy(dtype=float)
    times = peak_df["t_peak"].to_numpy(dtype=float)
    intervals = np.diff(times)

    row["n_peaks"] = int(len(heights))
    row["mean_peak_i"] = float(np.mean(heights))
    row["max_peak_i"] = float(np.max(heights))
    row["peak_height_cv"] = float(np.std(heights, ddof=1) / np.mean(heights)) if len(heights) >= 2 and np.mean(heights) > 0 else 0.0

    if len(intervals) >= 1:
        row["mean_peak_interval"] = float(np.mean(intervals))
        row["std_peak_interval"] = float(np.std(intervals, ddof=1)) if len(intervals) >= 2 else 0.0
        row["peak_interval_cv"] = float(row["std_peak_interval"] / row["mean_peak_interval"]) if row["mean_peak_interval"] > 0 else np.nan
    return row


def count_distinct_poincare_levels(values: np.ndarray, tolerance: float | None = None) -> int:
    """Count approximate distinct levels among annual Poincare samples."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0
    sorted_values = np.sort(values)
    value_range = float(sorted_values[-1] - sorted_values[0])
    scale = max(float(np.max(np.abs(sorted_values))), 1.0e-12)
    tol = tolerance if tolerance is not None else max(1.0e-9, 0.02 * value_range, 0.001 * scale)

    count = 1
    current_level = sorted_values[0]
    for value in sorted_values[1:]:
        if abs(value - current_level) > tol:
            count += 1
            current_level = value
    return int(count)


def classify_seasonal_pattern(
    alpha: float,
    poincare_values: np.ndarray,
    long_i: np.ndarray,
    peak_summary: dict,
) -> dict:
    """Classify long-run seasonal dynamics with robust peak-statistic rules.

    The classifier intentionally gives priority to long-run amplitude and peak
    interval regularity. The Poincare level count is retained as a diagnostic,
    but it is not used as the primary decision variable because very small
    numerical differences can otherwise create artificial levels.
    """
    long_i = np.asarray(long_i, dtype=float)
    poincare_values = np.asarray(poincare_values, dtype=float)

    long_min = float(np.min(long_i))
    long_max = float(np.max(long_i))
    long_mean = float(np.mean(long_i))
    long_range = long_max - long_min
    relative_amplitude = long_range / max(long_mean, 1.0e-12)
    poincare_levels = count_distinct_poincare_levels(poincare_values)
    poincare_range = float(np.max(poincare_values) - np.min(poincare_values)) if poincare_values.size else np.nan

    n_peaks = int(peak_summary.get("n_peaks", 0))
    interval = float(peak_summary.get("mean_peak_interval", np.nan))
    height_cv = float(peak_summary.get("peak_height_cv", np.nan))
    interval_cv = float(peak_summary.get("peak_interval_cv", np.nan))

    finite_interval = np.isfinite(interval)
    finite_height_cv = np.isfinite(height_cv)
    finite_interval_cv = np.isfinite(interval_cv)

    small_absolute = long_max < 2.5e-3 or long_range < 2.5e-3
    small_relative = relative_amplitude < 0.05
    regular_intervals = (not finite_interval_cv) or interval_cv <= 0.30
    highly_irregular = (finite_interval_cv and interval_cv > 0.40) or (finite_height_cv and height_cv > 0.90)

    if small_absolute or small_relative:
        label = "near equilibrium / small oscillation"
        reason = "long-run absolute or relative infection amplitude is small"
    elif n_peaks < 3:
        label = "near equilibrium / small oscillation" if relative_amplitude < 0.20 else "irregular or complex oscillation"
        reason = "too few recurrent peaks for a stable cycle diagnosis"
    elif highly_irregular:
        label = "irregular or complex oscillation"
        reason = "peak intervals or peak heights vary strongly"
    elif finite_interval and 0.8 <= interval <= 1.2 and regular_intervals:
        label = "annual cycle"
        reason = "mean peak interval is close to one year with low interval variability"
    elif finite_interval and 1.6 <= interval <= 2.4 and regular_intervals:
        label = "biennial cycle"
        reason = "mean peak interval is close to two years with low interval variability"
    elif finite_interval and interval > 2.4 and regular_intervals:
        label = "multi-year cycle"
        reason = "regular recurrent peaks occur on a multi-year timescale"
    else:
        label = "irregular or complex oscillation"
        reason = "peak intervals do not match a stable annual, biennial, or multi-year rule"

    short_labels = {
        "near equilibrium / small oscillation": "small oscillation",
        "annual cycle": "annual",
        "biennial cycle": "biennial",
        "multi-year cycle": "multi-year",
        "irregular or complex oscillation": "irregular / complex",
    }

    return {
        "alpha": float(alpha),
        "pattern": label,
        "regime_label": short_labels.get(label, label),
        "rule_reason": reason,
        "poincare_level_count": int(poincare_levels),
        "poincare_range": poincare_range,
        "long_i_min": long_min,
        "long_i_max": long_max,
        "long_i_mean": long_mean,
        "long_i_range": long_range,
        "relative_amplitude": relative_amplitude,
        "n_peaks": n_peaks,
        "mean_peak_interval": interval,
        "peak_height_cv": height_cv,
        "peak_interval_cv": interval_cv,
    }
