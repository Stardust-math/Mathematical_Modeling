"""Plotting functions for deterministic and stochastic SIR experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg
from config import FIG_DIR
from plotting_style import (
    CONTINUOUS_CMAP,
    MODEL_COLORS,
    PATTERN_COLORS,
    PATTERN_SHORT_LABELS,
    SCAN_PALETTE,
    SIR_COLORS,
    THIN_LINE_WIDTH,
    apply_axis_style,
    create_figure,
    create_horizontal_two_panel_figure,
    create_two_panel_figure,
    finalize_and_save,
    set_plot_style,
)


DAYS = cfg.DAYS_PER_YEAR


def _save(fig, filename_stem: str, fig_dir: Path = FIG_DIR) -> None:
    """Save figure as SVG and PNG."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    finalize_and_save(fig, fig_dir / filename_stem)


def _positive_for_log(values: np.ndarray, floor: float = 1.0e-10) -> np.ndarray:
    """Clip non-positive values only for log-scale visualization."""
    return np.clip(np.asarray(values, dtype=float), floor, None)


def _mask_below_log_floor(values: np.ndarray, floor: float = 1.0e-10) -> np.ndarray:
    """Mask values too small for meaningful log-scale display."""
    arr = np.asarray(values, dtype=float)
    return np.where(arr > float(floor), arr, np.nan)


def _three_panel_horizontal(width: float = 10.8, height: float = 3.6):
    """Create a clean horizontal three-panel figure."""
    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(width, height))
    for ax in axes:
        apply_axis_style(ax)
    return fig, axes


def plot_basic_sir_triplet(df: pd.DataFrame, title: str, filename_stem: str) -> None:
    """Figure 1: basic SIR curves with full days and early zoom panels."""
    fig, axes = create_horizontal_two_panel_figure(9.2, 3.8)
    ax_full, ax_zoom = axes
    for ax in axes:
        t_days = df["t"].to_numpy(dtype=float) * DAYS
        ax.plot(t_days, df["s"], color=SIR_COLORS["s"], label="s(t): susceptible")
        ax.plot(t_days, df["i"], color=SIR_COLORS["i"], label="i(t): infected")
        ax.plot(t_days, df["r"], color=SIR_COLORS["r"], label="r(t): removed")
        ax.set_xlabel("Time (days)")
        ax.set_ylim(-0.02, 1.02)
    ax_full.set_title("Full 1-year scale")
    ax_full.set_ylabel("Population proportion")
    ax_full.set_xlim(0.0, float(df["t"].max()) * DAYS)
    ax_zoom.set_title("Early-stage zoom")
    ax_zoom.set_xlim(0.0, min(cfg.BASIC_EARLY_ZOOM_DAYS, float(df["t"].max()) * DAYS))
    ax_full.legend(loc="upper center", bbox_to_anchor=(1.08, 1.22), ncol=3)
    fig.suptitle(title, y=1.03)
    _save(fig, filename_stem)


def plot_R0_comparison(df: pd.DataFrame, title: str, filename_stem: str) -> None:
    """Figure 2: compare i(t) under different R0 values with days-scale zoom."""
    fig, axes = create_horizontal_two_panel_figure(9.1, 3.8)
    ax_full, ax_zoom = axes
    for idx, (R0, g) in enumerate(df.groupby("R0", sort=True)):
        t_days = g["t"].to_numpy(dtype=float) * DAYS
        color = SCAN_PALETTE[idx % len(SCAN_PALETTE)]
        ax_full.plot(t_days, g["i"], color=color, label=rf"$R_0={R0:g}$")
        ax_zoom.plot(t_days, g["i"], color=color, label=rf"$R_0={R0:g}$")
    ax_full.set_xlabel("Time (days)")
    ax_full.set_ylabel("Infected proportion i(t)")
    ax_full.set_title("Full 1-year scale")
    ax_full.set_xlim(0.0, float(df["t"].max()) * DAYS)
    ax_full.set_ylim(bottom=0.0)
    ax_zoom.set_xlabel("Time (days)")
    ax_zoom.set_title("Early-stage zoom")
    ax_zoom.set_xlim(0.0, min(cfg.BASIC_EARLY_ZOOM_DAYS, float(df["t"].max()) * DAYS))
    ax_zoom.set_ylim(bottom=0.0)
    ax_zoom.legend(loc="upper right")
    fig.suptitle(title, y=1.03)
    _save(fig, filename_stem)


def plot_gamma_comparison(df: pd.DataFrame, title: str, filename_stem: str) -> None:
    """Figure 2b: compare basic SIR infection curves under different gamma values."""
    fig, ax = create_figure(7.2, 4.1)
    for idx, (gamma, g) in enumerate(df.groupby("gamma", sort=True)):
        days = g["t"].to_numpy(dtype=float) * DAYS
        mean_days = DAYS / float(gamma)
        label = rf"$\gamma={gamma:g}$/year; mean={mean_days:.1f} days"
        ax.plot(days, g["i"], color=SCAN_PALETTE[idx % len(SCAN_PALETTE)], label=label)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Infected proportion i(t)")
    ax.set_title(title)
    ax.set_xlim(0.0, min(float(df["t"].max()) * DAYS, 160.0))
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper right")
    _save(fig, filename_stem)


def plot_threshold_s0(df: pd.DataFrame, R0: float, filename_stem: str) -> None:
    """Figure 3: threshold experiment max i(t) versus s0."""
    fig, ax = create_figure(7.0, 4.2)
    x = df["s0"].to_numpy(dtype=float)
    y = df["peak_i"].to_numpy(dtype=float)
    threshold = 1.0 / float(R0)
    above = x > threshold
    below = ~above
    ax.scatter(x[below], y[below], color="#8C8C8C", s=20, label=r"$s_0 \leq 1/R_0$")
    ax.scatter(x[above], y[above], color=SIR_COLORS["i"], s=20, label=r"$s_0 > 1/R_0$")
    ax.plot(x, y, color="#444444", linewidth=THIN_LINE_WIDTH, alpha=0.9)
    ax.axvline(threshold, color=MODEL_COLORS["beta"], linestyle="--", linewidth=1.2, label=rf"$s_0=1/R_0={threshold:.3f}$")
    ax.set_xlabel(r"Initial susceptible proportion $s_0$")
    ax.set_ylabel(r"Peak infection $\max_t i(t)$")
    ax.set_title(rf"Outbreak threshold experiment ($R_0={R0:g}$)")
    ax.set_xlim(np.min(x) - 0.01, np.max(x) + 0.01)
    ax.legend(loc="upper left")
    _save(fig, filename_stem)


def plot_demographic_sir(df: pd.DataFrame, title: str, filename_stem: str) -> None:
    """Figure 4: long-term demographic SIR with late recurrent peaks visible."""
    set_plot_style()
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 7.1), sharex=False, gridspec_kw={"height_ratios": (1.0, 1.0, 1.0)})
    for ax in axes:
        apply_axis_style(ax)
    ax_sr, ax_i_full, ax_i_late = axes
    ax_sr.plot(df["t"], df["s"], color=SIR_COLORS["s"], label="s(t): susceptible")
    ax_sr.plot(df["t"], df["r"], color=SIR_COLORS["r"], label="r(t): removed")
    ax_sr.set_ylabel("s(t), r(t)")
    ax_sr.set_title("Susceptible and removed populations")
    ax_sr.set_xlim(df["t"].min(), df["t"].max())
    ax_sr.legend(loc="upper right")

    ax_i_full.plot(df["t"], _positive_for_log(df["i"]), color=SIR_COLORS["i"], label="i(t): infected")
    ax_i_full.set_yscale("log")
    ax_i_full.set_ylabel("i(t), log scale")
    ax_i_full.set_title("Infected population: initial outbreak and damped recurrence")
    ax_i_full.set_xlim(df["t"].min(), df["t"].max())
    ax_i_full.legend(loc="upper right")

    late = df[df["t"] >= cfg.DEMOGRAPHIC_LATE_START]
    ax_i_late.plot(late["t"], late["i"], color=SIR_COLORS["i"])
    ax_i_late.set_xlabel("Time (years)")
    ax_i_late.set_ylabel("i(t), late stage")
    ax_i_late.set_title(rf"Late-stage zoom ($t \geq {cfg.DEMOGRAPHIC_LATE_START:g}$ year)")
    ax_i_late.set_xlim(late["t"].min(), late["t"].max())
    ax_i_late.set_ylim(bottom=0.0)
    fig.suptitle(title, y=1.01)
    _save(fig, filename_stem)


def plot_basic_vs_demographic(df: pd.DataFrame, title: str, filename_stem: str) -> None:
    """Figure 5: early and late comparison of basic and demographic SIR."""
    fig, axes = create_horizontal_two_panel_figure(9.0, 3.8)
    ax_early, ax_late = axes
    early_cut = cfg.BASIC_EARLY_ZOOM_DAYS / DAYS
    for model, g in df.groupby("model", sort=False):
        color = MODEL_COLORS["basic"] if model == "basic_sir" else MODEL_COLORS["demographic"]
        label = "Basic SIR" if model == "basic_sir" else "Demographic SIR"
        early = g[g["t"] <= early_cut]
        late = g[g["t"] >= cfg.DEMOGRAPHIC_LATE_START]
        ax_early.plot(early["t"] * DAYS, early["i"], color=color, label=label)
        ax_late.plot(late["t"], _positive_for_log(late["i"]), color=color, label=label)
    ax_early.set_xlabel("Time (days)")
    ax_early.set_ylabel("Infected proportion i(t)")
    ax_early.set_title("Early outbreak comparison")
    ax_early.set_xlim(0.0, cfg.BASIC_EARLY_ZOOM_DAYS)
    ax_early.set_ylim(bottom=0.0)
    ax_early.legend(loc="upper right")
    ax_late.set_xlabel("Time (years)")
    ax_late.set_ylabel("i(t), log scale")
    ax_late.set_title(rf"Late-stage comparison ($t \geq {cfg.DEMOGRAPHIC_LATE_START:g}$ year)")
    ax_late.set_yscale("log")
    fig.suptitle(title, y=1.03)
    _save(fig, filename_stem)


def plot_demographic_R0_comparison(df: pd.DataFrame, title: str, filename_stem: str) -> None:
    """Figure 4b: late recurrent peaks for different demographic-SIR R0 values."""
    fig, ax = create_figure(7.4, 4.2)
    for idx, (R0, g) in enumerate(df.groupby("R0", sort=True)):
        late = g[g["t"] >= cfg.DEMOGRAPHIC_LATE_START]
        ax.plot(late["t"], _mask_below_log_floor(late["i"]), color=SCAN_PALETTE[idx % len(SCAN_PALETTE)], label=rf"$R_0={R0:g}$")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Infected proportion i(t), log scale")
    ax.set_title(title)
    ax.set_yscale("log")
    ax.set_xlim(cfg.DEMOGRAPHIC_LATE_START, df["t"].max())
    ax.legend(loc="upper right")
    _save(fig, filename_stem)


def plot_seasonal_alpha_curves(df: pd.DataFrame, title: str, filename_stem: str) -> None:
    """Figure 6: long-run seasonal infection curves."""
    fig, ax = create_figure(8.2, 4.5)
    for idx, (alpha, g) in enumerate(df.groupby("alpha", sort=True)):
        ax.plot(g["t"], g["i"], color=SCAN_PALETTE[idx % len(SCAN_PALETTE)], label=rf"$\alpha={alpha:.2f}$")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Infected proportion i(t)")
    ax.set_title(title)
    ax.set_xlim(df["t"].min(), df["t"].max())
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    _save(fig, filename_stem)


def plot_beta_infection_phase(t: np.ndarray, beta_t: np.ndarray, i_t: np.ndarray, alpha: float, filename_stem: str) -> None:
    """Figure 7: two-panel beta(t) and i(t) plot."""
    fig, axes = create_two_panel_figure(7.1, 5.3, sharex=True)
    ax1, ax2 = axes
    ax1.plot(t, beta_t, color=MODEL_COLORS["beta"], label=r"$\beta(t)$")
    ax1.set_ylabel(r"Transmission rate $\beta(t)$")
    ax1.set_title(rf"Seasonal forcing and infection response ($\alpha={alpha:.2f}$)")
    ax1.legend(loc="upper right")
    ax2.plot(t, i_t, color=SIR_COLORS["i"], label=r"$i(t)$")
    ax2.set_xlabel("Time (years)")
    ax2.set_ylabel("Infected proportion")
    ax2.legend(loc="upper right")
    _save(fig, filename_stem)


def plot_alpha_bifurcation(poincare_df: pd.DataFrame, filename_stem: str) -> None:
    """Figure 8: alpha bifurcation/Poincare sampling plot."""
    fig, ax = create_figure(7.2, 4.4)
    ax.scatter(poincare_df["alpha"], poincare_df["i_sample"], s=5, alpha=0.42, color=MODEL_COLORS["dark"], linewidths=0)
    ax.set_xlabel(r"Seasonal forcing strength $\alpha$")
    ax.set_ylabel(r"Annual same-phase samples of $i(t)$")
    ax.set_title(r"Poincare-style alpha scan for seasonal SIR")
    ax.set_xlim(poincare_df["alpha"].min() - 0.01, poincare_df["alpha"].max() + 0.01)
    ax.set_ylim(bottom=0.0)
    _save(fig, filename_stem)


def plot_peak_statistics(peak_summary_df: pd.DataFrame, filename_stem: str) -> None:
    """Figure 9: three-panel peak statistics across alpha."""
    fig, axes = _three_panel_horizontal(10.6, 3.5)
    data = peak_summary_df.sort_values("alpha")
    ax1, ax2, ax3 = axes
    ax1.plot(data["alpha"], data["mean_peak_i"], color=SIR_COLORS["i"], marker="o", markersize=3.2, linewidth=1.4)
    ax1.set_xlabel(r"$\alpha$")
    ax1.set_ylabel("Mean peak i")
    ax1.set_title("Mean peak height")
    ax1.set_ylim(bottom=0.0)

    ax2.plot(data["alpha"], data["mean_peak_interval"], color=MODEL_COLORS["basic"], marker="o", markersize=3.2, linewidth=1.4)
    ax2.set_xlabel(r"$\alpha$")
    ax2.set_ylabel("Mean interval (years)")
    ax2.set_title("Mean peak interval")
    ax2.set_ylim(bottom=0.0)

    cv_col = "peak_interval_cv" if "peak_interval_cv" in data.columns else "peak_height_cv"
    ax3.plot(data["alpha"], data[cv_col], color=MODEL_COLORS["demographic"], marker="s", markersize=3.0, linewidth=1.3)
    ax3.set_xlabel(r"$\alpha$")
    ax3.set_ylabel("Coefficient of variation")
    ax3.set_title("Peak-interval CV" if cv_col == "peak_interval_cv" else "Peak-height CV")
    ax3.set_ylim(bottom=0.0)
    _save(fig, filename_stem)


def plot_pattern_classification(classification_df: pd.DataFrame, filename_stem: str) -> None:
    """Figure 10: compact regime map for seasonal dynamic types."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.4, 2.25))
    apply_axis_style(ax, grid=False)

    data = classification_df.sort_values("alpha").reset_index(drop=True)
    alphas = data["alpha"].to_numpy(dtype=float)
    if len(alphas) == 1:
        width = 0.04
        left_edges = np.array([alphas[0] - width / 2.0])
        right_edges = np.array([alphas[0] + width / 2.0])
    else:
        midpoints = 0.5 * (alphas[1:] + alphas[:-1])
        left_edges = np.r_[alphas[0] - (midpoints[0] - alphas[0]), midpoints]
        right_edges = np.r_[midpoints, alphas[-1] + (alphas[-1] - midpoints[-1])]

    strip_y0 = 0.30
    strip_height = 0.40
    for left, right, (_, row) in zip(left_edges, right_edges, data.iterrows()):
        regime = row["pattern"]
        color = PATTERN_COLORS.get(regime, "#999999")
        ax.add_patch(
            mpatches.Rectangle(
                (left, strip_y0),
                right - left,
                strip_height,
                facecolor=color,
                edgecolor="white",
                linewidth=0.45,
            )
        )

    ax.set_xlim(left_edges[0], right_edges[-1])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([strip_y0 + strip_height / 2.0])
    ax.set_yticklabels(["regime"])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(r"Seasonal forcing strength $\alpha$", labelpad=8)

    if len(alphas) > 6:
        ax.set_xticks(np.linspace(float(alphas.min()), float(alphas.max()), 5))

    present = [p for p in PATTERN_COLORS if p in set(data["pattern"])]
    handles = [
        mpatches.Patch(color=PATTERN_COLORS[p], label=PATTERN_SHORT_LABELS.get(p, p))
        for p in present
    ]

    fig.suptitle("Seasonal-forcing regime map", y=0.98)

    if handles:
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.875),
            ncol=min(5, len(handles)),
            frameon=False,
            columnspacing=1.15,
            handlelength=1.40,
            handletextpad=0.50,
            borderaxespad=0.2,
        )

    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.25,
        top=0.62,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_stem = FIG_DIR / filename_stem
    fig.savefig(f"{out_stem}.svg", bbox_inches="tight")
    fig.savefig(f"{out_stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_stochastic_trajectories(stochastic_df: pd.DataFrame, deterministic_df: pd.DataFrame, filename_stem: str) -> None:
    """Figure 11: stochastic trajectories and deterministic reference, early and long-term panels."""
    fig, axes = create_horizontal_two_panel_figure(9.2, 3.9)
    ax_early, ax_long = axes
    early_cut = cfg.STOCHASTIC_TRAJECTORY_EARLY_DAYS / DAYS
    for _, g in stochastic_df.groupby("replicate", sort=True):
        ax_early.plot(g["t"] * DAYS, g["i_prop"], color=SIR_COLORS["i"], alpha=0.20, linewidth=0.7, zorder=1)
        ax_long.plot(g["t"], g["i_prop"], color=SIR_COLORS["i"], alpha=0.16, linewidth=0.7, zorder=1)
    det_early = deterministic_df[deterministic_df["t"] <= early_cut]
    ax_early.plot(det_early["t"] * DAYS, det_early["i"], color=MODEL_COLORS["dark"], linewidth=2.2, label="Deterministic", zorder=3)
    ax_long.plot(deterministic_df["t"], deterministic_df["i"], color=MODEL_COLORS["dark"], linewidth=2.2, label="Deterministic", zorder=3)
    ax_early.plot([], [], color=SIR_COLORS["i"], alpha=0.35, linewidth=1.0, label="Stochastic paths")
    ax_early.set_xlabel("Time (days)")
    ax_early.set_ylabel("Infected proportion")
    ax_early.set_title("Early outbreak")
    ax_early.set_xlim(0.0, min(cfg.STOCHASTIC_TRAJECTORY_EARLY_DAYS, stochastic_df["t"].max() * DAYS))
    ax_early.set_ylim(bottom=0.0)
    ax_early.legend(loc="upper right")
    ax_long.set_xlabel("Time (years)")
    ax_long.set_ylabel("Infected proportion")
    ax_long.set_title("Longer-term extinction and recurrence")
    ax_long.set_xlim(stochastic_df["t"].min(), stochastic_df["t"].max())
    ax_long.set_ylim(bottom=0.0)
    fig.suptitle("Stochastic trajectories vs deterministic demographic SIR", y=1.03)
    _save(fig, filename_stem)


def plot_extinction_probability_heatmap(summary_df: pd.DataFrame, filename_stem: str) -> None:
    """Figure 12: early fade-out probability heatmap over N and I0."""
    value_col = "early_fadeout_probability" if "early_fadeout_probability" in summary_df.columns else "extinction_probability"
    pivot = summary_df.pivot(index="I0", columns="N", values=value_col).sort_index()
    fig, ax = create_figure(6.8, 4.4)
    im = ax.imshow(pivot.to_numpy(dtype=float), origin="lower", aspect="auto", cmap=CONTINUOUS_CMAP, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(int(v)) for v in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(int(v)) for v in pivot.index])
    ax.set_xlabel("Population size N")
    ax.set_ylabel("Initial infected count I0")
    ax.set_title("Failure of establishment across N and I0")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Early fade-out probability")
    for row_idx, I0 in enumerate(pivot.index):
        for col_idx, N in enumerate(pivot.columns):
            value = pivot.loc[I0, N]
            if pd.notna(value):
                ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white" if value > 0.55 else "black")
    _save(fig, filename_stem)


def plot_extinction_time_distribution(detail_df: pd.DataFrame, filename_stem: str) -> None:
    """Figure 13: boxplot of extinction times among fade-out runs."""
    if "early_fadeout" in detail_df.columns:
        fadeout_df = detail_df[(detail_df["early_fadeout"] == True) & detail_df["extinction_time"].notna()].copy()
    else:
        fadeout_df = detail_df[(detail_df["extinct"] == True) & detail_df["extinction_time"].notna()].copy()
    fig, ax = create_figure(7.0, 4.2)
    groups, labels = [], []
    for N, g in fadeout_df.groupby("N", sort=True):
        values = g["extinction_time"].dropna().to_numpy(dtype=float)
        if values.size:
            groups.append(values)
            labels.append(str(int(N)))
    if groups:
        box = ax.boxplot(
            groups,
            tick_labels=labels,
            patch_artist=True,
            showfliers=False,
            widths=0.55,
            medianprops={"color": MODEL_COLORS["dark"], "linewidth": 1.3},
            whiskerprops={"color": MODEL_COLORS["dark"], "linewidth": 1.0},
            capprops={"color": MODEL_COLORS["dark"], "linewidth": 1.0},
        )
        for patch in box["boxes"]:
            patch.set_facecolor("#D8E2EC")
            patch.set_edgecolor(MODEL_COLORS["dark"])
            patch.set_linewidth(1.0)
    else:
        ax.text(0.5, 0.5, "No fade-out extinctions recorded", transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("Population size N")
    ax.set_ylabel("Time to I=0 among fade-out runs (years)")
    ax.set_title("Extinction time after failure of establishment")
    ax.set_ylim(bottom=0.0)
    _save(fig, filename_stem)


def plot_alpha_extinction_probability(alpha_df: pd.DataFrame, filename_stem: str) -> None:
    """Figure 14: early fade-out probability under seasonal forcing strengths."""
    fig, ax = create_figure(6.8, 4.1)
    ordered = alpha_df.sort_values("alpha")
    x = ordered["alpha"].to_numpy(dtype=float)
    y = ordered["early_fadeout_probability"].to_numpy(dtype=float)
    if "early_fadeout_se" in ordered.columns:
        yerr = 1.96 * ordered["early_fadeout_se"].to_numpy(dtype=float)
        ax.errorbar(x, y, yerr=yerr, color=SIR_COLORS["i"], marker="o", markersize=4.2, linewidth=1.7, capsize=3)
    else:
        ax.plot(x, y, color=SIR_COLORS["i"], marker="o", markersize=4.2, linewidth=1.7)
    ax.set_xlabel(r"Seasonal forcing strength $\alpha$")
    ax.set_ylabel("Early fade-out probability")
    ax.set_title("Seasonal forcing and establishment risk (phase = 0)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(x.min() - 0.04, x.max() + 0.04)
    for xi, yi in zip(x, y):
        ax.text(xi, min(yi + 0.055, 0.98), f"{yi:.2f}", ha="center", va="bottom", fontsize=8)
    _save(fig, filename_stem)


def plot_phase_sensitivity(phase_df: pd.DataFrame, filename_stem: str) -> None:
    """Figure 14b: phase sensitivity of stochastic establishment risk."""
    fig, ax = create_figure(6.9, 4.1)
    ordered = phase_df.sort_values(["phase", "alpha"])
    for idx, (phase, g) in enumerate(ordered.groupby("phase", sort=True)):
        label = g["phase_label"].iloc[0] if "phase_label" in g.columns else rf"phase={phase:.2f}"
        ax.plot(
            g["alpha"],
            g["early_fadeout_probability"],
            color=SCAN_PALETTE[idx % len(SCAN_PALETTE)],
            marker="o",
            markersize=4.0,
            linewidth=1.55,
            label=label,
        )
    ax.set_xlabel(r"Seasonal forcing strength $\alpha$")
    ax.set_ylabel("Early fade-out probability")
    ax.set_title("Sensitivity to the initial seasonal phase")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="upper right")
    _save(fig, filename_stem)