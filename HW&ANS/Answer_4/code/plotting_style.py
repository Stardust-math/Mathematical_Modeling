"""Unified nature-like plotting style for the SIR project."""

from __future__ import annotations

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

SIR_COLORS = {
    "s": "#4C9F70",
    "i": "#C44E52",
    "r": "#4C72B0",
}

MODEL_COLORS = {
    "basic": "#4C72B0",
    "demographic": "#C44E52",
    "beta": "#6C757D",
    "dark": "#2F3437",
    "light_gray": "#D9DDE3",
}

SCAN_PALETTE = [
    "#4C72B0",
    "#55A868",
    "#8172B2",
    "#CCB974",
    "#64B5CD",
    "#DD8452",
    "#C44E52",
]

PATTERN_COLORS = {
    "near equilibrium / small oscillation": "#B8C0C7",
    "annual cycle": "#66C2A5",
    "biennial cycle": "#FC8D62",
    "multi-year cycle": "#8DA0CB",
    "irregular or complex oscillation": "#A65628",
}

PATTERN_SHORT_LABELS = {
    "near equilibrium / small oscillation": "small oscillation",
    "annual cycle": "annual",
    "biennial cycle": "biennial",
    "multi-year cycle": "multi-year",
    "irregular or complex oscillation": "irregular / complex",
}

CONTINUOUS_CMAP = "cividis"
FONT_FAMILY = ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"]

FONT_SIZE = 9.5
TITLE_SIZE = 11.0
LABEL_SIZE = 10.0
TICK_SIZE = 9.0
LEGEND_SIZE = 8.6
LINE_WIDTH = 1.8
THIN_LINE_WIDTH = 1.2
MARKER_SIZE = 4.0
AXIS_LINE_WIDTH = 0.8
GRID_LINE_WIDTH = 0.6
PNG_DPI = 300


def set_plot_style() -> None:
    """Apply a restrained scientific plotting style."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.size": FONT_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "axes.linewidth": AXIS_LINE_WIDTH,
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
            "svg.fonttype": "none",
            "savefig.dpi": PNG_DPI,
            "figure.dpi": 120,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.color": "#D9DDE3",
            "grid.linewidth": GRID_LINE_WIDTH,
            "grid.alpha": 0.65,
            "legend.frameon": False,
            "legend.borderaxespad": 0.4,
            "legend.handlelength": 2.0,
            "legend.handletextpad": 0.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
        }
    )


def apply_axis_style(ax, grid: bool = True) -> None:
    """Apply consistent axis styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(AXIS_LINE_WIDTH)
    ax.spines["bottom"].set_linewidth(AXIS_LINE_WIDTH)
    ax.grid(True, which="major", axis="both", zorder=0) if grid else ax.grid(False)


def create_figure(width: float = 6.8, height: float = 4.2):
    """Create a one-panel figure."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(width, height))
    apply_axis_style(ax)
    return fig, ax


def create_two_panel_figure(
    width: float = 7.2,
    height: float = 5.2,
    sharex: bool = True,
    height_ratios: tuple[float, float] = (1.0, 1.2),
):
    """Create a vertical two-panel figure."""
    set_plot_style()
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(width, height),
        sharex=sharex,
        gridspec_kw={"height_ratios": height_ratios},
    )
    for ax in axes:
        apply_axis_style(ax)
    return fig, axes


def create_horizontal_two_panel_figure(width: float = 8.4, height: float = 3.9):
    """Create a horizontal two-panel figure."""
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(width, height))
    for ax in axes:
        apply_axis_style(ax)
    return fig, axes


def finalize_and_save(fig, path_stem) -> None:
    """Save figure to editable SVG and 300 dpi PNG."""
    fig.tight_layout()
    fig.savefig(f"{path_stem}.svg", bbox_inches="tight")
    fig.savefig(f"{path_stem}.png", dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)
