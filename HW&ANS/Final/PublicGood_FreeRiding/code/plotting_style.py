from __future__ import annotations
import matplotlib as mpl


def set_paper_style(font_family: str = 'DejaVu Sans') -> None:
    mpl.rcParams.update({
        'font.family': font_family,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.dpi': 120,
        'savefig.dpi': 300,
        'svg.fonttype': 'none',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.22,
        'lines.linewidth': 1.8,
    })
