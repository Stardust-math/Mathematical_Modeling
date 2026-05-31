from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # registers 3D projection

from .config import path_in_project
from .plotting_style import set_paper_style


POLICY_COLORS = {
    'baseline': '#4C78A8',
    'reputation': '#54A24B',
    'combined_portfolio': '#E45756',
    'penalty': '#F58518',
    'subsidy': '#72B7B2',
    'matching_fund': '#B279A2',
    'threshold_governance': '#9D755D',
}


def _policy_label(name: str) -> str:
    return name.replace('_', ' ').title()


_PAPER_FIGURE_ALIASES_3D = {
    'figs/synthetic/fig_dynamic_phase_3d': 'figs/paper/fig12_dynamic_phase_3d',
    'figs/synthetic/fig_policy_response_surface_3d': 'figs/paper/fig13_policy_response_surface_3d',
    'figs/synthetic/fig_pareto_front_3d': 'figs/paper/fig14_pareto_front_3d',
}


def _save_3d(fig, relpath_no_ext: str, dpi: int = 450) -> None:
    targets = [relpath_no_ext]
    alias = _PAPER_FIGURE_ALIASES_3D.get(relpath_no_ext)
    if alias is not None and alias not in targets:
        targets.append(alias)

    # 3D axes are more likely than 2D plots to lose z-axis labels after LaTeX
    # scaling or tight cropping. Keep a stable canvas with explicit margins
    # rather than relying on tight_layout.
    fig.subplots_adjust(left=0.11, right=0.84, bottom=0.10, top=0.88)
    for target in targets:
        out = path_in_project(target)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out) + '.pdf', bbox_inches=None, pad_inches=0.0)
        fig.savefig(str(out) + '.svg', bbox_inches=None, pad_inches=0.0)
        fig.savefig(str(out) + '.png', dpi=dpi, bbox_inches=None, pad_inches=0.0)
    plt.close(fig)


def _set_3d_zlabel(fig, ax, label: str) -> None:
    """Place a readable z-axis label inside the figure canvas.

    Matplotlib's default 3D z-label placement is often pushed outside the
    saved bounding box, especially after the figure is inserted into LaTeX.
    A small figure-level label keeps the exported image self-contained while
    preserving the 3D axis ticks.
    """
    ax.set_zlabel('')
    fig.text(0.055, 0.48, label, rotation=90, va='center', ha='center', fontsize=10)


def _surface_arrays(df: pd.DataFrame, z_col: str, color_col: str):
    z_pivot = df.pivot_table(index='penalty', columns='subsidy', values=z_col, aggfunc='mean').sort_index().sort_index(axis=1)
    c_pivot = df.pivot_table(index='penalty', columns='subsidy', values=color_col, aggfunc='mean').reindex(index=z_pivot.index, columns=z_pivot.columns)
    X, Y = np.meshgrid(z_pivot.columns.to_numpy(float), z_pivot.index.to_numpy(float))
    Z = z_pivot.to_numpy(float)
    C = c_pivot.to_numpy(float)
    return X, Y, Z, C


def plot_policy_response_surface_3d(df: pd.DataFrame, relpath: str = 'figs/synthetic/fig_policy_response_surface_3d') -> None:
    """Plot pure subsidy-penalty response surface with welfare as height and average maintenance pressure as color."""
    if df is None or len(df) == 0:
        return
    required = {'subsidy', 'penalty', 'avg_welfare_tail', 'avg_H_tail'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'Policy response surface data missing columns: {sorted(missing)}')

    set_paper_style()
    X, Y, Z, C = _surface_arrays(df, 'avg_welfare_tail', 'avg_H_tail')
    norm = Normalize(vmin=float(np.nanmin(C)), vmax=float(np.nanmax(C)))
    cmap = cm.get_cmap('cividis')
    facecolors = cmap(norm(C))

    fig = plt.figure(figsize=(9.6, 7.2))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(
        X, Y, Z,
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        linewidth=0.25,
        edgecolor='0.65',
        antialiased=True,
        shade=False,
        alpha=0.96,
    )
    best_idx = int(np.nanargmax(df['avg_welfare_tail'].to_numpy(float)))
    best = df.iloc[best_idx]
    ax.scatter(
        [best['subsidy']], [best['penalty']], [best['avg_welfare_tail']],
        s=70,
        c='#D62728',
        edgecolor='black',
        linewidth=0.7,
        depthshade=False,
        label='Highest welfare grid point',
    )
    ax.text(
        float(best['subsidy']),
        float(best['penalty']),
        float(best['avg_welfare_tail']) + 0.6,
        f"s={best['subsidy']:.2f}, p={best['penalty']:.2f}",
        fontsize=8,
        ha='center',
    )

    ax.set_xlabel('Subsidy intensity s', labelpad=12)
    ax.set_ylabel('Penalty intensity p', labelpad=12)
    _set_3d_zlabel(fig, ax, 'Average tail welfare')
    ax.set_title('Pure Subsidy–Penalty Response Surface', pad=14, fontsize=12, fontweight='bold')
    ax.view_init(elev=26, azim=-132)
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.96), frameon=False)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(C)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.68, pad=0.10)
    cbar.set_label('Average maintenance pressure')
    _save_3d(fig, relpath)


def plot_pareto_front_3d(df: pd.DataFrame, relpath: str = 'figs/synthetic/fig_pareto_front_3d') -> None:
    """Plot a 3D projection of the full six-indicator Pareto dominance calculation."""
    if df is None or len(df) == 0:
        return
    required = {'avg_policy_cost_tail', 'avg_welfare_tail', 'avg_H_tail', 'avg_effort_gini_tail', 'is_pareto'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'Pareto data missing columns: {sorted(missing)}')

    set_paper_style()
    data = df.copy()
    pareto = data[data['is_pareto'].astype(bool)]
    dominated = data[~data['is_pareto'].astype(bool)]

    fig = plt.figure(figsize=(9.6, 7.2))
    ax = fig.add_subplot(111, projection='3d')
    norm = Normalize(vmin=float(data['avg_effort_gini_tail'].min()), vmax=float(data['avg_effort_gini_tail'].max()))
    cmap = cm.get_cmap('cividis')

    sc = ax.scatter(
        dominated['avg_policy_cost_tail'],
        dominated['avg_welfare_tail'],
        dominated['avg_H_tail'],
        c=dominated['avg_effort_gini_tail'],
        cmap=cmap,
        norm=norm,
        s=26,
        alpha=0.46,
        depthshade=False,
        label='Dominated sampled portfolio',
    )
    ax.scatter(
        pareto['avg_policy_cost_tail'],
        pareto['avg_welfare_tail'],
        pareto['avg_H_tail'],
        c=pareto['avg_effort_gini_tail'],
        cmap=cmap,
        norm=norm,
        s=58,
        edgecolor='black',
        linewidth=0.75,
        alpha=0.98,
        depthshade=False,
        label='Pareto-efficient under six metrics',
    )
    ax.set_xlabel('Average policy cost', labelpad=12)
    ax.set_ylabel('Average welfare', labelpad=12)
    _set_3d_zlabel(fig, ax, 'Average maintenance pressure')
    ax.set_title('3D Projection of the Pareto Policy Trade-off', pad=14, fontsize=12, fontweight='bold')
    ax.view_init(elev=24, azim=-122)
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=False)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.68, pad=0.10)
    cbar.set_label('Average effort Gini')
    _save_3d(fig, relpath)


def plot_dynamic_phase_3d(df: pd.DataFrame, relpath: str = 'figs/synthetic/fig_dynamic_phase_3d') -> None:
    """Plot stock-pressure-welfare trajectories for selected policies in the critical scenario."""
    if df is None or len(df) == 0:
        return
    required = {'time', 'scenario', 'policy', 'G', 'H', 'welfare'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'Dynamic phase data missing columns: {sorted(missing)}')

    set_paper_style()
    fig = plt.figure(figsize=(9.6, 7.2))
    ax = fig.add_subplot(111, projection='3d')
    policy_order = ['baseline', 'reputation', 'combined_portfolio']
    for policy in policy_order:
        g = df[df['policy'] == policy].sort_values('time')
        if len(g) == 0:
            continue
        color = POLICY_COLORS.get(policy, None)
        ax.plot(g['G'], g['H'], g['welfare'], lw=2.2, color=color, label=_policy_label(policy))
        ax.scatter([g['G'].iloc[0]], [g['H'].iloc[0]], [g['welfare'].iloc[0]], marker='^', s=42, color=color, edgecolor='black', linewidth=0.4, depthshade=False)
        ax.scatter([g['G'].iloc[-1]], [g['H'].iloc[-1]], [g['welfare'].iloc[-1]], marker='o', s=48, color=color, edgecolor='black', linewidth=0.4, depthshade=False)

    ax.set_xlabel('Public-good stock G(t)', labelpad=12)
    ax.set_ylabel('Maintenance pressure H(t)', labelpad=12)
    _set_3d_zlabel(fig, ax, 'Social welfare W(t)')
    ax.set_title('Dynamic Stock–Pressure–Welfare Trajectories', pad=14, fontsize=12, fontweight='bold')
    ax.view_init(elev=23, azim=-136)
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), frameon=False)
    _save_3d(fig, relpath)
