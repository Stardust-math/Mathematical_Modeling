from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from .config import path_in_project
from .plotting_style import set_paper_style
from .constants import SCENARIO_ORDER


SCENARIO_COLORS = {
    'small_volunteer': '#4C78A8',
    'rapid_growth': '#F58518',
    'critical_infrastructure': '#54A24B',
    'burnout_prone': '#E45756',
}


def _save(fig, relpath_no_ext: str, dpi: int = 450):
    out = path_in_project(relpath_no_ext)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Avoid global tight_layout and tight bounding-box rendering here: both can
    # become unstable or very slow for polar axes and heavily annotated figures
    # during full-project regeneration. Figure-specific spacing is controlled
    # inside each plotting function.
    fig.savefig(str(out) + '.svg')
    fig.savefig(str(out) + '.png', dpi=dpi)
    plt.close(fig)


def _policy_label(x: str) -> str:
    return x.replace('_', ' ').title()


def _scenario_label(x: str) -> str:
    return x.replace('_', ' ').title()


def plot_model_framework():
    """Create a clean workflow diagram for the paper.

    The figure intentionally contains only the diagram itself. Explanatory
    comments should be written in the LaTeX caption/body rather than inside
    the image, so that the image remains compact and does not overflow.
    """
    set_paper_style()
    fig, ax = plt.subplots(figsize=(11.8, 2.6))
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    xs = [0.025, 0.220, 0.415, 0.610, 0.805]
    box_w, box_h, box_y = 0.170, 0.62, 0.19
    labels = [
        ('Data Layer', 'Synthetic\nscenarios'),
        ('Model I', 'Stock-pressure\nsystem'),
        ('Model II', 'Individual-rational\nresponse'),
        ('Model III', 'Planner benchmark\nand welfare loss'),
        ('Model IV', 'Policy, robustness\nand trade-offs'),
    ]
    colors = ['#d7ecff', '#e7f4e4', '#fff0d6', '#f3e6ff', '#ffe6ea']

    for i, ((title, subtitle), color) in enumerate(zip(labels, colors)):
        x = xs[i]
        rect = Rectangle((x, box_y), box_w, box_h, facecolor=color, edgecolor='black', lw=1.0)
        ax.add_patch(rect)
        ax.text(x + box_w / 2, box_y + 0.40, title,
                ha='center', va='center', fontsize=11.5, fontweight='bold')
        ax.text(x + box_w / 2, box_y + 0.23, subtitle,
                ha='center', va='center', fontsize=8.8, linespacing=1.08)
        if i < len(xs) - 1:
            arr = FancyArrowPatch(
                (x + box_w + 0.006, box_y + box_h / 2),
                (xs[i + 1] - 0.006, box_y + box_h / 2),
                arrowstyle='-|>', mutation_scale=13, lw=1.4, color='black'
            )
            ax.add_patch(arr)

    _save(fig, 'figs/paper/model_framework')


def plot_causal_loop_diagram():
    """Create a cleaner hexagon-style causal-loop diagram.

    The six nodes are arranged around a hexagonal ring so the arrows and sign
    markers have more whitespace and do not crowd the labels. The figure keeps
    only the structural elements needed for the paper: nodes, arrows, and
    positive/negative signs.
    """
    set_paper_style()
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10.6, 6.2))
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    pos = {
        'G': (0.50, 0.84),
        'D': (0.82, 0.66),
        'H': (0.74, 0.28),
        'F': (0.43, 0.16),
        'P': (0.16, 0.28),
        'Q': (0.18, 0.66),
    }
    texts = {
        'Q': 'Contribution\n$Q_t$',
        'G': 'Public-good stock\n$G_t$',
        'D': 'User demand\n$D_t$',
        'H': 'Maintenance pressure\n$H_t$',
        'F': 'Free-riding pressure\n$FR_t$',
        'P': 'Governance\nmechanism',
    }
    colors = {
        'Q': '#e7f4e4', 'G': '#d7ecff', 'D': '#fff0d6',
        'H': '#ffe6ea', 'F': '#f3e6ff', 'P': '#e9e9e9'
    }
    sizes = {
        'G': (0.19, 0.11), 'D': (0.19, 0.11), 'H': (0.20, 0.11),
        'F': (0.20, 0.11), 'P': (0.17, 0.11), 'Q': (0.19, 0.11)
    }

    patches = {}
    for key, (x, y) in pos.items():
        w, h = sizes[key]
        patch = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle='round,pad=0.016,rounding_size=0.018',
            facecolor=colors[key], edgecolor='black', lw=1.0, zorder=3
        )
        ax.add_patch(patch)
        patches[key] = patch
        ax.text(x, y, texts[key], ha='center', va='center', fontsize=9.8,
                linespacing=1.05, zorder=4)

    def arrow(a, b, rad=0.0, sign=None, sign_pos=None):
        arr = FancyArrowPatch(
            posA=pos[a], posB=pos[b], patchA=patches[a], patchB=patches[b],
            connectionstyle=f'arc3,rad={rad}', arrowstyle='-|>',
            mutation_scale=12.5, lw=1.35, color='black',
            shrinkA=7, shrinkB=7, zorder=2
        )
        ax.add_patch(arr)
        if sign is not None and sign_pos is not None:
            ax.text(sign_pos[0], sign_pos[1], sign, fontsize=13,
                    fontweight='bold', ha='center', va='center', zorder=5,
                    bbox=dict(boxstyle='round,pad=0.10', facecolor='white', edgecolor='none'))

    arrow('Q', 'G', rad=0.02, sign='+', sign_pos=(0.33, 0.79))
    arrow('G', 'D', rad=0.02, sign='+', sign_pos=(0.67, 0.79))
    arrow('D', 'H', rad=0.04, sign='+', sign_pos=(0.84, 0.47))
    arrow('H', 'G', rad=-0.10, sign='−', sign_pos=(0.65, 0.55))
    arrow('G', 'F', rad=0.12, sign='−', sign_pos=(0.36, 0.52))
    arrow('F', 'Q', rad=0.08, sign='−', sign_pos=(0.26, 0.42))
    arrow('P', 'Q', rad=0.02, sign='+', sign_pos=(0.10, 0.48))
    arrow('P', 'F', rad=0.00, sign='−', sign_pos=(0.29, 0.22))
    arrow('P', 'H', rad=-0.12, sign='−', sign_pos=(0.47, 0.30))

    _save(fig, 'figs/paper/causal_loop_diagram')


def plot_scenario_dashboard(trajectories: pd.DataFrame):
    set_paper_style()
    baseline = trajectories[trajectories['policy'] == 'baseline'].copy()
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), sharex=True)
    metrics = [
        ('G', 'Stock', 'Public-good stock G(t)'),
        ('H', 'Pressure', 'Maintenance pressure H(t)'),
        ('free_riding_ratio', 'Free-riding', 'Free-riding ratio'),
        ('welfare', 'Welfare', 'Social welfare'),
    ]
    handles, labels = [], []
    for ax, (col, title, ylabel) in zip(axes.ravel(), metrics):
        for scenario in SCENARIO_ORDER:
            g = baseline[baseline['scenario'] == scenario]
            if len(g):
                line, = ax.plot(
                    g['time'],
                    g[col],
                    label=_scenario_label(scenario),
                    color=SCENARIO_COLORS.get(scenario),
                    lw=2.1,
                )
                if title == 'Stock':
                    handles.append(line)
                    labels.append(_scenario_label(scenario))
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(axis='both', labelsize=9)
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=4,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.965),
        fontsize=9,
    )
    fig.suptitle('Scenario Overview under Baseline Governance', fontsize=14, fontweight='bold', y=1.02)
    fig.subplots_adjust(top=0.86, hspace=0.28, wspace=0.24)
    _save(fig, 'figs/synthetic/synthetic_scenario_dashboard')


def plot_contribution_distribution(agents: pd.DataFrame, efforts: pd.DataFrame):
    if efforts is None or len(efforts) == 0:
        return
    set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2))
    axes[0].hist(efforts['effort'], bins=18)
    axes[0].set_title('Baseline Individual Contribution Distribution')
    axes[0].set_xlabel('Effort'); axes[0].set_ylabel('Count')
    fr_share = efforts.groupby('initial_type')['free_rider'].mean().reset_index()
    axes[1].bar(fr_share['initial_type'].str.replace('_', ' '), fr_share['free_rider'])
    axes[1].set_title('Free-riding Share by Agent Type')
    axes[1].set_ylabel('Share classified as free riders')
    axes[1].tick_params(axis='x', rotation=15)
    _save(fig, 'figs/synthetic/synthetic_contribution_distribution')


def plot_free_riding_distribution(trajectories: pd.DataFrame):
    set_paper_style()
    tail = trajectories.groupby(['scenario', 'policy'])['free_riding_ratio'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for scenario in SCENARIO_ORDER:
        g = tail[tail['scenario'] == scenario]
        if len(g):
            ax.plot(g['policy'].str.replace('_', '\n'), g['free_riding_ratio'], marker='o', color=SCENARIO_COLORS.get(scenario), label=_scenario_label(scenario))
    ax.set_title('Free-riding Ratio across Policies and Scenarios')
    ax.set_ylabel('Average free-riding ratio')
    ax.legend(frameon=False, ncol=2)
    _save(fig, 'figs/synthetic/synthetic_free_riding_distribution')


def plot_stock_and_pressure(trajectories: pd.DataFrame):
    set_paper_style()
    baseline = trajectories[trajectories['policy'] == 'baseline'].copy()
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for scenario in SCENARIO_ORDER:
        g = baseline[baseline['scenario'] == scenario]
        if len(g):
            ax.plot(g['time'], g['G'], color=SCENARIO_COLORS.get(scenario), label=_scenario_label(scenario))
    ax.set_title('Public-good Stock Trajectory under Baseline Governance')
    ax.set_xlabel('Time'); ax.set_ylabel('Public-good stock G(t)')
    ax.legend(frameon=False, ncol=2)
    _save(fig, 'figs/synthetic/synthetic_stock_trajectory')

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for scenario in SCENARIO_ORDER:
        g = baseline[baseline['scenario'] == scenario]
        if len(g):
            ax.plot(g['time'], g['H'], color=SCENARIO_COLORS.get(scenario), label=_scenario_label(scenario))
    ax.set_title('Maintenance Pressure Trajectory under Baseline Governance')
    ax.set_xlabel('Time'); ax.set_ylabel('Maintenance pressure H(t)')
    ax.legend(frameon=False, ncol=2)
    _save(fig, 'figs/synthetic/synthetic_maintenance_pressure')


def plot_nash_social_comparison(df: pd.DataFrame):
    set_paper_style()
    x = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
    width = 0.36
    axes[0].bar(x - width / 2, df['Q_NE'], width, label='Nash-style IR')
    axes[0].bar(x + width / 2, df['Q_SO'], width, label='Planner benchmark')
    axes[0].set_xticks(x); axes[0].set_xticklabels([_scenario_label(s) for s in df['scenario']], rotation=15)
    axes[0].set_ylabel('Aggregate contribution Q')
    axes[0].set_title('Contribution: Nash-style IR vs Planner Benchmark')
    axes[0].legend(frameon=False)

    axes[1].bar(x - width / 2, df['W_NE'], width, label='Nash-style IR')
    axes[1].bar(x + width / 2, df['W_SO'], width, label='Planner benchmark')
    axes[1].set_xticks(x); axes[1].set_xticklabels([_scenario_label(s) for s in df['scenario']], rotation=15)
    axes[1].set_ylabel('One-step welfare')
    axes[1].set_title('Welfare: Nash-style IR vs Planner Benchmark')
    axes[1].legend(frameon=False)
    _save(fig, 'figs/synthetic/synthetic_nash_social_comparison')


def plot_nash_social_trajectory(df: pd.DataFrame):
    set_paper_style()
    scenario = 'critical_infrastructure' if 'critical_infrastructure' in df['scenario'].unique() else df['scenario'].iloc[0]
    g = df[df['scenario'] == scenario].copy()
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), sharex=True)
    mapping = [('G', 'Public-good stock G(t)'), ('H', 'Maintenance pressure H(t)'), ('Q', 'Aggregate contribution Q'), ('welfare', 'Social welfare')]
    for ax, (col, title) in zip(axes.ravel(), mapping):
        for mode, label in [('nash', 'Nash-style IR'), ('social', 'Planner benchmark')]:
            temp = g[g['mode'] == mode]
            ax.plot(temp['time'], temp[col], label=label)
        ax.set_title(title)
        ax.set_xlabel('Time'); ax.set_ylabel(title)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(f'Dynamic Comparison between Nash-style IR and Planner Benchmark ({_scenario_label(scenario)})', fontsize=13, fontweight='bold')
    _save(fig, 'figs/synthetic/synthetic_nash_vs_social_trajectory')


def plot_welfare_loss_decomposition(df: pd.DataFrame):
    set_paper_style()
    x = np.arange(len(df))
    width = 0.22
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.bar(x - width, df['free_riding_gap'], width, label='Free-riding gap')
    ax.bar(x, df['under_provision_ratio'], width, label='Under-provision ratio')
    ax.bar(x + width, df['welfare_loss_ratio'], width, label='Welfare loss ratio')
    ax.plot(x, df['maintenance_pressure_index_NE'], marker='o', linestyle='--', label='Maintenance pressure (Nash)')
    ax.set_xticks(x)
    ax.set_xticklabels([_scenario_label(s) for s in df['scenario']], rotation=15)
    ax.set_ylabel('Normalized magnitude')
    ax.set_title('Free-riding Gap and Welfare Loss Decomposition')
    ax.legend(frameon=False, ncol=2)
    _save(fig, 'figs/synthetic/synthetic_welfare_loss_decomposition')


def plot_policy_comparison(summary: pd.DataFrame):
    set_paper_style()
    pol = summary.copy().sort_values('avg_welfare_tail', ascending=False).reset_index(drop=True)
    numeric_cols = ['avg_welfare_tail', 'avg_policy_cost_tail', 'free_riding_reduction_vs_baseline']
    for col in numeric_cols:
        pol[col] = pd.to_numeric(pol[col], errors='coerce')

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    y = np.arange(len(pol))
    axes[0].barh(y, pol['avg_welfare_tail'].to_numpy(float))
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([_policy_label(p) for p in pol['policy']])
    axes[0].invert_yaxis()
    axes[0].set_title('Long-run Welfare across Policies')
    axes[0].set_xlabel('Average tail welfare')

    xvals = pol['avg_policy_cost_tail'].to_numpy(float)
    yvals = pol['free_riding_reduction_vs_baseline'].to_numpy(float)
    axes[1].scatter(xvals, yvals, s=70)
    # Keep only decision-relevant labels in the scatter panel to avoid the
    # repeated label overlap that appeared in earlier paper renders.
    label_set = {'baseline', 'reputation', 'combined_portfolio'}
    for _, row in pol[pol['policy'].isin(label_set)].iterrows():
        axes[1].annotate(
            _policy_label(row['policy']),
            (float(row['avg_policy_cost_tail']), float(row['free_riding_reduction_vs_baseline'])),
            xytext=(5, 5), textcoords='offset points', fontsize=8
        )
    axes[1].set_xlabel('Average tail policy cost')
    axes[1].set_ylabel('Free-riding reduction vs baseline')
    axes[1].set_title('Policy Cost vs. Free-riding Reduction')
    _save(fig, 'figs/synthetic/synthetic_policy_comparison')


def plot_policy_radar(summary: pd.DataFrame):
    set_paper_style()
    metrics = [
        'avg_G_tail',
        'avg_welfare_tail',
        'stability_tail',
        'free_riding_reduction_vs_baseline',
        'pressure_reduction_vs_baseline',
        'avg_effort_gini_tail',
    ]
    labels = ['Stock', 'Welfare', 'Stability', 'Low free-riding', 'Low pressure', 'Fairness']
    data = summary.copy()
    for m in metrics:
        vals = data[m].to_numpy(float)
        mn, mx = np.nanmin(vals), np.nanmax(vals)
        if mx - mn < 1e-9:
            data[m + '_score'] = 0.5
        elif m == 'avg_effort_gini_tail':
            data[m + '_score'] = 1 - (vals - mn) / (mx - mn)
        else:
            data[m + '_score'] = (vals - mn) / (mx - mn)

    top = data.sort_values('avg_welfare_tail', ascending=False).head(5)
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    # Keep the legend inside the saved canvas. The previous version placed the
    # legend outside the axes, which caused the word "Reputation" to be clipped
    # in the paper render because the generic save routine does not use a tight
    # bounding box for all 2D figures.
    fig = plt.figure(figsize=(8.2, 5.8))
    ax = plt.subplot(111, polar=True)
    # Reserve a right-side legend column inside the canvas instead of placing
    # the legend outside the saved bounding box.
    ax.set_position([0.08, 0.13, 0.62, 0.76])

    handles = []
    legend_labels = []
    for _, row in top.iterrows():
        vals = [row[m + '_score'] for m in metrics]
        vals += vals[:1]
        line, = ax.plot(angles, vals, label=_policy_label(row['policy']))
        ax.fill(angles, vals, alpha=0.08)
        handles.append(line)
        legend_labels.append(_policy_label(row['policy']))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title('Scenario-specific Policy Trade-off Radar', pad=18)
    ax.set_ylim(0.0, 1.0)
    fig.legend(
        handles,
        legend_labels,
        loc='center left',
        bbox_to_anchor=(0.72, 0.50),
        ncol=1,
        frameon=False,
        fontsize=8.5,
        handlelength=1.6,
        labelspacing=0.9,
    )
    _save(fig, 'figs/synthetic/synthetic_scenario_radar')


def plot_heatmap(df: pd.DataFrame, xcol: str, ycol: str, zcol: str, title: str, relpath: str, cbar_label: str | None = None):
    set_paper_style()
    piv = df.pivot_table(index=ycol, columns=xcol, values=zcol, aggfunc='mean')
    fig, ax = plt.subplots(figsize=(6.4, 5.1))
    im = ax.imshow(piv.values, origin='lower', aspect='auto')
    step_x = max(1, len(piv.columns) // 6)
    step_y = max(1, len(piv.index) // 6)
    ax.set_xticks(np.arange(len(piv.columns))[::step_x])
    ax.set_xticklabels([f'{v:.2f}' for v in piv.columns[::step_x]])
    ax.set_yticks(np.arange(len(piv.index))[::step_y])
    ax.set_yticklabels([f'{v:.2f}' for v in piv.index[::step_y]])
    ax.set_xlabel(xcol.replace('_', ' ').title()); ax.set_ylabel(ycol.replace('_', ' ').title())
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label or zcol)
    _save(fig, relpath)


def plot_sensitivity_heatmap(sens: pd.DataFrame):
    set_paper_style()
    pivot = sens.pivot_table(index='parameter', values=['final_G', 'avg_H_tail', 'avg_free_riding_tail', 'avg_welfare_tail'], aggfunc=lambda x: x.max() - x.min())
    fig, ax = plt.subplots(figsize=(8.0, 4.9))
    im = ax.imshow(pivot.values, aspect='auto')
    ax.set_xticks(np.arange(pivot.shape[1])); ax.set_xticklabels(pivot.columns, rotation=25, ha='right')
    ax.set_yticks(np.arange(pivot.shape[0])); ax.set_yticklabels(pivot.index)
    ax.set_title('Parameter Sensitivity Heatmap')
    fig.colorbar(im, ax=ax, label='Range of response')
    _save(fig, 'figs/synthetic/synthetic_parameter_sensitivity_heatmap')


def plot_phase_portrait(model, policy, trajectory: pd.DataFrame | None = None, relpath='figs/synthetic/synthetic_phase_portrait'):
    set_paper_style()
    G = np.linspace(0.05, getattr(model, 'G_capacity', 1.4) * 0.95, 14)
    H = np.linspace(0.02, getattr(model, 'H_capacity', 1.4) * 0.95, 14)
    GG, HH = np.meshgrid(G, H)
    D_ref = trajectory['D'].iloc[0] if trajectory is not None and len(trajectory) else 0.55
    Q_ref = trajectory['Q'].mean() if trajectory is not None and len(trajectory) else 10.0
    U, V = model.vector_field(GG, HH, D=float(D_ref), Q=float(Q_ref), policy=policy)
    mag = np.sqrt(U ** 2 + V ** 2) + 1e-9
    U2, V2 = U / mag, V / mag
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.quiver(GG, HH, U2, V2, angles='xy', scale_units='xy', scale=18, width=0.003)
    if trajectory is not None and len(trajectory):
        ax.plot(trajectory['G'], trajectory['H'], color='crimson', lw=2.0, label='Baseline trajectory')
        ax.scatter([trajectory['G'].iloc[-1]], [trajectory['H'].iloc[-1]], color='black', s=25, label='Terminal state')
        ax.legend(frameon=False)
    ax.set_xlabel('Public-good stock G(t)'); ax.set_ylabel('Maintenance pressure H(t)')
    ax.set_title('Phase Portrait of Stock–Pressure Dynamics')
    _save(fig, relpath)


def plot_pareto_front(df: pd.DataFrame):
    set_paper_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.9))
    allp = df[~df['is_pareto']]
    pareto = df[df['is_pareto']]
    sc = ax.scatter(allp['avg_policy_cost_tail'], allp['avg_welfare_tail'], c=allp['avg_free_riding_tail'], s=20, alpha=0.40)
    ax.scatter(pareto['avg_policy_cost_tail'], pareto['avg_welfare_tail'], c='crimson', s=35, label='Pareto-efficient policies')
    top = pareto.sort_values('avg_welfare_tail', ascending=False).head(4)
    for _, row in top.iterrows():
        ax.annotate(f"#{int(row['policy_id'])}", (row['avg_policy_cost_tail'], row['avg_welfare_tail']), fontsize=8)
    ax.set_xlabel('Long-run policy cost'); ax.set_ylabel('Long-run social welfare')
    ax.set_title('Pareto Frontier of Governance Policy Portfolios')
    ax.legend(frameon=False)
    fig.colorbar(sc, ax=ax, label='Average free-riding ratio')
    _save(fig, 'figs/synthetic/synthetic_pareto_front')


def plot_monte_carlo(mc: pd.DataFrame):
    set_paper_style()
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    policies = sorted(mc['policy'].unique())
    data = [mc.loc[mc['policy'] == p, 'avg_welfare_tail'].values for p in policies]
    ax.boxplot(data, labels=[_policy_label(p) for p in policies], showmeans=True)
    ax.set_title('Monte Carlo Robustness of Policy Welfare')
    ax.set_ylabel('Average tail welfare')
    ax.tick_params(axis='x', rotation=20)
    _save(fig, 'figs/synthetic/synthetic_monte_carlo_uncertainty')


def plot_uncertainty_bands(mc_traj: pd.DataFrame):
    if mc_traj is None or len(mc_traj) == 0:
        return
    set_paper_style()
    for metric, relpath, title, ylabel in [
        ('G', 'figs/synthetic/synthetic_monte_carlo_G_uncertainty_band', 'Monte Carlo Uncertainty Band for Public-good Stock', 'Public-good stock G(t)'),
        ('welfare', 'figs/synthetic/synthetic_monte_carlo_welfare_uncertainty_band', 'Monte Carlo Uncertainty Band for Social Welfare', 'Social welfare')
    ]:
        fig, ax = plt.subplots(figsize=(7.8, 4.7))
        for policy in ['baseline', 'reputation', 'combined_portfolio']:
            g = mc_traj[mc_traj['policy'] == policy].groupby('time')[metric].agg(
                mean='mean', q_low=lambda x: x.quantile(0.025), q_high=lambda x: x.quantile(0.975)
            ).reset_index()
            ax.plot(g['time'], g['mean'], label=_policy_label(policy))
            ax.fill_between(g['time'], g['q_low'], g['q_high'], alpha=0.18)
        ax.set_title(title)
        ax.set_xlabel('Time'); ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
        _save(fig, relpath)


def plot_policy_ablation(ablation: pd.DataFrame):
    set_paper_style()
    a = ablation.copy().sort_values('welfare_gain_vs_baseline', ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    axes[0].bar(a['policy'].str.replace('_', '\n'), a['welfare_gain_vs_baseline'])
    axes[0].set_title('Policy Ablation: Welfare Gain vs Baseline')
    axes[0].set_ylabel('Welfare gain')
    axes[1].bar(a['policy'].str.replace('_', '\n'), a['free_riding_reduction_vs_baseline'])
    axes[1].set_title('Policy Ablation: Free-riding Reduction vs Baseline')
    axes[1].set_ylabel('Reduction in free-riding ratio')
    _save(fig, 'figs/synthetic/synthetic_policy_ablation')


def plot_maintenance_ranking(summary: pd.DataFrame):
    set_paper_style()
    s = summary.copy().sort_values('avg_H_tail')
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.barh(s['policy'].str.replace('_', ' '), s['avg_H_tail'])
    ax.set_title('Maintenance Pressure Ranking across Policies')
    ax.set_xlabel('Average tail maintenance pressure')
    _save(fig, 'figs/synthetic/synthetic_maintenance_pressure_ranking')



def write_caption_index():
    rows = [
        ('model_framework', 'paper', 'Overall workflow from synthetic scenario construction to dynamic modeling, equilibrium analysis, policy evaluation, and robustness analysis.'),
        ('causal_loop_diagram', 'paper', 'Feedback structure linking contribution, public-good stock, demand, maintenance pressure, free-riding, and governance.'),
        ('synthetic_scenario_dashboard', 'synthetic', 'Baseline scenario overview showing stock, pressure, free-riding, and welfare dynamics across scenarios.'),
        ('synthetic_stock_trajectory', 'synthetic', 'Baseline public-good stock trajectories across scenarios, illustrating under-provision and divergence in long-run provision quality.'),
        ('synthetic_nash_social_comparison', 'synthetic', 'One-step comparison of aggregate contribution and welfare under Nash-style individual rationality and the stage-wise social-planner benchmark.'),
        ('synthetic_nash_vs_social_trajectory', 'synthetic', 'Dynamic trajectory comparison between Nash-style individual rationality and the stage-wise social-planner benchmark for the critical scenario.'),
        ('synthetic_welfare_loss_decomposition', 'synthetic', 'Scenario-wise decomposition of free-riding gap, under-provision, and welfare loss.'),
        ('synthetic_policy_comparison', 'synthetic', 'Policy comparison using long-run welfare and free-riding reduction relative to baseline.'),
        ('synthetic_policy_ablation', 'synthetic', 'Ablation-style comparison of policy contributions to welfare improvement and free-riding reduction.'),
        ('synthetic_subsidy_penalty_welfare_heatmap', 'synthetic', 'Two-dimensional response surface of long-run welfare under subsidy-penalty combinations.'),
        ('synthetic_subsidy_penalty_freeriding_heatmap', 'synthetic', 'Two-dimensional response surface of long-run free-riding under subsidy-penalty combinations.'),
        ('synthetic_parameter_sensitivity_heatmap', 'synthetic', 'Sensitivity heatmap summarizing how key model responses vary across parameter perturbations.'),
        ('synthetic_phase_portrait', 'synthetic', 'Phase portrait of public-good stock and maintenance pressure with an overlaid baseline trajectory.'),
        ('synthetic_pareto_front', 'synthetic', 'Pareto frontier highlighting trade-offs between policy cost, welfare, and free-riding.'),
        ('synthetic_monte_carlo_G_uncertainty_band', 'synthetic', 'Mean and 95% simulation interval for public-good stock under representative policies.'),
        ('synthetic_monte_carlo_welfare_uncertainty_band', 'synthetic', 'Mean and 95% simulation interval for social welfare under representative policies.'),
        ('synthetic_scenario_radar', 'synthetic', 'Radar-style multi-criteria comparison of representative policies for the critical scenario.'),
        ('fig_policy_response_surface_3d', 'synthetic', '3D subsidy-penalty response surface with welfare as height and average maintenance pressure as color.'),
        ('fig_pareto_front_3d', 'synthetic', '3D projection of the six-indicator Pareto policy trade-off.'),
        ('fig_dynamic_phase_3d', 'synthetic', 'Dynamic stock-pressure-welfare trajectories for baseline, reputation, and combined-portfolio policies.')
    ]
    pd.DataFrame(rows, columns=['figure_name', 'mode', 'paper_use']).to_csv(path_in_project('report_assets/figure_captions.csv'), index=False)
