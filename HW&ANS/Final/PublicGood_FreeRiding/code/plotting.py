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
    # Use tighter padding for publication figures so the content appears larger
    # once inserted into the paper, while keeping a slightly looser margin for
    # dense result plots.
    pdf_targets = ('figs/paper/', 'synthetic_policy_decision_matrix')
    pad = 0.012 if 'figs/paper/' in relpath_no_ext else 0.022
    if any(key in relpath_no_ext for key in pdf_targets):
        fig.savefig(str(out) + '.pdf', bbox_inches='tight', pad_inches=pad)
    fig.savefig(str(out) + '.svg', bbox_inches='tight', pad_inches=pad)
    fig.savefig(str(out) + '.png', dpi=dpi, bbox_inches='tight', pad_inches=pad)
    plt.close(fig)


def _policy_label(x: str) -> str:
    return x.replace('_', ' ').title()


def _scenario_label(x: str) -> str:
    return x.replace('_', ' ').title()


def plot_model_framework():
    """Create a cleaner DSPF workflow chart for the paper."""
    set_paper_style()
    from matplotlib.patches import FancyBboxPatch, Circle

    fig, ax = plt.subplots(figsize=(12.0, 4.35))
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    palette = {
        'setup': ('#EEF5FF', '#4E79A7'),
        'core': ('#EEF8EE', '#59A14F'),
        'bench': ('#FFF6E8', '#F28E2B'),
        'policy': ('#F7F0FF', '#8E6BBE'),
        'neutral': ('#FAFAFA', '#8C8C8C'),
    }

    cols = [0.028, 0.273, 0.518, 0.763]
    col_w = 0.19
    main_y, main_h = 0.30, 0.56
    blocks = [
        ('1', 'Synthetic setup', 'setup', ['Scenario design', 'Agent heterogeneity', 'Fixed seed']),
        ('2', 'Dynamic system', 'core', ['Contribution rule', 'Stock update', 'Pressure update', 'State feedback']),
        ('3', 'Benchmarks', 'bench', ['Individual rationality', 'Planner solution', 'Welfare gap']),
        ('4', 'Policy analysis', 'policy', ['Instruments', 'Score matrix', 'Pareto rule']),
    ]
    footer = ['Synthetic mode', 'Feasibility', 'Robustness', 'Selection']

    def rounded_box(x, y, w, h, fc, ec, lw=1.0, radius=0.018, z=2):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f'round,pad=0.010,rounding_size={radius}',
            facecolor=fc, edgecolor=ec, lw=lw, zorder=z
        )
        ax.add_patch(patch)
        return patch

    for x, (num, title, key, subitems) in zip(cols, blocks):
        fc, ec = palette[key]
        rounded_box(x, main_y, col_w, main_h, fc, ec, lw=1.15, radius=0.020)

        header_h = 0.098
        rounded_box(x + 0.010, main_y + main_h - header_h - 0.012, col_w - 0.020, header_h,
                    '#FFFFFF', ec, lw=0.75, radius=0.014, z=3)
        badge = Circle((x + 0.030, main_y + main_h - 0.061), 0.016,
                       facecolor=ec, edgecolor=ec, lw=0.8, zorder=4)
        ax.add_patch(badge)
        ax.text(x + 0.030, main_y + main_h - 0.061, num,
                ha='center', va='center', fontsize=8.2, fontweight='bold', color='white', zorder=5)
        ax.text(x + col_w/2 + 0.012, main_y + main_h - 0.061, title,
                ha='center', va='center', fontsize=9.1, fontweight='bold', color='#222222', zorder=5)

        y0 = main_y + main_h - 0.183
        step = 0.102 if len(subitems) <= 3 else 0.082
        box_h = 0.066 if len(subitems) <= 3 else 0.059
        for j, it in enumerate(subitems):
            yy = y0 - j * step
            rounded_box(x + 0.025, yy - box_h/2, col_w - 0.050, box_h,
                        '#FFFFFF', ec, lw=0.60, radius=0.011, z=3)
            ax.text(x + col_w/2, yy, it, ha='center', va='center',
                    fontsize=7.7, color='#222222', zorder=4)

    for i in range(3):
        x1 = cols[i] + col_w + 0.010
        x2 = cols[i+1] - 0.010
        y = main_y + main_h/2 - 0.01
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='-|>', lw=1.25, color='#333333',
                                    shrinkA=0, shrinkB=0, mutation_scale=13))

    foot_y, foot_h = 0.095, 0.105
    rounded_box(0.028, foot_y, 0.925, foot_h, palette['neutral'][0], palette['neutral'][1], lw=0.9, radius=0.015)
    centers = [0.135, 0.377, 0.619, 0.861]
    for cx, label in zip(centers, footer):
        rounded_box(cx - 0.073, foot_y + 0.024, 0.146, 0.045, '#FFFFFF', '#A7A7A7', lw=0.60, radius=0.010, z=3)
        ax.text(cx, foot_y + 0.046, label, ha='center', va='center', fontsize=7.7, color='#333333', zorder=4)
    for i in range(3):
        ax.annotate('', xy=(centers[i+1] - 0.082, foot_y + 0.046),
                    xytext=(centers[i] + 0.082, foot_y + 0.046),
                    arrowprops=dict(arrowstyle='->', lw=0.90, color='#777777', mutation_scale=10))
    for x in [c + col_w/2 for c in cols]:
        ax.annotate('', xy=(x, foot_y + foot_h + 0.003), xytext=(x, main_y - 0.008),
                    arrowprops=dict(arrowstyle='->', lw=0.75, color='#A0A0A0', mutation_scale=8, linestyle='--'))

    _save(fig, 'figs/paper/model_framework')

def plot_causal_loop_diagram():
    """Create a code-consistent hexagon-style causal-loop diagram."""
    set_paper_style()
    from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(10.4, 5.9))
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    nodes = {
        'G': {'center': (0.50, 0.82), 'wh': (0.26, 0.10), 'label': 'Public-good stock\n$G_t$', 'fc': '#EEF5FF', 'ec': '#4E79A7'},
        'D': {'center': (0.78, 0.65), 'wh': (0.18, 0.10), 'label': 'Demand\n$D_t$', 'fc': '#FFF6E8', 'ec': '#F28E2B'},
        'H': {'center': (0.78, 0.33), 'wh': (0.24, 0.10), 'label': 'Maintenance pressure\n$H_t$', 'fc': '#FFF0F0', 'ec': '#E15759'},
        'F': {'center': (0.50, 0.16), 'wh': (0.23, 0.10), 'label': 'Free-riding pressure\n$FR_t$', 'fc': '#F7F0FF', 'ec': '#8E6BBE'},
        'P': {'center': (0.22, 0.33), 'wh': (0.24, 0.10), 'label': 'Governance\nmechanism', 'fc': '#F7F7F7', 'ec': '#777777'},
        'Q': {'center': (0.22, 0.65), 'wh': (0.21, 0.10), 'label': 'Contribution\n$Q_t$', 'fc': '#EEF8EE', 'ec': '#59A14F'},
    }

    def draw_node(key):
        spec = nodes[key]
        cx, cy = spec['center']
        w, h = spec['wh']
        x, y = cx - w / 2, cy - h / 2
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle='round,pad=0.010,rounding_size=0.022',
            facecolor=spec['fc'], edgecolor=spec['ec'], lw=1.35, zorder=5
        )
        ax.add_patch(patch)
        ax.text(cx, cy, spec['label'], ha='center', va='center', fontsize=9.0,
                linespacing=1.02, color='#222222', zorder=6)

    def anchor(key, side):
        cx, cy = nodes[key]['center']
        w, h = nodes[key]['wh']
        x1, x2 = cx - w / 2, cx + w / 2
        y1, y2 = cy - h / 2, cy + h / 2
        one_third_x = w / 6.0
        one_third_y = h / 6.0
        pts = {
            'left': (x1, cy),
            'right': (x2, cy),
            'top': (cx, y2),
            'bottom': (cx, y1),
            'top_left_third': (cx - one_third_x, y2),
            'top_right_third': (cx + one_third_x, y2),
            'bottom_left_third': (cx - one_third_x, y1),
            'bottom_right_third': (cx + one_third_x, y1),
            'left_upper': (x1, cy + one_third_y),
            'left_lower': (x1, cy - one_third_y),
            'right_upper': (x2, cy + one_third_y),
            'right_lower': (x2, cy - one_third_y),
        }
        return pts[side]

    for key in ['G', 'D', 'H', 'F', 'P', 'Q']:
        draw_node(key)

    def draw_arrow(start, end, color, rad=0.0, lw=1.7):
        arrow = FancyArrowPatch(
            start, end,
            arrowstyle='-|>', mutation_scale=16,
            connectionstyle=f'arc3,rad={rad}',
            linewidth=lw, color=color, zorder=3,
            shrinkA=4, shrinkB=5,
        )
        ax.add_patch(arrow)

    def sign_badge(x, y, sign, color):
        badge = Circle((x, y), 0.0155, facecolor='white', edgecolor=color, lw=1.0, zorder=7)
        ax.add_patch(badge)
        ax.text(x, y - 0.001, sign, ha='center', va='center', fontsize=9.8,
                fontweight='bold', color=color, zorder=8)

    def loop_badge(txt, x, y, color):
        tag = FancyBboxPatch(
            (x - 0.026, y - 0.013), 0.052, 0.026,
            boxstyle='round,pad=0.007,rounding_size=0.016',
            facecolor='white', edgecolor=color, lw=0.9, zorder=8
        )
        ax.add_patch(tag)
        ax.text(x, y, txt, ha='center', va='center', fontsize=7.7,
                fontweight='bold', color=color, zorder=9)

    green, blue, orange, red, purple, gray = '#59A14F', '#4E79A7', '#F28E2B', '#E15759', '#8E6BBE', '#777777'

    # Code-consistent directional links.
    draw_arrow(anchor('Q', 'top'), anchor('G', 'left'), green, rad=0.02)
    sign_badge(0.346, 0.742, '+', green)

    draw_arrow(anchor('G', 'right'), anchor('D', 'top'), blue, rad=0.02)
    sign_badge(0.676, 0.742, '+', blue)

    draw_arrow(anchor('D', 'bottom'), anchor('H', 'top'), orange, rad=0.00)
    sign_badge(0.875, 0.490, '+', orange)

    draw_arrow(anchor('H', 'top_left_third'), anchor('G', 'bottom'), red, rad=-0.36)
    sign_badge(0.650, 0.570, '-', red)

    draw_arrow(anchor('Q', 'bottom_right_third'), anchor('F', 'top'), purple, rad=-0.08)
    sign_badge(0.392, 0.402, '-', purple)

    draw_arrow(anchor('P', 'top'), anchor('Q', 'bottom'), gray, rad=0.00)
    sign_badge(0.092, 0.492, '+', gray)

    draw_arrow(anchor('P', 'bottom'), anchor('F', 'left'), gray, rad=-0.12)
    sign_badge(0.318, 0.224, '-', gray)

    draw_arrow(anchor('P', 'right_upper'), anchor('H', 'left_lower'), gray, rad=-0.22)
    sign_badge(0.492, 0.318, '-', gray)

    # Loop labels are placed in open regions, close to their corresponding loops,
    # but not on top of arrow shafts or node borders.
    loop_badge('R1', 0.620, 0.690, blue)
    loop_badge('B1', 0.360, 0.560, purple)
    loop_badge('B2', 0.565, 0.465, red)

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


def plot_policy_decision_matrix(summary: pd.DataFrame):
    """Create a normalized multi-criteria decision matrix for policies."""
    set_paper_style()
    data = summary.copy()
    policy_order = [
        'baseline', 'subsidy', 'penalty', 'reputation', 'matching_fund',
        'threshold_governance', 'combined_portfolio'
    ]
    data['policy'] = pd.Categorical(data['policy'], categories=policy_order, ordered=True)
    data = data.sort_values('policy')

    metrics = [
        ('avg_welfare_tail', 'Welfare', 'higher'),
        ('avg_policy_cost_tail', 'Low cost', 'lower'),
        ('free_riding_reduction_vs_baseline', 'FR reduction', 'higher'),
        ('pressure_reduction_vs_baseline', 'Pressure relief', 'higher'),
        ('avg_effort_gini_tail', 'Fairness', 'lower'),
        ('stability_tail', 'Stability', 'higher'),
    ]
    score = pd.DataFrame(index=[_policy_label(p) for p in data['policy'].astype(str)])
    for col, label, direction in metrics:
        vals = pd.to_numeric(data[col], errors='coerce').to_numpy(float)
        mn, mx = np.nanmin(vals), np.nanmax(vals)
        if mx - mn < 1e-12:
            normalized = np.full_like(vals, 0.5, dtype=float)
        elif direction == 'lower':
            normalized = 1.0 - (vals - mn) / (mx - mn)
        else:
            normalized = (vals - mn) / (mx - mn)
        score[label] = normalized

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    im = ax.imshow(score.values, vmin=0, vmax=1, cmap='YlGnBu', aspect='auto')
    ax.set_xticks(np.arange(score.shape[1]))
    ax.set_xticklabels(score.columns, rotation=25, ha='right', fontsize=9.0)
    ax.set_yticks(np.arange(score.shape[0]))
    ax.set_yticklabels(score.index, fontsize=9.0)

    ax.set_xticks(np.arange(-.5, score.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, score.shape[0], 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.2)
    ax.tick_params(which='minor', bottom=False, left=False)

    for i in range(score.shape[0]):
        for j in range(score.shape[1]):
            val = score.iloc[i, j]
            color = 'white' if val > 0.58 else '#222222'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8.4,
                    color=color, fontweight='bold' if val > 0.75 else 'normal')

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label('Normalized score', fontsize=9)
    fig.subplots_adjust(left=0.22, right=0.91, top=0.96, bottom=0.18)
    _save(fig, 'figs/synthetic/synthetic_policy_decision_matrix')

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
        ('synthetic_subsidy_penalty_welfare_heatmap', 'synthetic', 'Two-dimensional response surface of long-run welfare under pure subsidy-penalty combinations from the no-policy baseline.'),
        ('synthetic_subsidy_penalty_freeriding_heatmap', 'synthetic', 'Two-dimensional response surface of long-run free-riding under pure subsidy-penalty combinations from the no-policy baseline.'),
        ('synthetic_parameter_sensitivity_heatmap', 'synthetic', 'Sensitivity heatmap summarizing how key model responses vary across parameter perturbations.'),
        ('synthetic_phase_portrait', 'synthetic', 'Phase portrait of public-good stock and maintenance pressure with an overlaid baseline trajectory.'),
        ('synthetic_pareto_front', 'synthetic', 'Pareto frontier highlighting trade-offs between policy cost, welfare, and free-riding.'),
        ('synthetic_monte_carlo_G_uncertainty_band', 'synthetic', 'Mean and 95% simulation interval for public-good stock under representative policies.'),
        ('synthetic_monte_carlo_welfare_uncertainty_band', 'synthetic', 'Mean and 95% simulation interval for social welfare under representative policies.'),
        ('synthetic_policy_decision_matrix', 'synthetic', 'Normalized multi-criteria decision matrix comparing representative policies for the critical scenario.'),
        ('fig_policy_response_surface_3d', 'synthetic', '3D pure subsidy-penalty response surface with welfare as height and average maintenance pressure as color.'),
        ('fig_pareto_front_3d', 'synthetic', '3D projection of the six-indicator Pareto policy trade-off.'),
        ('fig_dynamic_phase_3d', 'synthetic', 'Dynamic stock-pressure-welfare trajectories for baseline, reputation, and combined-portfolio policies.')
    ]
    pd.DataFrame(rows, columns=['figure_name', 'mode', 'paper_use']).to_csv(path_in_project('report_assets/figure_captions.csv'), index=False)
