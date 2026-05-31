from __future__ import annotations

from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import path_in_project
from .dynamic_stock_model import PublicGoodDynamicModel
from .game_model import FreeRidingGame
from .metrics import gini, nash_social_metrics, summarize_trajectory
from .policy_mechanisms import get_policy, policy_cost
from .social_optimum import SocialPlanner
from .solvers import one_step_equilibrium_summary, simulate_policy_trajectory


_PAPER_FIGURE_ALIASES_ROBUSTNESS = {
    'synthetic_behavioral_weight_robustness': 'fig19_behavioral_weight_robustness',
    'synthetic_solver_damping_robustness': 'fig20_solver_damping_robustness',
    'synthetic_behavioral_weight_oat_underprovision': 'fig21_behavioral_weight_oat_underprovision',
    'synthetic_behavioral_weight_oat_welfare_loss': 'fig22_behavioral_weight_oat_welfare_loss',
}


def _save_robustness_figure(fig, base_name: str, dpi: int = 300) -> None:
    out = path_in_project('figs/synthetic')
    out.mkdir(parents=True, exist_ok=True)
    paper_out = path_in_project('figs/paper')
    paper_out.mkdir(parents=True, exist_ok=True)
    targets = [out / base_name]
    alias = _PAPER_FIGURE_ALIASES_ROBUSTNESS.get(base_name)
    if alias is not None:
        targets.append(paper_out / alias)
    for target in targets:
        fig.savefig(str(target) + '.pdf', bbox_inches='tight', pad_inches=0.022)
        fig.savefig(str(target) + '.png', dpi=dpi, bbox_inches='tight', pad_inches=0.022)
        fig.savefig(str(target) + '.svg', bbox_inches='tight', pad_inches=0.022)
    plt.close(fig)


def _scale_response_weights(base_weights: Dict[str, float], multiplier: float) -> Dict[str, float]:
    return {k: float(v) * float(multiplier) for k, v in base_weights.items()}


def _one_step_nash_summary_only(
    agents: pd.DataFrame,
    scenario_params: Dict,
    policy: Dict[str, float],
    config: Dict,
    scenario: str,
) -> dict:
    """Compute only the Nash-style one-step row.

    Behavioral-response robustness perturbs only the Nash-style response weights;
    the stage-wise social-planner benchmark is invariant to those weights. This
    helper avoids repeatedly solving the same L-BFGS-B social-planner problem
    inside robustness loops while keeping the reported Nash row identical to the
    implementation used elsewhere.
    """
    contribution_max = float(config['contribution_max'])
    eff_array = agents['efficiency'].to_numpy(float)
    model = PublicGoodDynamicModel(scenario_params, q_max=float(np.sum(eff_array * contribution_max)))
    game = FreeRidingGame(
        agents,
        scenario_params,
        contribution_max,
        config['free_rider_threshold'],
        config['high_benefit_quantile'],
    )
    planner = SocialPlanner(
        agents,
        scenario_params,
        contribution_max,
        config['free_rider_threshold'],
        config['high_benefit_quantile'],
    )
    G, H, D = scenario_params['G0'], scenario_params['H0'], scenario_params['D0']
    effort = game.solve_nash_equilibrium(G, H, D, policy)
    free = game.classify_free_riders(effort)
    Q = float(np.dot(eff_array, effort))
    G_next, H_next, D_next, qn = model.step_state(G, H, D, Q, policy, rng=None)
    pcost = policy_cost(policy, effort, free)
    return {
        'scenario': scenario,
        'policy': 'baseline',
        'mode': 'nash',
        'G': G_next,
        'H': H_next,
        'D': D_next,
        'Q': Q,
        'Q_norm': qn,
        'avg_effort': float(np.mean(effort)),
        'effort_gini': gini(effort),
        'free_riding_ratio': float(np.mean(free)),
        'welfare': planner.social_welfare(effort, G_next, H_next, pcost),
        'policy_cost': pcost,
    }


def _benchmark_validity(nash_row: dict, social_row: dict) -> dict:
    warning = ''
    valid = True
    if social_row['welfare'] + 1e-8 < nash_row['welfare']:
        valid = False
        warning += 'W_SO < W_NE; '
    if social_row['Q'] + 1e-8 < nash_row['Q']:
        valid = False
        warning += 'Q_SO < Q_NE; '
    return {
        'social_optimum_valid': valid,
        'optimizer_method': 'scipy.optimize.minimize (L-BFGS-B); social row reused because behavioral weights affect Nash response only',
        'warning': warning.strip(),
    }


def run_behavioral_response_robustness(
    agents: pd.DataFrame,
    base_params: Dict,
    config: Dict,
    scenario: str,
    multipliers: Iterable[float],
    policies: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stress-test synthetic Nash-style behavioral response weights.

    The first returned table checks the one-step individual-rational benchmark
    against the social-planner benchmark under common multipliers of the five
    behavioral weights. The second table checks whether the long-run policy
    ordering in the critical scenario is qualitatively stable under the same
    perturbations.
    """
    base_weights = dict(config.get('nash_response_weights', {}))
    if not base_weights:
        base_weights = {
            'stock_benefit': 0.90,
            'pressure_relief': 0.85,
            'penalty_base': 0.35,
            'penalty_gap': 2.80,
            'threshold_push': 0.10,
        }

    baseline_policy = get_policy(config, 'baseline')
    _, social_reference_row, _ = one_step_equilibrium_summary(agents, base_params, baseline_policy, config, scenario)

    one_step_rows = []
    policy_rows = []
    for gamma in multipliers:
        print(f'[synthetic] Behavioral joint-weight multiplier: {float(gamma):.3f}', flush=True)
        params = dict(base_params)
        params['nash_response_weights'] = _scale_response_weights(base_weights, float(gamma))
        params['nash_update_damping_raw_weight'] = float(config.get('nash_update_damping_raw_weight', 0.42))

        nrow = _one_step_nash_summary_only(agents, params, baseline_policy, config, scenario)
        row = nash_social_metrics(nrow, social_reference_row)
        row.update(_benchmark_validity(nrow, social_reference_row))
        row.update({
            'scenario': scenario,
            'weight_multiplier': float(gamma),
            'weights_scaled': 'all five Nash-style behavioral weights',
        })
        one_step_rows.append(row)

        for pname in policies:
            policy = get_policy(config, pname)
            df = simulate_policy_trajectory(
                agents,
                params,
                policy,
                config,
                scenario,
                pname,
                mode='nash',
                rng=np.random.default_rng(int(config['random_seed']) + 202),
            )
            s = summarize_trajectory(df)
            s.update({'scenario': scenario, 'policy': pname, 'weight_multiplier': float(gamma)})
            policy_rows.append(s)

    one_step = pd.DataFrame(one_step_rows)
    policy_summary = pd.DataFrame(policy_rows)
    if not policy_summary.empty:
        policy_summary['welfare_rank_within_multiplier'] = policy_summary.groupby('weight_multiplier')['avg_welfare_tail'].rank(
            ascending=False,
            method='min',
        )
    return one_step, policy_summary


def run_solver_damping_robustness(
    agents: pd.DataFrame,
    base_params: Dict,
    config: Dict,
    scenario: str,
    damping_weights: Iterable[float],
) -> pd.DataFrame:
    """Check the damped marginal-response solver under alternative damping rates."""
    rows = []
    policy = get_policy(config, 'baseline')
    for w in damping_weights:
        params = dict(base_params)
        params['nash_response_weights'] = dict(config.get('nash_response_weights', {}))
        params['nash_update_damping_raw_weight'] = float(w)
        game = FreeRidingGame(
            agents,
            params,
            config['contribution_max'],
            config['free_rider_threshold'],
            config['high_benefit_quantile'],
        )
        diag = game.nash_update_diagnostics(params['G0'], params['H0'], params['D0'], policy)
        nrow, srow, check = one_step_equilibrium_summary(agents, params, policy, config, scenario)
        ns = nash_social_metrics(nrow, srow)
        rows.append({
            'scenario': scenario,
            'damping_raw_weight': float(w),
            'damped_sweeps': diag['damped_sweeps'],
            'converged': diag['converged'],
            'max_raw_residual': diag['max_residual'],
            'max_damped_residual': diag['max_damped_residual'],
            'Q_IR': diag['Q_IR'],
            'mean_effort': diag['mean_effort'],
            'Q_SO': ns['Q_SO'],
            'under_provision_ratio': ns['under_provision_ratio'],
            'welfare_loss_ratio': ns['welfare_loss_ratio'],
            'social_optimum_valid': check['social_optimum_valid'],
        })
    return pd.DataFrame(rows)


_WEIGHT_AUDIT_ROWS = [
    {
        'weight_key': 'stock_benefit',
        'symbol': 'theta_G',
        'model_channel': 'perceived marginal stock-benefit response',
        'calibration_role': 'Keeps the private perceived stock-benefit channel on the same effort-response scale as cost and pressure channels.',
    },
    {
        'weight_key': 'pressure_relief',
        'symbol': 'theta_H',
        'model_channel': 'perceived marginal maintenance-pressure relief',
        'calibration_role': 'Keeps the pressure-relief channel comparable to the stock-benefit channel without treating it as an empirical estimate.',
    },
    {
        'weight_key': 'penalty_base',
        'symbol': 'xi_0',
        'model_channel': 'base targeted deterrence for high-benefit low-effort agents',
        'calibration_role': 'Sets the baseline upward push generated by targeted penalty incentives.',
    },
    {
        'weight_key': 'penalty_gap',
        'symbol': 'xi_1',
        'model_channel': 'extra penalty response when effort is far below the threshold',
        'calibration_role': 'Controls how strongly the marginal response increases with the threshold gap.',
    },
    {
        'weight_key': 'threshold_push',
        'symbol': 'xi_L',
        'model_channel': 'mild generic push for below-threshold agents',
        'calibration_role': 'Prevents threshold governance from acting only on high-benefit agents while keeping this channel intentionally weak.',
    },
]


def build_nash_weight_scale_audit(config: Dict, multipliers: Iterable[float]) -> pd.DataFrame:
    """Document the synthetic calibration status of each behavioral-response weight.

    The returned table is intentionally descriptive: it separates a modeling
    calibration choice from empirical estimation and records the perturbation
    range used by the robustness experiment. This CSV is used by the paper as a
    parameter-audit layer, not as an additional empirical data source.
    """
    base_weights = dict(config.get('nash_response_weights', {})) or {
        'stock_benefit': 0.90,
        'pressure_relief': 0.85,
        'penalty_base': 0.35,
        'penalty_gap': 2.80,
        'threshold_push': 0.10,
    }
    multipliers = [float(x) for x in multipliers]
    rows = []
    for spec in _WEIGHT_AUDIT_ROWS:
        key = spec['weight_key']
        base_value = float(base_weights[key])
        rows.append({
            **spec,
            'baseline_value': base_value,
            'source_type': 'synthetic calibration / behavioral-response scale setting',
            'empirical_estimate': False,
            'one_at_a_time_multiplier_min': min(multipliers),
            'one_at_a_time_multiplier_max': max(multipliers),
            'tested_value_min': base_value * min(multipliers),
            'tested_value_max': base_value * max(multipliers),
            'interpretation_boundary': 'Values are not fitted to observations; robustness checks test whether the under-provision conclusion depends on the baseline calibration.',
        })
    return pd.DataFrame(rows)


def run_one_at_a_time_behavioral_weight_robustness(
    agents: pd.DataFrame,
    base_params: Dict,
    config: Dict,
    scenario: str,
    multipliers: Iterable[float],
) -> pd.DataFrame:
    """Perturb each Nash-style behavioral weight separately.

    Joint scaling checks whether the overall behavioral responsiveness matters.
    This one-at-a-time experiment is stricter: only one weight is changed in
    each run, while all other behavioral-response weights stay at the baseline.
    The social-planner benchmark is intentionally left unchanged.
    """
    base_weights = dict(config.get('nash_response_weights', {})) or {
        'stock_benefit': 0.90,
        'pressure_relief': 0.85,
        'penalty_base': 0.35,
        'penalty_gap': 2.80,
        'threshold_push': 0.10,
    }
    multipliers = [float(x) for x in multipliers]
    baseline_policy = get_policy(config, 'baseline')
    _, social_reference_row, _ = one_step_equilibrium_summary(agents, base_params, baseline_policy, config, scenario)
    rows = []
    for weight_key in base_weights:
        print(f'[synthetic] One-at-a-time behavioral weight: {weight_key}', flush=True)
        for multiplier in multipliers:
            weights = dict(base_weights)
            weights[weight_key] = float(base_weights[weight_key]) * multiplier
            params = dict(base_params)
            params['nash_response_weights'] = weights
            params['nash_update_damping_raw_weight'] = float(config.get('nash_update_damping_raw_weight', 0.42))
            nrow = _one_step_nash_summary_only(agents, params, baseline_policy, config, scenario)
            row = nash_social_metrics(nrow, social_reference_row)
            row.update(_benchmark_validity(nrow, social_reference_row))
            row.update({
                'scenario': scenario,
                'perturbed_weight': weight_key,
                'multiplier': multiplier,
                'baseline_weight_value': float(base_weights[weight_key]),
                'tested_weight_value': float(weights[weight_key]),
                'all_other_weights': 'held at baseline values',
            })
            for key, val in weights.items():
                row[f'weight_{key}'] = float(val)
            rows.append(row)
    df = pd.DataFrame(rows)
    preferred_cols = [
        'scenario', 'perturbed_weight', 'multiplier', 'baseline_weight_value', 'tested_weight_value',
        'Q_NE', 'Q_SO', 'free_riding_gap', 'under_provision_ratio', 'welfare_loss_ratio',
        'free_riding_ratio_NE', 'maintenance_pressure_index_NE', 'social_optimum_valid',
    ]
    remaining = [c for c in df.columns if c not in preferred_cols]
    return df[preferred_cols + remaining]


def plot_behavioral_robustness(one_step: pd.DataFrame, policy_summary: pd.DataFrame) -> None:
    """Create compact figures for the added robustness appendix/paper section."""
    out = path_in_project('figs/synthetic')
    out.mkdir(parents=True, exist_ok=True)

    if one_step is not None and not one_step.empty:
        fig, ax1 = plt.subplots(figsize=(6.8, 4.2))
        x = one_step['weight_multiplier'].to_numpy(float)
        ax1.plot(x, one_step['under_provision_ratio'].to_numpy(float), marker='o', label='Under-provision ratio')
        ax1.set_xlabel('Behavioral-weight multiplier')
        ax1.set_ylabel('Under-provision ratio')
        ax1.grid(True, alpha=0.25)
        ax2 = ax1.twinx()
        ax2.plot(x, one_step['welfare_loss_ratio'].to_numpy(float), marker='s', linestyle='--', label='Welfare-loss ratio')
        ax2.set_ylabel('Welfare-loss ratio')
        lines = ax1.get_lines() + ax2.get_lines()
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='best', frameon=True)
        fig.tight_layout()
        _save_robustness_figure(fig, 'synthetic_behavioral_weight_robustness')

    if policy_summary is not None and not policy_summary.empty:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for pname, grp in policy_summary.sort_values('weight_multiplier').groupby('policy'):
            ax.plot(grp['weight_multiplier'], grp['avg_welfare_tail'], marker='o', label=pname.replace('_', ' '))
        ax.set_xlabel('Behavioral-weight multiplier')
        ax.set_ylabel('Tail-average welfare')
        ax.grid(True, alpha=0.25)
        ax.legend(loc='best', frameon=True)
        fig.tight_layout()
        _save_robustness_figure(fig, 'synthetic_policy_ranking_under_behavioral_weights')


def plot_one_at_a_time_behavioral_weight_robustness(oat: pd.DataFrame) -> None:
    """Plot the one-at-a-time behavioral-weight robustness check."""
    out = path_in_project('figs/synthetic')
    out.mkdir(parents=True, exist_ok=True)
    if oat is None or oat.empty:
        return

    label_map = {
        'stock_benefit': 'stock benefit',
        'pressure_relief': 'pressure relief',
        'penalty_base': 'penalty base',
        'penalty_gap': 'penalty gap',
        'threshold_push': 'threshold push',
    }

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for key, grp in oat.sort_values('multiplier').groupby('perturbed_weight'):
        ax.plot(
            grp['multiplier'],
            grp['under_provision_ratio'],
            marker='o',
            label=label_map.get(key, key.replace('_', ' ')),
        )
    ax.set_xlabel('One-at-a-time multiplier')
    ax.set_ylabel('Under-provision ratio')
    ax.set_title('One-at-a-Time Behavioral-Weight Robustness')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', frameon=True, fontsize=8)
    fig.tight_layout()
    _save_robustness_figure(fig, 'synthetic_behavioral_weight_oat_underprovision')

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for key, grp in oat.sort_values('multiplier').groupby('perturbed_weight'):
        ax.plot(
            grp['multiplier'],
            grp['welfare_loss_ratio'],
            marker='s',
            label=label_map.get(key, key.replace('_', ' ')),
        )
    ax.set_xlabel('One-at-a-time multiplier')
    ax.set_ylabel('Welfare-loss ratio')
    ax.set_title('Welfare-Loss Robustness to Individual Weight Perturbations')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', frameon=True, fontsize=8)
    fig.tight_layout()
    _save_robustness_figure(fig, 'synthetic_behavioral_weight_oat_welfare_loss')


def plot_damping_robustness(damping: pd.DataFrame) -> None:
    out = path_in_project('figs/synthetic')
    out.mkdir(parents=True, exist_ok=True)
    if damping is None or damping.empty:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(damping['damping_raw_weight'], damping['max_raw_residual'], marker='o', label='Raw residual')
    ax.plot(damping['damping_raw_weight'], damping['max_damped_residual'], marker='s', linestyle='--', label='Damped residual')
    ax.set_xlabel('Raw best-response weight in damped update')
    ax.set_ylabel('Maximum one-step residual')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', frameon=True)
    fig.tight_layout()
    _save_robustness_figure(fig, 'synthetic_solver_damping_robustness')
