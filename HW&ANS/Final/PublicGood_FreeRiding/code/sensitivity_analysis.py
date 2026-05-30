from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict
from .solvers import simulate_policy_trajectory
from .policy_mechanisms import get_policy
from .metrics import summarize_trajectory


def run_1d_sensitivity(agents: pd.DataFrame, base_params: Dict, config: Dict, scenario: str) -> pd.DataFrame:
    rows = []
    base_policy = get_policy(config, 'baseline')
    grid_points = int(config['sensitivity']['grid_points'])
    for param in config['sensitivity']['parameters']:
        if param in ['subsidy', 'penalty', 'reputation', 'matching']:
            values = np.linspace(0.0, 0.6, grid_points)
        else:
            base_val = float(base_params.get(param, 1.0))
            values = np.linspace(max(0.001, 0.6 * base_val), 1.4 * base_val, grid_points)
        for value in values:
            params = dict(base_params)
            policy = dict(base_policy)
            if param == 'cost_scale':
                mod_agents = agents.copy(); mod_agents['cost'] = mod_agents['cost'] * (value / max(float(base_params.get('cost_scale', 1.0)), 1e-9))
            elif param == 'benefit_scale':
                mod_agents = agents.copy(); mod_agents['benefit'] = mod_agents['benefit'] * (value / max(float(base_params.get('benefit_scale', 1.0)), 1e-9))
            elif param in policy:
                mod_agents = agents
                policy[param] = float(value)
            else:
                mod_agents = agents
                params[param] = float(value)
            df = simulate_policy_trajectory(mod_agents, params, policy, config, scenario, f'sensitivity_{param}', rng=np.random.default_rng(config['random_seed'] + 7))
            summary = summarize_trajectory(df)
            summary.update({'parameter': param, 'value': float(value)})
            rows.append(summary)
    return pd.DataFrame(rows)


def _paired_policy_noise_seed(config: Dict, scenario: str) -> int:
    """Return the deterministic noise seed used for paired policy comparisons.

    Policy grids should differ only in the policy levers being scanned. Using the
    same scenario-level exogenous noise path as the main policy-comparison table
    makes the grid origin (s=0, p=0) exactly comparable with the no-policy
    baseline when all other levers remain at zero.
    """
    scenarios = list(config.get('scenarios', []))
    scenario_idx = scenarios.index(scenario) if scenario in scenarios else 0
    return int(config['random_seed'] + 1_000 + scenario_idx)


def _pure_subsidy_penalty_policy(config: Dict, subsidy: float, penalty: float) -> Dict[str, float]:
    """Create a pure two-lever policy from the no-policy baseline.

    Only subsidy and penalty are changed. Matching, reputation, budget support,
    backlog reduction, and the explicit policy threshold are left at the baseline
    value. The Nash-style solver still uses the model's free-rider diagnostic
    threshold internally, but no additional threshold-governance or pressure-relief
    channel is introduced by this grid.
    """
    policy = dict(get_policy(config, 'baseline'))
    policy['subsidy'] = float(subsidy)
    policy['penalty'] = float(penalty)
    return policy


def run_subsidy_penalty_heatmap(agents: pd.DataFrame, params: Dict, config: Dict, scenario: str) -> pd.DataFrame:
    """Run a pure subsidy-penalty grid for two-dimensional sensitivity heatmaps."""
    rows = []
    values = np.linspace(0.0, 0.6, int(config['sensitivity']['grid_points']))
    noise_seed = _paired_policy_noise_seed(config, scenario)
    for _idx_s, s in enumerate(values):
        for _idx_p, p in enumerate(values):
            _grid_idx = _idx_s * len(values) + _idx_p + 1
            if _grid_idx == 1 or _grid_idx % 25 == 0 or _grid_idx == len(values) * len(values):
                print(f'[synthetic] 2D subsidy-penalty grid progress: {_grid_idx}/{len(values) * len(values)}', flush=True)
            policy = _pure_subsidy_penalty_policy(config, s, p)
            df = simulate_policy_trajectory(
                agents,
                params,
                policy,
                config,
                scenario,
                'subsidy_penalty_grid',
                rng=np.random.default_rng(noise_seed),
            )
            summary = summarize_trajectory(df)
            summary.update({
                'subsidy': float(s),
                'penalty': float(p),
                'threshold': float(policy.get('threshold', 0.0)),
                'backlog_reduction': float(policy.get('backlog_reduction', 0.0)),
                'policy_template': 'pure_subsidy_penalty_from_baseline',
            })
            rows.append(summary)
    return pd.DataFrame(rows)


def run_policy_response_surface_3d(agents: pd.DataFrame, params: Dict, config: Dict, scenario: str) -> pd.DataFrame:
    """Generate a finer pure subsidy-penalty grid for the 3D response surface.

    This function intentionally writes no files by itself. The caller is responsible
    for saving the returned DataFrame with a synthetic_ prefix. Its mechanism is
    kept consistent with run_subsidy_penalty_heatmap: subsidy and penalty are the
    only scanned policy levers, so the surface is a refined visualization of the
    same two-lever sensitivity experiment rather than a mixed threshold/backlog
    governance template.
    """
    rows = []
    viz_cfg = config.get('visualization_3d', {})
    grid_points = int(viz_cfg.get('response_grid_points', max(17, int(config['sensitivity']['grid_points']))))
    values = np.linspace(0.0, 0.6, grid_points)
    noise_seed = _paired_policy_noise_seed(config, scenario)
    for _idx_s, s in enumerate(values):
        for _idx_p, p in enumerate(values):
            _grid_idx = _idx_s * len(values) + _idx_p + 1
            if _grid_idx == 1 or _grid_idx % 25 == 0 or _grid_idx == len(values) * len(values):
                print(f'[synthetic] 3D response grid progress: {_grid_idx}/{len(values) * len(values)}', flush=True)
            policy = _pure_subsidy_penalty_policy(config, s, p)
            df = simulate_policy_trajectory(
                agents,
                params,
                policy,
                config,
                scenario,
                'subsidy_penalty_response_3d',
                rng=np.random.default_rng(noise_seed),
            )
            summary = summarize_trajectory(df)
            summary.update({
                'subsidy': float(s),
                'penalty': float(p),
                'threshold': float(policy.get('threshold', 0.0)),
                'backlog_reduction': float(policy.get('backlog_reduction', 0.0)),
                'policy_template': 'pure_subsidy_penalty_from_baseline',
            })
            rows.append(summary)
    return pd.DataFrame(rows)
