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


def run_subsidy_penalty_heatmap(agents: pd.DataFrame, params: Dict, config: Dict, scenario: str) -> pd.DataFrame:
    rows = []
    values = np.linspace(0.0, 0.6, int(config['sensitivity']['grid_points']))
    for s in values:
        for p in values:
            policy = dict(get_policy(config, 'baseline'))
            policy['subsidy'] = float(s)
            policy['penalty'] = float(p)
            policy['threshold'] = max(config['free_rider_threshold'], 0.14)
            policy['backlog_reduction'] = 0.01 + 0.02 * p
            df = simulate_policy_trajectory(agents, params, policy, config, scenario, 'subsidy_penalty_grid', rng=np.random.default_rng(config['random_seed'] + 11))
            summary = summarize_trajectory(df)
            summary.update({'subsidy': float(s), 'penalty': float(p)})
            rows.append(summary)
    return pd.DataFrame(rows)



def run_policy_response_surface_3d(agents: pd.DataFrame, params: Dict, config: Dict, scenario: str) -> pd.DataFrame:
    """Generate a finer subsidy-penalty grid for the 3D response-surface figure.

    This function intentionally writes no files by itself. The caller is responsible for
    saving the returned DataFrame with a synthetic_ prefix. Its policy mechanism is kept
    consistent with run_subsidy_penalty_heatmap so that the 3D plot is a refined view of
    the same sensitivity experiment rather than a new model.
    """
    rows = []
    viz_cfg = config.get('visualization_3d', {})
    grid_points = int(viz_cfg.get('response_grid_points', max(17, int(config['sensitivity']['grid_points']))))
    values = np.linspace(0.0, 0.6, grid_points)
    for s in values:
        for p in values:
            policy = dict(get_policy(config, 'baseline'))
            policy['subsidy'] = float(s)
            policy['penalty'] = float(p)
            policy['threshold'] = max(config['free_rider_threshold'], 0.14)
            policy['backlog_reduction'] = 0.01 + 0.02 * p
            df = simulate_policy_trajectory(
                agents,
                params,
                policy,
                config,
                scenario,
                'subsidy_penalty_response_3d',
                rng=np.random.default_rng(config['random_seed'] + 311),
            )
            summary = summarize_trajectory(df)
            summary.update({'subsidy': float(s), 'penalty': float(p)})
            rows.append(summary)
    return pd.DataFrame(rows)
