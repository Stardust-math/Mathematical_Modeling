from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from .synthetic_generator import generate_agents
from .policy_mechanisms import get_policy
from .solvers import simulate_policy_trajectory
from .metrics import summarize_trajectory


def jitter_params(base: Dict, rng: np.random.Generator) -> Dict:
    p = dict(base)
    for key in ['delta', 'alpha', 'eta', 'rho', 'lambda', 'kappa', 'demand_growth']:
        scale = 0.18 if key != 'demand_growth' else 0.35
        p[key] = float(max(0.001, base[key] * rng.lognormal(0.0, scale)))
    p['noise'] = float(base.get('noise', 0.006))
    return p


def run_monte_carlo(config: Dict, scenario: str, policies: list[str], collect_trajectories: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    traj_frames = []
    base_params = config['scenario_parameters'][scenario]
    n_runs = int(config['monte_carlo']['num_runs'])
    base_seed = int(config['random_seed'] + config['monte_carlo']['seed_offset'])
    for r in range(n_runs):
        if r == 0 or (r + 1) % max(1, n_runs // 5) == 0 or r == n_runs - 1:
            print(f'[synthetic] Monte Carlo progress: {r + 1}/{n_runs}', flush=True)
        # One setup RNG determines the parameter perturbation and synthetic agents.
        # A separate paired trajectory RNG is re-created for every policy below,
        # so policies in the same Monte Carlo run face the same exogenous demand
        # noise sequence. This makes policy differences cleaner and more auditable.
        rng_setup = np.random.default_rng(base_seed + r)
        params = jitter_params(base_params, rng_setup)
        cfg2 = dict(config)
        cfg2['scenario_parameters'] = {scenario: params}
        cfg2['scenarios'] = [scenario]
        cfg2['time_horizon'] = int(min(config.get('time_horizon', 60), config.get('monte_carlo', {}).get('time_horizon', 30)))
        cfg2['num_agents'] = int(config.get('monte_carlo', {}).get('num_agents', max(20, config.get('num_agents', 50))))
        agents = generate_agents(cfg2, scenario, rng_setup)
        paired_noise_seed = base_seed + 100_000 + r
        for pname in policies:
            policy = get_policy(config, pname)
            rng_traj = np.random.default_rng(paired_noise_seed)
            df = simulate_policy_trajectory(agents, params, policy, cfg2, scenario, pname, rng=rng_traj)
            summary = summarize_trajectory(df)
            summary.update({'run': r})
            rows.append(summary)
            if collect_trajectories and pname in ['baseline', 'reputation', 'combined_portfolio']:
                temp = df[['time', 'scenario', 'policy', 'G', 'H', 'welfare']].copy()
                temp['run'] = r
                traj_frames.append(temp)
    summary_df = pd.DataFrame(rows)
    traj_df = pd.concat(traj_frames, ignore_index=True) if traj_frames else pd.DataFrame(columns=['time', 'scenario', 'policy', 'G', 'H', 'welfare', 'run'])
    return summary_df, traj_df
