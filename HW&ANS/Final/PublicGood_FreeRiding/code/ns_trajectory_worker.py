from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow the script to be launched directly by subprocess from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code.config import load_json_config, path_in_project
from code.policy_mechanisms import get_policy
from code.solvers import simulate_policy_trajectory
from code.synthetic_generator import generate_all_agents


def _scenario_params(config: dict, scenario: str) -> dict:
    p = dict(config['scenario_parameters'][scenario])
    p['nash_response_weights'] = dict(config.get('nash_response_weights', {}))
    p['nash_update_max_iter'] = int(config.get('nash_update_max_iter', 60))
    p['nash_update_tol'] = float(config.get('nash_update_tol', 1e-7))
    p['nash_update_damping_raw_weight'] = float(config.get('nash_update_damping_raw_weight', 0.42))
    return p


def main(config_path: str) -> None:
    config = load_json_config(config_path)
    agents_all = generate_all_agents(config)
    frames = []
    for scenario in config['scenarios']:
        agents = agents_all[agents_all['scenario'] == scenario].copy()
        params = _scenario_params(config, scenario)
        policy = get_policy(config, 'baseline')
        for mode in ['nash', 'social']:
            frames.append(
                simulate_policy_trajectory(
                    agents,
                    params,
                    policy,
                    config,
                    scenario,
                    'baseline',
                    mode=mode,
                    rng=np.random.default_rng(config['random_seed'] + 777),
                )
            )
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(path_in_project('data/processed/synthetic/synthetic_nash_social_trajectories.csv'), index=False)
    out.to_csv(path_in_project('results/synthetic/synthetic_nash_social_trajectories.csv'), index=False)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python code/ns_trajectory_worker.py <config_path>')
    main(sys.argv[1])
