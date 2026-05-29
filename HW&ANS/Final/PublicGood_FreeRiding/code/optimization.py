from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict
from .solvers import simulate_policy_trajectory
from .metrics import summarize_trajectory


def run_random_pareto_search(agents: pd.DataFrame, params: Dict, config: Dict, scenario: str) -> pd.DataFrame:
    rng = np.random.default_rng(config['random_seed'] + 551)
    rows = []
    n = int(config['optimization']['pareto_samples'])
    for i in range(n):
        policy = {
            'subsidy': float(rng.uniform(0, 0.35)),
            'penalty': float(rng.uniform(0, 0.40)),
            'reputation': float(rng.uniform(0, 0.35)),
            'matching': float(rng.uniform(0, 0.35)),
            'budget': float(rng.uniform(0, 0.05)),
            'backlog_reduction': float(rng.uniform(0, 0.07)),
            'threshold': float(rng.uniform(0.05, 0.22))
        }
        # Hold the exogenous demand-noise sequence fixed across sampled policy
        # portfolios, so Pareto dominance reflects policy parameters rather than
        # different random shocks.
        df = simulate_policy_trajectory(agents, params, policy, config, scenario, 'random_policy', rng=np.random.default_rng(config['random_seed'] + 1_551))
        s = summarize_trajectory(df)
        s.update(policy)
        s['policy_id'] = i
        rows.append(s)
    res = pd.DataFrame(rows)
    objs = res[['avg_welfare_tail', 'avg_G_tail', 'avg_policy_cost_tail', 'avg_free_riding_tail', 'avg_H_tail', 'avg_effort_gini_tail']].to_numpy(float)
    dominated = np.zeros(len(res), dtype=bool)
    for i in range(len(res)):
        for j in range(len(res)):
            if i == j:
                continue
            better_or_equal = (
                objs[j, 0] >= objs[i, 0] and objs[j, 1] >= objs[i, 1] and objs[j, 2] <= objs[i, 2]
                and objs[j, 3] <= objs[i, 3] and objs[j, 4] <= objs[i, 4] and objs[j, 5] <= objs[i, 5]
            )
            strictly = (
                objs[j, 0] > objs[i, 0] or objs[j, 1] > objs[i, 1] or objs[j, 2] < objs[i, 2]
                or objs[j, 3] < objs[i, 3] or objs[j, 4] < objs[i, 4] or objs[j, 5] < objs[i, 5]
            )
            if better_or_equal and strictly:
                dominated[i] = True
                break
    res['is_pareto'] = ~dominated
    return res
