from __future__ import annotations
import numpy as np
from typing import Dict


def policy_cost(policy: Dict[str, float], effort, free_riders=None) -> float:
    total_effort = float(np.sum(effort))
    free_count = 0 if free_riders is None else int(np.sum(free_riders))
    cost = 0.0
    cost += float(policy.get('subsidy', 0.0)) * total_effort
    cost += 0.35 * float(policy.get('matching', 0.0)) * total_effort
    cost += 0.20 * float(policy.get('reputation', 0.0)) * np.log1p(total_effort)
    cost += float(policy.get('budget', 0.0)) * 50.0
    cost += float(policy.get('backlog_reduction', 0.0)) * 35.0
    cost += 0.03 * float(policy.get('penalty', 0.0)) * free_count
    return float(cost)


def get_policy(config: Dict, name: str) -> Dict[str, float]:
    if name not in config['policies']:
        raise KeyError(f'Unknown policy: {name}')
    return dict(config['policies'][name])
