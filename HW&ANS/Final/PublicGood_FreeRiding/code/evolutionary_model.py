from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict


def simulate_replicator(policy_name: str, policy: Dict[str, float], T: int = 120, seed: int = 20260614) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.array([0.18, 0.42, 0.40], dtype=float)  # active, conditional, free rider
    rows = []
    for t in range(T):
        public_good_bonus = 0.7 * x[0] + 0.35 * x[1]
        penalty = policy.get('penalty', 0.0)
        reputation = policy.get('reputation', 0.0)
        subsidy = policy.get('subsidy', 0.0)
        matching = policy.get('matching', 0.0)
        pi_active = 0.55 + public_good_bonus + reputation + 0.5 * subsidy + 0.25 * matching - 0.28
        pi_cond = 0.48 + public_good_bonus + 0.45 * reputation + 0.25 * subsidy + 0.2 * matching - 0.16 + 0.2 * (x[0] - x[2])
        pi_free = 0.52 + public_good_bonus - penalty * (0.6 + x[2])
        payoffs = np.array([pi_active, pi_cond, pi_free]) + rng.normal(0, 0.005, 3)
        avg = float(x @ payoffs)
        x = x * np.exp(0.25 * (payoffs - avg))
        x = x / x.sum()
        rows.append({'time': t, 'policy': policy_name, 'active_contributors': x[0], 'conditional_contributors': x[1], 'free_riders': x[2], 'avg_payoff': avg})
    return pd.DataFrame(rows)
