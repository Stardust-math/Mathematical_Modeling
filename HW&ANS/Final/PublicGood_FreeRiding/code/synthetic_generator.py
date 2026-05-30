from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from .constants import AGENT_COLUMNS


def generate_agents(config: Dict, scenario: str, rng: np.random.Generator) -> pd.DataFrame:
    n = int(config['num_agents'])
    dist = config['agent_distributions']
    sparams = config['scenario_parameters'][scenario]
    benefit = rng.lognormal(dist['benefit_lognormal_mean'], dist['benefit_lognormal_sigma'], n) * sparams.get('benefit_scale', 1.0)
    cost = rng.gamma(dist['cost_gamma_shape'], dist['cost_gamma_scale'], n) * sparams.get('cost_scale', 1.0)
    efficiency = rng.uniform(dist['efficiency_low'], dist['efficiency_high'], n)
    mu = rng.uniform(dist['pressure_low'], dist['pressure_high'], n)
    types = rng.choice(['active_contributor', 'conditional_contributor', 'passive_user'], size=n, p=[0.18, 0.42, 0.40])
    df = pd.DataFrame({
        'agent_id': np.arange(n),
        'scenario': scenario,
        'benefit': benefit,
        'cost': cost,
        'efficiency': efficiency,
        'pressure_sensitivity': mu,
        'initial_type': types
    })
    return df[AGENT_COLUMNS]


def generate_all_agents(config: Dict) -> pd.DataFrame:
    rng = np.random.default_rng(int(config['random_seed']))
    frames = []
    for scenario in config['scenarios']:
        frames.append(generate_agents(config, scenario, rng))
    return pd.concat(frames, ignore_index=True)


def scenario_initial_state(config: Dict, scenario: str) -> Dict[str, float]:
    p = config['scenario_parameters'][scenario]
    return {'G': float(p['G0']), 'H': float(p['H0']), 'D': float(p['D0'])}
