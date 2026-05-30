from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from .dynamic_stock_model import PublicGoodDynamicModel
from .game_model import FreeRidingGame
from .social_optimum import SocialPlanner
from .policy_mechanisms import policy_cost
from .metrics import gini


def simulate_policy_trajectory(agents: pd.DataFrame, scenario_params: Dict, policy: Dict, config: Dict, scenario: str, policy_name: str, mode: str = 'nash', rng=None) -> pd.DataFrame:
    T = int(config['time_horizon'])
    contribution_max = float(config['contribution_max'])
    eff_array = agents['efficiency'].to_numpy(float)
    model = PublicGoodDynamicModel(scenario_params, q_max=float(np.sum(eff_array * contribution_max)))
    game = FreeRidingGame(agents, scenario_params, contribution_max, config['free_rider_threshold'], config['high_benefit_quantile'])
    planner = SocialPlanner(agents, scenario_params, contribution_max, config['free_rider_threshold'], config['high_benefit_quantile'])
    G, H, D = scenario_params['G0'], scenario_params['H0'], scenario_params['D0']
    rows = []
    x0_social = None
    for t in range(T):
        if mode == 'social':
            effort = planner.solve_social_optimum(G, H, D, policy, x0=x0_social)
            x0_social = effort.copy()
        else:
            effort = game.solve_nash_equilibrium(G, H, D, policy)
        free = game.classify_free_riders(effort)
        effort_list = effort.astype(float, copy=False)
        Q = float(np.dot(eff_array, effort_list))
        G_next, H_next, D_next, qn = model.step_state(G, H, D, Q, policy, rng=rng)
        pcost = policy_cost(policy, effort, free)
        welfare = planner.social_welfare(effort, G_next, H_next, pcost)
        rows.append({
            'time': t, 'scenario': scenario, 'policy': policy_name, 'mode': mode,
            'G': G_next, 'H': H_next, 'D': D_next, 'Q': Q, 'Q_norm': qn,
            'avg_effort': float(np.mean(effort_list)), 'effort_gini': gini(effort_list),
            'free_riding_ratio': float(sum(bool(v) for v in free) / len(free)), 'welfare': welfare, 'policy_cost': pcost,
            'stability_score': model.stability_score(G_next, H_next, D_next, Q, policy)
        })
        G, H, D = G_next, H_next, D_next
    return pd.DataFrame(rows)


def one_step_equilibrium_summary(agents: pd.DataFrame, scenario_params: Dict, policy: Dict, config: Dict, scenario: str) -> Tuple[dict, dict, dict]:
    contribution_max = float(config['contribution_max'])
    eff_array = agents['efficiency'].to_numpy(float)
    model = PublicGoodDynamicModel(scenario_params, q_max=float(np.sum(eff_array * contribution_max)))
    game = FreeRidingGame(agents, scenario_params, contribution_max, config['free_rider_threshold'], config['high_benefit_quantile'])
    planner = SocialPlanner(agents, scenario_params, contribution_max, config['free_rider_threshold'], config['high_benefit_quantile'])
    G, H, D = scenario_params['G0'], scenario_params['H0'], scenario_params['D0']
    rows = []
    e_nash = game.solve_nash_equilibrium(G, H, D, policy)
    e_social = planner.solve_social_optimum(G, H, D, policy, x0=e_nash)
    for mode, effort in [('nash', e_nash), ('social', e_social)]:
        free = game.classify_free_riders(effort)
        effort_list = effort.astype(float, copy=False)
        Q = float(np.dot(eff_array, effort_list))
        G_next, H_next, D_next, qn = model.step_state(G, H, D, Q, policy, rng=None)
        pcost = policy_cost(policy, effort, free)
        rows.append({
            'scenario': scenario, 'policy': 'baseline', 'mode': mode, 'G': G_next, 'H': H_next, 'D': D_next, 'Q': Q, 'Q_norm': qn,
            'avg_effort': float(np.mean(effort_list)), 'effort_gini': gini(effort_list), 'free_riding_ratio': float(sum(bool(v) for v in free) / len(free)),
            'welfare': planner.social_welfare(effort, G_next, H_next, pcost), 'policy_cost': pcost
        })
    check = planner.validate_against_nash(rows[0], rows[1])
    return rows[0], rows[1], check
