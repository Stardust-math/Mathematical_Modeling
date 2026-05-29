from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy.optimize import minimize
from .dynamic_stock_model import PublicGoodDynamicModel, value_function
from .policy_mechanisms import policy_cost


class SocialPlanner:
    def __init__(self, agents: pd.DataFrame, params: Dict[str, float], contribution_max: float, free_rider_threshold: float = 0.08, high_benefit_quantile: float = 0.55):
        self.agents = agents.reset_index(drop=True).copy()
        self.params = params
        self.contribution_max = float(contribution_max)
        self.free_rider_threshold = float(free_rider_threshold)
        self.high_benefit_cutoff = float(self.agents['benefit'].quantile(high_benefit_quantile))
        self.q_max = float((self.agents['efficiency'] * self.contribution_max).sum())
        self.last_result = None

    def _free_flags(self, effort: np.ndarray) -> np.ndarray:
        return (effort < self.free_rider_threshold) & (self.agents['benefit'].to_numpy(float) >= self.high_benefit_cutoff)

    def social_welfare(self, effort: np.ndarray, G_next: float, H_next: float, policy_cost_value: float) -> float:
        b = self.agents['benefit'].to_numpy(float)
        c = self.agents['cost'].to_numpy(float)
        mu = self.agents['pressure_sensitivity'].to_numpy(float)
        welfare = np.sum(b * value_function(G_next) - 0.5 * c * effort ** 2 - mu * H_next)
        return float(welfare - policy_cost_value)

    def solve_social_optimum(self, G: float, H: float, D: float, policy: Dict[str, float], x0: np.ndarray | None = None) -> np.ndarray:
        n = len(self.agents)
        a = self.agents['efficiency'].to_numpy(float)
        b = self.agents['benefit'].to_numpy(float)
        c = np.maximum(self.agents['cost'].to_numpy(float), 1e-8)
        mu = self.agents['pressure_sensitivity'].to_numpy(float)
        model = PublicGoodDynamicModel(self.params, q_max=self.q_max)
        bounds = [(0.0, self.contribution_max)] * n
        if x0 is None:
            x0 = np.clip(0.20 + 0.15 * (b / np.maximum(c, 1e-8)) / np.max(b / np.maximum(c, 1e-8)), 0.0, self.contribution_max)

        def objective(eff):
            eff = np.clip(np.asarray(eff, dtype=float), 0.0, self.contribution_max)
            Q = float(np.sum(a * eff))
            G_next, H_next, _, _ = model.step_state(G, H, D, Q, policy, rng=None)
            free_flags = self._free_flags(eff)
            pcost = policy_cost(policy, eff, free_flags)
            welfare = np.sum(b * value_function(G_next) - 0.5 * c * eff ** 2 - mu * H_next) - pcost
            return -float(welfare)

        result = minimize(objective, x0=np.asarray(x0, dtype=float), method='L-BFGS-B', bounds=bounds, options={'maxiter': 120, 'ftol': 1e-10})
        eff = np.clip(result.x, 0.0, self.contribution_max)
        self.last_result = result
        return eff

    def validate_against_nash(self, nash_row: Dict[str, float], social_row: Dict[str, float]) -> Dict[str, float | str | bool]:
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
            'optimizer_method': 'scipy.optimize.minimize (L-BFGS-B)',
            'warning': warning.strip()
        }
