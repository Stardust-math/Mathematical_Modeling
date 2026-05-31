from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict
from .dynamic_stock_model import value_derivative, value_function, contribution_transform_derivative


class FreeRidingGame:
    def __init__(self, agents: pd.DataFrame, params: Dict[str, float], contribution_max: float, free_rider_threshold: float, high_benefit_quantile: float):
        self.agents = agents.reset_index(drop=True).copy()
        self.params = params
        self.contribution_max = float(contribution_max)
        self.free_rider_threshold = float(free_rider_threshold)
        self.benefit = self.agents['benefit'].to_numpy(float)
        self.cost = np.maximum(self.agents['cost'].to_numpy(float), 1e-6)
        self.efficiency = self.agents['efficiency'].to_numpy(float)
        self.pressure_sensitivity = self.agents['pressure_sensitivity'].to_numpy(float)
        self.high_benefit_cutoff = float(np.quantile(self.benefit, high_benefit_quantile))
        self.high_benefit_mask = (self.benefit >= self.high_benefit_cutoff).astype(float)
        self.q_max = float(np.sum(self.efficiency * self.contribution_max))
        weights = self.params.get('nash_response_weights', {}) or {}
        self.stock_benefit_weight = float(weights.get('stock_benefit', 0.90))
        self.pressure_relief_weight = float(weights.get('pressure_relief', 0.85))
        self.penalty_base_weight = float(weights.get('penalty_base', 0.35))
        self.penalty_gap_weight = float(weights.get('penalty_gap', 2.80))
        self.threshold_push_weight = float(weights.get('threshold_push', 0.10))
        self.damping_raw_weight = float(self.params.get('nash_update_damping_raw_weight', 0.42))
        self.damping_raw_weight = float(np.clip(self.damping_raw_weight, 0.05, 0.95))
        self.nash_update_max_iter = int(self.params.get('nash_update_max_iter', 60))
        self.nash_update_tol = float(self.params.get('nash_update_tol', 1e-7))

    def raw_marginal_response(self, effort: np.ndarray, G: float, policy: Dict[str, float]) -> np.ndarray:
        """Return the implemented raw Nash-style marginal response before damping.

        This helper is shared by the solver and the diagnostic CSV used in the
        paper. Keeping the formula in one place prevents mismatches between the
        reported best-response residuals and the actual equilibrium proxy.
        """
        e = np.asarray(effort, dtype=float)
        b = self.benefit
        c = self.cost
        a = self.efficiency
        mu = self.pressure_sensitivity
        subsidy = policy.get('subsidy', 0.0)
        penalty = policy.get('penalty', 0.0)
        reputation = policy.get('reputation', 0.0)
        matching = policy.get('matching', 0.0)
        threshold = max(policy.get('threshold', 0.0), self.free_rider_threshold)

        Q = float(np.dot(a, e))
        qprime = contribution_transform_derivative(Q, self.q_max)
        vprime = value_derivative(G)
        G_capacity = max(float(self.params.get('G_capacity', 1.5)), 1e-9)
        stock_power = float(self.params.get('stock_saturation_power', 2.0))
        stock_ratio = np.clip(float(G) / G_capacity, 0.0, 1.0)
        stock_capacity_factor = np.clip(1.0 - stock_ratio ** stock_power, 0.0, 1.0)
        marginal_public = a * (
            self.stock_benefit_weight * b * self.params['alpha'] * (1 + matching)
            * stock_capacity_factor * vprime * qprime
            + self.pressure_relief_weight * mu * self.params['kappa'] * qprime
        )
        rep_marginal = reputation / (1.0 + e)
        high = self.high_benefit_mask
        gap = np.maximum(threshold - e, 0.0)
        penalty_marginal = penalty * (self.penalty_base_weight + self.penalty_gap_weight * gap) * high
        threshold_push = np.where(e < threshold, self.threshold_push_weight * penalty, 0.0)
        raw = (marginal_public + subsidy + rep_marginal + penalty_marginal + threshold_push) / c
        return np.clip(raw, 0.0, self.contribution_max)

    def solve_nash_equilibrium(self, G: float, H: float, D: float, policy: Dict[str, float], max_iter: int | None = None, tol: float | None = None) -> np.ndarray:
        if max_iter is None:
            max_iter = self.nash_update_max_iter
        if tol is None:
            tol = self.nash_update_tol
        n = len(self.agents)
        e = np.full(n, 0.06, dtype=float)
        for _ in range(max_iter):
            new_e = self.raw_marginal_response(e, G, policy)
            if np.max(np.abs(new_e - e)) < tol:
                e = new_e
                break
            w = self.damping_raw_weight
            e = (1.0 - w) * e + w * new_e
        return e

    def nash_update_diagnostics(self, G: float, H: float, D: float, policy: Dict[str, float], max_iter: int | None = None, tol: float | None = None) -> Dict[str, float]:
        """Run the Nash-style update and report the residual used by the paper.

        The reported residual is the raw one-step best-response residual after
        the final damped sweep: max_i |BR_i(e_final) - e_final_i|. This is the
        same quantity described in the numerical stability table.
        """
        if max_iter is None:
            max_iter = self.nash_update_max_iter
        if tol is None:
            tol = self.nash_update_tol
        n = len(self.agents)
        e = np.full(n, 0.06, dtype=float)
        sweeps = 0
        converged = False
        for k in range(max_iter):
            new_e = self.raw_marginal_response(e, G, policy)
            sweeps = k + 1
            if np.max(np.abs(new_e - e)) < tol:
                e = new_e
                converged = True
                break
            w = self.damping_raw_weight
            e = (1.0 - w) * e + w * new_e

        final_raw = self.raw_marginal_response(e, G, policy)
        max_raw_residual = float(np.max(np.abs(final_raw - e)))
        w = self.damping_raw_weight
        final_damped = (1.0 - w) * e + w * final_raw
        max_damped_residual = float(np.max(np.abs(final_damped - e)))
        a = self.efficiency
        return {
            'damped_sweeps': int(sweeps),
            'converged': bool(converged),
            'max_residual': max_raw_residual,
            'max_damped_residual': max_damped_residual,
            'Q_IR': float(np.sum(a * e)),
            'mean_effort': float(np.mean(e)),
        }

    def classify_free_riders(self, effort: np.ndarray) -> np.ndarray:
        return (effort < self.free_rider_threshold) & (self.benefit >= self.high_benefit_cutoff)

    def utility(self, effort: np.ndarray, G_next: float, H_next: float, policy: Dict[str, float]) -> np.ndarray:
        """Evaluate the implemented synthetic individual utility.

        The penalty-related term is written so that its local derivative is
        consistent with the marginal response used in solve_nash_equilibrium:
        high-benefit agents receive the targeted threshold incentive and all
        below-threshold agents receive the small threshold-push term. This helper
        is mainly diagnostic; the reported Nash-style benchmark is computed by
        the damped marginal-response update above.
        """
        effort = np.asarray(effort, dtype=float)
        b = self.benefit
        c = self.cost
        mu = self.pressure_sensitivity
        threshold = max(policy.get('threshold', 0.0), self.free_rider_threshold)
        base = b * value_function(G_next) - 0.5 * c * effort ** 2 - mu * H_next
        base += policy.get('subsidy', 0.0) * effort
        base += policy.get('reputation', 0.0) * np.log1p(effort)

        penalty = float(policy.get('penalty', 0.0))
        high = self.high_benefit_mask
        gap = np.maximum(threshold - effort, 0.0)
        base += penalty * (high * (self.penalty_base_weight * effort - 0.5 * self.penalty_gap_weight * gap ** 2) - self.threshold_push_weight * gap)
        return base
