from __future__ import annotations
import numpy as np
from typing import Dict, Tuple


def contribution_transform(Q: float, Q_max: float) -> float:
    return float(np.log1p(max(Q, 0.0)) / max(np.log1p(max(Q_max, 1e-9)), 1e-9))


def contribution_transform_derivative(Q: float, Q_max: float) -> float:
    return float(1.0 / ((1.0 + max(Q, 0.0)) * max(np.log1p(max(Q_max, 1e-9)), 1e-9)))


def value_function(G: float, omega: float = 3.6, kind: str = 'log') -> float:
    G = float(np.clip(G, 0, 2.0))
    if kind == 'exp':
        return float(1.0 - np.exp(-omega * G))
    return float(np.log1p(omega * G))


def value_derivative(G: float, omega: float = 3.6, kind: str = 'log') -> float:
    G = float(np.clip(G, 0, 2.0))
    if kind == 'exp':
        return float(omega * np.exp(-omega * G))
    return float(omega / (1.0 + omega * G))


class PublicGoodDynamicModel:
    def __init__(self, params: Dict[str, float], q_max: float):
        self.params = params
        self.q_max = max(float(q_max), 1e-6)
        self.G_capacity = float(params.get('G_capacity', 1.5))
        self.H_capacity = float(params.get('H_capacity', 1.5))
        self.D_capacity = float(params.get('D_capacity', 1.5))

    def q_norm(self, Q: float) -> float:
        return contribution_transform(Q, self.q_max)

    def _project_state(self, x: float, low: float, high: float) -> float:
        """Project a state variable onto its feasible interval.

        The projection is only a feasibility safeguard. The economically
        meaningful capacity effect is represented explicitly in the transition
        equations through saturation factors, rather than being hidden in a
        clipping function.
        """
        if high <= low:
            return float(low)
        return float(np.clip(x, low, high))

    def stock_saturation_factor(self, G: float) -> float:
        """Diminishing contribution productivity near stock capacity.

        When the public-good stock approaches its capacity, additional
        effective contribution generates a smaller increment. This makes the
        capacity constraint part of the model mechanism, not only a numerical
        post-processing step.
        """
        power = float(self.params.get('stock_saturation_power', 2.0))
        ratio = np.clip(float(G) / max(self.G_capacity, 1e-9), 0.0, 1.0)
        return float(np.clip(1.0 - ratio ** power, 0.0, 1.0))

    def pressure_room_factor(self, H: float) -> float:
        """Remaining room for maintenance pressure accumulation.

        Demand-induced pressure is damped when the pressure index is already
        close to its scenario-specific capacity.
        """
        power = float(self.params.get('pressure_saturation_power', 1.0))
        ratio = np.clip(float(H) / max(self.H_capacity, 1e-9), 0.0, 1.0)
        return float(np.clip(1.0 - ratio ** power, 0.0, 1.0))


    def stock_saturation_derivative(self, G: float) -> float:
        power = float(self.params.get('stock_saturation_power', 2.0))
        if power <= 0:
            return 0.0
        ratio = np.clip(float(G) / max(self.G_capacity, 1e-9), 0.0, 1.0)
        if ratio <= 0.0 and power < 1.0:
            return 0.0
        return float(-power * (ratio ** max(power - 1.0, 0.0)) / max(self.G_capacity, 1e-9))

    def pressure_room_derivative(self, H: float) -> float:
        power = float(self.params.get('pressure_saturation_power', 1.0))
        if power <= 0:
            return 0.0
        ratio = np.clip(float(H) / max(self.H_capacity, 1e-9), 0.0, 1.0)
        if ratio <= 0.0 and power < 1.0:
            return 0.0
        return float(-power * (ratio ** max(power - 1.0, 0.0)) / max(self.H_capacity, 1e-9))

    def step_state(self, G: float, H: float, D: float, Q: float, policy: Dict[str, float], rng=None) -> Tuple[float, float, float, float]:
        p = self.params
        qn = self.q_norm(Q)
        matching = float(policy.get('matching', 0.0))
        budget = float(policy.get('budget', 0.0))
        backlog_reduction = float(policy.get('backlog_reduction', 0.0))

        stock_capacity_factor = self.stock_saturation_factor(G)
        pressure_room = self.pressure_room_factor(H)

        G_raw = (
            (1 - p['delta']) * G
            + p['alpha'] * (1 + matching) * qn * stock_capacity_factor
            - p['eta'] * H
            + budget
        )
        H_raw = (
            (1 - p['rho']) * H
            + p['lambda'] * D * pressure_room
            - p['kappa'] * qn
            - backlog_reduction
        )

        G_next = self._project_state(G_raw, 0.0, self.G_capacity)
        H_next = self._project_state(H_raw, 0.0, self.H_capacity)

        noise = p.get('noise', 0.0)
        eps = 0.0 if rng is None else float(rng.normal(0.0, noise))
        D_next = D * (1 + p.get('demand_growth', 0.0) + p.get('chi', 0.0) * (G / max(self.G_capacity, 1e-9)) - p.get('psi', 0.0) * (H / max(self.H_capacity, 1e-9))) + eps
        D_next = float(np.clip(D_next, 0.02, self.D_capacity))
        return float(np.clip(G_next, 0.0, self.G_capacity)), float(np.clip(H_next, 0.0, self.H_capacity)), D_next, qn

    def vector_field(self, G_grid, H_grid, D: float, Q: float, policy: Dict[str, float]):
        U = np.zeros_like(G_grid)
        V = np.zeros_like(H_grid)
        for i in range(G_grid.shape[0]):
            for j in range(G_grid.shape[1]):
                gn, hn, _, _ = self.step_state(float(G_grid[i, j]), float(H_grid[i, j]), D, Q, policy, rng=None)
                U[i, j] = gn - G_grid[i, j]
                V[i, j] = hn - H_grid[i, j]
        return U, V

    def jacobian_numeric(self, G: float, H: float, D: float, Q: float, policy: Dict[str, float], eps: float = 1e-5):
        base = np.array(self.step_state(G, H, D, Q, policy, rng=None)[:3])
        J = np.zeros((3, 3))
        x = np.array([G, H, D], dtype=float)
        for k in range(3):
            xp = x.copy(); xp[k] += eps
            fp = np.array(self.step_state(xp[0], xp[1], xp[2], Q, policy, rng=None)[:3])
            J[:, k] = (fp - base) / eps
        return J

    def jacobian_analytic(self, G: float, H: float, D: float, Q: float, policy: Dict[str, float]):
        """Analytic local Jacobian of the unclipped saturating transition.

        This avoids repeatedly calling finite differences during long simulation
        experiments. At exact projection boundaries the derivative is interpreted
        as the interior-side local derivative, which is sufficient for the
        reported stability diagnostic.
        """
        p = self.params
        qn = self.q_norm(Q)
        matching = float(policy.get('matching', 0.0))
        stock_factor = self.stock_saturation_factor(G)
        pressure_room = self.pressure_room_factor(H)
        d_stock = self.stock_saturation_derivative(G)
        d_room = self.pressure_room_derivative(H)

        J = np.zeros((3, 3), dtype=float)
        J[0, 0] = (1.0 - p['delta']) + p['alpha'] * (1.0 + matching) * qn * d_stock
        J[0, 1] = -p['eta']
        J[0, 2] = 0.0

        J[1, 0] = 0.0
        J[1, 1] = (1.0 - p['rho']) + p['lambda'] * D * d_room
        J[1, 2] = p['lambda'] * pressure_room

        growth_factor = (
            1.0
            + p.get('demand_growth', 0.0)
            + p.get('chi', 0.0) * (G / max(self.G_capacity, 1e-9))
            - p.get('psi', 0.0) * (H / max(self.H_capacity, 1e-9))
        )
        J[2, 0] = D * p.get('chi', 0.0) / max(self.G_capacity, 1e-9)
        J[2, 1] = -D * p.get('psi', 0.0) / max(self.H_capacity, 1e-9)
        J[2, 2] = growth_factor
        return J

    def stability_score(self, G: float, H: float, D: float, Q: float, policy: Dict[str, float]) -> float:
        J = self.jacobian_analytic(G, H, D, Q, policy)
        radius = max(abs(np.linalg.eigvals(J)))
        return float(1.0 - radius)

