from abc import ABC, abstractmethod
import numpy as np

class BaseCurveModel(ABC):
    def __init__(self, closed=False, name="base"):
        self.closed = bool(closed)
        self.name = name

    @abstractmethod
    def evaluate(self, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sample(self, n_samples=400):
        u = np.linspace(0.0, 1.0, n_samples, endpoint=not self.closed)
        return np.asarray(self.evaluate(u), dtype=float)
