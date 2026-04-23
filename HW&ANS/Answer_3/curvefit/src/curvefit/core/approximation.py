from dataclasses import dataclass
import numpy as np
from scipy.interpolate import splprep, splev
from .base import BaseCurveModel
from .parameterization import compute_parameterization


@dataclass
class PolynomialApproximationCurve(BaseCurveModel):
    coef_x: np.ndarray
    coef_y: np.ndarray

    def __init__(self, coef_x, coef_y, closed=False, name="poly_ls"):
        super().__init__(closed=closed, name=name)
        self.coef_x = np.asarray(coef_x, dtype=float)
        self.coef_y = np.asarray(coef_y, dtype=float)

    def evaluate(self, u):
        u = np.asarray(u, dtype=float)
        return np.column_stack([np.polyval(self.coef_x, u), np.polyval(self.coef_y, u)])


class BSplineApproximationCurve(BaseCurveModel):
    def __init__(self, tck, closed=False, name="bspline_ls"):
        super().__init__(closed=closed, name=name)
        self.tck = tck

    def evaluate(self, u):
        x, y = splev(np.asarray(u, dtype=float), self.tck)
        return np.column_stack([x, y])


def fit_polynomial_least_squares(points, parameterization="chord", closed=False, degree=5):
    pts = np.asarray(points, dtype=float)
    t = compute_parameterization(pts, parameterization, closed=closed)
    degree = int(max(1, min(degree, len(pts) - 1)))
    return PolynomialApproximationCurve(
        np.polyfit(t, pts[:, 0], degree),
        np.polyfit(t, pts[:, 1], degree),
        closed=closed,
    )


def fit_bspline_least_squares(points, parameterization="chord", closed=False, degree=3, smoothing=0.0):
    pts = np.asarray(points, dtype=float)
    t = compute_parameterization(pts, parameterization, closed=closed)
    degree = int(max(1, min(degree, len(pts) - 1)))
    tck, _u = splprep(
        [pts[:, 0], pts[:, 1]],
        u=t,
        s=float(max(0.0, smoothing)),
        k=degree,
        per=int(closed),
    )
    return BSplineApproximationCurve(tck, closed=closed)


def fit_approximation(method, points, parameterization="chord", closed=False, **kwargs):
    m = method.lower()
    if m in {"poly_ls", "poly", "polynomial_ls"}:
        return fit_polynomial_least_squares(points, parameterization=parameterization, closed=closed, **kwargs)
    if m in {"bspline_ls", "bspline"}:
        return fit_bspline_least_squares(points, parameterization=parameterization, closed=closed, **kwargs)
    raise ValueError(f"Unknown approximation method: {method}")
