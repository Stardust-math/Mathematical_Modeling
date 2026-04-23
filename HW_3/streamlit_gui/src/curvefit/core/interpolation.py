from dataclasses import dataclass
import numpy as np
from scipy.interpolate import CubicSpline, splprep, splev
from .base import BaseCurveModel
from .parameterization import compute_parameterization


@dataclass
class CubicSplineCurve(BaseCurveModel):
    sx: CubicSpline
    sy: CubicSpline

    def __init__(self, sx, sy, closed=False, name="cubic_spline"):
        super().__init__(closed=closed, name=name)
        self.sx, self.sy = sx, sy

    def evaluate(self, u):
        u = np.asarray(u, dtype=float)
        return np.column_stack([self.sx(u), self.sy(u)])


class BSplineInterpolantCurve(BaseCurveModel):
    def __init__(self, tck, closed=False, name="bspline_interp"):
        super().__init__(closed=closed, name=name)
        self.tck = tck

    def evaluate(self, u):
        u = np.asarray(u, dtype=float)
        x, y = splev(u, self.tck)
        return np.column_stack([x, y])


def _remove_consecutive_duplicates(points, closed=False, tol=1e-12):
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if len(pts) < 2:
        raise ValueError("at least two points are required")

    kept = [pts[0]]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - kept[-1]) > tol:
            kept.append(pts[i])
    pts = np.asarray(kept, dtype=float)

    if closed and len(pts) >= 2 and np.linalg.norm(pts[0] - pts[-1]) <= tol:
        pts = pts[:-1]

    if closed and len(pts) < 3:
        raise ValueError("closed curve interpolation needs at least 3 distinct points")
    if (not closed) and len(pts) < 2:
        raise ValueError("open curve interpolation needs at least 2 distinct points")
    return pts


def _make_strictly_increasing(t, eps=1e-10):
    t = np.asarray(t, dtype=float).copy()
    if len(t) == 0:
        return t
    t[0] = 0.0
    for i in range(1, len(t)):
        if not np.isfinite(t[i]) or t[i] <= t[i - 1]:
            t[i] = t[i - 1] + eps
    span = t[-1] - t[0]
    if span <= 0:
        return np.linspace(0.0, 1.0, len(t))
    return (t - t[0]) / span


def _unique_parameter_samples(t, pts, closed=False, tol=1e-12):
    t = np.asarray(t, dtype=float)
    pts = np.asarray(pts, dtype=float)

    keep = [0]
    for i in range(1, len(t)):
        if t[i] - t[keep[-1]] > tol:
            keep.append(i)

    t2 = t[keep]
    pts2 = pts[keep]

    if closed and len(pts2) < 3:
        raise ValueError("closed curve interpolation needs at least 3 unique parameter samples")
    if (not closed) and len(pts2) < 2:
        raise ValueError("open curve interpolation needs at least 2 unique parameter samples")

    t2 = _make_strictly_increasing(t2)
    return t2, pts2


def _closed_periodic_data(points, parameterization):
    pts = _remove_consecutive_duplicates(points, closed=True)

    if np.linalg.norm(pts[0] - pts[-1]) > 1e-12:
        pts_aug = np.vstack([pts, pts[:1]])
    else:
        pts_aug = pts.copy()
        pts_aug[-1] = pts_aug[0]

    t = compute_parameterization(pts_aug, parameterization, closed=True)
    t = _make_strictly_increasing(t)
    t, pts_aug = _unique_parameter_samples(t, pts_aug, closed=True)

    if np.linalg.norm(pts_aug[0] - pts_aug[-1]) > 1e-12:
        pts_aug = np.vstack([pts_aug, pts_aug[:1]])
        t = np.concatenate([t, [1.0 + max(1e-8, np.mean(np.diff(t)))]])
        t = _make_strictly_increasing(t)

    pts_aug[-1] = pts_aug[0]
    return pts_aug, t


def _prepare_bspline_interp_data(points, parameterization="chord", closed=False):
    pts = _remove_consecutive_duplicates(points, closed=closed)
    t = compute_parameterization(pts, parameterization, closed=closed)
    t = _make_strictly_increasing(t)
    t, pts = _unique_parameter_samples(t, pts, closed=closed)
    return pts, t


def fit_cubic_spline_interpolant(points, parameterization="chord", closed=False):
    pts = _remove_consecutive_duplicates(points, closed=closed)

    if closed:
        pts, t = _closed_periodic_data(pts, parameterization)
        bc = "periodic"
    else:
        t = compute_parameterization(pts, parameterization, closed=False)
        t = _make_strictly_increasing(t)
        t, pts = _unique_parameter_samples(t, pts, closed=False)
        bc = "natural"

    return CubicSplineCurve(
        CubicSpline(t, pts[:, 0], bc_type=bc),
        CubicSpline(t, pts[:, 1], bc_type=bc),
        closed=closed,
    )


def fit_bspline_interpolant(points, parameterization="chord", closed=False, degree=3):
    pts, t = _prepare_bspline_interp_data(points, parameterization=parameterization, closed=closed)
    max_degree = len(pts) - 1 if not closed else len(pts) - 1
    degree = int(max(1, min(int(degree), 3, max_degree)))
    if len(pts) <= degree:
        degree = max(1, len(pts) - 1)

    tck, _u = splprep(
        [pts[:, 0], pts[:, 1]],
        u=t,
        s=0.0,
        k=degree,
        per=int(closed),
    )
    return BSplineInterpolantCurve(tck, closed=closed)


def fit_interpolant(method, points, parameterization="chord", closed=False, **kwargs):
    m = method.lower()
    if m in {"cubic_spline", "spline", "cubic"}:
        return fit_cubic_spline_interpolant(points, parameterization=parameterization, closed=closed)
    if m in {"bspline_interp", "bspline_interpolation", "parametric_bspline", "b_spline_interp"}:
        return fit_bspline_interpolant(points, parameterization=parameterization, closed=closed, **kwargs)
    raise ValueError(f"Unknown interpolation method: {method}")
