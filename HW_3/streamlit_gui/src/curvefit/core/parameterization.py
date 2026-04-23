import numpy as np

def _to_unit_interval(t):
    t = np.asarray(t, dtype=float)
    t = t - t[0]
    denom = t[-1] - t[0]
    return np.linspace(0.0, 1.0, len(t)) if denom < 1e-12 else t / denom

def uniform_parameterization(points):
    return np.linspace(0.0, 1.0, len(points))

def chord_length_parameterization(points):
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    seg = np.maximum(seg, 1e-12)
    return _to_unit_interval(np.concatenate([[0.0], np.cumsum(seg)]))

def centripetal_parameterization(points):
    seg = np.sqrt(np.linalg.norm(np.diff(points, axis=0), axis=1))
    seg = np.maximum(seg, 1e-12)
    return _to_unit_interval(np.concatenate([[0.0], np.cumsum(seg)]))

def _turning_angles(points, closed=False):
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    ang = np.zeros(n, dtype=float)
    def angle(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        c = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
        return float(np.arccos(c))
    if closed:
        for i in range(n):
            ang[i] = angle(pts[i] - pts[(i - 1) % n], pts[(i + 1) % n] - pts[i])
    else:
        for i in range(1, n - 1):
            ang[i] = angle(pts[i] - pts[i - 1], pts[i + 1] - pts[i])
    return ang

def foley_nielsen_parameterization(points, closed=False):
    # Angle-augmented practical Foley–Nielsen style variant used in this project.
    pts = np.asarray(points, dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    seg = np.maximum(seg, 1e-12)
    ang = _turning_angles(pts, closed=closed)
    alpha = 0.5
    inc = []
    for i, d in enumerate(seg):
        correction = 1.0 + alpha * (ang[i] + ang[i + 1]) / np.pi
        inc.append(max(d * correction, 1e-12))
    return _to_unit_interval(np.concatenate([[0.0], np.cumsum(np.asarray(inc))]))

def compute_parameterization(points, method, closed=False):
    m = method.lower()
    if m in {"uniform"}:
        return uniform_parameterization(points)
    if m in {"chord", "chord_length"}:
        return chord_length_parameterization(points)
    if m in {"centripetal"}:
        return centripetal_parameterization(points)
    if m in {"foley", "foley_nielsen", "foley-nielsen"}:
        return foley_nielsen_parameterization(points, closed=closed)
    raise ValueError(f"Unknown parameterization: {method}")
