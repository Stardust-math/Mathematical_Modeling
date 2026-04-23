from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np

def _validate_points(points):
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Points must have shape (N,2).")
    if arr.shape[0] < 2:
        raise ValueError("At least two points are required.")
    if not np.isfinite(arr).all():
        raise ValueError("Points contain non-finite values.")
    return arr

def normalize_points(points, method="bbox"):
    pts = _validate_points(points)
    if method == "bbox":
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        center = 0.5 * (mn + mx)
        scale = float(np.max(mx - mn))
        scale = scale if scale > 1e-12 else 1.0
    elif method == "zscore":
        center = pts.mean(axis=0)
        scale = float(np.std(pts))
        scale = scale if scale > 1e-12 else 1.0
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return (pts - center) / scale, {"method": method, "center": center, "scale": scale}

def resample_polyline(points, n_samples, closed=False):
    pts = _validate_points(points)
    work = np.vstack([pts, pts[:1]]) if closed else pts
    seg = np.linalg.norm(np.diff(work, axis=0), axis=1)
    total = float(seg.sum())
    if total < 1e-12:
        return np.repeat(pts[:1], n_samples, axis=0)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, total, n_samples + int(closed))
    if closed:
        targets = targets[:-1]
    out = []
    j = 0
    for t in targets:
        while j + 1 < len(s) and s[j + 1] < t:
            j += 1
        j = min(j, len(work) - 2)
        denom = s[j + 1] - s[j]
        a = 0.0 if denom < 1e-12 else (t - s[j]) / denom
        out.append((1 - a) * work[j] + a * work[j + 1])
    return np.asarray(out, dtype=float)

def add_gaussian_noise(points, sigma, random_state=None):
    pts = _validate_points(points)
    if sigma <= 0:
        return pts.copy()
    rng = np.random.default_rng(random_state)
    return pts + rng.normal(0.0, sigma, size=pts.shape)

def load_points(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        arr = np.loadtxt(path, delimiter=",")
    elif path.suffix.lower() == ".txt":
        arr = np.loadtxt(path)
    elif path.suffix.lower() == ".npy":
        arr = np.load(path)
    elif path.suffix.lower() == ".json":
        arr = np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=float)
    else:
        raise ValueError(f"Unsupported file: {path.suffix}")
    return _validate_points(arr)

@dataclass
class PointSet:
    points: np.ndarray
    closed: bool = False
    normalized: bool = False
    normalization_meta: dict | None = None
    name: str = "points"
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.points = _validate_points(self.points)

    def normalized_copy(self, method="bbox"):
        pts, meta = normalize_points(self.points, method=method)
        return PointSet(pts, self.closed, True, meta, self.name, dict(self.extra))

    def resampled_copy(self, n_samples):
        pts = resample_polyline(self.points, n_samples=n_samples, closed=self.closed)
        return PointSet(pts, self.closed, self.normalized, self.normalization_meta, self.name, dict(self.extra))
