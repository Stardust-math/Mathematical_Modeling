from dataclasses import dataclass
import time
import numpy as np
from scipy.spatial.distance import cdist

def _pairwise_min_distances(a, b):
    d = cdist(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return d.min(axis=1), d.min(axis=0)

def chamfer_distance(a, b):
    d1, d2 = _pairwise_min_distances(a, b)
    return float(d1.mean() + d2.mean())

def hausdorff_distance(a, b):
    d1, d2 = _pairwise_min_distances(a, b)
    return float(max(d1.max(), d2.max()))

def _point_to_segment_distances(points, seg_start, seg_end):
    p = points[:, None, :]
    a = seg_start[None, :, :]
    b = seg_end[None, :, :]
    ab = b - a
    denom = np.sum(ab * ab, axis=2, keepdims=True)
    denom = np.where(denom < 1e-12, 1.0, denom)
    t = np.sum((p - a) * ab, axis=2, keepdims=True) / denom
    t = np.clip(t, 0.0, 1.0)
    proj = a + t * ab
    return np.linalg.norm(p - proj, axis=2).min(axis=1)

def average_point_to_curve_distance(points, curve_points, closed=False):
    curve = np.asarray(curve_points, dtype=float)
    work = np.vstack([curve, curve[:1]]) if closed else curve
    d = _point_to_segment_distances(np.asarray(points, dtype=float), work[:-1], work[1:])
    return float(d.mean())

def smoothness_second_difference(curve_points, closed=False):
    curve = np.asarray(curve_points, dtype=float)
    if len(curve) < 3:
        return 0.0
    if closed:
        second = np.roll(curve, 1, axis=0) - 2.0 * curve + np.roll(curve, -1, axis=0)
    else:
        second = curve[:-2] - 2.0 * curve[1:-1] + curve[2:]
    return float(np.mean(np.linalg.norm(second, axis=1)))

def runtime_of(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return time.perf_counter() - t0, out

@dataclass
class EvaluationResult:
    avg_point_to_curve: float
    chamfer: float
    hausdorff: float
    smoothness: float
    runtime_sec: float
    def to_dict(self):
        return self.__dict__.copy()

def evaluate_reconstruction(input_points, reconstructed_curve, reference_curve, closed=False, runtime_sec=0.0):
    return EvaluationResult(
        avg_point_to_curve=average_point_to_curve_distance(input_points, reconstructed_curve, closed=closed),
        chamfer=chamfer_distance(reconstructed_curve, reference_curve),
        hausdorff=hausdorff_distance(reconstructed_curve, reference_curve),
        smoothness=smoothness_second_difference(reconstructed_curve, closed=closed),
        runtime_sec=float(runtime_sec),
    )
