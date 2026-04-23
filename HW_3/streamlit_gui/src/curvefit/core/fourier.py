from dataclasses import dataclass
import numpy as np
from scipy.interpolate import CubicSpline
from .base import BaseCurveModel

def _prepare_closed_curve(points, n_resample=512):
    pts = np.asarray(points, dtype=float)
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-12:
        pts = np.vstack([pts, pts[:1]])
    t = np.linspace(0.0, 1.0, len(pts))
    sx = CubicSpline(t, pts[:,0], bc_type="periodic")
    sy = CubicSpline(t, pts[:,1], bc_type="periodic")
    u = np.linspace(0.0, 1.0, n_resample, endpoint=False)
    return np.column_stack([sx(u), sy(u)])

@dataclass
class FourierCurve(BaseCurveModel):
    coeffs: np.ndarray
    freqs: np.ndarray
    def __init__(self, coeffs, freqs, name="fourier"):
        super().__init__(closed=True, name=name)
        self.coeffs = np.asarray(coeffs, dtype=complex)
        self.freqs = np.asarray(freqs, dtype=int)
    def evaluate(self, u):
        u = np.asarray(u, dtype=float).reshape(-1,1)
        z = np.sum(self.coeffs[None,:] * np.exp(2j*np.pi*u*self.freqs[None,:]), axis=1)
        return np.column_stack([z.real, z.imag])

class FourierApproximator:
    def __init__(self, points, n_resample=512):
        dense = _prepare_closed_curve(points, n_resample=n_resample)
        self.dense_curve = dense
        z = dense[:,0] + 1j*dense[:,1]
        n = len(z)
        fft = np.fft.fft(z) / n
        freqs = np.fft.fftfreq(n, d=1.0/n).astype(int)
        order = np.argsort(np.abs(freqs))
        self.coeffs_full = fft[order]
        self.freqs_full = freqs[order]

    def build_curve(self, k):
        k = int(max(0, k))
        mask = (self.freqs_full == 0) if k == 0 else (np.abs(self.freqs_full) <= k)
        return FourierCurve(self.coeffs_full[mask], self.freqs_full[mask], name=f"fourier_k{k}")

    def reconstruct(self, k, n_samples=400):
        return self.build_curve(k).sample(n_samples)

    def spectrum(self):
        order = np.argsort(self.freqs_full)
        return self.freqs_full[order], np.abs(self.coeffs_full[order])

    def epicycle_chain(self, t, k):
        curve = self.build_curve(k)
        order = np.argsort(np.abs(curve.freqs))
        coeffs, freqs = curve.coeffs[order], curve.freqs[order]
        center = 0.0 + 0.0j
        chain = []
        for c, f in zip(coeffs, freqs):
            end = center + c * np.exp(2j*np.pi*f*t)
            chain.append({"center": np.array([center.real, center.imag]), "end": np.array([end.real, end.imag]), "radius": float(np.abs(c)), "freq": int(f), "coef": c})
            center = end
        return chain
