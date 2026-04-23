import numpy as np
from curvefit.core.data import add_gaussian_noise

def _sample_parameter(n_points, sampling="uniform", random_state=None, closed=False):
    rng = np.random.default_rng(random_state)
    if sampling == "uniform":
        return np.linspace(0.0, 1.0, n_points, endpoint=not closed)
    if sampling == "nonuniform":
        raw = np.sort(rng.random(n_points))
        raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
        if not closed:
            raw[0], raw[-1] = 0.0, 1.0
        return raw
    raise ValueError(f"Unknown sampling: {sampling}")

def _s_curve(t):
    x = 2.0*t - 1.0
    return np.column_stack([x, np.sin(np.pi*x)])
def _sine_modulated(t):
    x = 2.0*t - 1.0
    y = 0.5*np.sin(4*np.pi*t) + 0.25*np.sin(9*np.pi*t)
    return np.column_stack([x, y])
def _cubic_poly(t):
    x = 2.0*t - 1.0
    return np.column_stack([x, 0.8*x**3 - 0.4*x])
def _circle(t):
    th = 2*np.pi*t
    return np.column_stack([np.cos(th), np.sin(th)])
def _ellipse(t):
    th = 2*np.pi*t
    return np.column_stack([1.4*np.cos(th), 0.8*np.sin(th)])
def _cardioid(t):
    th = 2*np.pi*t
    r = 1.0 - np.cos(th)
    return np.column_stack([r*np.cos(th), r*np.sin(th)])
def _rose(t):
    th = 2*np.pi*t
    r = np.cos(5*th)
    return np.column_stack([r*np.cos(th), r*np.sin(th)])
def _wavy_circle(t):
    th = 2*np.pi*t
    r = 1.0 + 0.2*np.cos(6*th) + 0.1*np.sin(3*th)
    return np.column_stack([r*np.cos(th), r*np.sin(th)])

SHAPES = {
    "s_curve": (_s_curve, False),
    "sine_modulated": (_sine_modulated, False),
    "cubic_poly": (_cubic_poly, False),
    "circle": (_circle, True),
    "ellipse": (_ellipse, True),
    "cardioid": (_cardioid, True),
    "rose": (_rose, True),
    "wavy_circle": (_wavy_circle, True),
}
def list_available_shapes():
    return list(SHAPES.keys())

def generate_synthetic_curve(shape, n_points=40, sampling="uniform", noise_sigma=0.0, random_state=None):
    func, closed = SHAPES[shape]
    t = _sample_parameter(n_points, sampling=sampling, random_state=random_state, closed=closed)
    points = func(t)
    if noise_sigma > 0:
        points = add_gaussian_noise(points, sigma=noise_sigma, random_state=random_state)
    return points, closed
