import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from curvefit.core.interpolation import fit_interpolant
from curvefit.core.approximation import fit_approximation
from curvefit.core.fourier import FourierApproximator
from curvefit.data.synthetic import generate_synthetic_curve


def main():
    points, closed = generate_synthetic_curve("s_curve", n_points=30, sampling="nonuniform", random_state=0)
    for method in ["cubic_spline", "bspline_interp", "poly_ls", "bspline_ls"]:
        model = (
            fit_interpolant(method, points, parameterization="chord", closed=closed)
            if method in {"cubic_spline", "bspline_interp"}
            else fit_approximation(method, points, parameterization="chord", closed=closed)
        )
        print(method, model.sample(120).shape)

    points2, closed2 = generate_synthetic_curve("circle", n_points=60, sampling="nonuniform", random_state=0)
    fa = FourierApproximator(points2)
    print("fourier", fa.reconstruct(k=8, n_samples=120).shape, closed2)


if __name__ == "__main__":
    main()
