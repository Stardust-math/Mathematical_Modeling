from dataclasses import dataclass
import pandas as pd
import numpy as np
from curvefit.config import FIGURES_DIR, RESULTS_DIR
from curvefit.core.interpolation import fit_interpolant
from curvefit.core.approximation import fit_approximation
from curvefit.core.metrics import runtime_of, evaluate_reconstruction
from curvefit.data.synthetic import generate_synthetic_curve
from curvefit.utils.io_utils import ensure_dir, save_dataframe_csv
from curvefit.viz.plots import (
    plot_points_and_curve,
    plot_method_comparison,
    plot_local_zoom,
    plot_parameterization_comparison,
    plot_line_chart,
    plot_boxplot,
    plot_violinplot,
)

INTERP_METHODS = ["cubic_spline", "bspline_interp"]
APPROX_METHODS = ["poly_ls", "bspline_ls"]
ALL_METHODS = INTERP_METHODS + APPROX_METHODS
PARAMS = ["uniform", "chord", "centripetal", "foley_nielsen"]


def _default_fit_kwargs(method, n_points):
    if method == "bspline_interp":
        return {"degree": 3}
    if method == "poly_ls":
        return {"degree": min(7, max(3, n_points // 4))}
    if method == "bspline_ls":
        return {"degree": 3, "smoothing": 0.001 * n_points}
    return {}


@dataclass
class MainExperimentRunner:
    results_dir: str = RESULTS_DIR / "main"
    figures_dir: str = FIGURES_DIR / "main"
    curve_samples: int = 500
    reference_samples: int = 2000

    def __post_init__(self):
        ensure_dir(self.results_dir)
        ensure_dir(self.figures_dir)

    def _fit_and_evaluate(self, method, points, reference_curve, parameterization, closed, n_points):
        kwargs = _default_fit_kwargs(method, n_points)
        if method in INTERP_METHODS:
            elapsed, model = runtime_of(
                fit_interpolant,
                method,
                points,
                parameterization=parameterization,
                closed=closed,
                **kwargs,
            )
        else:
            elapsed, model = runtime_of(
                fit_approximation,
                method,
                points,
                parameterization=parameterization,
                closed=closed,
                **kwargs,
            )
        curve = model.sample(self.curve_samples)
        metric = evaluate_reconstruction(points, curve, reference_curve, closed=closed, runtime_sec=elapsed).to_dict()
        return curve, metric

    def run_no_noise_basic_reconstruction(self):
        rows = []
        for shape in ["s_curve", "circle", "ellipse", "sine_modulated"]:
            points, closed = generate_synthetic_curve(shape, n_points=40, sampling="uniform", noise_sigma=0.0, random_state=1)
            ref, _ = generate_synthetic_curve(shape, n_points=self.reference_samples, sampling="uniform", noise_sigma=0.0, random_state=1)
            curve_dict = {}
            for method in ALL_METHODS:
                curve, metric = self._fit_and_evaluate(method, points, ref, "chord", closed, len(points))
                curve_dict[method] = curve
                row = {
                    "experiment": "no_noise_basic",
                    "shape": shape,
                    "closed": closed,
                    "method": method,
                    "parameterization": "chord",
                    "n_points": len(points),
                    "noise_sigma": 0.0,
                }
                row.update(metric)
                rows.append(row)
                fig = plot_points_and_curve(
                    points,
                    curve,
                    title=f"{shape} | {method}",
                    closed=closed,
                    save_path=self.figures_dir / "no_noise_basic" / f"{shape}_{method}.svg",
                )
                import matplotlib.pyplot as plt
                plt.close(fig)
            fig = plot_method_comparison(
                points,
                curve_dict,
                title=f"Method comparison | {shape}",
                closed=closed,
                save_path=self.figures_dir / "no_noise_basic" / f"{shape}_comparison.svg",
            )
            import matplotlib.pyplot as plt
            plt.close(fig)
            all_curves = np.vstack(list(curve_dict.values()))
            xmin, xmax = np.percentile(all_curves[:, 0], [35, 65])
            ymin, ymax = np.percentile(all_curves[:, 1], [35, 65])
            fig = plot_local_zoom(
                points,
                curve_dict,
                (xmin, xmax, ymin, ymax),
                title=f"Local zoom | {shape}",
                closed=closed,
                save_path=self.figures_dir / "no_noise_basic" / f"{shape}_local_zoom.svg",
            )
            plt.close(fig)
        df = pd.DataFrame(rows)
        save_dataframe_csv(df, self.results_dir / "no_noise_basic.csv")
        return df

    def run_interpolation_vs_approximation_comparison(self):
        rows = []
        for shape in ["s_curve", "cubic_poly", "circle", "wavy_circle"]:
            for noise_sigma in [0.0, 0.02]:
                points, closed = generate_synthetic_curve(shape, n_points=48, sampling="nonuniform", noise_sigma=noise_sigma, random_state=2)
                ref, _ = generate_synthetic_curve(shape, n_points=self.reference_samples, sampling="uniform", noise_sigma=0.0, random_state=2)
                for method in ALL_METHODS:
                    curve, metric = self._fit_and_evaluate(method, points, ref, "chord", closed, len(points))
                    row = {
                        "experiment": "interp_vs_approx",
                        "shape": shape,
                        "closed": closed,
                        "method": method,
                        "family": "interpolation" if method in INTERP_METHODS else "approximation",
                        "parameterization": "chord",
                        "n_points": len(points),
                        "noise_sigma": noise_sigma,
                    }
                    row.update(metric)
                    rows.append(row)
        df = pd.DataFrame(rows)
        save_dataframe_csv(df, self.results_dir / "interp_vs_approx.csv")
        return df

    def run_parameterization_comparison(self):
        rows = []
        for shape in ["sine_modulated", "ellipse", "cardioid"]:
            points, closed = generate_synthetic_curve(shape, n_points=42, sampling="nonuniform", noise_sigma=0.0, random_state=3)
            ref, _ = generate_synthetic_curve(shape, n_points=self.reference_samples, sampling="uniform", noise_sigma=0.0, random_state=3)
            curve_dict = {}
            for parameterization in PARAMS:
                curve, metric = self._fit_and_evaluate("cubic_spline", points, ref, parameterization, closed, len(points))
                curve_dict[parameterization] = curve
                row = {
                    "experiment": "parameterization_comparison",
                    "shape": shape,
                    "closed": closed,
                    "method": "cubic_spline",
                    "parameterization": parameterization,
                    "n_points": len(points),
                    "noise_sigma": 0.0,
                }
                row.update(metric)
                rows.append(row)
            fig = plot_parameterization_comparison(
                points,
                curve_dict,
                title=f"Parameterization comparison | {shape}",
                closed=closed,
                save_path=self.figures_dir / "parameterization" / f"{shape}_parameterization_comparison.svg",
            )
            import matplotlib.pyplot as plt
            plt.close(fig)
        df = pd.DataFrame(rows)
        save_dataframe_csv(df, self.results_dir / "parameterization_comparison.csv")
        return df

    def run_node_count_experiment(self):
        rows = []
        for shape in ["s_curve", "wavy_circle"]:
            for n_points in [12, 20, 32, 48, 72, 100]:
                points, closed = generate_synthetic_curve(shape, n_points=n_points, sampling="nonuniform", noise_sigma=0.0, random_state=n_points)
                ref, _ = generate_synthetic_curve(shape, n_points=self.reference_samples, sampling="uniform", noise_sigma=0.0, random_state=n_points)
                for method in ALL_METHODS:
                    curve, metric = self._fit_and_evaluate(method, points, ref, "chord", closed, n_points)
                    row = {
                        "experiment": "node_count",
                        "shape": shape,
                        "closed": closed,
                        "method": method,
                        "parameterization": "chord",
                        "n_points": n_points,
                        "noise_sigma": 0.0,
                    }
                    row.update(metric)
                    rows.append(row)
        df = pd.DataFrame(rows)
        save_dataframe_csv(df, self.results_dir / "node_count.csv")
        for shape in df["shape"].unique():
            sub = df[df["shape"] == shape]
            import matplotlib.pyplot as plt
            fig = plot_line_chart(
                sub,
                x="n_points",
                y="chamfer",
                hue="method",
                title=f"Node count vs error | {shape}",
                save_path=self.figures_dir / "node_count" / f"{shape}_node_count_vs_error.svg",
            )
            plt.close(fig)
            fig = plot_line_chart(
                sub,
                x="n_points",
                y="runtime_sec",
                hue="method",
                title=f"Node count vs runtime | {shape}",
                save_path=self.figures_dir / "node_count" / f"{shape}_node_count_vs_runtime.svg",
            )
            plt.close(fig)
        return df

    def run_noise_robustness_experiment(self):
        rows = []
        for shape in ["s_curve", "circle"]:
            for seed in range(5):
                for noise_sigma in [0.0, 0.01, 0.02, 0.04, 0.06]:
                    points, closed = generate_synthetic_curve(shape, n_points=40, sampling="nonuniform", noise_sigma=noise_sigma, random_state=100 + seed)
                    ref, _ = generate_synthetic_curve(shape, n_points=self.reference_samples, sampling="uniform", noise_sigma=0.0, random_state=100 + seed)
                    for method in ALL_METHODS:
                        curve, metric = self._fit_and_evaluate(method, points, ref, "chord", closed, len(points))
                        row = {
                            "experiment": "noise_robustness",
                            "shape": shape,
                            "closed": closed,
                            "method": method,
                            "parameterization": "chord",
                            "n_points": len(points),
                            "noise_sigma": noise_sigma,
                            "trial": seed,
                        }
                        row.update(metric)
                        rows.append(row)
        df = pd.DataFrame(rows)
        save_dataframe_csv(df, self.results_dir / "noise_robustness.csv")
        import matplotlib.pyplot as plt
        for shape in df["shape"].unique():
            sub = df[df["shape"] == shape]
            mean_df = sub.groupby(["noise_sigma", "method"], as_index=False)["chamfer"].mean()
            fig = plot_line_chart(
                mean_df,
                x="noise_sigma",
                y="chamfer",
                hue="method",
                title=f"Noise vs error | {shape}",
                save_path=self.figures_dir / "noise_robustness" / f"{shape}_noise_vs_error.svg",
            )
            plt.close(fig)
            box_df = sub[sub["noise_sigma"] > 0].assign(method_noise=lambda x: x["method"] + "_σ=" + x["noise_sigma"].astype(str))
            fig = plot_boxplot(
                box_df,
                x="method_noise",
                y="chamfer",
                title=f"Error boxplot | {shape}",
                save_path=self.figures_dir / "noise_robustness" / f"{shape}_error_boxplot.svg",
            )
            plt.close(fig)
            fig = plot_violinplot(
                box_df,
                x="method_noise",
                y="chamfer",
                title=f"Error violin plot | {shape}",
                save_path=self.figures_dir / "noise_robustness" / f"{shape}_error_violin.svg",
            )
            plt.close(fig)
        return df

    def run_all(self):
        return {
            "no_noise_basic": self.run_no_noise_basic_reconstruction(),
            "interp_vs_approx": self.run_interpolation_vs_approximation_comparison(),
            "parameterization_comparison": self.run_parameterization_comparison(),
            "node_count": self.run_node_count_experiment(),
            "noise_robustness": self.run_noise_robustness_experiment(),
        }