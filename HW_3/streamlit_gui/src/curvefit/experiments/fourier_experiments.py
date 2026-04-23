from dataclasses import dataclass
import pandas as pd
from curvefit.config import ANIMATIONS_DIR, FIGURES_DIR, RESULTS_DIR
from curvefit.core.fourier import FourierApproximator
from curvefit.core.metrics import evaluate_reconstruction
from curvefit.data.synthetic import generate_synthetic_curve
from curvefit.utils.io_utils import ensure_dir, save_dataframe_csv
from curvefit.viz.plots import animate_epicycle, plot_line_chart, plot_points_and_curve, plot_spectrum, save_epicycle_keyframes

@dataclass
class FourierExperimentRunner:
    results_dir: str = RESULTS_DIR / "fourier"
    figures_dir: str = FIGURES_DIR / "fourier"
    animations_dir: str = ANIMATIONS_DIR / "fourier"
    reference_samples: int = 2000
    curve_samples: int = 600
    def __post_init__(self):
        ensure_dir(self.results_dir); ensure_dir(self.figures_dir); ensure_dir(self.animations_dir)
    def run_k_comparison(self):
        rows = []
        for shape in ["circle", "ellipse", "cardioid", "rose", "wavy_circle"]:
            points, closed = generate_synthetic_curve(shape, n_points=80, sampling="nonuniform", noise_sigma=0.0, random_state=10)
            ref, _ = generate_synthetic_curve(shape, n_points=self.reference_samples, sampling="uniform", noise_sigma=0.0, random_state=10)
            approximator = FourierApproximator(points, n_resample=512)
            for k in [1,2,3,5,8,12,20,30]:
                curve = approximator.reconstruct(k=k, n_samples=self.curve_samples)
                metric = evaluate_reconstruction(points, curve, ref, closed=True, runtime_sec=0.0).to_dict()
                row = {"shape": shape, "K": k}; row.update(metric); rows.append(row)
                import matplotlib.pyplot as plt
                fig = plot_points_and_curve(points, curve, title=f"{shape} | Fourier K={k}", closed=True, save_path=self.figures_dir / "k_comparison" / f"{shape}_K{k}.png"); plt.close(fig)
            freqs, amps = approximator.spectrum()
            import matplotlib.pyplot as plt
            fig = plot_spectrum(freqs, amps, title=f"Spectrum | {shape}", save_path=self.figures_dir / "spectrum" / f"{shape}_spectrum.png"); plt.close(fig)
            save_epicycle_keyframes(approximator, k=12, n_frames=6, out_dir=self.figures_dir / "epicycle_keyframes" / shape, prefix=f"{shape}_epicycle")
            animate_epicycle(approximator, k=12, n_frames=90, save_path=self.animations_dir / f"{shape}_epicycle.gif")
        df = pd.DataFrame(rows); save_dataframe_csv(df, self.results_dir / "fourier_k_comparison.csv")
        import matplotlib.pyplot as plt
        for shape in df["shape"].unique():
            sub = df[df["shape"] == shape]
            fig = plot_line_chart(sub, x="K", y="chamfer", hue="shape", title=f"K vs error | {shape}", save_path=self.figures_dir / "summary" / f"{shape}_K_vs_error.png"); plt.close(fig)
        return df
    def run_all(self):
        return {"fourier_k_comparison": self.run_k_comparison()}
