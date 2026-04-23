import sys
from pathlib import Path
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from curvefit.config import FIGURES_DIR, RESULTS_DIR
from curvefit.utils.io_utils import ensure_dir
from curvefit.viz.plots import plot_boxplot, plot_violinplot, plot_line_chart
def main():
    out_dir = ensure_dir(FIGURES_DIR / "paper")
    node_csv = RESULTS_DIR / "main" / "node_count.csv"
    noise_csv = RESULTS_DIR / "main" / "noise_robustness.csv"
    cmp_csv = RESULTS_DIR / "main" / "interp_vs_approx.csv"
    import matplotlib.pyplot as plt
    if node_csv.exists():
        df = pd.read_csv(node_csv)
        for shape in df["shape"].unique():
            sub = df[df["shape"] == shape]
            fig = plot_line_chart(sub, x="n_points", y="chamfer", hue="method", title=f"Node count vs error | {shape}", save_path=out_dir / f"{shape}_node_error.svg"); plt.close(fig)
            fig = plot_line_chart(sub, x="n_points", y="runtime_sec", hue="method", title=f"Node count vs runtime | {shape}", save_path=out_dir / f"{shape}_node_time.svg"); plt.close(fig)
    if noise_csv.exists():
        df = pd.read_csv(noise_csv)
        mean_df = df.groupby(["shape", "noise_sigma", "method"], as_index=False)["chamfer"].mean()
        for shape in mean_df["shape"].unique():
            sub = mean_df[mean_df["shape"] == shape]
            fig = plot_line_chart(sub, x="noise_sigma", y="chamfer", hue="method", title=f"Noise vs error | {shape}", save_path=out_dir / f"{shape}_noise_error.svg"); plt.close(fig)
    if cmp_csv.exists():
        df = pd.read_csv(cmp_csv)
        fig = plot_boxplot(df, x="method", y="chamfer", title="Method comparison boxplot", save_path=out_dir / "method_boxplot.svg"); plt.close(fig)
        fig = plot_violinplot(df, x="method", y="chamfer", title="Method comparison violin plot", save_path=out_dir / "method_violinplot.svg"); plt.close(fig)
if __name__ == "__main__":
    main()