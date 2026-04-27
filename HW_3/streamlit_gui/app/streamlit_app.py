import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from curvefit.core.data import normalize_points
from curvefit.core.interpolation import fit_interpolant
from curvefit.core.approximation import fit_approximation
from curvefit.core.fourier import FourierApproximator
from curvefit.core.metrics import evaluate_reconstruction
from curvefit.data.synthetic import generate_synthetic_curve, list_available_shapes
from curvefit.viz.plots import plot_points_and_curve, plot_spectrum, plot_epicycle_snapshot


METHOD_LABELS = {
    "cubic_spline": "Cubic Spline Interpolation",
    "bspline_interp": "Parametric B-Spline Interpolation",
    "poly_ls": "Least-Squares Polynomial Approximation",
    "bspline_ls": "Least-Squares B-Spline Approximation",
}


def _load_uploaded_points(file):
    suffix = Path(file.name).suffix.lower()
    if suffix == ".csv":
        return np.loadtxt(io.BytesIO(file.getvalue()), delimiter=",")
    if suffix == ".txt":
        return np.loadtxt(io.BytesIO(file.getvalue()))
    if suffix == ".npy":
        return np.load(io.BytesIO(file.getvalue()))
    raise ValueError("Only .csv, .txt, and .npy files are supported.")


def _fit_curve(method, points, parameterization, closed, degree, smoothing):
    if method in {"cubic_spline", "bspline_interp"}:
        kwargs = {"degree": min(5, max(1, degree))} if method == "bspline_interp" else {}
        return fit_interpolant(method, points, parameterization=parameterization, closed=closed, **kwargs)
    kwargs = {"degree": degree} if method == "poly_ls" else {"degree": min(5, max(1, degree)), "smoothing": smoothing}
    return fit_approximation(method, points, parameterization=parameterization, closed=closed, **kwargs)


def _fourier_process_steps(K):
    K = int(max(0, K))
    candidates = [0, 1, max(1, K // 4), max(1, K // 2), K]
    return sorted({k for k in candidates if 0 <= k <= K})


def main():
    st.set_page_config(page_title="Curve Fitting GUI", layout="wide")
    st.title("Planar Curve Fitting and Fourier Visualization")

    with st.sidebar:
        data_mode = st.radio("Data Source", ["Synthetic Data", "Upload File"])
        normalize = st.checkbox("Normalize Point Set", value=False)

        if data_mode == "Synthetic Data":
            shape = st.selectbox("Curve Shape", list_available_shapes(), index=0)
            n_points = st.slider("Number of Sample Points", 8, 200, 40)
            sampling = st.selectbox("Sampling Mode", ["uniform", "nonuniform"], index=0)
            noise_sigma = st.slider("Noise Level", 0.0, 0.10, 0.0, 0.005)
            points, closed = generate_synthetic_curve(
                shape,
                n_points=n_points,
                sampling=sampling,
                noise_sigma=noise_sigma,
                random_state=42,
            )
        else:
            upload = st.file_uploader("Upload Point File", type=["csv", "txt", "npy"])
            closed = st.checkbox("Closed Curve", value=False)
            points = _load_uploaded_points(upload) if upload is not None else None

        parameterization = st.selectbox(
            "Parameterization",
            ["uniform", "chord", "centripetal", "foley_nielsen"],
            index=1,
        )
        method = st.selectbox(
            "Method",
            ["cubic_spline", "bspline_interp", "poly_ls", "bspline_ls"],
            format_func=lambda x: METHOD_LABELS.get(x, x),
            index=0,
        )
        degree = st.slider("Polynomial / Spline Degree Parameter", 1, 10, 5)
        smoothing = st.slider("Smoothing Parameter", 0.0, 0.20, 0.0, 0.005)
        n_curve_samples = st.slider("Number of Reconstructed Curve Samples", 100, 1500, 500, 50)
        use_fourier = st.checkbox("Enable Fourier Module for Closed Curves", value=False)
        K = st.slider("Number of Harmonics K", 1, 60, 12)
        fourier_resample = st.slider("Internal Fourier Resampling Size", 128, 2048, 512, 64)

    if points is None:
        st.info("Please upload a point file or switch to synthetic data.")
        return

    points = np.asarray(points, dtype=float)
    if normalize:
        points, _meta = normalize_points(points)

    tab1, tab2, tab3 = st.tabs(["Reconstruction", "Fourier", "Raw Data"])

    with tab3:
        fig = plot_points_and_curve(points, points, title="Input Points", closed=closed)
        st.pyplot(fig)
        plt.close(fig)

    with tab1:
        try:
            model = _fit_curve(method, points, parameterization, closed, degree, smoothing)
            curve = model.sample(n_curve_samples)
            metrics = evaluate_reconstruction(points, curve, curve, closed=closed, runtime_sec=0.0).to_dict()
            fig = plot_points_and_curve(
                points,
                curve,
                title=f"{METHOD_LABELS.get(method, method)} | {parameterization}",
                closed=closed,
            )
            st.pyplot(fig)
            plt.close(fig)
            st.dataframe(pd.DataFrame([metrics]))
        except Exception as e:
            st.error(f"Main reconstruction failed: {e}")

    with tab2:
        if not use_fourier:
            st.info("Enable the Fourier module from the sidebar to display this tab.")
        elif not closed:
            st.warning("The Fourier module is only available for closed curves.")
        else:
            try:
                approximator = FourierApproximator(points, n_resample=fourier_resample)

                curve = approximator.reconstruct(k=K, n_samples=n_curve_samples)
                fig = plot_points_and_curve(points, curve, title=f"Fourier Reconstruction | K={K}", closed=True)
                st.pyplot(fig)
                plt.close(fig)

                st.subheader("Fourier reconstruction process")
                st.caption("Increasing K adds more Fourier harmonics, so the curve changes from a coarse low-frequency outline to a detailed reconstruction.")
                k_steps = _fourier_process_steps(K)
                cols = st.columns(len(k_steps))
                for col, k_step in zip(cols, k_steps):
                    step_curve = approximator.reconstruct(k=k_step, n_samples=n_curve_samples)
                    step_fig = plot_points_and_curve(points, step_curve, title=f"K={k_step}", closed=True)
                    col.pyplot(step_fig)
                    plt.close(step_fig)

                freqs, amps = approximator.spectrum()
                fig = plot_spectrum(freqs, amps, title="Fourier Spectrum")
                st.pyplot(fig)
                plt.close(fig)

                t_show = st.slider("Epicycle Time t", 0.0, 1.0, 0.2, 0.02, key="t_show")
                chain = approximator.epicycle_chain(t_show, k=K)
                n_trace = max(2, int(max(t_show, 1e-12) * n_curve_samples))
                traced = approximator.partial_trace(k=K, t_end=t_show, n_samples=n_trace)
                fig = plot_epicycle_snapshot(
                    chain,
                    traced_curve=traced,
                    title=f"Epicycle Snapshot | t={t_show:.2f}, K={K}",
                )
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Fourier module failed: {e}")


if __name__ == "__main__":
    main()
