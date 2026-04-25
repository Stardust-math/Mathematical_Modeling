# Planar Curve Reconstruction, Approximation, and Periodic Curve Synthesis

## 1. Project Overview

This project studies planar curve recovery from sampled two-dimensional point sets and implements a complete, modular, and reproducible Python framework for experiments. The work is organized around three major components.

First, the mandatory part focuses on **curve reconstruction and approximation** for both open and closed curves. The framework includes:
- four parameterization strategies: uniform, chord-length, centripetal, and Foley--Nielsen parameterization;
- two interpolation methods: cubic spline interpolation and parametric B-spline interpolation;
- two approximation methods: least-squares polynomial approximation and least-squares B-spline approximation;
- several evaluation metrics, including average point-to-curve distance, Chamfer distance, Hausdorff distance, smoothness, and runtime.

Second, the project provides **batch experiment scripts and paper-style figure generation**. Experimental results are saved automatically as CSV files, while the corresponding plots are written to the figures directory for later use in the report.

Third, for periodic closed curves, the project implements **discrete Fourier / trigonometric reconstruction and epicycle visualization**. This part supports different harmonic numbers \(K\), coefficient spectrum analysis, and circular-motion-based visual synthesis of the target contour.

In addition, the project includes a **Streamlit GUI** that allows the user to upload point sets, choose the curve type and fitting method, and inspect the reconstruction result interactively.

---

## 2. Directory Structure

```text
curvefit/
├─ app/
│  └─ streamlit_app.py                 # Streamlit GUI entry
├─ outputs/
│  ├─ animations/
│  │  └─ fourier/                      # epicycle animations for periodic curves
│  ├─ figures/
│  │  ├─ fourier/                      # periodic-curve reconstructions, spectra, key frames
│  │  │  ├─ epicycle_keyframes/         # selected epicycle frames for each test curve
│  │  │  ├─ k_comparison/              # Fourier reconstruction results under different K
│  │  │  ├─ spectrum/                  # coefficient spectrum figures
│  │  │  └─ summary/                   # K-vs-error summary figures
│  │  ├─ main/                         # figures from the main experiments
│  │  │  ├─ no_noise_basic/            # basic interpolation/approximation comparisons
│  │  │  ├─ parameterization/          # parameterization comparison figures
│  │  │  ├─ node_count/                # node-count sensitivity figures
│  │  │  └─ noise_robustness/          # noise robustness and distribution figures
│  │  └─ paper/                        # paper-style summary figures
│  └─ results/
│     ├─ fourier/
│     │  └─ fourier_k_comparison.csv   # periodic-curve experiment results
│     └─ main/
│        ├─ interp_vs_approx.csv
│        ├─ node_count.csv
│        ├─ noise_robustness.csv
│        ├─ no_noise_basic.csv
│        └─ parameterization_comparison.csv
├─ scripts/
│  ├─ generate_paper_figures.py        # paper-style figure generation
│  ├─ run_fourier_experiments.py       # periodic-curve experiment entry
│  ├─ run_main_experiments.py          # main experiment entry
│  └─ smoke_test.py                    # smoke test
├─ src/
│  └─ curvefit/
│     ├─ core/
│     │  ├─ approximation.py           # least-squares approximation methods
│     │  ├─ base.py                    # shared curve model interface
│     │  ├─ data.py                    # point-set and curve data utilities
│     │  ├─ fourier.py                 # Fourier reconstruction and epicycle logic
│     │  ├─ interpolation.py           # interpolation methods
│     │  ├─ metrics.py                 # evaluation metrics
│     │  ├─ parameterization.py        # parameterization methods
│     │  └─ __init__.py
│     ├─ data/
│     │  ├─ synthetic.py               # synthetic test curve generation
│     │  └─ __init__.py
│     ├─ experiments/
│     │  ├─ fourier_experiments.py     # periodic-curve experiment logic
│     │  ├─ main_experiments.py        # main experiment logic
│     │  └─ __init__.py
│     ├─ utils/
│     │  ├─ io_utils.py                # file and directory utilities
│     │  └─ __init__.py
│     ├─ viz/
│     │  ├─ plots.py                   # plotting and visualization functions
│     │  └─ __init__.py
│     ├─ config.py                     # global configuration
│     └─ __init__.py
├─ tools/
│  └─ ui_helpers.bat                   # helper bat utilities
├─ environment.yml                     # conda environment definition
├─ setup_env.bat                       # one-click environment setup
├─ run_smoke_test.bat                  # smoke test
├─ run_main_experiments.bat            # main experiments
├─ run_fourier_experiments.bat         # periodic-curve experiments
├─ generate_paper_figures.bat          # paper-figure generation
└─ run_gui.bat                         # GUI launcher
```

---

## 3. How to Run the Project

### 3.1 Run by BAT files (recommended on Windows)

The following BAT files can be executed directly by double-clicking them or from the command line:

```text
setup_env.bat
run_smoke_test.bat
run_main_experiments.bat
run_fourier_experiments.bat
generate_paper_figures.bat
run_gui.bat
```

Recommended order:

```text
1. setup_env.bat
2. run_smoke_test.bat
3. run_main_experiments.bat
4. run_fourier_experiments.bat
5. generate_paper_figures.bat
6. run_gui.bat
```

Notes:
- **A black console window is normal and indicates that the program is running.**
- The main experiments and periodic-curve experiments automatically save CSV results and plots under `outputs/results/` and `outputs/figures/`.
- The GUI will open in the browser after startup.

---

### 3.2 Run from the terminal

Create and activate the conda environment first:

```bash
conda env create -f environment.yml
conda activate curvefit_env
```

Then run the following commands in the project root directory:

```bash
python scripts/smoke_test.py
python scripts/run_main_experiments.py
python scripts/run_fourier_experiments.py
python scripts/generate_paper_figures.py
streamlit run app/streamlit_app.py
```

---

## 4. Output Locations

- Main experiment CSV files: `outputs/results/main/`
- Periodic-curve CSV files: `outputs/results/fourier/`
- Main experiment figures: `outputs/figures/main/`
- Periodic-curve figures: `outputs/figures/fourier/`
- Paper-style summary figures: `outputs/figures/paper/`
- Animation outputs: `outputs/animations/fourier/`
