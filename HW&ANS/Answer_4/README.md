# SIR Periodic Outbreak Modeling and Stochastic Epidemic Simulation

## 1. Project Overview

This project studies epidemic outbreak dynamics under several SIR-type models and implements a complete, modular, and reproducible Python framework for the fourth mathematical modeling assignment. The work is organized around five major components.

First, the project implements the **basic SIR model** and uses it to examine the mechanism of a single epidemic outbreak. The experiments compare different basic reproduction numbers \(R_0\), recovery rates \(\gamma\), and initial susceptible proportions, with attention to the infection peak, peak time, final susceptible level, and epidemic threshold condition.

Second, the project extends the basic model to a **demographic SIR model with births and deaths**. This part studies how demographic replenishment of susceptible individuals can generate recurrent epidemic peaks after the first outbreak, and compares the long-term behavior with the non-demographic SIR model.

Third, the project introduces **seasonal transmission forcing** by allowing the transmission rate to vary periodically in time. The corresponding experiments investigate how the forcing amplitude \(\alpha\) changes the long-term infection trajectory, the phase relationship between transmission and infection, and the transition from nearly annual oscillations to more complex multi-year patterns.

Fourth, the project includes a **seasonal parameter scan** over the forcing amplitude. The scan produces Poincare-type long-term samples, peak-statistics summaries, and a qualitative pattern-classification figure, which together describe the periodic outbreak structure under different forcing strengths.

Finally, the project implements **Gillespie stochastic simulation** for finite-population epidemic dynamics. This part compares stochastic trajectories with deterministic solutions and studies early fade-out probability under different population sizes, initial infection numbers, seasonal forcing amplitudes, and seasonal phases.

All experiment outputs are saved automatically as CSV files under `results/`, while the corresponding figures are written to `figs/` in both PNG and SVG formats. The SVG figures are intended for direct use in the LaTeX report.

---

## 2. Directory Structure

```text
HW4_SIR_Periodic_Outbreak/
├─ code/
│  ├─ config.py                         # global paths, parameters, random seed, and file names
│  ├─ sir_models.py                     # basic, demographic, and seasonal SIR right-hand sides
│  ├─ solvers.py                        # RK4 and solve_ivp-based deterministic ODE solvers
│  ├─ metrics.py                        # peak, threshold, extinction, and summary metrics
│  ├─ gillespie.py                      # Gillespie stochastic simulation algorithm
│  ├─ plotting_style.py                 # shared plotting style settings
│  ├─ plotting.py                       # figure-generation functions
│  ├─ experiments_deterministic.py      # deterministic experiment workflow
│  ├─ experiments_stochastic.py         # stochastic experiment workflow
│  ├─ main_deterministic.py             # deterministic-only entry point
│  ├─ main_stochastic.py                # stochastic-only entry point
│  └─ main_all.py                       # one-click entry point for all experiments
├─ figs/
│  ├─ fig01_basic_sir.png / .svg
│  ├─ fig02_R0_comparison.png / .svg
│  ├─ fig02b_gamma_comparison.png / .svg
│  ├─ fig03_threshold_s0.png / .svg
│  ├─ fig04_demographic_sir.png / .svg
│  ├─ fig04b_demographic_R0_comparison.png / .svg
│  ├─ fig05_basic_vs_demographic.png / .svg
│  ├─ fig06_seasonal_alpha_curves.png / .svg
│  ├─ fig07_beta_infection_phase.png / .svg
│  ├─ fig08_alpha_bifurcation.png / .svg
│  ├─ fig09_peak_statistics.png / .svg
│  ├─ fig10_pattern_classification.png / .svg
│  ├─ fig11_stochastic_vs_deterministic.png / .svg
│  ├─ fig12_extinction_heatmap.png / .svg
│  ├─ fig13_extinction_time_distribution.png / .svg
│  ├─ fig14_alpha_extinction_probability.png / .svg
│  └─ fig14b_phase_sensitivity.png / .svg
├─ results/
│  ├─ basic_sir_summary.csv
│  ├─ basic_sir_timeseries.csv
│  ├─ R0_comparison_summary.csv
│  ├─ R0_comparison_timeseries.csv
│  ├─ gamma_comparison_summary.csv
│  ├─ gamma_comparison_timeseries.csv
│  ├─ threshold_s0_summary.csv
│  ├─ demographic_summary.csv
│  ├─ demographic_timeseries.csv
│  ├─ demographic_late_peak_summary.csv
│  ├─ demographic_R0_comparison_summary.csv
│  ├─ demographic_R0_comparison_timeseries.csv
│  ├─ basic_vs_demographic_summary.csv
│  ├─ basic_vs_demographic_timeseries.csv
│  ├─ seasonal_alpha_curves_summary.csv
│  ├─ seasonal_alpha_curves_timeseries.csv
│  ├─ beta_infection_phase_timeseries.csv
│  ├─ alpha_scan_poincare.csv
│  ├─ seasonal_peak_events.csv
│  ├─ seasonal_peak_summary.csv
│  ├─ seasonal_pattern_classification.csv
│  ├─ stochastic_deterministic_reference.csv
│  ├─ stochastic_trajectory_summary.csv
│  ├─ stochastic_trajectory_timeseries.csv
│  ├─ stochastic_extinction_summary.csv
│  ├─ stochastic_extinction_detail.csv
│  ├─ stochastic_alpha_extinction.csv
│  ├─ stochastic_alpha_extinction_detail.csv
│  └─ stochastic_phase_sensitivity.csv
├─ environment.yml                      # conda environment definition
├─ run_all.bat                          # Windows one-click launcher
└─ README.md                            # project description and running instructions
```

---

## 3. How to Run the Project

### 3.1 Run by BAT file (recommended on Windows)

The project provides a Windows BAT launcher. It can be executed directly by double-clicking it or from the command line:

```text
run_all.bat
```

By default, the script runs the full experiment pipeline. It checks whether the conda environment `hw4-sir-periodic` already exists; if not, it creates the environment from `environment.yml` and then runs the full pipeline.

To run the quick test mode, use:

```bat
run_all.bat quick
```

Notes:
- **A black console window is normal and indicates that the program is running.**
- Full-mode experiments take longer than quick-mode experiments because the stochastic simulations and seasonal scans are larger.
- Figures are saved under `figs/`, and CSV files are saved under `results/`.

---

### 3.2 Run from the terminal

Create and activate the conda environment first:

```bash
conda env create -f environment.yml
conda activate hw4-sir-periodic
```

Then run the following commands in the project root directory.

Run all experiments:

```bash
python code/main_all.py --mode full --method rk4
```

Run a quick test:

```bash
python code/main_all.py --mode quick --method rk4
```

Run only deterministic experiments:

```bash
python code/main_deterministic.py --mode full --method rk4
```

Run only stochastic Gillespie experiments:

```bash
python code/main_stochastic.py --mode full
```

The option `--method rk4` uses the fourth-order Runge--Kutta solver. The deterministic scripts also support SciPy-based ODE solving by using one of the following method names:

```bash
--method solve_ivp
--method ivp
--method scipy
```

---

## 4. Output Locations

- Deterministic and stochastic figures: `figs/`
- CSV time-series data: `results/*_timeseries.csv`
- CSV summary tables: `results/*_summary.csv`
- Detailed stochastic repeated-trial records: `results/*_detail.csv`
- Seasonal parameter-scan samples: `results/alpha_scan_poincare.csv`
- Seasonal peak and pattern results: `results/seasonal_peak_*.csv`, `results/seasonal_pattern_classification.csv`

Each figure is saved in both `.png` and `.svg` formats. The `.png` files are convenient for quick preview, while the `.svg` files are suitable for insertion into the LaTeX report.

---

## 5. Recommended Reproduction Order

A typical reproduction workflow is:

```text
1. conda env create -f environment.yml
2. conda activate hw4-sir-periodic
3. python code/main_all.py --mode quick --method rk4
4. python code/main_all.py --mode full --method rk4
```

The quick mode is used to verify that the environment and scripts work correctly. The full mode regenerates the formal figures and CSV tables used in the report. Running the scripts again will overwrite files with the same names in `figs/` and `results/`, so these two directories should be backed up first if the original outputs need to be preserved.
