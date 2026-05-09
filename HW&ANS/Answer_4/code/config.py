"""Project-wide configuration for SIR periodic-outbreak experiments."""

from __future__ import annotations

from pathlib import Path
import random

import numpy as np


# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
FIG_DIR = PROJECT_ROOT / "figs"
RESULT_DIR = PROJECT_ROOT / "results"


# =============================================================================
# Reproducibility
# =============================================================================
RANDOM_SEED = 20260508


def set_random_seed(seed: int = RANDOM_SEED) -> np.random.Generator:
    """Set Python and NumPy random seeds and return a NumPy generator."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


# =============================================================================
# Epidemiological parameters, time unit: year
# =============================================================================
GAMMA = 52.0                         # one-week mean infectious period
MU = 1.0 / 70.0                      # one 70-year life expectancy
R0_VALUES = [2.0, 8.0, 15.0]
SEASONAL_ALPHA_VALUES = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.60]

I0 = 1.0e-4
R_INIT = 0.0
S_INIT = 1.0 - I0 - R_INIT
DEFAULT_Y0 = np.array([S_INIT, I0, R_INIT], dtype=float)

DAYS_PER_YEAR = 365.0


# =============================================================================
# Numerical settings
# =============================================================================
DEFAULT_DT = 1.0 / 730.0        # about 0.5 day, used for short/stiff waves
LONG_DT = 1.0 / 365.0           # daily, used for long seasonal scans
SAVE_DT_DAILY = 1.0 / 365.0
SAVE_DT_WEEKLY = 7.0 / 365.0

POPULATION_TOL = 1.0e-6
NEGATIVE_TOL = 1.0e-9


# =============================================================================
# Deterministic experiment settings
# =============================================================================
BASIC_SINGLE_R0 = 8.0
BASIC_T_END = 1.0
BASIC_COMPARE_T_END = 1.0
BASIC_EARLY_ZOOM_DAYS = 90.0

GAMMA_COMPARISON_R0 = 8.0
GAMMA_COMPARISON_VALUES = [26.0, 52.0, 104.0]
GAMMA_COMPARISON_T_END = 0.50

THRESHOLD_R0 = 8.0
THRESHOLD_T_END = 2.0
THRESHOLD_S0_VALUES = np.linspace(0.02, 0.999, 70)

VITAL_R0 = 15.0
VITAL_T_END = 40.0
BASIC_VS_VITAL_T_END = 10.0
DEMOGRAPHIC_LATE_START = 1.0
DEMOGRAPHIC_R0_VALUES = [2.0, 8.0, 15.0]
DEMOGRAPHIC_R0_T_END_FULL = 40.0
DEMOGRAPHIC_R0_T_END_QUICK = 20.0

SEASONAL_R0 = 15.0
SEASONAL_T_END = 60.0
SEASONAL_PLOT_START = 40.0
SEASONAL_PLOT_END = 50.0

FIG07_ALPHA = 0.30
FIG07_T_START = 42.0
FIG07_T_END = 46.0

ALPHA_SCAN_MIN = 0.0
ALPHA_SCAN_MAX = 0.8
ALPHA_SCAN_N_FULL = 61
ALPHA_SCAN_N_QUICK = 17
ALPHA_SCAN_T_END_FULL = 120.0
ALPHA_SCAN_TRANSIENT_FULL = 60.0
ALPHA_SCAN_SAMPLE_START_FULL = 61.0
ALPHA_SCAN_SAMPLE_END_FULL = 120.0
ALPHA_SCAN_T_END_QUICK = 60.0
ALPHA_SCAN_TRANSIENT_QUICK = 30.0
ALPHA_SCAN_SAMPLE_START_QUICK = 31.0
ALPHA_SCAN_SAMPLE_END_QUICK = 60.0
ALPHA_SCAN_DT = LONG_DT
PEAK_MIN_DISTANCE_YEARS = 0.25


# =============================================================================
# Stochastic experiment settings
# =============================================================================
# The stochastic trajectory plot still uses a clearly supercritical outbreak so
# that deterministic and stochastic paths can be compared visually.
STOCHASTIC_TRAJECTORY_R0 = 8.0
STOCHASTIC_TRAJECTORY_N = 10_000
STOCHASTIC_TRAJECTORY_I0 = 20
STOCHASTIC_TRAJECTORY_REPEATS_FULL = 30
STOCHASTIC_TRAJECTORY_REPEATS_QUICK = 5
STOCHASTIC_TRAJECTORY_T_MAX_FULL = 8.0
STOCHASTIC_TRAJECTORY_T_MAX_QUICK = 2.0
STOCHASTIC_TRAJECTORY_SAMPLE_DT = 3.0 / 365.0
STOCHASTIC_TRAJECTORY_EARLY_DAYS = 120.0

# The extinction experiments initialize the population close to the epidemic
# threshold: S0/N is only slightly larger than 1/R0. This makes stochastic
# establishment versus early fade-out visible instead of producing all-ones
# final-extinction probabilities.
STOCHASTIC_EXTINCTION_R0 = 8.0
STOCHASTIC_CRITICAL_S_MULTIPLIER = 1.20
STOCHASTIC_OUTBREAK_THRESHOLD = 0.005
STOCHASTIC_N_VALUES_FULL = [1_000, 3_000, 10_000, 30_000]
STOCHASTIC_N_VALUES_QUICK = [1_000, 10_000]
STOCHASTIC_I0_VALUES_FULL = [1, 2, 5, 10]
STOCHASTIC_I0_VALUES_QUICK = [1, 5]
STOCHASTIC_REPEATS_FULL = 100
STOCHASTIC_REPEATS_QUICK = 12
STOCHASTIC_T_MAX_FULL = 2.0
STOCHASTIC_T_MAX_QUICK = 1.0

STOCHASTIC_ALPHA_VALUES_FULL = [0.0, 0.1, 0.3, 0.6]
STOCHASTIC_ALPHA_VALUES_QUICK = [0.0, 0.3, 0.6]
STOCHASTIC_ALPHA_N = 10_000
STOCHASTIC_ALPHA_I0 = 5
STOCHASTIC_PHASE_ALPHA_VALUES = [0.1, 0.3, 0.6]
STOCHASTIC_PHASE_VALUES = [0.0, 0.5 * np.pi, np.pi]
STOCHASTIC_PHASE_REPEATS_FULL = 50
STOCHASTIC_PHASE_REPEATS_QUICK = 12


# =============================================================================
# Figure settings and required names
# =============================================================================
PNG_DPI = 300
FIG01_NAME = "fig01_basic_sir"
FIG02_NAME = "fig02_R0_comparison"
FIG02B_NAME = "fig02b_gamma_comparison"
FIG03_NAME = "fig03_threshold_s0"
FIG04_NAME = "fig04_demographic_sir"
FIG04B_NAME = "fig04b_demographic_R0_comparison"
FIG05_NAME = "fig05_basic_vs_demographic"
FIG06_NAME = "fig06_seasonal_alpha_curves"
FIG07_NAME = "fig07_beta_infection_phase"
FIG08_NAME = "fig08_alpha_bifurcation"
FIG09_NAME = "fig09_peak_statistics"
FIG10_NAME = "fig10_pattern_classification"
FIG11_NAME = "fig11_stochastic_vs_deterministic"
FIG12_NAME = "fig12_extinction_heatmap"
FIG13_NAME = "fig13_extinction_time_distribution"
FIG14_NAME = "fig14_alpha_extinction_probability"
FIG14B_NAME = "fig14b_phase_sensitivity"
