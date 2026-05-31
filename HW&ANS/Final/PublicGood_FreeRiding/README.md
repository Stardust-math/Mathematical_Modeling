# PublicGood_FreeRiding

This project implements a simulation-based mathematical modeling framework for the topic **Free Riding in Dynamic Public-Good Provision**. It is designed to support an English MCM-style modeling paper rather than to estimate a particular real public-good system.

## 1. Project Goal

The project studies how decentralized individual rationality can lead to under-provision of a public good when agents privately bear contribution costs but collectively share public-good benefits. The model compares:

1. a **Nash-style individual-rational benchmark**, computed by damped marginal best-response updates; and
2. a **stage-wise social-planner benchmark**, computed by numerical optimization under the same current-state conditions.

The benchmark is deliberately described as stage-wise. It is not an infinite-horizon dynamic-programming optimum and should not be interpreted as a proof of a unique analytical Nash equilibrium.

## 2. Modeling Scope

The implemented Dynamic Stock--Pressure Free-Riding (DSPF) framework contains:

- heterogeneous agents with benefit, cost, efficiency, and pressure-sensitivity parameters;
- dynamic public-good stock with natural decay and capacity saturation;
- maintenance pressure and demand feedback;
- free-rider classification based on high benefit and low effort;
- policy mechanisms including subsidy, penalty, reputation, matching fund, threshold governance, and combined portfolio;
- sensitivity, robustness, Monte Carlo, Pareto, and response-surface experiments;
- a parameter-audit layer for the Nash-style behavioral-response weights, including both joint scaling and one-at-a-time perturbation checks;
- a runtime-profile CSV for documenting the expected cost of quick checks versus the full paper-scale workflow.

All data are synthetic. The numerical results are controlled simulation outputs, not empirical observations.

## 3. Directory Structure

```text
PublicGood_FreeRiding/
├── code/                         # Model, simulation, optimization, and plotting modules
├── configs/                      # Synthetic scenario and figure configuration files
├── data/processed/synthetic/     # Generated synthetic agents, trajectories, and diagnostics
├── figs/                         # Generated figures used or available for the paper
│   ├── paper/                    # Framework and causal-loop diagrams
│   └── synthetic/                # Numerical experiment figures
├── logs/experiment_logs/         # Created by main.py to store run summaries
├── report_assets/                # Figure and table captions / paper-use recommendations
├── results/synthetic/            # CSV outputs from experiments
├── main.py                       # Runs experiments and regenerates all figures
├── environment.yml               # Conda environment specification
└── README.md                     # This file
```

## 4. Reproducibility Settings

The main reproducibility settings are stored in `configs/synthetic_config.json`:

- random seed: `20260614`;
- execution profile: `paper`;
- number of agents in the paper profile: `35`;
- time horizon in the paper profile: `30`;
- Monte Carlo runs in the paper profile: `50`;
- Pareto samples in the paper profile: `80`;
- fine subsidy-penalty response surface in the paper profile: `13 × 13`;
- quick profile for workflow checks: `12` agents, a `12`-period horizon, `2` Monte Carlo runs, `8` Pareto samples, and a `5 × 5` 3D response surface;
- one-at-a-time behavioral-weight multipliers: `0.70, 0.85, 1.00, 1.15, 1.30` for each of the five Nash-style response weights.

Because the social-planner benchmark uses floating-point numerical optimization, exact bitwise equality may vary slightly across Python / NumPy / SciPy versions. The paper values should be treated as reproducible to the displayed rounding precision.

## 5. How to Run

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate publicgood-freeriding
```

Recommended workflow:

```bash
python main.py --profile quick --no-figures
```

This command is a fast sanity check for the full code path. The `quick` profile intentionally uses fewer agents, fewer Monte Carlo runs, fewer Pareto samples, a shorter horizon, and a coarser 3D response-surface grid, so its numerical outputs should not be used to update the paper tables.

Run the full paper-scale numerical workflow without figures:

```bash
python main.py --profile paper --no-figures
```

The Nash/social dynamic-trajectory CSV is intentionally reusable. If `results/synthetic/synthetic_nash_social_trajectories.csv` already exists, `main.py` reuses it to avoid repeatedly launching the slower stage-wise social-planner trajectory computation. To refresh that file explicitly, run:

```bash
python code/ns_trajectory_worker.py configs/_active_paper_config.json
```

Run the full paper-scale workflow with all figures regenerated:

```bash
python main.py --profile paper
```

The paper profile includes the full Monte Carlo, Pareto, robustness, one-at-a-time behavioral-weight perturbation, and fine-grid response-surface experiments. The actual elapsed times are written to `results/synthetic/synthetic_runtime_profile.csv` and summarized in `logs/experiment_logs/run_summary.json`.

To regenerate figures only after the CSV outputs already exist, use Python to call `code.generate_figures.generate_figures_from_outputs()`. The plotting implementation is kept inside the internal plotting modules; no separate user-facing figure-generation command is required for the standard workflow.

## 6. Main Output Files


The subsidy-penalty heatmap and 3D response surface are implemented as **pure two-lever scans**: subsidy and penalty are varied from the no-policy baseline, while matching, reputation, budget support, explicit threshold governance, and backlog reduction remain at their baseline values. The grid uses the same paired scenario-level exogenous noise seed as the main policy-comparison run, so the grid origin `(subsidy=0, penalty=0)` is directly comparable with the deterministic baseline policy for that scenario.

Important numerical outputs include:

- `results/synthetic/synthetic_nash_vs_social_optimum.csv`;
- `results/synthetic/synthetic_nash_social_trajectories.csv`;
- `results/synthetic/synthetic_policy_comparison.csv`;
- `results/synthetic/synthetic_sensitivity_1d.csv`;
- `results/synthetic/synthetic_sensitivity_2d_subsidy_penalty.csv`;
- `results/synthetic/synthetic_robustness_summary.csv`;
- `results/synthetic/synthetic_monte_carlo_summary.csv`;
- `results/synthetic/synthetic_pareto_front.csv`;
- `results/synthetic/synthetic_behavioral_weight_robustness.csv`;
- `results/synthetic/synthetic_behavioral_weight_oat_robustness.csv`;
- `results/synthetic/synthetic_nash_weight_scale_audit.csv`;
- `results/synthetic/synthetic_runtime_profile.csv`.

Important paper figures include:

- `figs/paper/model_framework.svg`;
- `figs/paper/causal_loop_diagram.svg`;
- `figs/synthetic/synthetic_nash_social_comparison.svg`;
- `figs/synthetic/synthetic_nash_vs_social_trajectory.svg`;
- `figs/synthetic/synthetic_policy_comparison.svg`;
- `figs/synthetic/synthetic_policy_decision_matrix.svg`;
- `figs/synthetic/synthetic_pareto_front.svg`;
- `figs/synthetic/fig_pareto_front_3d.svg`;
- `figs/synthetic/fig_policy_response_surface_3d.svg`;
- `figs/synthetic/synthetic_monte_carlo_uncertainty.svg`;
- `figs/synthetic/synthetic_monte_carlo_welfare_uncertainty_band.svg`;
- `figs/synthetic/synthetic_behavioral_weight_oat_underprovision.svg`;
- `figs/synthetic/synthetic_behavioral_weight_oat_welfare_loss.svg`.

## 7. Interpretation Boundary

The model is intended for mechanism-oriented analysis. It can support statements such as:

- under the synthetic parameter setting, Nash-style individual rationality under-provides relative to the stage-wise planner benchmark;
- policy instruments differ in welfare, cost, free-riding reduction, pressure relief, fairness, and stability;
- reputation is cost-effective in the critical-infrastructure scenario, while combined portfolios are more suitable when maintenance pressure dominates.

It should not be used to claim empirical estimates for a real community, infrastructure system, open-source project, or environmental-governance setting without additional data calibration.

## 8. Notes on Behavioral-Response Weights

The five Nash-style behavioral-response weights are not empirical estimates. They are synthetic calibration constants used to place the perceived stock-benefit channel, pressure-relief channel, targeted penalty channel, threshold-gap channel, and mild threshold-push channel on comparable effort-response scales. The project now exports two checks for these values:

- `synthetic_behavioral_weight_robustness.csv`: joint scaling of all five weights;
- `synthetic_behavioral_weight_oat_robustness.csv`: one-at-a-time perturbation of each weight while the other four remain at baseline.

These checks are intended to support cautious paper statements such as: the under-provision conclusion is stable under the tested synthetic behavioral-weight perturbations. They should not be described as empirical validation.
