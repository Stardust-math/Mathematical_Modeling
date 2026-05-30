from __future__ import annotations

import time
import gc
import matplotlib.pyplot as plt
import pandas as pd
from .config import load_json_config, path_in_project
from .dynamic_stock_model import PublicGoodDynamicModel
from .policy_mechanisms import get_policy
from .plotting import (
    plot_model_framework, plot_causal_loop_diagram, plot_scenario_dashboard,
    plot_contribution_distribution, plot_free_riding_distribution, plot_stock_and_pressure,
    plot_nash_social_comparison, plot_nash_social_trajectory, plot_welfare_loss_decomposition,
    plot_policy_comparison, plot_policy_decision_matrix, plot_heatmap, plot_sensitivity_heatmap,
    plot_phase_portrait, plot_pareto_front, plot_monte_carlo, plot_uncertainty_bands,
    plot_policy_ablation, plot_maintenance_ranking, write_caption_index,
)
from .plotting_3d import (
    plot_policy_response_surface_3d, plot_pareto_front_3d, plot_dynamic_phase_3d,
)
from .robustness_extensions import (
    plot_behavioral_robustness,
    plot_one_at_a_time_behavioral_weight_robustness,
    plot_damping_robustness,
)


def _scenario_params(config: dict, scenario: str) -> dict:
    params = dict(config['scenario_parameters'][scenario])
    params['cost_scale'] = params.get('cost_scale', 1.0)
    params['benefit_scale'] = params.get('benefit_scale', 1.0)
    return params


def _call_plot(name: str, fn, *args, **kwargs):
    t = time.time()
    print(f'[figures] start {name}', flush=True)
    plt.close('all')
    result = fn(*args, **kwargs)
    plt.close('all')
    gc.collect()
    print(f'[figures] done {name} in {time.time() - t:.2f}s', flush=True)
    return result


def generate_figures_from_outputs(config_path: str = 'configs/synthetic_config.json') -> None:
    """Regenerate all figures from already saved synthetic CSV outputs.

    Figure rendering is intentionally isolated from the numerical experiment
    loop. This keeps full-project runs reproducible and avoids Matplotlib state
    accumulation during long simulation sessions.
    """
    t0 = time.time()
    config = load_json_config(config_path)
    scenario = 'critical_infrastructure'

    agents_all = pd.read_csv(path_in_project('data/processed/synthetic/synthetic_agents.csv'))
    agents = agents_all[agents_all['scenario'] == scenario].copy()
    efforts_path = path_in_project('data/processed/synthetic/synthetic_baseline_efforts.csv')
    efforts = pd.read_csv(efforts_path) if efforts_path.exists() else pd.DataFrame()

    trajectories = pd.read_csv(path_in_project('results/synthetic/synthetic_policy_trajectories.csv'))
    baseline = trajectories[trajectories['policy'] == 'baseline'].copy()
    ns = pd.read_csv(path_in_project('results/synthetic/synthetic_nash_vs_social_optimum.csv'))
    ns_traj_df = pd.read_csv(path_in_project('results/synthetic/synthetic_nash_social_trajectories.csv'))
    summary = pd.read_csv(path_in_project('results/synthetic/synthetic_policy_comparison.csv'))
    crit_summary = summary[summary['scenario'] == scenario].copy()
    sens = pd.read_csv(path_in_project('results/synthetic/synthetic_sensitivity_1d.csv'))
    heat = pd.read_csv(path_in_project('results/synthetic/synthetic_sensitivity_2d_subsidy_penalty.csv'))
    pareto = pd.read_csv(path_in_project('results/synthetic/synthetic_pareto_front.csv'))
    response_surface_3d = pd.read_csv(path_in_project('results/synthetic/synthetic_policy_response_surface_3d.csv'))
    dynamic_phase_3d = pd.read_csv(path_in_project('results/synthetic/synthetic_dynamic_phase_3d.csv'))
    mc = pd.read_csv(path_in_project('results/synthetic/synthetic_monte_carlo_runs.csv'))
    mc_traj = pd.read_csv(path_in_project('results/synthetic/synthetic_monte_carlo_trajectories.csv'))
    behavior_one_step_path = path_in_project('results/synthetic/synthetic_behavioral_weight_robustness.csv')
    behavior_policy_path = path_in_project('results/synthetic/synthetic_behavioral_policy_ranking.csv')
    behavior_oat_path = path_in_project('results/synthetic/synthetic_behavioral_weight_oat_robustness.csv')
    damping_path = path_in_project('results/synthetic/synthetic_solver_damping_robustness.csv')
    behavior_one_step = pd.read_csv(behavior_one_step_path) if behavior_one_step_path.exists() else pd.DataFrame()
    behavior_policy = pd.read_csv(behavior_policy_path) if behavior_policy_path.exists() else pd.DataFrame()
    behavior_oat = pd.read_csv(behavior_oat_path) if behavior_oat_path.exists() else pd.DataFrame()
    damping = pd.read_csv(damping_path) if damping_path.exists() else pd.DataFrame()

    params = _scenario_params(config, scenario)
    model = PublicGoodDynamicModel(params, q_max=float((agents['efficiency'] * config['contribution_max']).sum()))

    _call_plot('model_framework', plot_model_framework)
    _call_plot('causal_loop_diagram', plot_causal_loop_diagram)
    _call_plot('scenario_dashboard', plot_scenario_dashboard, trajectories)
    if len(efforts):
        _call_plot('contribution_distribution', plot_contribution_distribution, agents, efforts)
    _call_plot('free_riding_distribution', plot_free_riding_distribution, trajectories)
    _call_plot('stock_and_pressure', plot_stock_and_pressure, trajectories)
    _call_plot('nash_social_comparison', plot_nash_social_comparison, ns)
    _call_plot('nash_social_trajectory', plot_nash_social_trajectory, ns_traj_df)
    _call_plot('welfare_loss_decomposition', plot_welfare_loss_decomposition, ns)
    _call_plot('policy_comparison', plot_policy_comparison, crit_summary)
    _call_plot('policy_decision_matrix', plot_policy_decision_matrix, crit_summary)
    _call_plot('policy_ablation', plot_policy_ablation, crit_summary)
    _call_plot('maintenance_ranking', plot_maintenance_ranking, crit_summary)
    _call_plot(
        'subsidy_penalty_welfare_heatmap', plot_heatmap,
        heat, 'subsidy', 'penalty', 'avg_welfare_tail',
        'Long-run Welfare under Pure Subsidy-Penalty Combinations',
        'figs/synthetic/synthetic_subsidy_penalty_welfare_heatmap',
        cbar_label='Long-run welfare',
    )
    _call_plot(
        'subsidy_penalty_freeriding_heatmap', plot_heatmap,
        heat, 'subsidy', 'penalty', 'avg_free_riding_tail',
        'Free-riding Ratio under Pure Subsidy-Penalty Combinations',
        'figs/synthetic/synthetic_subsidy_penalty_freeriding_heatmap',
        cbar_label='Average free-riding ratio',
    )
    _call_plot('sensitivity_heatmap', plot_sensitivity_heatmap, sens)
    _call_plot('phase_portrait', plot_phase_portrait, model, get_policy(config, 'baseline'), baseline[baseline['scenario'] == scenario])
    _call_plot('pareto_front', plot_pareto_front, pareto)
    _call_plot('policy_response_surface_3d', plot_policy_response_surface_3d, response_surface_3d)
    _call_plot('pareto_front_3d', plot_pareto_front_3d, pareto)
    _call_plot('dynamic_phase_3d', plot_dynamic_phase_3d, dynamic_phase_3d)
    _call_plot('monte_carlo', plot_monte_carlo, mc)
    _call_plot('uncertainty_bands', plot_uncertainty_bands, mc_traj)
    if not behavior_one_step.empty or not behavior_policy.empty:
        _call_plot('behavioral_weight_robustness', plot_behavioral_robustness, behavior_one_step, behavior_policy)
    if not behavior_oat.empty:
        _call_plot('behavioral_weight_oat_robustness', plot_one_at_a_time_behavioral_weight_robustness, behavior_oat)
    if not damping.empty:
        _call_plot('solver_damping_robustness', plot_damping_robustness, damping)
    _call_plot('caption_index', write_caption_index)
    print(f'[figures] regenerated all figures from CSV outputs in {time.time() - t0:.2f}s')

