from __future__ import annotations
import json
import time
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from .config import load_json_config, path_in_project
from .constants import POLICY_ORDER
from .synthetic_generator import generate_all_agents
from .policy_mechanisms import get_policy
from .solvers import simulate_policy_trajectory, one_step_equilibrium_summary
from .metrics import summarize_trajectory, nash_social_metrics, add_policy_relative_metrics
from .sensitivity_analysis import run_1d_sensitivity, run_subsidy_penalty_heatmap
from .monte_carlo import run_monte_carlo
from .optimization import run_random_pareto_search
from .evolutionary_model import simulate_replicator
from .game_model import FreeRidingGame


def _scenario_params(config, scenario):
    p = dict(config['scenario_parameters'][scenario])
    p['cost_scale'] = p.get('cost_scale', 1.0)
    p['benefit_scale'] = p.get('benefit_scale', 1.0)
    return p


def _write_manifest(config, trajectories, outputs):
    manifest = pd.DataFrame([
        ('random_seed', config['random_seed']),
        ('time_horizon', config['time_horizon']),
        ('num_agents_per_scenario', config['num_agents']),
        ('scenarios', ', '.join(config['scenarios'])),
        ('policies', ', '.join(POLICY_ORDER)),
        ('parameter_source_type', 'synthetic / simulation-based'),
        ('execution_profile', config.get('active_execution_profile', 'default')),
        ('output_files_count', len(outputs)),
        ('reproducibility_note', 'All synthetic experiments use fixed random seeds and explicit synthetic scenario labels.')
    ], columns=['field', 'value'])
    manifest.to_csv(path_in_project('results/synthetic/synthetic_experiment_manifest.csv'), index=False)

    audit_rows = []
    for scenario in config['scenarios']:
        audit_rows.append({
            'scenario': scenario,
            'random_seed': config['random_seed'],
            'time_horizon': config['time_horizon'],
            'num_agents': config['num_agents'],
            'parameter_source_type': 'synthetic',
            'data_label': 'simulation-based',
            'output_tag': 'synthetic',
            'note': 'Synthetic scenario for mechanism study; not real-world observation.'
        })
    pd.DataFrame(audit_rows).to_csv(path_in_project('data/processed/synthetic/synthetic_data_audit.csv'), index=False)


def run_synthetic_experiments(config_path: str = 'configs/synthetic_config.json', generate_figures: bool = False) -> dict:
    config = load_json_config(config_path)
    agents_all = generate_all_agents(config)
    agents_all.to_csv(path_in_project('data/processed/synthetic/synthetic_agents.csv'), index=False)
    scenario_param_rows = []
    for sc in config['scenarios']:
        r = {'scenario': sc}
        r.update(config['scenario_parameters'][sc])
        scenario_param_rows.append(r)
    pd.DataFrame(scenario_param_rows).to_csv(path_in_project('data/processed/synthetic/synthetic_scenario_parameters.csv'), index=False)

    print('[synthetic] generating synthetic agents and scenarios...')

    # Policy trajectories
    print('[synthetic] simulating policy trajectories...')
    all_traj = []
    summary_rows = []
    for scenario_idx, scenario in enumerate(config['scenarios']):
        agents = agents_all[agents_all['scenario'] == scenario].copy()
        params = _scenario_params(config, scenario)
        paired_noise_seed = int(config['random_seed'] + 1_000 + scenario_idx)
        for pname in POLICY_ORDER:
            policy = get_policy(config, pname)
            # Use the same exogenous noise path for every policy within the same
            # scenario. This makes deterministic policy comparisons paired rather
            # than partly driven by different random demand shocks.
            df = simulate_policy_trajectory(agents, params, policy, config, scenario, pname, mode='nash', rng=np.random.default_rng(paired_noise_seed))
            all_traj.append(df)
            summary_rows.append(summarize_trajectory(df))
    trajectories = pd.concat(all_traj, ignore_index=True)
    trajectories.to_csv(path_in_project('data/processed/synthetic/synthetic_policy_trajectories.csv'), index=False)
    trajectories.to_csv(path_in_project('results/synthetic/synthetic_policy_trajectories.csv'), index=False)
    summary = pd.DataFrame(summary_rows)
    summary = add_policy_relative_metrics(summary)
    summary.to_csv(path_in_project('results/synthetic/synthetic_policy_comparison.csv'), index=False)

    baseline = trajectories[trajectories['policy'] == 'baseline'].copy()
    baseline.to_csv(path_in_project('data/processed/synthetic/synthetic_baseline_trajectory.csv'), index=False)
    baseline.to_csv(path_in_project('results/synthetic/synthetic_baseline_trajectory.csv'), index=False)

    print('[synthetic] computing Nash-style individual rationality and stage-wise social-planner benchmark...')
    ns_rows = []
    checks = []
    nash_diagnostic_rows = []
    efforts_rows = []
    ns_traj = []
    for scenario in config['scenarios']:
        agents = agents_all[agents_all['scenario'] == scenario].copy()
        params = _scenario_params(config, scenario)
        policy = get_policy(config, 'baseline')
        nrow, srow, check = one_step_equilibrium_summary(agents, params, policy, config, scenario)
        row = nash_social_metrics(nrow, srow)
        row.update(check)
        ns_rows.append(row)
        checks.append({
            'scenario': scenario,
            'Q_NE': row['Q_NE'], 'Q_SO': row['Q_SO'], 'W_NE': row['W_NE'], 'W_SO': row['W_SO'],
            'Q_gap': row['Q_SO'] - row['Q_NE'], 'W_gap': row['W_SO'] - row['W_NE'],
            'welfare_loss_ratio': row['welfare_loss_ratio'],
            'social_optimum_valid': check['social_optimum_valid'],
            'optimizer_method': check['optimizer_method'], 'warning': check['warning']
        })
        game = FreeRidingGame(agents, params, config['contribution_max'], config['free_rider_threshold'], config['high_benefit_quantile'])
        diag = game.nash_update_diagnostics(params['G0'], params['H0'], params['D0'], policy)
        diag.update({'scenario': scenario})
        nash_diagnostic_rows.append(diag)
        if scenario == 'critical_infrastructure':
            e = game.solve_nash_equilibrium(params['G0'], params['H0'], params['D0'], policy)
            temp = agents[['agent_id', 'scenario', 'benefit', 'cost', 'efficiency', 'initial_type']].copy()
            temp['effort'] = e
            temp['free_rider'] = game.classify_free_riders(e)
            efforts_rows.append(temp)
        for mode in ['nash', 'social']:
            ns_traj.append(simulate_policy_trajectory(agents, params, policy, config, scenario, 'baseline', mode=mode, rng=np.random.default_rng(config['random_seed'] + 777)))
    ns = pd.DataFrame(ns_rows)
    ns.to_csv(path_in_project('results/synthetic/synthetic_nash_vs_social_optimum.csv'), index=False)
    ns.to_csv(path_in_project('data/processed/synthetic/synthetic_nash_vs_social_optimum.csv'), index=False)
    pd.DataFrame(checks).to_csv(path_in_project('results/synthetic/synthetic_social_optimizer_check.csv'), index=False)
    nash_diagnostics = pd.DataFrame(nash_diagnostic_rows)
    nash_diagnostics = nash_diagnostics[['scenario', 'damped_sweeps', 'converged', 'max_residual', 'max_damped_residual', 'Q_IR', 'mean_effort']]
    nash_diagnostics.to_csv(path_in_project('results/synthetic/synthetic_nash_update_diagnostics.csv'), index=False)
    nash_diagnostics.to_csv(path_in_project('data/processed/synthetic/synthetic_nash_update_diagnostics.csv'), index=False)
    ns_traj_df = pd.concat(ns_traj, ignore_index=True)
    ns_traj_df.to_csv(path_in_project('data/processed/synthetic/synthetic_nash_social_trajectories.csv'), index=False)
    ns_traj_df.to_csv(path_in_project('results/synthetic/synthetic_nash_social_trajectories.csv'), index=False)
    if efforts_rows:
        efforts = pd.concat(efforts_rows, ignore_index=True)
        efforts.to_csv(path_in_project('data/processed/synthetic/synthetic_baseline_efforts.csv'), index=False)
    else:
        efforts = pd.DataFrame()

    ns[['scenario', 'free_riding_gap', 'under_provision_ratio', 'welfare_loss_ratio', 'maintenance_pressure_index_NE', 'maintenance_pressure_index_SO']].to_csv(
        path_in_project('results/synthetic/synthetic_welfare_loss_decomposition.csv'), index=False
    )

    crit_summary = summary[summary['scenario'] == 'critical_infrastructure'].copy()
    crit_summary.to_csv(path_in_project('results/synthetic/synthetic_policy_ablation.csv'), index=False)

    print('[synthetic] running sensitivity analysis...')
    _t_sens = time.time()
    scenario = 'critical_infrastructure'
    agents = agents_all[agents_all['scenario'] == scenario].copy()
    params = _scenario_params(config, scenario)
    sens = run_1d_sensitivity(agents, params, config, scenario)
    sens.to_csv(path_in_project('results/synthetic/synthetic_sensitivity_1d.csv'), index=False)
    heat = run_subsidy_penalty_heatmap(agents, params, config, scenario)
    heat.to_csv(path_in_project('results/synthetic/synthetic_sensitivity_2d_subsidy_penalty.csv'), index=False)

    # The existing subsidy-penalty grid is reused for the 3D surface so that the
    # new visualization remains a strict 3D view of the already reported 2D
    # sensitivity experiment, without changing the numerical meaning of the old CSV.
    response_surface_3d = heat.copy()
    response_surface_3d['source_grid'] = 'synthetic_sensitivity_2d_subsidy_penalty.csv'
    response_surface_3d.to_csv(path_in_project('results/synthetic/synthetic_policy_response_surface_3d.csv'), index=False)
    response_surface_3d.to_csv(path_in_project('data/processed/synthetic/synthetic_policy_response_surface_3d.csv'), index=False)
    print(f'[synthetic] sensitivity analysis completed in {time.time() - _t_sens:.2f}s')

    print('[synthetic] running Monte Carlo analysis...')
    _t_mc = time.time()
    mc, mc_traj = run_monte_carlo(config, scenario, ['baseline', 'reputation', 'combined_portfolio'])
    mc.to_csv(path_in_project('results/synthetic/synthetic_monte_carlo_runs.csv'), index=False)
    mc_summary = mc.groupby('policy').agg(
        avg_welfare_mean=('avg_welfare_tail', 'mean'), avg_welfare_std=('avg_welfare_tail', 'std'),
        final_G_mean=('final_G', 'mean'), free_riding_mean=('avg_free_riding_tail', 'mean'),
        policy_cost_mean=('avg_policy_cost_tail', 'mean')
    ).reset_index()
    mc_summary.to_csv(path_in_project('results/synthetic/synthetic_monte_carlo_summary.csv'), index=False)
    mc_traj.to_csv(path_in_project('results/synthetic/synthetic_monte_carlo_trajectories.csv'), index=False)
    print(f'[synthetic] Monte Carlo analysis completed in {time.time() - _t_mc:.2f}s')

    print('[synthetic] running Pareto policy search...')
    _t_pareto = time.time()
    pareto = run_random_pareto_search(agents, params, config, scenario)
    pareto.to_csv(path_in_project('results/synthetic/synthetic_pareto_front.csv'), index=False)
    pareto.to_csv(path_in_project('results/synthetic/synthetic_pareto_front_3d_view.csv'), index=False)
    print(f'[synthetic] Pareto policy search completed in {time.time() - _t_pareto:.2f}s')

    evo_frames = []
    for pname in ['baseline', 'reputation', 'penalty', 'combined_portfolio']:
        evo_frames.append(simulate_replicator(pname, get_policy(config, pname), T=config['time_horizon'], seed=config['random_seed']))
    evo = pd.concat(evo_frames, ignore_index=True)
    evo.to_csv(path_in_project('results/synthetic/synthetic_evolutionary_dynamics.csv'), index=False)

    rob_rows = []
    for variant, mult in [('low_decay', 0.70), ('high_decay', 1.30), ('low_cost', 0.80), ('high_cost', 1.20), ('demand_shock', 1.45)]:
        p2 = dict(params)
        a2 = agents.copy()
        if 'decay' in variant:
            p2['delta'] *= mult
        if 'cost' in variant:
            a2['cost'] *= mult
        if variant == 'demand_shock':
            p2['D0'] = min(float(p2.get('D_capacity', 1.5)), p2['D0'] * mult)
            p2['lambda'] *= 1.15
        for pname in ['baseline', 'combined_portfolio']:
            df = simulate_policy_trajectory(a2, p2, get_policy(config, pname), config, scenario, pname, mode='nash', rng=np.random.default_rng(config['random_seed'] + 333))
            s = summarize_trajectory(df)
            s.update({'robustness_variant': variant})
            rob_rows.append(s)
    robustness = pd.DataFrame(rob_rows)
    robustness.to_csv(path_in_project('results/synthetic/synthetic_robustness_summary.csv'), index=False)

    dynamic_phase_policies = config.get('visualization_3d', {}).get('dynamic_phase_policies', ['baseline', 'reputation', 'combined_portfolio'])
    dynamic_phase_3d = trajectories[(trajectories['scenario'] == scenario) & (trajectories['policy'].isin(dynamic_phase_policies))].copy()
    dynamic_phase_3d.to_csv(path_in_project('results/synthetic/synthetic_dynamic_phase_3d.csv'), index=False)
    dynamic_phase_3d.to_csv(path_in_project('data/processed/synthetic/synthetic_dynamic_phase_3d.csv'), index=False)

    if generate_figures:
        print('[synthetic] generating figures from saved CSV outputs...')
        _t_fig = time.time()
        cmd = [sys.executable, '-u', '-m', 'code.generate_figures']
        try:
            subprocess.run(cmd, cwd=path_in_project('.'), check=True)
        except subprocess.CalledProcessError as exc:
            cmd_str = ' '.join(str(part) for part in cmd)
            raise RuntimeError(
                f'Figure generation command failed with exit status {exc.returncode}: {cmd_str}'
            ) from exc
        print(f'[synthetic] figure generation completed in {time.time() - _t_fig:.2f}s')
    else:
        print('[synthetic] numerical CSV outputs completed; the top-level workflow will regenerate figures from saved CSV outputs.')

    table_index = pd.DataFrame([
        ('synthetic_policy_comparison.csv', 'Policy ranking by long-run metrics'),
        ('synthetic_nash_vs_social_optimum.csv', 'Nash-style individual rationality versus stage-wise social-planner benchmark'),
        ('synthetic_nash_social_trajectories.csv', 'Dynamic trajectories under Nash-style individual rationality and the stage-wise social-planner benchmark'),
        ('synthetic_welfare_loss_decomposition.csv', 'Free-riding gap, welfare loss, maintenance pressure'),
        ('synthetic_social_optimizer_check.csv', 'Social-planner optimizer validity check'),
        ('synthetic_nash_update_diagnostics.csv', 'Nash-style best-response update diagnostics for the numerical stability table'),
        ('synthetic_sensitivity_1d.csv', 'One-dimensional sensitivity results'),
        ('synthetic_monte_carlo_summary.csv', 'Monte Carlo robustness summary'),
        ('synthetic_pareto_front.csv', 'Random-search Pareto policy set'),
        ('synthetic_policy_response_surface_3d.csv', 'Fine-grid subsidy-penalty response surface for 3D visualization'),
        ('synthetic_pareto_front_3d_view.csv', 'Pareto-search data used for the 3D trade-off projection'),
        ('synthetic_dynamic_phase_3d.csv', 'Critical-scenario trajectories used for the 3D dynamic phase plot')
    ], columns=['file', 'paper_use'])
    table_index.to_csv(path_in_project('report_assets/table_captions.csv'), index=False)

    figure_usage = """# Figure Usage Recommendations for Newly Added 3D Visualizations

All files listed here are generated from synthetic simulation outputs and should be interpreted as mechanism-oriented numerical evidence rather than empirical estimates.

## fig_policy_response_surface_3d
- Suggested placement: Section 10.2, after the subsidy-penalty response-surface table.
- Main message: welfare varies nonlinearly over joint subsidy-penalty intensities, while color records the corresponding average maintenance pressure.
- Recommended role: main text if space is available; otherwise appendix with the 2D heatmap retained in the main text.

## fig_pareto_front_3d
- Suggested placement: Section 9.4, immediately after the existing 2D Pareto projection.
- Main message: policy choice is a multi-objective trade-off among cost, welfare, free riding, pressure, stock, and inequality.
- Recommended role: appendix or supplementary unless the paper needs a stronger visual explanation of Pareto dominance.

## fig_dynamic_phase_3d
- Suggested placement: Section 9.1, after the baseline dynamics dashboard, or in an appendix if the 2D dashboard remains the main figure.
- Main message: free riding changes the system trajectory through coupled stock-pressure-welfare feedback rather than through a static one-period effect.
- Recommended role: main text if Figure 4 is too dense; otherwise appendix.
"""
    path_in_project('report_assets/figure_usage_recommendations.md').write_text(figure_usage, encoding='utf-8')

    outputs = [str(p.relative_to(path_in_project('.'))) for p in path_in_project('results/synthetic').glob('*.csv')]
    outputs += [str(p.relative_to(path_in_project('.'))) for p in path_in_project('figs/synthetic').glob('*.*')]
    outputs += [str(p.relative_to(path_in_project('.'))) for p in path_in_project('figs/paper').glob('*.*')]
    _write_manifest(config, trajectories, outputs)

    return {'trajectories': trajectories, 'summary': summary, 'nash_social': ns, 'monte_carlo': mc_summary, 'pareto': pareto}
