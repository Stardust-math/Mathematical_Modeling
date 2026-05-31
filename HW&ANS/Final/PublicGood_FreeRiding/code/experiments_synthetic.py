from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .config import load_json_config, path_in_project
from .constants import POLICY_ORDER
from .synthetic_generator import generate_all_agents
from .policy_mechanisms import get_policy
from .solvers import simulate_policy_trajectory, one_step_equilibrium_summary
from .metrics import summarize_trajectory, nash_social_metrics, add_policy_relative_metrics
from .sensitivity_analysis import run_1d_sensitivity, run_subsidy_penalty_heatmap, run_policy_response_surface_3d
from .monte_carlo import run_monte_carlo
from .optimization import run_random_pareto_search
from .evolutionary_model import simulate_replicator
from .game_model import FreeRidingGame
from .robustness_extensions import (
    build_nash_weight_scale_audit,
    run_behavioral_response_robustness,
    run_one_at_a_time_behavioral_weight_robustness,
    run_solver_damping_robustness,
    plot_behavioral_robustness,
    plot_one_at_a_time_behavioral_weight_robustness,
    plot_damping_robustness,
)


def _scenario_params(config, scenario):
    p = dict(config['scenario_parameters'][scenario])
    p['cost_scale'] = p.get('cost_scale', 1.0)
    p['benefit_scale'] = p.get('benefit_scale', 1.0)
    p['nash_response_weights'] = dict(config.get('nash_response_weights', {}))
    p['nash_update_damping_raw_weight'] = float(config.get('nash_update_damping_raw_weight', 0.42))
    p['nash_update_max_iter'] = int(config.get('nash_update_max_iter', 60))
    p['nash_update_tol'] = float(config.get('nash_update_tol', 1e-7))
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
    runtime_rows = []
    _run_t0 = time.perf_counter()

    def _mark_runtime(step: str, start_time: float, note: str = '') -> None:
        runtime_rows.append({
            'execution_profile': config.get('active_execution_profile', config.get('execution_profile', 'default')),
            'step': step,
            'seconds': round(time.perf_counter() - start_time, 4),
            'note': note,
        })

    _t_step = time.perf_counter()
    agents_all = generate_all_agents(config)
    agents_all.to_csv(path_in_project('data/processed/synthetic/synthetic_agents.csv'), index=False)
    scenario_param_rows = []
    for sc in config['scenarios']:
        r = {'scenario': sc}
        r.update(config['scenario_parameters'][sc])
        scenario_param_rows.append(r)
    pd.DataFrame(scenario_param_rows).to_csv(path_in_project('data/processed/synthetic/synthetic_scenario_parameters.csv'), index=False)
    _mark_runtime('agent_and_scenario_generation', _t_step, 'synthetic agents, scenario-parameter table, and processed agent CSV')

    print('[synthetic] generating synthetic agents and scenarios...')

    # Policy trajectories
    _t_step = time.perf_counter()
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
    _mark_runtime('policy_trajectory_simulation', _t_step, 'all scenario-policy Nash-style trajectories and policy comparison table')

    baseline = trajectories[trajectories['policy'] == 'baseline'].copy()
    baseline.to_csv(path_in_project('data/processed/synthetic/synthetic_baseline_trajectory.csv'), index=False)
    baseline.to_csv(path_in_project('results/synthetic/synthetic_baseline_trajectory.csv'), index=False)

    _t_step = time.perf_counter()
    # The following critical-infrastructure setup is reused by the Monte Carlo,
    # Pareto, and response-surface modules.
    scenario = 'critical_infrastructure'
    agents = agents_all[agents_all['scenario'] == scenario].copy()
    params = _scenario_params(config, scenario)

    print('[synthetic] running Monte Carlo analysis...')
    _t_mc = time.perf_counter()
    mc, mc_traj = run_monte_carlo(config, scenario, ['baseline', 'reputation', 'combined_portfolio'])
    mc.to_csv(path_in_project('results/synthetic/synthetic_monte_carlo_runs.csv'), index=False)
    mc_summary = mc.groupby('policy').agg(
        avg_welfare_mean=('avg_welfare_tail', 'mean'), avg_welfare_std=('avg_welfare_tail', 'std'),
        final_G_mean=('final_G', 'mean'), free_riding_mean=('avg_free_riding_tail', 'mean'),
        policy_cost_mean=('avg_policy_cost_tail', 'mean')
    ).reset_index()
    mc_summary.to_csv(path_in_project('results/synthetic/synthetic_monte_carlo_summary.csv'), index=False)
    mc_traj.to_csv(path_in_project('results/synthetic/synthetic_monte_carlo_trajectories.csv'), index=False)
    print(f'[synthetic] Monte Carlo analysis completed in {time.perf_counter() - _t_mc:.2f}s')
    _mark_runtime('monte_carlo_uncertainty', _t_mc, 'Monte Carlo runs, summary, and trajectory bands')

    print('[synthetic] running Pareto policy search...')
    _t_pareto = time.perf_counter()
    pareto = run_random_pareto_search(agents, params, config, scenario)
    pareto.to_csv(path_in_project('results/synthetic/synthetic_pareto_front.csv'), index=False)
    pareto.to_csv(path_in_project('results/synthetic/synthetic_pareto_front_3d_view.csv'), index=False)
    print(f'[synthetic] Pareto policy search completed in {time.perf_counter() - _t_pareto:.2f}s')
    _mark_runtime('pareto_policy_search', _t_pareto, 'random policy portfolios and full multi-indicator Pareto screen')

    print('[synthetic] checking Nash/social dynamic trajectories...')
    _t_ns_traj = time.perf_counter()
    ns_traj_path = path_in_project('results/synthetic/synthetic_nash_social_trajectories.csv')
    force_ns_traj = bool(config.get('force_regenerate_nash_social_trajectories', False))
    if force_ns_traj or not ns_traj_path.exists():
        subprocess.run(
            [sys.executable, str(path_in_project('code/ns_trajectory_worker.py')), config_path],
            cwd=str(path_in_project('.')),
            check=True,
        )
        _ns_note = 'dynamic Nash-style and stage-wise social-planner baseline trajectories generated in a fresh worker process'
    else:
        _ns_note = 'existing dynamic Nash/social trajectory CSV reused; run code/ns_trajectory_worker.py to refresh it explicitly'
    _mark_runtime('nash_social_trajectories', _t_ns_traj, _ns_note)

    crit_summary = summary[summary['scenario'] == 'critical_infrastructure'].copy()
    crit_summary.to_csv(path_in_project('results/synthetic/synthetic_policy_ablation.csv'), index=False)

    print('[synthetic] running sensitivity analysis...')
    _t_sens = time.perf_counter()
    scenario = 'critical_infrastructure'
    agents = agents_all[agents_all['scenario'] == scenario].copy()
    params = _scenario_params(config, scenario)
    sens = run_1d_sensitivity(agents, params, config, scenario)
    sens.to_csv(path_in_project('results/synthetic/synthetic_sensitivity_1d.csv'), index=False)
    print(f'[synthetic] 1D sensitivity rows: {len(sens)}', flush=True)
    heat = run_subsidy_penalty_heatmap(agents, params, config, scenario)
    heat.to_csv(path_in_project('results/synthetic/synthetic_sensitivity_2d_subsidy_penalty.csv'), index=False)
    print(f'[synthetic] 2D subsidy-penalty rows: {len(heat)}', flush=True)

    response_surface_3d = run_policy_response_surface_3d(agents, params, config, scenario)
    response_surface_3d['source_grid'] = 'synthetic_policy_response_surface_3d_fine_grid'
    response_surface_3d.to_csv(path_in_project('results/synthetic/synthetic_policy_response_surface_3d.csv'), index=False)
    response_surface_3d.to_csv(path_in_project('data/processed/synthetic/synthetic_policy_response_surface_3d.csv'), index=False)
    print(f'[synthetic] 3D response surface rows: {len(response_surface_3d)}', flush=True)
    print(f'[synthetic] sensitivity analysis completed in {time.perf_counter() - _t_sens:.2f}s')
    _mark_runtime('sensitivity_and_response_surface', _t_sens, '1D sensitivity, 2D heatmap, and fine-grid 3D pure subsidy-penalty response surface')

    _t_step = time.perf_counter()
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
    ns = pd.DataFrame(ns_rows)
    ns.to_csv(path_in_project('results/synthetic/synthetic_nash_vs_social_optimum.csv'), index=False)
    ns.to_csv(path_in_project('data/processed/synthetic/synthetic_nash_vs_social_optimum.csv'), index=False)
    pd.DataFrame(checks).to_csv(path_in_project('results/synthetic/synthetic_social_optimizer_check.csv'), index=False)
    nash_diagnostics = pd.DataFrame(nash_diagnostic_rows)
    nash_diagnostics = nash_diagnostics[['scenario', 'damped_sweeps', 'converged', 'max_residual', 'max_damped_residual', 'Q_IR', 'mean_effort']]
    nash_diagnostics.to_csv(path_in_project('results/synthetic/synthetic_nash_update_diagnostics.csv'), index=False)
    nash_diagnostics.to_csv(path_in_project('data/processed/synthetic/synthetic_nash_update_diagnostics.csv'), index=False)
    if efforts_rows:
        efforts = pd.concat(efforts_rows, ignore_index=True)
        efforts.to_csv(path_in_project('data/processed/synthetic/synthetic_baseline_efforts.csv'), index=False)
    else:
        efforts = pd.DataFrame()

    ns[['scenario', 'free_riding_gap', 'under_provision_ratio', 'welfare_loss_ratio', 'maintenance_pressure_index_NE', 'maintenance_pressure_index_SO']].to_csv(
        path_in_project('results/synthetic/synthetic_welfare_loss_decomposition.csv'), index=False
    )
    _mark_runtime('benchmark_comparison', _t_step, 'Nash-style individual rationality, stage-wise planner, diagnostics, and trajectories')


    _t_evo = time.perf_counter()
    evo_frames = []
    for pname in ['baseline', 'reputation', 'penalty', 'combined_portfolio']:
        evo_frames.append(simulate_replicator(pname, get_policy(config, pname), T=config['time_horizon'], seed=config['random_seed']))
    evo = pd.concat(evo_frames, ignore_index=True)
    evo.to_csv(path_in_project('results/synthetic/synthetic_evolutionary_dynamics.csv'), index=False)
    _mark_runtime('evolutionary_dynamics', _t_evo, 'replicator-style auxiliary policy dynamics')

    _t_rob = time.perf_counter()
    # Perturbed-scenario robustness is deliberately anchored to the same
    # critical-infrastructure testbed used for the main deterministic policy
    # comparison. Do not reuse the loop variable left over from the preceding
    # all-scenario benchmark loop; after that loop it points to the final
    # scenario in config['scenarios'] and can silently mislabel this table.
    robustness_scenario = config.get('robustness_scenario', 'critical_infrastructure')
    robustness_agents = agents_all[agents_all['scenario'] == robustness_scenario].copy()
    robustness_params = _scenario_params(config, robustness_scenario)
    rob_rows = []
    for variant, mult in [('low_decay', 0.70), ('high_decay', 1.30), ('low_cost', 0.80), ('high_cost', 1.20), ('demand_shock', 1.45)]:
        p2 = dict(robustness_params)
        a2 = robustness_agents.copy()
        if 'decay' in variant:
            p2['delta'] *= mult
        if 'cost' in variant:
            a2['cost'] *= mult
        if variant == 'demand_shock':
            p2['D0'] = min(float(p2.get('D_capacity', 1.5)), p2['D0'] * mult)
            p2['lambda'] *= 1.15
        for pname in ['baseline', 'combined_portfolio']:
            df = simulate_policy_trajectory(
                a2,
                p2,
                get_policy(config, pname),
                config,
                robustness_scenario,
                pname,
                mode='nash',
                rng=np.random.default_rng(config['random_seed'] + 333),
            )
            s = summarize_trajectory(df)
            s.update({'robustness_variant': variant})
            rob_rows.append(s)
    robustness = pd.DataFrame(rob_rows)
    robustness.to_csv(path_in_project('results/synthetic/synthetic_robustness_summary.csv'), index=False)
    _mark_runtime(
        'perturbed_scenario_robustness',
        _t_rob,
        f'decay, cost, and demand perturbations for baseline versus combined portfolio in {robustness_scenario}',
    )

    print('[synthetic] running behavioral-response and solver-damping robustness checks...')
    _t_extra = time.perf_counter()
    robust_cfg = config.get('robustness_extensions', {})
    robust_scenario = robust_cfg.get('scenario', scenario)
    robust_agents = agents_all[agents_all['scenario'] == robust_scenario].copy()
    robust_params = _scenario_params(config, robust_scenario)
    behavior_one_step, behavior_policy = run_behavioral_response_robustness(
        robust_agents,
        robust_params,
        config,
        robust_scenario,
        robust_cfg.get('behavioral_weight_multipliers', [0.8, 1.0, 1.2]),
        robust_cfg.get('policies', ['baseline', 'reputation', 'combined_portfolio']),
    )
    behavior_one_step.to_csv(path_in_project('results/synthetic/synthetic_behavioral_weight_robustness.csv'), index=False)
    behavior_policy.to_csv(path_in_project('results/synthetic/synthetic_behavioral_policy_ranking.csv'), index=False)

    oat_multipliers = robust_cfg.get('behavioral_weight_oat_multipliers', [0.70, 0.85, 1.00, 1.15, 1.30])
    behavior_oat = run_one_at_a_time_behavioral_weight_robustness(
        robust_agents,
        robust_params,
        config,
        robust_scenario,
        oat_multipliers,
    )
    behavior_oat.to_csv(path_in_project('results/synthetic/synthetic_behavioral_weight_oat_robustness.csv'), index=False)
    weight_audit = build_nash_weight_scale_audit(config, oat_multipliers)
    weight_audit.to_csv(path_in_project('results/synthetic/synthetic_nash_weight_scale_audit.csv'), index=False)
    weight_audit.to_csv(path_in_project('data/processed/synthetic/synthetic_nash_weight_scale_audit.csv'), index=False)

    damping = run_solver_damping_robustness(
        robust_agents,
        robust_params,
        config,
        robust_scenario,
        robust_cfg.get('damping_raw_weights', [0.30, 0.42, 0.55]),
    )
    damping.to_csv(path_in_project('results/synthetic/synthetic_solver_damping_robustness.csv'), index=False)
    if generate_figures:
        plot_behavioral_robustness(behavior_one_step, behavior_policy)
        plot_one_at_a_time_behavioral_weight_robustness(behavior_oat)
        plot_damping_robustness(damping)
    print(f'[synthetic] added robustness checks completed in {time.perf_counter() - _t_extra:.2f}s')
    _mark_runtime('behavioral_and_solver_robustness', _t_extra, 'joint weight scaling, one-at-a-time weight perturbations, weight-scale audit, and damping checks')

    dynamic_phase_policies = config.get('visualization_3d', {}).get('dynamic_phase_policies', ['baseline', 'reputation', 'combined_portfolio'])
    dynamic_phase_scenario = config.get('visualization_3d', {}).get('dynamic_phase_scenario', 'critical_infrastructure')
    dynamic_phase_3d = trajectories[
        (trajectories['scenario'] == dynamic_phase_scenario)
        & (trajectories['policy'].isin(dynamic_phase_policies))
    ].copy()
    dynamic_phase_3d.to_csv(path_in_project('results/synthetic/synthetic_dynamic_phase_3d.csv'), index=False)
    dynamic_phase_3d.to_csv(path_in_project('data/processed/synthetic/synthetic_dynamic_phase_3d.csv'), index=False)

    if generate_figures:
        print('[synthetic] generating figures from saved CSV outputs...')
        _t_fig = time.perf_counter()
        from .generate_figures import generate_figures_from_outputs
        generate_figures_from_outputs(config_path=config_path)
        print(f'[synthetic] figure generation completed in {time.perf_counter() - _t_fig:.2f}s')
        _mark_runtime('figure_generation', _t_fig, 'all paper and synthetic figures regenerated from saved CSV outputs')
    else:
        print('[synthetic] numerical CSV outputs completed.')

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
        ('synthetic_policy_response_surface_3d.csv', 'Fine-grid pure subsidy-penalty response surface for 3D visualization'),
        ('synthetic_robustness_summary.csv', 'Perturbed critical-infrastructure scenario robustness for baseline versus combined portfolio'),
        ('synthetic_pareto_front_3d_view.csv', 'Pareto-search data used for the 3D trade-off projection'),
        ('synthetic_dynamic_phase_3d.csv', 'Critical-infrastructure trajectories used for the 3D dynamic phase plot'),
        ('synthetic_behavioral_weight_robustness.csv', 'Robustness of the Nash-style benchmark to behavioral-response weight perturbations'),
        ('synthetic_behavioral_policy_ranking.csv', 'Policy ranking under behavioral-response weight perturbations'),
        ('synthetic_solver_damping_robustness.csv', 'Damped best-response solver robustness under alternative damping rates'),
        ('synthetic_behavioral_weight_oat_robustness.csv', 'One-at-a-time perturbation check for each Nash-style behavioral-response weight'),
        ('synthetic_nash_weight_scale_audit.csv', 'Audit table documenting the source, interpretation, and tested range of behavioral-response weights'),
        ('synthetic_runtime_profile.csv', 'Runtime profile for reproducibility and full-workflow reporting')
    ], columns=['file', 'paper_use'])
    table_index.to_csv(path_in_project('report_assets/table_captions.csv'), index=False)

    outputs = [str(p.relative_to(path_in_project('.'))) for p in path_in_project('results/synthetic').glob('*.csv')]
    outputs += [str(p.relative_to(path_in_project('.'))) for p in path_in_project('figs/synthetic').glob('*.*')]
    outputs += [str(p.relative_to(path_in_project('.'))) for p in path_in_project('figs/paper').glob('*.*')]
    _mark_runtime('manifest_and_caption_assets', _run_t0, 'total wall time through manifest, caption index, and output discovery')
    runtime = pd.DataFrame(runtime_rows)
    runtime.to_csv(path_in_project('results/synthetic/synthetic_runtime_profile.csv'), index=False)
    outputs.append('results/synthetic/synthetic_runtime_profile.csv')

    _write_manifest(config, trajectories, outputs)

    return {
        'trajectories': trajectories,
        'summary': summary,
        'nash_social': ns,
        'monte_carlo': mc_summary,
        'pareto': pareto,
        'runtime_profile': runtime,
    }
