from __future__ import annotations
import numpy as np
import pandas as pd


def gini(x) -> float:
    arr = sorted(max(float(v), 0.0) for v in x)
    n = len(arr)
    total = sum(arr)
    if n == 0 or abs(total) < 1e-12:
        return 0.0
    weighted = sum((i + 1) * val for i, val in enumerate(arr))
    return float((2.0 * weighted) / (n * total) - (n + 1.0) / n)


def summarize_trajectory(df: pd.DataFrame) -> dict:
    tail = df.tail(max(8, len(df) // 5))
    return {
        'scenario': df['scenario'].iloc[0],
        'policy': df['policy'].iloc[0],
        'mode': df['mode'].iloc[0] if 'mode' in df.columns else 'nash',
        'final_G': float(df['G'].iloc[-1]),
        'final_H': float(df['H'].iloc[-1]),
        'final_D': float(df['D'].iloc[-1]),
        'avg_G_tail': float(tail['G'].mean()),
        'avg_H_tail': float(tail['H'].mean()),
        'avg_D_tail': float(tail['D'].mean()),
        'avg_free_riding_tail': float(tail['free_riding_ratio'].mean()),
        'avg_welfare_tail': float(tail['welfare'].mean()),
        'avg_policy_cost_tail': float(tail['policy_cost'].mean()),
        'avg_Q_tail': float(tail['Q'].mean()),
        'avg_effort_tail': float(tail['avg_effort'].mean()),
        'avg_effort_gini_tail': float(tail['effort_gini'].mean()),
        'stability_tail': float(tail['stability_score'].mean())
    }


def add_policy_relative_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    base = out[out['policy'] == 'baseline'][['scenario', 'avg_free_riding_tail', 'avg_H_tail', 'avg_welfare_tail', 'avg_policy_cost_tail']].rename(columns={
        'avg_free_riding_tail': 'baseline_free_riding',
        'avg_H_tail': 'baseline_pressure',
        'avg_welfare_tail': 'baseline_welfare',
        'avg_policy_cost_tail': 'baseline_policy_cost'
    })
    out = out.merge(base, on='scenario', how='left')
    out['free_riding_reduction_vs_baseline'] = out['baseline_free_riding'] - out['avg_free_riding_tail']
    out['pressure_reduction_vs_baseline'] = out['baseline_pressure'] - out['avg_H_tail']
    out['welfare_gain_vs_baseline'] = out['avg_welfare_tail'] - out['baseline_welfare']
    out['cost_effectiveness'] = np.where(out['avg_policy_cost_tail'] > 1e-9, out['welfare_gain_vs_baseline'] / out['avg_policy_cost_tail'], np.nan)
    return out


def nash_social_metrics(nash_row: dict, social_row: dict) -> dict:
    q_ne = float(nash_row['Q']); q_so = float(social_row['Q'])
    g_ne = float(nash_row['G']); g_so = float(social_row['G'])
    w_ne = float(nash_row['welfare']); w_so = float(social_row['welfare'])
    out = {
        'scenario': nash_row['scenario'],
        'Q_NE': q_ne, 'Q_SO': q_so,
        'G_NE': g_ne, 'G_SO': g_so,
        'H_NE': float(nash_row['H']), 'H_SO': float(social_row['H']),
        'W_NE': w_ne, 'W_SO': w_so,
        'free_riding_ratio_NE': float(nash_row['free_riding_ratio']),
        'free_riding_ratio_SO': float(social_row['free_riding_ratio']),
        'free_riding_gap': float(max(0.0, (q_so - q_ne) / max(q_so, 1e-9))),
        'under_provision_ratio': float(max(0.0, (g_so - g_ne) / max(g_so, 1e-9))),
        'welfare_loss_ratio': float(max(0.0, (w_so - w_ne) / max(abs(w_so), 1e-9))),
        'maintenance_pressure_index_NE': float(nash_row['H']),
        'maintenance_pressure_index_SO': float(social_row['H'])
    }
    return out
