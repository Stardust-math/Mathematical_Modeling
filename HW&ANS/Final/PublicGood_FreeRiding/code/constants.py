from __future__ import annotations

MODE_SYNTHETIC = 'synthetic'
SYNTHETIC_PREFIX = 'synthetic'

STATE_COLUMNS = [
    'time', 'scenario', 'policy', 'mode',
    'G', 'H', 'D', 'Q',
    'free_riding_ratio', 'welfare',
    'policy_cost', 'stability_score'
]
AGENT_COLUMNS = [
    'agent_id', 'scenario', 'benefit', 'cost',
    'efficiency', 'pressure_sensitivity', 'initial_type'
]
POLICY_ORDER = [
    'baseline', 'subsidy', 'penalty', 'reputation',
    'matching_fund', 'threshold_governance', 'combined_portfolio'
]
SCENARIO_ORDER = [
    'small_volunteer', 'rapid_growth',
    'critical_infrastructure', 'burnout_prone'
]
