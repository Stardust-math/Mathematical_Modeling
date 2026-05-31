from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def apply_execution_profile(config: Dict[str, Any]) -> Dict[str, Any]:
    profile_name = config.get('execution_profile')
    profiles = config.get('profiles', {})
    if profile_name and profile_name in profiles:
        merged = _deep_update(config, profiles[profile_name])
        merged['active_execution_profile'] = profile_name
        return merged
    config['active_execution_profile'] = profile_name or 'default'
    return config


def load_json_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with p.open('r', encoding='utf-8') as f:
        cfg = json.load(f)
    return apply_execution_profile(cfg)


def ensure_directories() -> None:
    dirs = [
        'data/processed/synthetic',
        'results/synthetic',
        'results/tables',
        'figs/synthetic',
        'figs/paper',
        'logs/experiment_logs',
        'logs/error_logs',
        'report_assets'
    ]
    for d in dirs:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)


def path_in_project(relative: str | Path) -> Path:
    return PROJECT_ROOT / relative
