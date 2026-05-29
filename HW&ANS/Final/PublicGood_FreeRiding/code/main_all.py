from __future__ import annotations

import json
import sys
from datetime import datetime
from .config import ensure_directories, path_in_project
from .experiments_synthetic import run_synthetic_experiments


def _count_files(folder, suffixes=None):
    p = path_in_project(folder)
    if not p.exists():
        return 0
    if suffixes is None:
        return sum(1 for item in p.rglob('*') if item.is_file())
    return sum(1 for item in p.rglob('*') if item.is_file() and item.suffix.lower() in suffixes)


def main():
    ensure_directories()
    status = {
        'started_at_utc': datetime.utcnow().isoformat(),
        'mode': 'synthetic_only',
    }

    # One-step workflow: regenerate numerical CSV outputs and then regenerate
    # all figures from those saved outputs. Figure rendering is still isolated
    # inside code.generate_figures, but users only need to run `python main.py`.
    run_synthetic_experiments(generate_figures=True)
    status['synthetic'] = 'completed'
    status['figures'] = 'regenerated from saved CSV outputs by main.py'
    status['finished_at_utc'] = datetime.utcnow().isoformat()

    with path_in_project('logs/experiment_logs/run_summary.json').open('w', encoding='utf-8') as f:
        json.dump(status, f, ensure_ascii=False, indent=2)

    print('\nPublicGood_FreeRiding synthetic-only run summary')
    print(json.dumps(status, indent=2))
    print(f"Project root: {path_in_project('.').resolve()}")
    print(f"Results directory: {path_in_project('results').resolve()}")
    print(f"Figures directory: {path_in_project('figs').resolve()}")
    print(f"Processed synthetic data directory: {path_in_project('data/processed/synthetic').resolve()}")
    print(f"CSV files generated/found: {_count_files('results', {'.csv'}) + _count_files('data/processed', {'.csv'}) + _count_files('report_assets', {'.csv'})}")
    print(f"Figure files generated/found: {_count_files('figs', {'.svg', '.png'})}")
    return status


if __name__ == '__main__':
    result = main()
    print(json.dumps(result, indent=2))
    sys.stdout.flush()
