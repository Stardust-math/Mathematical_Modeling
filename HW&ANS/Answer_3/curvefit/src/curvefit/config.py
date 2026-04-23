from pathlib import Path
PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
RESULTS_DIR = OUTPUT_ROOT / "results"
FIGURES_DIR = OUTPUT_ROOT / "figures"
ANIMATIONS_DIR = OUTPUT_ROOT / "animations"
for p in [OUTPUT_ROOT, RESULTS_DIR, FIGURES_DIR, ANIMATIONS_DIR]:
    p.mkdir(parents=True, exist_ok=True)
DEFAULT_RANDOM_SEED = 42
