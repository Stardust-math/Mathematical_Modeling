import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from curvefit.experiments.fourier_experiments import FourierExperimentRunner
def main():
    out = FourierExperimentRunner().run_all()
    for k, df in out.items():
        print(f"[DONE] {k}: {len(df)} rows")
if __name__ == "__main__":
    main()
