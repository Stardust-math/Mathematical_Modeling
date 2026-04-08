"""RPCA 图像修复 —— 启动入口"""

from pathlib import Path

from gui import RPCAApp


if __name__ == "__main__":
    FIG_ROOT = Path(__file__).resolve().parent / "figs"
    app = RPCAApp(FIG_ROOT)
    app.run()