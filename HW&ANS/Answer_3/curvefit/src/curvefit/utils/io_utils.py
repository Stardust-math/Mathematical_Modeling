import re, time
from contextlib import contextmanager
from pathlib import Path
import pandas as pd

def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "_", text.strip().lower())
    return re.sub(r"_+", "_", text).strip("_")

def save_dataframe_csv(df: pd.DataFrame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path

@contextmanager
def time_block():
    bag = {"start": time.perf_counter(), "elapsed": None}
    try:
        yield bag
    finally:
        bag["elapsed"] = time.perf_counter() - bag["start"]
