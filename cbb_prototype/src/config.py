"""Paths, constants and a tiny helper for the shared results.json."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "output"
TEMPLATES = ROOT / "templates"
for _p in (RAW, PROC, OUT):
    _p.mkdir(parents=True, exist_ok=True)

SEASON = 2024  # the 2023-24 season (ESPN/hoopR label seasons by the year they end)
SEED = 42
RESULTS_PATH = OUT / "results.json"

# Chart colours: the validated reference palette (categorical slots 1-3, muted ink, gridline)
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_INK, C_MUTED, C_GRID = "#0b0b0b", "#898781", "#e1e0d9"


def to_py(o):
    """Make numpy / pandas scalars JSON-serialisable."""
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if np.isnan(v) else v
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return o


def load_results() -> dict:
    return json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}


def update_results(key: str, value) -> None:
    """Every pipeline stage writes its numbers here; README/summary/card read only from this file."""
    res = load_results()
    res[key] = to_py(value)
    RESULTS_PATH.write_text(json.dumps(res, indent=2))


def reset_results() -> None:
    RESULTS_PATH.write_text("{}")
