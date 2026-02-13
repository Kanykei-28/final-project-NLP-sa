from __future__ import annotations
from pathlib import Path

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = project_root()
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
FIGURES_DIR = OUTPUTS_DIR / "figures"