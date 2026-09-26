"""Loads thresholds from spec/thresholds.yaml, the single source of truth (see CLAUDE.md)."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS_PATH = REPO_ROOT / "spec" / "thresholds.yaml"
DATA_PATH = REPO_ROOT / "ev_battery_charging_data.csv"
MODELS_DIR = REPO_ROOT / "models"


def load_thresholds(path: Path = THRESHOLDS_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
