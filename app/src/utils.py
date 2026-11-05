"""Shared dataset configuration, loading, and label mapping for CreditGuard AI."""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SETTINGS_PATH = DATA_DIR / "dataset_config.json"
DEFAULT_DATASET_NAMES = ("transactions.csv", "creditcard.csv")
MAPPING_KEYS = ("transaction_id", "time", "amount", "target")


def load_dataset_settings() -> dict[str, Any]:
    """Return saved dataset settings, with an empty mapping if none is saved."""
    if not SETTINGS_PATH.is_file():
        return {}
    try:
        value = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def save_dataset_settings(settings: dict[str, Any]) -> None:
    """Persist settings as a small local JSON file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged = load_dataset_settings()
    merged.update(settings)
    SETTINGS_PATH.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def available_datasets() -> list[Path]:
    """Return CSV files in the project data folder."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(DATA_DIR.glob("*.csv"), key=lambda path: path.name.lower())


def resolve_dataset_path(settings: dict[str, Any] | None = None) -> Path | None:
    """Resolve the configured CSV, or the first supported default in data/."""
    settings = settings if settings is not None else load_dataset_settings()
    configured = settings.get("dataset_path")
    if isinstance(configured, str) and configured.strip():
        candidate = Path(configured)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(DATA_DIR.resolve())
        except ValueError:
            candidate = None
        if candidate is not None and candidate.is_file() and candidate.suffix.lower() == ".csv":
            return candidate

    for name in DEFAULT_DATASET_NAMES:
        candidate = DATA_DIR / name
        if candidate.is_file():
            return candidate
    return None


@st.cache_data(show_spinner="Loading dataset...")
def _read_csv_cached(path: str, modified_time_ns: int) -> pd.DataFrame:
    """Read a local CSV once per file version; the timestamp invalidates the cache."""
    return pd.read_csv(path)


def load_dataset(path: Path) -> pd.DataFrame:
    path = path.resolve()
    return _read_csv_cached(str(path), path.stat().st_mtime_ns)


def read_uploaded_csv(contents: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(contents))


def preview_csv(path: Path, rows: int = 12) -> pd.DataFrame:
    return pd.read_csv(path, nrows=rows)


def infer_column_mapping(columns: list[str]) -> dict[str, str | None]:
    """Suggest mappings from common transaction column names."""
    by_normalized = {str(column).strip().lower(): str(column) for column in columns}
    aliases = {
        "transaction_id": ("transaction id", "transaction_id", "id", "trans_id"),
        "time": ("time", "timestamp", "date", "datetime", "transaction time"),
        "amount": ("amount", "transaction amount", "transaction_amount"),
        "target": ("class", "label", "actual label", "is_fraud", "fraud", "target"),
    }
    result: dict[str, str | None] = {}
    for key, candidates in aliases.items():
        result[key] = next((by_normalized[name] for name in candidates if name in by_normalized), None)
    return result


def resolve_column_mapping(columns: list[str], settings: dict[str, Any] | None = None) -> dict[str, str | None]:
    settings = settings if settings is not None else load_dataset_settings()
    inferred = infer_column_mapping(columns)
    saved = settings.get("column_mapping", {})
    if not isinstance(saved, dict):
        return inferred
    valid = set(columns)
    for key in MAPPING_KEYS:
        value = saved.get(key)
        if isinstance(value, str) and value in valid:
            inferred[key] = value
    return inferred


def target_values(series: pd.Series) -> list[str]:
    values = series.dropna().unique().tolist()
    return sorted((str(value) for value in values), key=str.casefold)


def infer_positive_class(values: list[str]) -> str | None:
    """Suggest the positive class only when it resembles a known fraud encoding."""
    normalized = {value.strip().lower(): value for value in values}
    for candidate in ("1", "fraud", "fraudulent", "yes", "true", "positive"):
        if candidate in normalized:
            return normalized[candidate]
    return values[-1] if len(values) == 2 else None


def map_target_labels(series: pd.Series, positive_class: str | None) -> pd.Series:
    """Map the selected ground-truth positive class to Fraud; preserve missing labels."""
    if positive_class is None:
        return pd.Series("Unknown", index=series.index, dtype="object")
    positive_numeric = pd.to_numeric(pd.Series([positive_class]), errors="coerce").iloc[0]
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == series.notna().sum() and not pd.isna(positive_numeric):
        mapped = numeric.eq(positive_numeric).map({True: "Fraud", False: "Normal"})
        return mapped.where(series.notna(), "Unknown")
    positive_text = str(positive_class).strip().casefold()
    cleaned = series.astype("string").str.strip().str.casefold()
    mapped = cleaned.eq(positive_text).map({True: "Fraud", False: "Normal"})
    return mapped.where(series.notna(), "Unknown")

