"""Load prepared scenario splits from the existing preprocessing work.

This module deliberately does not clean, split, scale, or resample data.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd


SCENARIO_FOLDERS = {
    "S1": "S1_Original",
    "S2": "S2_Augmentation",
    "S3": "S3_Preprocessing_Augmentation",
}
SPLITS = ("train", "val", "test")


@dataclass
class ScenarioSplits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame | None
    y_test: pd.Series | None
    scenario: str
    metadata: dict[str, Any]


def find_scenario_folder(root: str | Path, scenario: str) -> Path | None:
    """Find an exported scenario inside a package root or scenarios directory."""
    if scenario not in SCENARIO_FOLDERS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose S1, S2, or S3.")
    root = Path(root).expanduser()
    folder_name = SCENARIO_FOLDERS[scenario]
    candidates = (root / "scenarios" / folder_name, root / folder_name)
    expected_files = [
        name for split in SPLITS
        for name in (f"X_{split}.csv", f"y_{split}.csv")
    ]
    for candidate in candidates:
        if all((candidate / name).is_file() for name in expected_files):
            return candidate
    return None


def scenario_package_available(root: str | Path) -> bool:
    return any(find_scenario_folder(root, scenario) is not None for scenario in SCENARIO_FOLDERS)


def load_scenario_splits(
    root: str | Path,
    scenario: str = "S1",
    *,
    include_test: bool = False,
) -> ScenarioSplits:
    """Load previously prepared X/y split files without repeating notebook steps.

    Accepts scenario folders directly under root or under root/scenarios/.
    Metadata and scaler artifacts are optional because scenario CSVs are already prepared.
    """
    root = Path(root).expanduser()
    folder = find_scenario_folder(root, scenario)
    if folder is None:
        expected = root / "scenarios" / SCENARIO_FOLDERS[scenario]
        raise FileNotFoundError(
            f"Prepared {scenario} files were not found in {expected}. "
            "Place the exported scenario package in data/preprocessed first."
        )

    x_sets: dict[str, pd.DataFrame] = {}
    y_sets: dict[str, pd.Series] = {}
    reference_columns: list[str] | None = None
    allowed_classes = {0, 1}

    splits_to_load = SPLITS if include_test else ("train", "val")
    for split in splits_to_load:
        x_path = folder / f"X_{split}.csv"
        y_path = folder / f"y_{split}.csv"
        if not y_path.is_file():
            raise FileNotFoundError(f"Prepared target file is missing: {y_path}")

        X = pd.read_csv(x_path)
        y_frame = pd.read_csv(y_path)
        if y_frame.shape[1] != 1:
            raise ValueError(f"{y_path.name} must contain exactly one target column.")
        y = y_frame.iloc[:, 0].rename("Class")

        if len(X) != len(y):
            raise ValueError(f"{split} feature and target row counts do not match.")
        if "Class" in X.columns:
            raise ValueError("A prepared feature file contains the target column Class.")
        if reference_columns is None:
            reference_columns = [str(column) for column in X.columns]
        elif list(X.columns) != reference_columns:
            raise ValueError(f"Feature columns or order differ in {x_path.name}.")
        if X.isna().any().any():
            raise ValueError(f"{x_path.name} contains missing feature values.")
        if not set(pd.unique(y)).issubset(allowed_classes):
            raise ValueError(f"{y_path.name} contains labels other than 0 and 1.")

        x_sets[split] = X
        y_sets[split] = y.astype("int8")

    metadata: dict[str, Any] = {}
    metadata_candidates = (
        root / "artifacts" / "scenario_metadata.json",
        folder.parent / "artifacts" / "scenario_metadata.json",
    )
    for metadata_path in metadata_candidates:
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            break

    return ScenarioSplits(
        X_train=x_sets["train"],
        y_train=y_sets["train"],
        X_val=x_sets["val"],
        y_val=y_sets["val"],
        X_test=x_sets.get("test"),
        y_test=y_sets.get("test"),
        scenario=scenario,
        metadata=metadata,
    )

