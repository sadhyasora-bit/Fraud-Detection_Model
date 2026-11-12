
"""Profile original transaction features from Notebook 02."""

from pathlib import Path

import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Original split dataset from Notebook 02
DATA_DIR = PROJECT_ROOT / "data" / "preprocessed" / "original_split"

# Load original training features
X_train = pd.read_csv(DATA_DIR / "X_train.csv")

# Validate required columns
required_features = ["Time", "Amount"]

missing_features = [
    col for col in required_features
    if col not in X_train.columns
]

if missing_features:
    raise ValueError(
        f"Missing required features: {missing_features}"
    )

amount = X_train["Amount"]
time = X_train["Time"]

# Validate numeric values
if not np.isfinite(X_train[required_features].to_numpy()).all():
    raise ValueError("Time or Amount contains non-finite values.")

print("\n===== DATASET INFORMATION =====")
print("Dataset path:", DATA_DIR)
print("Training shape:", X_train.shape)

print("\n===== AMOUNT DISTRIBUTION (ORIGINAL SCALE) =====")
print(
    amount.describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
)

print("\n===== TIME DISTRIBUTION (ORIGINAL SCALE) =====")
print(
    time.describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
)

print("\n===== AMOUNT VALIDATION =====")
print("Zero Amount:", (amount == 0).sum())
print("Negative Amount:", (amount < 0).sum())

print("\n===== TIME VALIDATION =====")
print("Negative Time:", (time < 0).sum())

print("\n===== PROFILING COMPLETED =====")
