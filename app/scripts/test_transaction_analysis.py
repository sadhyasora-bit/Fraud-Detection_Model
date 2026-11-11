from pathlib import Path

from src.ml_data import load_scenario_splits
from src.model_inference import load_s1_scaler
from src.agent.tools.transaction_analysis import (
    run_transaction_analysis,
)

# Define project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load S1 validation data
splits = load_scenario_splits(
    root=PROJECT_ROOT / "data" / "preprocessed",
    scenario="S1",
    include_test=False,
)

# Take one validation sample
sample = splits.X_val.iloc[[0]].copy()

# Inverse transform to original feature scale
scaler = load_s1_scaler()

raw_sample = sample.copy()
raw_sample.iloc[:, :] = scaler.inverse_transform(sample)

# Run Transaction Analysis Tool
result = run_transaction_analysis(raw_sample)

print("\n===== TRANSACTION ANALYSIS RESULT =====")

for key, value in result.items():
    print(f"{key}: {value}")