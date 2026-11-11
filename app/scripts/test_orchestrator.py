
from pathlib import Path

from src.ml_data import load_scenario_splits
from src.model_inference import load_s1_scaler
from src.agent.orchestrator import run_agent


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load S1 validation data
splits = load_scenario_splits(
    root=PROJECT_ROOT / "data" / "preprocessed",
    scenario="S1",
    include_test=False,
)

scaler = load_s1_scaler()

# Select one normal and one fraud sample
test_indices = {
    "NORMAL": splits.y_val[splits.y_val == 0].index[0],
    "FRAUD": splits.y_val[splits.y_val == 1].index[0],
}

for label, index in test_indices.items():

    # Get validation sample
    sample = splits.X_val.loc[[index]].copy()

    # Convert to original feature scale
    raw_sample = sample.copy()
    raw_sample.iloc[:, :] = scaler.inverse_transform(sample)

    # Run Agent Orchestrator
    result = run_agent(
        transaction=raw_sample,
        threshold=0.50,
        analysis_trigger=0.10,
    )

    print(f"\n{'=' * 50}")
    print(f"ACTUAL TRANSACTION: {label}")
    print("=" * 50)

    print("\nExecution Trace:")
    for step in result["execution_trace"]:
        print(f"- {step}")

    print("\nML Prediction:")
    print(result["ml_prediction"])

    print("\nTransaction Analysis:")
    print(result["transaction_analysis"])

    print("\nRisk Assessment:")
    print(result["risk_assessment"])

    print("\nFinal Decision:")
    print(result["decision"])
