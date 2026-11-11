from pathlib import Path

from src.ml_data import load_scenario_splits
from src.model_inference import load_s1_scaler
from src.agent.tools.ml_prediction import run_ml_prediction
from src.agent.tools.threshold_analysis import run_threshold_analysis
from src.agent.tools.transaction_analysis import run_transaction_analysis
from src.agent.tools.risk_assessment import run_risk_assessment


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load validation data
splits = load_scenario_splits(
    root=PROJECT_ROOT / "data" / "preprocessed",
    scenario="S1",
    include_test=False,
)

scaler = load_s1_scaler()

# Test one normal and one fraud transaction
test_indices = {
    "NORMAL": splits.y_val[splits.y_val == 0].index[0],
    "FRAUD": splits.y_val[splits.y_val == 1].index[0],
}

for label, index in test_indices.items():

    # Get scaled transaction
    sample = splits.X_val.loc[[index]].copy()

    # Convert to original feature scale
    raw_sample = sample.copy()
    raw_sample.iloc[:, :] = scaler.inverse_transform(sample)

    # Run ML Prediction Tool
    ml_result = run_ml_prediction(raw_sample)

    # Run Threshold Analysis Tool
    threshold_result = run_threshold_analysis(
        fraud_probability=ml_result["fraud_probability"],
        threshold=ml_result["threshold"],
    )

    # Run Transaction Analysis Tool
    transaction_result = run_transaction_analysis(raw_sample)

    # Run Risk Assessment Tool
    risk_result = run_risk_assessment(
        ml_result=ml_result,
        threshold_result=threshold_result,
        transaction_result=transaction_result,
    )

    print(f"\n===== ACTUAL {label} =====")

    print("\nML Prediction:")
    print(ml_result)

    print("\nThreshold Analysis:")
    print(threshold_result)

    print("\nTransaction Analysis:")
    print(transaction_result)

    print("\nRisk Assessment:")
    print(risk_result)