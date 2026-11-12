
"""End-to-end test for CreditGuard AI using real XGBoost inference."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.model_inference import predict_fraud
from src.agent.orchestrator import run_agent


# ============================================================
# 1. LOAD TEST DATA
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "preprocessed" / "original_split"

X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv")

# Use the first column of y_test as the target.
y_true = y_test.iloc[:, 0].astype(int).to_numpy()

if len(X_test) != len(y_true):
    raise ValueError("X_test and y_test have different row counts.")

print("=" * 65)
print("CREDITGUARD AI - END-TO-END TEST")
print("=" * 65)
print(f"Test samples : {len(X_test):,}")
print(f"Fraud        : {(y_true == 1).sum():,}")
print(f"Normal       : {(y_true == 0).sum():,}")


# ============================================================
# 2. GET REAL MODEL PREDICTIONS
# ============================================================

print("\n[1/3] Running real XGBoost inference...")

model_output = predict_fraud(
    transaction=X_test,
    threshold=0.50,
    already_scaled=False,
)

probabilities = np.asarray(model_output["probabilities"])
y_pred = np.asarray(model_output["predictions"])

print("Real model inference completed.")


# ============================================================
# 3. SELECT TP, TN, FP, FN
# ============================================================

cases = {
    "TP": np.where((y_true == 1) & (y_pred == 1))[0],
    "TN": np.where((y_true == 0) & (y_pred == 0))[0],
    "FP": np.where((y_true == 0) & (y_pred == 1))[0],
    "FN": np.where((y_true == 1) & (y_pred == 0))[0],
}

print("\n[2/3] Selecting test cases...")

selected_indices = {}

for case_name, indices in cases.items():
    if len(indices) == 0:
        print(f"{case_name}: NOT AVAILABLE")
        continue

    # Select a representative sample:
    # TP/FN: highest fraud probability.
    # TN/FP: highest probability within the corresponding group.
    selected = indices[np.argmax(probabilities[indices])]
    selected_indices[case_name] = int(selected)

    print(
        f"{case_name}: row={selected}, "
        f"actual={y_true[selected]}, "
        f"prediction={y_pred[selected]}, "
        f"probability={probabilities[selected]:.6f}"
    )


# ============================================================
# 4. RUN ACTUAL AGENT ORCHESTRATOR
# ============================================================

print("\n[3/3] Running Agent Orchestrator...")

results = []

for case_name, index in selected_indices.items():

    transaction = X_test.iloc[[index]].copy()

    print("\n" + "-" * 65)
    print(f"TEST CASE: {case_name}")
    print("-" * 65)

    try:
        # No mocking: run_agent calls the real ML Prediction Tool.
        result = run_agent(transaction)

        ml_result = result.get("ml_prediction", {})
        decision_result = result.get("decision", {})

        agent_probability = ml_result.get("fraud_probability")
        agent_prediction = ml_result.get("prediction")

        results.append({
            "case": case_name,
            "row": index,
            "actual": int(y_true[index]),
            "model_prediction": int(y_pred[index]),
            "agent_prediction": agent_prediction,
            "model_probability": float(probabilities[index]),
            "agent_probability": agent_probability,
            "agent_status": result.get("status"),
            "decision": decision_result.get("recommended_action"),
            "trace": result.get("execution_trace"),
        })

        print(f"Agent status     : {result.get('status')}")
        print(f"Agent prediction : {agent_prediction}")
        print(f"Agent probability: {agent_probability}")
        print(f"Decision         : {decision_result.get('recommended_action')}")
        print(f"Decision priority: {decision_result.get('priority')}")
        print(f"Decision reason  : {decision_result.get('reason')}")
        print(f"Execution trace  : {result.get('execution_trace')}")

        if agent_probability is not None:
            probability_match = np.isclose(
                float(agent_probability),
                float(probabilities[index]),
                atol=1e-7,
            )
            print(f"Probability match: {'PASS' if probability_match else 'FAIL'}")

    except Exception as exc:
        print(f"TEST ERROR: {type(exc).__name__}: {exc}")

        results.append({
            "case": case_name,
            "row": index,
            "actual": int(y_true[index]),
            "model_prediction": int(y_pred[index]),
            "agent_status": "exception",
            "error": str(exc),
        })


# ============================================================
# 5. SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("END-TO-END TEST SUMMARY")
print("=" * 65)

summary = pd.DataFrame(results)

if not summary.empty:
    display_columns = [
        col for col in [
            "case",
            "actual",
            "model_prediction",
            "agent_prediction",
            "agent_status",
            "decision",
        ]
        if col in summary.columns
    ]

    print(summary[display_columns].to_string(index=False))

    successful = (
        summary["agent_status"] == "success"
    ).sum()

    print(f"\nSuccessful cases: {successful}/{len(summary)}")

    if successful == len(summary):
        print("ALL SELECTED E2E CASES COMPLETED.")
    else:
        print("SOME E2E CASES FAILED. CHECK THE DETAILS ABOVE.")

else:
    print("No test cases were available.")

print("=" * 65)
