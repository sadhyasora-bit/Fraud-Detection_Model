
from unittest.mock import patch

import pandas as pd

from src.agent.orchestrator import run_agent


transaction = pd.DataFrame(
    [[0.0] * 30],
    columns=[
        "Time",
        *[f"V{i}" for i in range(1, 29)],
        "Amount",
    ],
)


# ==========================================
# TEST 1: Invalid threshold
# ==========================================

result = run_agent(
    transaction=transaction,
    threshold=1.5,
)

print("\n===== INVALID THRESHOLD TEST =====")
print("Status:", result["status"])
print("Failed Tool:", result["failed_tool"])
print("Error:", result["error"])

assert result["status"] == "failed"
assert result["failed_tool"] == "input_validation"

print("TEST RESULT: PASSED")


# ==========================================
# TEST 2: Invalid analysis trigger
# ==========================================

result = run_agent(
    transaction=transaction,
    analysis_trigger=0,
)

print("\n===== INVALID ANALYSIS TRIGGER TEST =====")
print("Status:", result["status"])
print("Failed Tool:", result["failed_tool"])

assert result["status"] == "failed"
assert result["failed_tool"] == "input_validation"

print("TEST RESULT: PASSED")


# ==========================================
# TEST 3: ML Prediction failure
# ==========================================

with patch(
    "src.agent.orchestrator.run_ml_prediction",
    side_effect=RuntimeError("Simulated ML model failure"),
):
    result = run_agent(transaction=transaction)

print("\n===== ML TOOL FAILURE TEST =====")
print("Status:", result["status"])
print("Failed Tool:", result["failed_tool"])
print("Error:", result["error"])
print("Execution Trace:", result["execution_trace"])

assert result["status"] == "failed"
assert result["failed_tool"] == "ml_prediction"
assert result["error"]["error_type"] == "RuntimeError"
assert "ml_prediction_failed" in result["execution_trace"]

print("TEST RESULT: PASSED")


# ==========================================
# TEST 4: Successful execution
# ==========================================

with patch(
    "src.agent.orchestrator.run_ml_prediction",
    return_value={
        "tool": "ml_prediction",
        "status": "success",
        "prediction": 0,
        "fraud_probability": 0.05,
        "threshold": 0.50,
        "classification": "NORMAL",
    },
):
    result = run_agent(transaction=transaction)

print("\n===== SUCCESSFUL EXECUTION TEST =====")
print("Status:", result["status"])
print("Execution Trace:", result["execution_trace"])
print("Decision:", result["decision"])

assert result["status"] == "success"
assert result["decision"]["recommended_action"] == "ALLOW"

print("TEST RESULT: PASSED")


print("\nALL ERROR HANDLING TESTS PASSED")
