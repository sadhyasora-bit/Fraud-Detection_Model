
import time
from unittest.mock import patch

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
)

from src.model_inference import predict_fraud
from src.agent.orchestrator import run_agent


# ==========================================
# 1. LOAD TEST DATA
# ==========================================

DATA_PATH = "data/preprocessed/S1_Original"

X_test = pd.read_csv(f"{DATA_PATH}/X_test.csv")
y_test = pd.read_csv(f"{DATA_PATH}/y_test.csv").iloc[:, 0]

print("=" * 65)
print("CREDITGUARD AI - REAL DATASET AGENT EVALUATION")
print("=" * 65)

print("Test samples:", len(X_test))
print("Features:", X_test.shape[1])
print("Fraud samples:", int(y_test.sum()))
print("Normal samples:", int((y_test == 0).sum()))


# ==========================================
# 2. ML PREDICTION
# ==========================================

print("\nRunning XGBoost predictions...")

start_time = time.perf_counter()

ml_output = predict_fraud(
    transaction=X_test,
    threshold=0.50,
    already_scaled=False,
)

ml_time = time.perf_counter() - start_time

probabilities = np.asarray(
    ml_output["probabilities"]
).reshape(-1)

predictions = np.asarray(
    ml_output["predictions"]
).reshape(-1)

if len(probabilities) != len(X_test):
    raise ValueError("Prediction count does not match test dataset.")

if len(predictions) != len(X_test):
    raise ValueError("Label prediction count does not match test dataset.")

if not np.isfinite(probabilities).all():
    raise ValueError("Invalid probability detected.")

print("ML prediction completed.")
print(f"ML processing time: {ml_time:.4f} seconds")


# ==========================================
# 3. AGENT ROUTING EVALUATION
# ==========================================

print("\nEvaluating Agent Orchestrator...")

original_audit_logger = (
    "src.agent.orchestrator.log_agent_event"
)

routing_correct = 0

total_executed_tools = 0
successful_executed_tools = 0

investigation_count = 0
decision_consistent = 0

successful_agent_count = 0
failure_count = 0
failure_details = {}

agent_times = []

# Simulate the ML result from the actual XGBoost output.
# The actual prediction model is NOT mocked.
# Only the already-computed ML output is passed into the orchestrator.

with patch(
    original_audit_logger,
    return_value=None,
):

    for index, row in X_test.iterrows():

        probability = float(probabilities[index])
        actual_label = int(y_test.iloc[index])

        mock_ml_result = {
            "tool": "ml_prediction",
            "status": "success",
            "prediction": int(predictions[index]),
            "fraud_probability": probability,
            "threshold": 0.50,
            "classification": (
                "FRAUD" if predictions[index] == 1 else "NORMAL"
            ),
        }

        start = time.perf_counter()

        with patch(
            "src.agent.orchestrator.run_ml_prediction",
            return_value=mock_ml_result,
        ):

            result = run_agent(
                transaction=row.to_frame().T,
                threshold=0.50,
                analysis_trigger=0.10,
            )

        elapsed = time.perf_counter() - start
        agent_times.append(elapsed)

        if result["status"] != "success":

            failure_count += 1

            failed_tool = result.get(
                "failed_tool",
                "unknown",
            )

            error_info = result.get("error", {})

            error_type = error_info.get(
                "error_type",
                "UnknownError",
            )

            error_message = error_info.get(
                "message",
                error_info.get("error", "No message"),
            )

            failure_key = (
                f"{failed_tool}: {error_type}: {error_message}"
            )

            failure_details[failure_key] = (
                failure_details.get(failure_key, 0) + 1
            )

            continue
        # Routing correctness
        expected_analysis = probability >= 0.10

        actual_analysis = (
            "transaction_analysis" in result["execution_trace"]
        )

        if actual_analysis == expected_analysis:
            routing_correct += 1

        # Successful Agent Orchestration
        successful_agent_count += 1

        # Tool execution success
        tool_results = [
            result.get("threshold_analysis"),
            result.get("transaction_analysis"),
            result.get("risk_assessment"),
            result.get("decision"),
        ]

        executed_results = [
            item for item in tool_results
            if item is not None
            and not item.get("analysis_skipped", False)
        ]

        successful_results = [
            item for item in executed_results
            if item.get("status") == "success"
        ]

        total_executed_tools += len(executed_results)

        successful_executed_tools += len(successful_results)
        # Investigation rate
        if actual_analysis:
            investigation_count += 1

        # Decision consistency with ML prediction
        decision = result["decision"]["recommended_action"]

        expected_decision = (
            "ALLOW" if probability < 0.20
            else "REVIEW" if probability < 0.70
            else "ESCALATE"
        )

        if decision == expected_decision:
            decision_consistent += 1


# ==========================================
# 4. MODEL PERFORMANCE
# ==========================================

print("\nCalculating model performance...")

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test, predictions, zero_division=0
)

recall = recall_score(
    y_test, predictions, zero_division=0
)

f1 = f1_score(
    y_test, predictions, zero_division=0
)

pr_auc = average_precision_score(
    y_test, probabilities
)

tn, fp, fn, tp = confusion_matrix(
    y_test, predictions, labels=[0, 1]
).ravel()


# ==========================================
# 5. AGENT METRICS
# ==========================================

total = len(X_test)

agent_completion_rate = (
    successful_agent_count / total * 100
    if total > 0 else 0
)

tool_selection_accuracy = (
    routing_correct / successful_agent_count * 100
    if successful_agent_count > 0 else 0
)

tool_execution_success_rate = (
    successful_executed_tools / total_executed_tools * 100
    if total_executed_tools > 0 else 0
)

decision_consistency_rate = (
    decision_consistent / successful_agent_count * 100
    if successful_agent_count > 0 else 0
)

investigation_rate = (
    investigation_count / total * 100
    if total > 0 else 0
)

average_agent_time = (
    np.mean(agent_times)
    if agent_times else 0
)


# ==========================================
# 6. DISPLAY RESULTS
# ==========================================

print("\n" + "=" * 65)
print("MODEL PERFORMANCE - REAL TEST SET")
print("=" * 65)

print(f"Accuracy:  {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall:    {recall:.6f}")
print(f"F1-score:  {f1:.6f}")
print(f"PR-AUC:    {pr_auc:.6f}")

print("\nConfusion Matrix:")
print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")


print("\n" + "=" * 65)
print("AGENT ORCHESTRATOR EVALUATION")
print("=" * 65)

print(f"Agent Completion Rate: {agent_completion_rate:.2f}%")
print(f"Tool Selection Accuracy: {tool_selection_accuracy:.2f}%")
print(f"Tool Execution Success: {tool_execution_success_rate:.2f}%")
print(f"Decision Consistency: {decision_consistency_rate:.2f}%")
print(f"Investigation Rate: {investigation_rate:.2f}%")

print(
    f"Successful Agent Runs: "
    f"{successful_agent_count}/{total}"
)

print(
    f"Executed Tools: "
    f"{successful_executed_tools}/{total_executed_tools}"
)

print(
    f"Average Agent Processing Time: "
    f"{average_agent_time * 1000:.3f} ms"
)

print(
    f"ML Prediction Processing Time: "
    f"{ml_time:.4f} seconds"
)