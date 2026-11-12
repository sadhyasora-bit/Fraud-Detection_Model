
"""Full-dataset E2E evaluation for CreditGuard AI.

Uses the real ML Prediction Tool and real Agent Orchestrator.
Audit file writes are temporarily disabled to avoid generating
hundreds of thousands of log records during bulk evaluation.
"""

import time
from collections import Counter
from unittest.mock import patch
from pathlib import Path

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

from src.agent.orchestrator import run_agent


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "preprocessed" / "original_split"

CLASSIFICATION_THRESHOLD = 0.50
ANALYSIS_TRIGGER = 0.10
REVIEW_THRESHOLD = 0.20
ESCALATE_THRESHOLD = 0.70


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0]

y_true = y_test.astype(int).to_numpy()

if len(X_test) != len(y_true):
    raise ValueError("X_test and y_test have different row counts.")

if X_test.isna().any().any():
    raise ValueError("X_test contains missing values.")

print("=" * 70)
print("CREDITGUARD AI - FULL DATASET E2E EVALUATION")
print("=" * 70)

print(f"Test samples : {len(X_test):,}")
print(f"Features     : {X_test.shape[1]}")
print(f"Fraud        : {(y_true == 1).sum():,}")
print(f"Normal       : {(y_true == 0).sum():,}")

print("\nEvaluation configuration:")
print(f"Classification threshold : {CLASSIFICATION_THRESHOLD}")
print(f"Analysis trigger         : {ANALYSIS_TRIGGER}")
print(f"Review threshold         : {REVIEW_THRESHOLD}")
print(f"Escalation threshold     : {ESCALATE_THRESHOLD}")


# ============================================================
# 3. EVALUATION COUNTERS
# ============================================================

total = len(X_test)

successful_agent_count = 0
failure_count = 0

total_executed_tools = 0
successful_executed_tools = 0

routing_correct = 0
decision_consistent = 0
investigation_count = 0

failure_details = Counter()
tool_execution_counts = Counter()
tool_success_counts = Counter()
decision_counts = Counter()

agent_times = []

agent_predictions = []
agent_probabilities = []
agent_decisions = []
agent_risk_levels = []
agent_actual_labels = []


# ============================================================
# 4. FULL REAL AGENT EXECUTION
# ============================================================

print("\nStarting full E2E agent execution...")
print("ML Prediction Tool is NOT mocked.")
print("Audit file writing is temporarily disabled.")

start_all = time.perf_counter()

# Suppress only physical audit log writes.
# All other tools and the ML inference pipeline remain real.
with patch(
    "src.agent.orchestrator.log_agent_event",
    return_value=None,
):

    for position in range(total):

        transaction = X_test.iloc[[position]].copy()
        actual_label = int(y_true[position])

        start = time.perf_counter()

        try:
            result = run_agent(
                transaction=transaction,
                threshold=CLASSIFICATION_THRESHOLD,
                analysis_trigger=ANALYSIS_TRIGGER,
                review_threshold=REVIEW_THRESHOLD,
                escalate_threshold=ESCALATE_THRESHOLD,
            )

        except Exception as exc:
            failure_count += 1
            failure_details[
                f"Unhandled exception: {type(exc).__name__}: {exc}"
            ] += 1
            continue

        elapsed = time.perf_counter() - start
        agent_times.append(elapsed)

        if result.get("status") != "success":

            failure_count += 1

            failed_tool = result.get("failed_tool", "unknown")
            error_info = result.get("error", {})

            failure_details[
                f"{failed_tool}: "
                f"{error_info.get('error_type', 'UnknownError')}: "
                f"{error_info.get('message', 'No message')}"
            ] += 1

            continue

        successful_agent_count += 1

        # ----------------------------------------------------
        # A. Collect actual ML output
        # ----------------------------------------------------

        ml_result = result["ml_prediction"]

        probability = float(ml_result["fraud_probability"])
        prediction = int(ml_result["prediction"])

        agent_probabilities.append(probability)
        agent_predictions.append(prediction)
        agent_actual_labels.append(actual_label)

        # ----------------------------------------------------
        # B. Check tool execution
        # ----------------------------------------------------

        tool_results = {
            "ml_prediction": result.get("ml_prediction"),
            "threshold_analysis": result.get("threshold_analysis"),
            "transaction_analysis": result.get("transaction_analysis"),
            "risk_assessment": result.get("risk_assessment"),
            "decision": result.get("decision"),
        }

        for tool_name, tool_result in tool_results.items():

            if tool_result is None:
                continue

            if tool_result.get("analysis_skipped", False):
                continue

            tool_execution_counts[tool_name] += 1
            total_executed_tools += 1

            if tool_result.get("status") == "success":
                tool_success_counts[tool_name] += 1
                successful_executed_tools += 1

        # ----------------------------------------------------
        # C. Check conditional routing
        # ----------------------------------------------------

        expected_analysis = probability >= ANALYSIS_TRIGGER

        actual_analysis = (
            result["transaction_analysis"].get("analysis_skipped")
            is not True
        )

        if actual_analysis:
            investigation_count += 1

        if actual_analysis == expected_analysis:
            routing_correct += 1

        # ----------------------------------------------------
        # D. Check decision consistency
        # ----------------------------------------------------

        decision_result = result["decision"]

        actual_decision = decision_result["recommended_action"]
        risk_level = result["risk_assessment"]["risk_level"]

        if probability >= ESCALATE_THRESHOLD:
            expected_decision = "ESCALATE"
        elif probability >= REVIEW_THRESHOLD:
            expected_decision = "REVIEW"
        else:
            expected_decision = "ALLOW"

        if actual_decision == expected_decision:
            decision_consistent += 1

        decision_counts[actual_decision] += 1

        agent_decisions.append(actual_decision)
        agent_risk_levels.append(risk_level)

        # ----------------------------------------------------
        # E. Progress reporting
        # ----------------------------------------------------

        if (position + 1) % 5000 == 0 or position + 1 == total:
            print(
                f"Processed: {position + 1:,}/{total:,} "
                f"| Success: {successful_agent_count:,} "
                f"| Failed: {failure_count:,}"
            )


total_agent_time = time.perf_counter() - start_all


# ============================================================
# 5. MODEL PERFORMANCE FROM REAL AGENT OUTPUT
# ============================================================

print("\nCalculating model performance...")

if not agent_predictions:
    raise RuntimeError("No successful agent predictions available.")

y_eval = np.asarray(agent_actual_labels)
pred_eval = np.asarray(agent_predictions)
prob_eval = np.asarray(agent_probabilities)

accuracy = accuracy_score(y_eval, pred_eval)

precision = precision_score(
    y_eval, pred_eval, zero_division=0
)

recall = recall_score(
    y_eval, pred_eval, zero_division=0
)

f1 = f1_score(
    y_eval, pred_eval, zero_division=0
)

pr_auc = average_precision_score(y_eval, prob_eval)

tn, fp, fn, tp = confusion_matrix(
    y_eval, pred_eval, labels=[0, 1]
).ravel()


# ============================================================
# 6. AGENT METRICS
# ============================================================

completion_rate = successful_agent_count / total * 100

tool_selection_accuracy = (
    routing_correct / successful_agent_count * 100
    if successful_agent_count else 0
)

tool_execution_success_rate = (
    successful_executed_tools / total_executed_tools * 100
    if total_executed_tools else 0
)

decision_consistency_rate = (
    decision_consistent / successful_agent_count * 100
    if successful_agent_count else 0
)

investigation_rate = investigation_count / total * 100

average_agent_time = (
    np.mean(agent_times) if agent_times else 0
)

median_agent_time = (
    np.median(agent_times) if agent_times else 0
)


# ============================================================
# 7. DECISION DISTRIBUTION BY ACTUAL LABEL
# ============================================================

decision_table = pd.crosstab(
    pd.Series(agent_actual_labels, name="Actual Label"),
    pd.Series(agent_decisions, name="Agent Decision"),
).reindex(
    index=[0, 1],
    columns=["ALLOW", "REVIEW", "ESCALATE"],
    fill_value=0,
)


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MODEL PERFORMANCE - REAL AGENT OUTPUT")
print("=" * 70)

print(f"Accuracy  : {accuracy:.6f}")
print(f"Precision : {precision:.6f}")
print(f"Recall    : {recall:.6f}")
print(f"F1-score  : {f1:.6f}")
print(f"PR-AUC    : {pr_auc:.6f}")

print("\nConfusion Matrix:")
print(f"TN: {tn:,}")
print(f"FP: {fp:,}")
print(f"FN: {fn:,}")
print(f"TP: {tp:,}")


print("\n" + "=" * 70)
print("AGENT ORCHESTRATOR EVALUATION")
print("=" * 70)

print(f"Agent Completion Rate    : {completion_rate:.2f}%")
print(f"Tool Selection Accuracy  : {tool_selection_accuracy:.2f}%")
print(f"Tool Execution Success   : {tool_execution_success_rate:.2f}%")
print(f"Decision Consistency     : {decision_consistency_rate:.2f}%")
print(f"Investigation Rate       : {investigation_rate:.2f}%")

print(
    f"\nSuccessful Agent Runs: "
    f"{successful_agent_count:,}/{total:,}"
)

print(f"Failed Agent Runs    : {failure_count:,}")

print(
    f"Successful Tool Calls: "
    f"{successful_executed_tools:,}/{total_executed_tools:,}"
)

print("\nTool Execution Breakdown:")

for tool_name in [
    "ml_prediction",
    "threshold_analysis",
    "transaction_analysis",
    "risk_assessment",
    "decision",
]:
    executed = tool_execution_counts[tool_name]
    succeeded = tool_success_counts[tool_name]

    print(
        f"{tool_name:25s}: "
        f"{succeeded:,}/{executed:,}"
    )

print("\nAgent Decision Distribution:")
for action in ["ALLOW", "REVIEW", "ESCALATE"]:
    print(f"{action:12s}: {decision_counts[action]:,}")

print("\nDecision Distribution by Actual Label:")
print(decision_table.to_string())

print("\nProcessing Time:")
print(f"Total agent time  : {total_agent_time:.3f} seconds")
print(f"Average per agent : {average_agent_time * 1000:.3f} ms")
print(f"Median per agent  : {median_agent_time * 1000:.3f} ms")

print("\nFailure Details:")
if failure_details:
    for error, count in failure_details.items():
        print(f"{count:,}x - {error}")
else:
    print("No failures detected.")

print("\n" + "=" * 70)
print("FULL DATASET E2E EVALUATION COMPLETED")
print("=" * 70)
