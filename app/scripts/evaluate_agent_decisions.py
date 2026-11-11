
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
    classification_report,
)

from src.model_inference import predict_fraud
from src.agent.orchestrator import run_agent


# ==========================================
# 1. LOAD TEST DATA
# ==========================================

DATA_PATH = "data/preprocessed/S1_Original"

X_test = pd.read_csv(f"{DATA_PATH}/X_test.csv")
y_test = pd.read_csv(f"{DATA_PATH}/y_test.csv").iloc[:, 0]

print("=" * 70)
print("CREDITGUARD AI - AGENT DECISION EVALUATION")
print("=" * 70)

print("Total transactions:", len(X_test))
print("Actual fraud:", int(y_test.sum()))
print("Actual normal:", int((y_test == 0).sum()))


# ==========================================
# 2. GENERATE REAL MODEL PREDICTIONS
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
    raise ValueError("Probability count mismatch.")

if len(predictions) != len(X_test):
    raise ValueError("Prediction count mismatch.")


# ==========================================
# 3. RUN AGENT DECISIONS
# ==========================================

print("\nEvaluating Agent decisions...")

decisions = []
decision_times = []
failure_count = 0

original_audit_logger = (
    "src.agent.orchestrator.log_agent_event"
)

with patch(
    original_audit_logger,
    return_value=None,
):

    for index, row in X_test.iterrows():

        probability = float(probabilities[index])

        mock_ml_result = {
            "tool": "ml_prediction",
            "status": "success",
            "prediction": int(predictions[index]),
            "fraud_probability": probability,
            "threshold": 0.50,
            "classification": (
                "FRAUD" if predictions[index] == 1
                else "NORMAL"
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

        decision_times.append(
            time.perf_counter() - start
        )

        if result["status"] != "success":
            failure_count += 1
            decisions.append("ERROR")
            continue

        decisions.append(
            result["decision"]["recommended_action"]
        )


# ==========================================
# 4. MAP DECISIONS TO EVALUATION CLASSES
# ==========================================

# ALLOW = NORMAL
# REVIEW = REVIEW
# ESCALATE = FRAUD

decision_mapping = {
    "ALLOW": "NORMAL",
    "REVIEW": "REVIEW",
    "ESCALATE": "FRAUD",
}

predicted_decisions = pd.Series(decisions).map(
    decision_mapping
).fillna("ERROR")

actual_decisions = pd.Series(
    np.where(y_test.to_numpy() == 1, "FRAUD", "NORMAL")
)


# ==========================================
# 5. MULTICLASS DECISION EVALUATION
# ==========================================

labels = ["NORMAL", "REVIEW", "FRAUD"]

decision_matrix = confusion_matrix(
    actual_decisions,
    predicted_decisions,
    labels=labels,
)

decision_accuracy = accuracy_score(
    actual_decisions,
    predicted_decisions,
)

print("\n" + "=" * 70)
print("DECISION CONFUSION MATRIX")
print("=" * 70)

matrix_df = pd.DataFrame(
    decision_matrix,
    index=[f"Actual {label}" for label in labels],
    columns=[f"Predicted {label}" for label in labels],
)

print(matrix_df)


# ==========================================
# 6. FRAUD ESCALATION EVALUATION
# ==========================================

actual_fraud = y_test.to_numpy() == 1
actual_normal = y_test.to_numpy() == 0

escalated = np.asarray(decisions) == "ESCALATE"
reviewed = np.asarray(decisions) == "REVIEW"
allowed = np.asarray(decisions) == "ALLOW"

fraud_total = int(actual_fraud.sum())
normal_total = int(actual_normal.sum())

fraud_escalated = int(
    np.sum(actual_fraud & escalated)
)

fraud_reviewed = int(
    np.sum(actual_fraud & reviewed)
)

fraud_allowed = int(
    np.sum(actual_fraud & allowed)
)

normal_escalated = int(
    np.sum(actual_normal & escalated)
)

normal_reviewed = int(
    np.sum(actual_normal & reviewed)
)

normal_allowed = int(
    np.sum(actual_normal & allowed)
)

fraud_escalation_rate = (
    fraud_escalated / fraud_total * 100
    if fraud_total else 0
)

missed_fraud_rate = (
    fraud_allowed / fraud_total * 100
    if fraud_total else 0
)

fraud_review_rate = (
    fraud_reviewed / fraud_total * 100
    if fraud_total else 0
)

false_escalation_rate = (
    normal_escalated / normal_total * 100
    if normal_total else 0
)

normal_review_rate = (
    normal_reviewed / normal_total * 100
    if normal_total else 0
)


# ==========================================
# 7. DECISION DISTRIBUTION
# ==========================================

decision_counts = pd.Series(decisions).value_counts()

total = len(X_test)

print("\n" + "=" * 70)
print("DECISION DISTRIBUTION")
print("=" * 70)

for decision in ["ALLOW", "REVIEW", "ESCALATE", "ERROR"]:

    count = int(decision_counts.get(decision, 0))

    percentage = (
        count / total * 100
        if total else 0
    )

    print(
        f"{decision:<12}: {count:>7} "
        f"({percentage:.4f}%)"
    )


# ==========================================
# 8. FINAL EVALUATION REPORT
# ==========================================

print("\n" + "=" * 70)
print("AGENT DECISION PERFORMANCE")
print("=" * 70)

print(f"Decision Accuracy: {decision_accuracy:.6f}")

print(
    f"Fraud Escalation Rate: "
    f"{fraud_escalation_rate:.2f}%"
)

print(
    f"Fraud Review Rate: "
    f"{fraud_review_rate:.2f}%"
)

print(
    f"Missed Fraud Rate: "
    f"{missed_fraud_rate:.2f}%"
)

print(
    f"False Escalation Rate: "
    f"{false_escalation_rate:.4f}%"
)

print(
    f"Normal Review Rate: "
    f"{normal_review_rate:.4f}%"
)

print(f"Agent Execution Failures: {failure_count}")

print(
    f"Average Agent Decision Time: "
    f"{np.mean(decision_times) * 1000:.3f} ms"
)

print(
    f"ML Prediction Processing Time: "
    f"{ml_time:.4f} seconds"
)


# ==========================================
# 9. CLASSIFICATION REPORT
# ==========================================

print("\n" + "=" * 70)
print("DECISION CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        actual_decisions,
        predicted_decisions,
        labels=labels,
        zero_division=0,
    )
)

print("\n" + "=" * 70)
print("EVALUATION COMPLETED")
print("=" * 70)
