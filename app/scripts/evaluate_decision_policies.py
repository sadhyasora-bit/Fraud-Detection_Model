
import time

import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
)


# ==========================================
# 1. LOAD DATA
# ==========================================

DATA_PATH = "data/preprocessed/S1_Original"

X_train = pd.read_csv(f"{DATA_PATH}/X_train.csv")
y_train = pd.read_csv(f"{DATA_PATH}/y_train.csv").iloc[:, 0]

X_val = pd.read_csv(f"{DATA_PATH}/X_val.csv")
y_val = pd.read_csv(f"{DATA_PATH}/y_val.csv").iloc[:, 0]

print("=" * 75)
print("CREDITGUARD AI - DECISION POLICY OPTIMIZATION")
print("=" * 75)

print(f"Training samples   : {len(X_train):,}")
print(f"Validation samples : {len(X_val):,}")
print(f"Validation fraud   : {int(y_val.sum()):,}")
print(f"Validation normal  : {int((y_val == 0).sum()):,}")


# ==========================================
# 2. TRAIN TEMPORARY MODEL
# ==========================================

print("\nTraining temporary XGBoost model...")

start_time = time.perf_counter()

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

if positive_count == 0:
    raise ValueError("Training data contains no fraud samples.")

model = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    scale_pos_weight=negative_count / positive_count,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

training_time = time.perf_counter() - start_time

print(f"Temporary model trained in {training_time:.2f} seconds.")


# ==========================================
# 3. GENERATE VALIDATION PROBABILITIES
# ==========================================

print("\nGenerating validation probabilities...")

start_time = time.perf_counter()

probabilities = model.predict_proba(X_val)[:, 1]

prediction_time = time.perf_counter() - start_time

actual = y_val.to_numpy().astype(int)

print(f"Prediction time: {prediction_time:.4f} seconds")


# ==========================================
# 4. DEFINE DECISION POLICIES
# ==========================================

POLICIES = {
    "Baseline": {
        "review": 0.50,
        "escalate": 0.80,
    },
    "Policy A": {
        "review": 0.40,
        "escalate": 0.80,
    },
    "Policy B": {
        "review": 0.30,
        "escalate": 0.80,
    },
    "Policy C": {
        "review": 0.30,
        "escalate": 0.70,
    },
    "Policy D": {
        "review": 0.20,
        "escalate": 0.70,
    },
}


# ==========================================
# 5. EVALUATE EACH POLICY
# ==========================================

results = []

for policy_name, thresholds in POLICIES.items():

    review_threshold = thresholds["review"]
    escalate_threshold = thresholds["escalate"]

    if not (
        0 < review_threshold < escalate_threshold <= 1
    ):
        raise ValueError(
            f"Invalid thresholds in {policy_name}"
        )

    # Decision policy
    decisions = np.select(
        [
            probabilities >= escalate_threshold,
            probabilities >= review_threshold,
        ],
        [
            "ESCALATE",
            "REVIEW",
        ],
        default="ALLOW",
    )

    actual_fraud = actual == 1
    actual_normal = actual == 0

    escalated = decisions == "ESCALATE"
    reviewed = decisions == "REVIEW"
    allowed = decisions == "ALLOW"

    fraud_total = int(actual_fraud.sum())
    normal_total = int(actual_normal.sum())

    # Fraud outcomes
    fraud_escalated = int(
        np.sum(actual_fraud & escalated)
    )

    fraud_reviewed = int(
        np.sum(actual_fraud & reviewed)
    )

    fraud_allowed = int(
        np.sum(actual_fraud & allowed)
    )

    # Normal outcomes
    normal_escalated = int(
        np.sum(actual_normal & escalated)
    )

    normal_reviewed = int(
        np.sum(actual_normal & reviewed)
    )

    normal_allowed = int(
        np.sum(actual_normal & allowed)
    )

    # Strict escalation evaluation
    escalation_precision = precision_score(
        actual,
        escalated.astype(int),
        zero_division=0,
    )

    escalation_recall = recall_score(
        actual,
        escalated.astype(int),
        zero_division=0,
    )

    escalation_f1 = f1_score(
        actual,
        escalated.astype(int),
        zero_division=0,
    )

    escalation_accuracy = accuracy_score(
        actual,
        escalated.astype(int),
    )

    # Intervention evaluation:
    # REVIEW and ESCALATE both require human attention.
    intervention = np.isin(
        decisions,
        ["REVIEW", "ESCALATE"],
    )

    intervention_recall = recall_score(
        actual,
        intervention.astype(int),
        zero_division=0,
    )

    # Rates
    fraud_escalation_rate = (
        fraud_escalated / fraud_total * 100
        if fraud_total else 0
    )

    fraud_review_rate = (
        fraud_reviewed / fraud_total * 100
        if fraud_total else 0
    )

    missed_fraud_rate = (
        fraud_allowed / fraud_total * 100
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

    investigation_load = (
        np.sum(intervention) / len(actual) * 100
    )

    results.append({
        "Policy": policy_name,
        "Review Threshold": review_threshold,
        "Escalate Threshold": escalate_threshold,
        "Escalation Precision": escalation_precision,
        "Escalation Recall": escalation_recall,
        "Escalation F1": escalation_f1,
        "Escalation Accuracy": escalation_accuracy,
        "Fraud Escalation Rate (%)": fraud_escalation_rate,
        "Fraud Review Rate (%)": fraud_review_rate,
        "Missed Fraud Rate (%)": missed_fraud_rate,
        "False Escalation Rate (%)": false_escalation_rate,
        "Normal Review Rate (%)": normal_review_rate,
        "Intervention Recall": intervention_recall,
        "Investigation Load (%)": investigation_load,
        "Fraud Escalated": fraud_escalated,
        "Fraud Reviewed": fraud_reviewed,
        "Fraud Allowed": fraud_allowed,
        "Normal Escalated": normal_escalated,
        "Normal Reviewed": normal_reviewed,
        "Normal Allowed": normal_allowed,
    })


# ==========================================
# 6. DISPLAY RESULTS
# ==========================================

results_df = pd.DataFrame(results)

percentage_columns = [
    "Fraud Escalation Rate (%)",
    "Fraud Review Rate (%)",
    "Missed Fraud Rate (%)",
    "False Escalation Rate (%)",
    "Normal Review Rate (%)",
    "Investigation Load (%)",
]

print("\n" + "=" * 75)
print("POLICY COMPARISON")
print("=" * 75)

display_columns = [
    "Policy",
    "Review Threshold",
    "Escalate Threshold",
    "Escalation Precision",
    "Escalation Recall",
    "Escalation F1",
    "Fraud Escalation Rate (%)",
    "Fraud Review Rate (%)",
    "Missed Fraud Rate (%)",
    "False Escalation Rate (%)",
    "Investigation Load (%)",
]

print(
    results_df[display_columns].to_string(
        index=False,
        formatters={
            "Escalation Precision": "{:.4f}".format,
            "Escalation Recall": "{:.4f}".format,
            "Escalation F1": "{:.4f}".format,
            **{
                column: "{:.4f}".format
                for column in percentage_columns
            },
        },
    )
)


# ==========================================
# 7. ACTION DISTRIBUTION
# ==========================================

print("\n" + "=" * 75)
print("DECISION DISTRIBUTION")
print("=" * 75)

distribution_rows = []

for policy_name, thresholds in POLICIES.items():

    decisions = np.select(
        [
            probabilities >= thresholds["escalate"],
            probabilities >= thresholds["review"],
        ],
        [
            "ESCALATE",
            "REVIEW",
        ],
        default="ALLOW",
    )

    counts = pd.Series(decisions).value_counts()

    distribution_rows.append({
        "Policy": policy_name,
        "ALLOW": int(counts.get("ALLOW", 0)),
        "REVIEW": int(counts.get("REVIEW", 0)),
        "ESCALATE": int(counts.get("ESCALATE", 0)),
    })

distribution_df = pd.DataFrame(distribution_rows)

print(distribution_df.to_string(index=False))


# ==========================================
# 8. SAVE RESULTS
# ==========================================

output_path = "data/results/decision_policy_comparison.csv"

results_df.to_csv(output_path, index=False)

print("\nResults saved to:", output_path)

print("\n" + "=" * 75)
print("DECISION POLICY OPTIMIZATION COMPLETED")
print("=" * 75)
