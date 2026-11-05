"""Compare baseline classifiers and inspect validation threshold tradeoffs."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from src.ml_data import load_scenario_splits
from src.utils import DATA_DIR


st.title("Experiment Lab")
st.caption("CreditGuard AI · Model comparison")
st.markdown(
    "Compare three classifiers using the same prepared train and validation split. "
    "The test split stays untouched."
)

scenario_labels = {
    "S1": "S1 · Original + RobustScaler",
    "S2": "S2 · SMOTE augmentation",
    "S3": "S3 · log1p(Amount) + SMOTE",
}
with st.container(border=True):
    st.subheader("Experiment setup")
    selected_scenario = st.segmented_control(
        "Preprocessing scenario",
        options=list(scenario_labels),
        format_func=lambda value: scenario_labels[value],
        default="S1",
        key="experiment_scenario",
    )
    st.caption(
        "Logistic Regression is the linear baseline; Random Forest tests bagged trees; "
        "XGBoost tests boosted trees with class-imbalance weighting."
    )
    run_comparison = st.button(
        "Train and compare on validation",
        type="primary",
        disabled=selected_scenario is None,
    )

if run_comparison:
    try:
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            average_precision_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
    except ImportError:
        st.error("Scikit-learn is required. Install the project dependencies, then rerun the app.")
        st.stop()

    with st.spinner(f"Loading {selected_scenario} and training all three classifiers…"):
        try:
            splits = load_scenario_splits(DATA_DIR / "preprocessed", selected_scenario, include_test=False)
            classifiers = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=42,
                ),
                "Random Forest": RandomForestClassifier(
                    n_estimators=120,
                    max_depth=12,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
                "XGBoost": XGBClassifier(
                    n_estimators=400,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="aucpr",
                    tree_method="hist",
                    scale_pos_weight=max(1.0, float((splits.y_train == 0).sum()) / max(1, int((splits.y_train == 1).sum()))),
                    random_state=42,
                    n_jobs=-1,
                ),
            }
            comparison = {}
            for name, model in classifiers.items():
                model.fit(splits.X_train, splits.y_train)
                probabilities = model.predict_proba(splits.X_val)[:, 1]
                predictions_at_default = (probabilities >= 0.50).astype("int8")
                comparison[name] = {
                    "probabilities": probabilities.tolist(),
                    "pr_auc": float(average_precision_score(splits.y_val, probabilities)),
                    "roc_auc": float(roc_auc_score(splits.y_val, probabilities)),
                    "precision_at_050": float(precision_score(splits.y_val, predictions_at_default, zero_division=0)),
                    "recall_at_050": float(recall_score(splits.y_val, predictions_at_default, zero_division=0)),
                    "f1_at_050": float(f1_score(splits.y_val, predictions_at_default, zero_division=0)),
                }
            st.session_state["creditguard_model_comparison"] = {
                "scenario": selected_scenario,
                "y_validation": splits.y_val.astype("int8").tolist(),
                "train_rows": len(splits.X_train),
                "validation_rows": len(splits.X_val),
                "normal_train": int((splits.y_train == 0).sum()),
                "fraud_train": int((splits.y_train == 1).sum()),
                "normal_validation": int((splits.y_val == 0).sum()),
                "fraud_validation": int((splits.y_val == 1).sum()),
                "comparison": comparison,
            }
            # Keep the Research Evaluation page synchronized with completed validation runs.
            manifest_path = DATA_DIR / "results" / "validation_comparisons.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            else:
                manifest = {"schema_version": 1, "positive_label": 1, "operating_threshold": 0.50, "scenarios": {}}
            manifest.setdefault("scenarios", {})[selected_scenario] = {
                "scenario_name": scenario_labels[selected_scenario].split("·", 1)[-1].strip(),
                "source": "Experiment Lab run",
                "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "train_rows": len(splits.X_train),
                "validation_rows": len(splits.X_val),
                "validation_normal": int((splits.y_val == 0).sum()),
                "validation_fraud": int((splits.y_val == 1).sum()),
                "models": {
                    name: {
                        "pr_auc": values["pr_auc"],
                        "roc_auc": values["roc_auc"],
                        "precision": values["precision_at_050"],
                        "recall": values["recall_at_050"],
                        "f1": values["f1_at_050"],
                    }
                    for name, values in comparison.items()
                },
            }
            temporary_manifest = manifest_path.with_suffix(".tmp")
            temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary_manifest.replace(manifest_path)
        except Exception as exc:
            st.error(f"Could not run the comparison: {exc}")
            st.stop()

result = st.session_state.get("creditguard_model_comparison")
if result and result["scenario"] == selected_scenario:
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

    st.divider()
    st.subheader(f"Validation comparison · {scenario_labels[result['scenario']]}")
    st.caption(
        f"Training rows: {result['train_rows']:,} "
        f"({result['normal_train']:,} Normal · {result['fraud_train']:,} Fraud) | "
        f"Validation rows: {result['validation_rows']:,} "
        f"({result['normal_validation']:,} Normal · {result['fraud_validation']:,} Fraud)"
    )

    default_threshold = 0.50
    validation_labels = result["y_validation"]
    comparison_rows = []
    for name, values in result["comparison"].items():
        default_prediction = [int(score >= default_threshold) for score in values["probabilities"]]
        comparison_rows.append({
            "Model": name,
            "PR-AUC": values["pr_auc"],
            "ROC-AUC": values["roc_auc"],
            "Precision @ 0.50": precision_score(validation_labels, default_prediction, zero_division=0),
            "Recall @ 0.50": recall_score(validation_labels, default_prediction, zero_division=0),
            "F1 @ 0.50": f1_score(validation_labels, default_prediction, zero_division=0),
        })
    summary = pd.DataFrame(comparison_rows).set_index("Model")
    st.markdown("**Model scores**")
    st.dataframe(
        summary.style.format("{:.3f}"),
        width="stretch",
        alt="Validation metrics comparing Logistic Regression, Random Forest, and XGBoost",
    )
    st.caption("PR-AUC and ROC-AUC evaluate ranking across thresholds. The other columns use a fixed 0.50 cutoff.")

    selected_model = st.selectbox(
        "Inspect decision threshold for",
        options=list(result["comparison"]),
        key="experiment_threshold_model",
    )
    threshold = st.slider(
        "Fraud probability threshold",
        min_value=0.01,
        max_value=0.99,
        value=0.50,
        step=0.01,
        help="Lower thresholds usually catch more fraud but can increase false alarms. This changes validation predictions only.",
    )
    probabilities = result["comparison"][selected_model]["probabilities"]
    predictions = [int(score >= threshold) for score in probabilities]
    tn, fp, fn, tp = confusion_matrix(validation_labels, predictions, labels=[0, 1]).ravel()
    metrics = st.columns(3)
    metrics[0].metric("Precision", f"{precision_score(validation_labels, predictions, zero_division=0):.3f}")
    metrics[1].metric("Recall", f"{recall_score(validation_labels, predictions, zero_division=0):.3f}")
    metrics[2].metric("F1", f"{f1_score(validation_labels, predictions, zero_division=0):.3f}")
    st.markdown("**Confusion matrix · validation at selected threshold**")
    matrix = pd.DataFrame(
        [[int(tn), int(fp)], [int(fn), int(tp)]],
        index=pd.Index(["Actual Normal", "Actual Fraud"], name="Actual / predicted"),
        columns=["Predicted Normal", "Predicted Fraud"],
    )
    st.dataframe(matrix, width="stretch", alt="Confusion matrix for the selected validation threshold")
    st.caption(
        "Changing the threshold does not retrain the model. Model comparison and threshold selection both use validation, "
        "so treat these as development results; keep the test split for one final evaluation after model selection. "
        "Dashboard labels remain actual dataset ground truth."
    )
elif result:
    st.info("The selected scenario differs from the last run. Train the comparison again to see its validation results.")
else:
    st.info("Choose a scenario and run the comparison. No model will train until you press the button.")
