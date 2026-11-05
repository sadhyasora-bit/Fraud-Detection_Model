"""One-time final test evaluation for the selected CreditGuard model."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from src.ml_data import load_scenario_splits
from src.utils import DATA_DIR


st.title("Model Performance")
st.caption("CreditGuard AI · Final held-out evaluation")
st.markdown(
    "This page runs the locked candidate once: **S1 + XGBoost + threshold 0.50**. "
    "It refits on train and validation, then evaluates on the held-out test split."
)
st.warning(
    "Test results are for final reporting only. Do not use them to change the model or threshold.",
    icon=":material/lock:",
)

report_path = DATA_DIR / "results" / "final_test_s1_xgboost_threshold_050.json"


def show_report(report: dict) -> None:
    """Render the persisted final test report."""
    st.subheader("Final test results")
    st.caption(f"Completed: {report['completed_utc']} · Candidate: S1 + XGBoost · Threshold: {report['threshold']:.2f}")
    st.caption(
        f"Fit rows: {report['fit_rows']:,} ({report['fit_normal']:,} Normal · {report['fit_fraud']:,} Fraud) | "
        f"Test rows: {report['test_rows']:,} ({report['test_normal']:,} Normal · {report['test_fraud']:,} Fraud)"
    )
    values = report["metrics"]
    cols = st.columns(3)
    cols[0].metric("Precision", f"{values['precision']:.3f}")
    cols[1].metric("Recall", f"{values['recall']:.3f}")
    cols[2].metric("F1", f"{values['f1']:.3f}")
    cols = st.columns(3)
    cols[0].metric("PR-AUC", f"{values['pr_auc']:.3f}")
    cols[1].metric("ROC-AUC", f"{values['roc_auc']:.3f}")
    cols[2].metric("Specificity", f"{values['specificity']:.3f}")
    st.markdown("**Confusion matrix · test set**")
    matrix = pd.DataFrame(
        [[values['tn'], values['fp']], [values['fn'], values['tp']]],
        index=pd.Index(["Actual Normal", "Actual Fraud"], name="Actual / predicted"),
        columns=["Predicted Normal", "Predicted Fraud"],
    )
    st.dataframe(matrix, width="stretch", alt="Confusion matrix for the final held-out test evaluation")
    st.download_button(
        "Download final evaluation report",
        data=json.dumps(report, ensure_ascii=False, indent=2),
        file_name=report_path.name,
        mime="application/json",
    )
    st.caption("This report is saved locally and the final-evaluation button is locked to prevent accidental test reuse.")


if report_path.exists():
    try:
        saved_report = json.loads(report_path.read_text(encoding="utf-8"))
        show_report(saved_report)
        st.info(f"Saved report: {report_path.relative_to(DATA_DIR.parent).as_posix()}")
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        st.error(f"The saved final report could not be read: {exc}")
else:
    with st.container(border=True):
        st.subheader("Locked final candidate")
        st.markdown("""
- Scenario: **S1 - Original + RobustScaler**
- Model: **XGBoost** with the Experiment Lab settings
- Threshold: **0.50**
- Fit: combine train and validation; do not tune from test
""")
        st.caption("The test files are not read by Experiment Lab. They are opened only after this final model has finished fitting.")
        acknowledged = st.checkbox(
            "I have finalized S1 + XGBoost + threshold 0.50 and will not use the test results to tune the model.",
            key="confirm_final_test_evaluation",
        )
        run_final = st.button(
            "Run final evaluation on test set",
            type="primary",
            disabled=not acknowledged,
        )

    if run_final:
        try:
            from sklearn.metrics import (
                accuracy_score,
                average_precision_score,
                confusion_matrix,
                f1_score,
                precision_score,
                recall_score,
                roc_auc_score,
            )
            from xgboost import XGBClassifier
        except ImportError:
            st.error("Scikit-learn and XGBoost are required. Install the project dependencies, then rerun the app.")
            st.stop()

        with st.spinner("Fitting the final S1 + XGBoost model, then evaluating the held-out test set…"):
            try:
                # Test rows remain unread until the final model is fit.
                fit_splits = load_scenario_splits(
                    DATA_DIR / "preprocessed", "S1", include_test=False
                )
                X_fit = pd.concat([fit_splits.X_train, fit_splits.X_val], ignore_index=True)
                y_fit = pd.concat([fit_splits.y_train, fit_splits.y_val], ignore_index=True)
                positive_count = int((y_fit == 1).sum())
                negative_count = int((y_fit == 0).sum())
                final_model = XGBClassifier(
                    n_estimators=400,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="aucpr",
                    tree_method="hist",
                    scale_pos_weight=max(1.0, float(negative_count) / max(1, positive_count)),
                    random_state=42,
                    n_jobs=-1,
                )
                final_model.fit(X_fit, y_fit)

                # The held-out test set is opened only after fitting is complete.
                test_splits = load_scenario_splits(
                    DATA_DIR / "preprocessed", "S1", include_test=True
                )
                if test_splits.X_test is None or test_splits.y_test is None:
                    raise ValueError("The S1 test split could not be loaded.")
                if list(test_splits.X_test.columns) != list(X_fit.columns):
                    raise ValueError("Test feature columns do not match the training features.")
                probabilities = final_model.predict_proba(test_splits.X_test)[:, 1]
                threshold = 0.50
                predictions = (probabilities >= threshold).astype("int8")
                tn, fp, fn, tp = confusion_matrix(
                    test_splits.y_test, predictions, labels=[0, 1]
                ).ravel()
                specificity = float(tn / (tn + fp)) if tn + fp else 0.0
                report = {
                    "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "scenario": "S1",
                    "model": "XGBoost",
                    "threshold": threshold,
                    "fit_rows": len(X_fit),
                    "fit_normal": negative_count,
                    "fit_fraud": positive_count,
                    "test_rows": len(test_splits.X_test),
                    "test_normal": int((test_splits.y_test == 0).sum()),
                    "test_fraud": int((test_splits.y_test == 1).sum()),
                    "parameters": {
                        "n_estimators": 400,
                        "max_depth": 5,
                        "learning_rate": 0.05,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "tree_method": "hist",
                        "scale_pos_weight": float(negative_count / max(1, positive_count)),
                        "random_state": 42,
                    },
                    "metrics": {
                        "accuracy": float(accuracy_score(test_splits.y_test, predictions)),
                        "precision": float(precision_score(test_splits.y_test, predictions, zero_division=0)),
                        "recall": float(recall_score(test_splits.y_test, predictions, zero_division=0)),
                        "f1": float(f1_score(test_splits.y_test, predictions, zero_division=0)),
                        "specificity": specificity,
                        "pr_auc": float(average_precision_score(test_splits.y_test, probabilities)),
                        "roc_auc": float(roc_auc_score(test_splits.y_test, probabilities)),
                        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
                    },
                }
                report_path.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive create prevents overwriting or silently repeating a final test report.
                with report_path.open("x", encoding="utf-8") as report_file:
                    json.dump(report, report_file, ensure_ascii=False, indent=2)
                st.session_state.pop("confirm_final_test_evaluation", None)
                st.rerun()
            except FileExistsError:
                st.error("A final test report already exists. Reload the page to view it; it will not be overwritten.")
            except Exception as exc:
                st.error(f"Final evaluation did not finish: {exc}")
