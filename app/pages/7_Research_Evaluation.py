"""Research protocol and evidence summary for CreditGuard AI."""
from __future__ import annotations

import json
import pandas as pd
import streamlit as st

from src.utils import DATA_DIR

st.title("Research Evaluation")
st.caption("CreditGuard AI · Experimental design and results")
st.markdown("This page documents the preprocessing scenarios, model comparison, validation findings, and the final held-out test result. Validation and test evidence are shown separately.")

results_dir = DATA_DIR / "results"
validation_path = results_dir / "validation_comparisons.json"
test_path = results_dir / "final_test_s1_xgboost_threshold_050.json"
validation = {}

st.subheader("Research question")
st.write("How do class balancing and the Amount transformation affect fraud detection, and which of three classifiers gives the most useful precision-recall trade-off?")

st.subheader("Study design")
cols = st.columns(3)
with cols[0]:
    with st.container(border=True):
        st.markdown("**Dataset and target**")
        st.write("European credit card transactions")
        st.caption("Class 1 = observed Fraud · Class 0 = Normal")
        st.caption("Derived ML data excludes exact duplicates; raw dashboard data is preserved.")
with cols[1]:
    with st.container(border=True):
        st.markdown("**Split and safeguards**")
        st.write("Stratified random split · 70/15/15 · seed 42")
        st.caption("Scaler fit on train only; SMOTE only on train. Test held out until final evaluation.")
with cols[2]:
    with st.container(border=True):
        st.markdown("**Compared factors**")
        st.write("3 preprocessing scenarios × 3 classifiers")
        st.caption("Logistic Regression · Random Forest · XGBoost")

with st.expander("Scenario definitions"):
    scenarios = pd.DataFrame([
        {"Scenario": "S1", "Preparation": "RobustScaler; no resampling"},
        {"Scenario": "S2", "Preparation": "RobustScaler; SMOTE on training split"},
        {"Scenario": "S3", "Preparation": "log1p(Amount), RobustScaler, and training-only SMOTE"},
    ])
    st.dataframe(scenarios, hide_index=True, width="stretch", alt="Definitions of the three existing data preparation scenarios")

st.subheader("Validation comparison")
st.caption("PR-AUC is emphasized because Fraud is rare. Precision, recall, and F1 below use a common 0.50 threshold. Initial values were transcribed from the result screenshots you supplied; reruns from Experiment Lab replace that scenario with newly calculated results.")
if validation_path.is_file():
    try:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        rows = []
        for scenario in ("S1", "S2", "S3"):
            scenario_data = validation.get("scenarios", {}).get(scenario)
            if not scenario_data:
                continue
            for model in ("Logistic Regression", "Random Forest", "XGBoost"):
                values = scenario_data.get("models", {}).get(model)
                if values:
                    rows.append({
                        "Scenario": scenario,
                        "Preparation": scenario_data.get("scenario_name", scenario),
                        "Model": model,
                        "PR-AUC": values.get("pr_auc"),
                        "ROC-AUC": values.get("roc_auc"),
                        "Precision @ 0.50": values.get("precision"),
                        "Recall @ 0.50": values.get("recall"),
                        "F1 @ 0.50": values.get("f1"),
                    })
        if rows:
            comparison = pd.DataFrame(rows)
            st.dataframe(
                comparison,
                hide_index=True,
                width="stretch",
                column_config={
                    "PR-AUC": st.column_config.NumberColumn(format="%.3f"),
                    "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
                    "Precision @ 0.50": st.column_config.NumberColumn(format="%.3f"),
                    "Recall @ 0.50": st.column_config.NumberColumn(format="%.3f"),
                    "F1 @ 0.50": st.column_config.NumberColumn(format="%.3f"),
                },
                alt="Validation comparison for all recorded scenario and classifier runs",
            )
            best_pr = max(rows, key=lambda row: row["PR-AUC"])
            best_f1 = max(rows, key=lambda row: row["F1 @ 0.50"])
            best_recall = max(rows, key=lambda row: row["Recall @ 0.50"])
            insight_cols = st.columns(3)
            insight_cols[0].metric("Highest PR-AUC", f"{best_pr['PR-AUC']:.3f}", f"{best_pr['Scenario']} · {best_pr['Model']}")
            insight_cols[1].metric("Highest F1 @ 0.50", f"{best_f1['F1 @ 0.50']:.3f}", f"{best_f1['Scenario']} · {best_f1['Model']}")
            insight_cols[2].metric("Highest recall @ 0.50", f"{best_recall['Recall @ 0.50']:.3f}", f"{best_recall['Scenario']} · {best_recall['Model']}")
            st.info("In the supplied validation runs, S1 + XGBoost leads on PR-AUC and F1 at threshold 0.50. The highest recall belongs to Logistic Regression, but its precision is much lower. This is a validation finding, not a claim that one model is universally best.")
        else:
            st.info("No validation comparison records are available yet. Run a scenario in Experiment Lab.")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        st.error(f"Could not read validation comparison records: {exc}")
else:
    st.info("Validation comparison records have not been created yet. Run the scenarios from Experiment Lab first.")

st.subheader("Final held-out test evaluation")
if test_path.is_file():
    try:
        final_report = json.loads(test_path.read_text(encoding="utf-8"))
        metrics = final_report["metrics"]
        st.success(f"Final evaluation is recorded for {final_report['scenario']} + {final_report['model']} at threshold {final_report['threshold']:.2f}. Completed {final_report['completed_utc']} UTC.")
        st.caption(f"The final model was fit on {final_report['fit_rows']:,} train + validation rows. The held-out test contains {final_report['test_rows']:,} rows, including {final_report['test_fraud']:,} Fraud cases.")
        metric_cols = st.columns(4)
        metric_cols[0].metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
        metric_cols[1].metric("Precision", f"{metrics['precision']:.3f}")
        metric_cols[2].metric("Recall", f"{metrics['recall']:.3f}")
        metric_cols[3].metric("F1", f"{metrics['f1']:.3f}")
        matrix = pd.DataFrame(
            [[metrics['tn'], metrics['fp']], [metrics['fn'], metrics['tp']]],
            index=pd.Index(["Actual Normal", "Actual Fraud"], name="Actual / predicted"),
            columns=["Predicted Normal", "Predicted Fraud"],
        )
        st.markdown("**Final test confusion matrix**")
        st.dataframe(matrix, width="stretch", alt="Confusion matrix from the one-time final test evaluation")
        st.caption(f"At threshold {final_report['threshold']:.2f}, the model caught {metrics['tp']} of {final_report['test_fraud']} Fraud cases, missed {metrics['fn']}, and raised {metrics['fp']} false alarms. The test set must not be used to tune the model or threshold.")
        st.download_button(
            "Download research evaluation summary",
            data=json.dumps({"validation": validation, "final_test": final_report}, ensure_ascii=False, indent=2),
            file_name="creditguard_research_evaluation.json",
            mime="application/json",
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        st.error(f"Could not read the final test report: {exc}")
else:
    st.warning("The final test report is not available yet. Run the locked evaluation from Model Performance after finalizing the candidate.")
    st.page_link("pages/5_Model_Performance.py", label="Open Model Performance", icon=":material/assessment:")

st.subheader("Interpretation and limitations")
st.markdown("""
- The validation and test sets each contain only 71 Fraud cases, so a small number of transactions can shift recall and F1 noticeably.
- The split is stratified and random; it does not measure performance on future transactions under time drift.
- Precision-recall results describe this dataset and setup; they are not evidence of production performance.
- Dashboard labels remain actual dataset labels. Model outputs are reported only in the experiment and evaluation pages.
""")
