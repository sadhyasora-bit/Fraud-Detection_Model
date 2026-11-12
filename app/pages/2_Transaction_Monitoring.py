
"""Transaction Monitoring page for CreditGuard AI."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.utils import load_dataset_settings, load_dataset, resolve_dataset_path
from src.model_inference import load_fraud_model
from src.agent.orchestrator import run_agent


st.set_page_config(
    page_title="CreditGuard AI | Transaction Monitoring",
    page_icon="🛡️",
    layout="wide",
)


# ==================================================
# CONFIGURATION
# ==================================================

DEFAULT_THRESHOLD = 0.50
DEFAULT_ANALYSIS_TRIGGER = 0.10
DEFAULT_REVIEW_THRESHOLD = 0.20
DEFAULT_ESCALATE_THRESHOLD = 0.70


# ==================================================
# MODEL FEATURE CONTRACT
# ==================================================

@st.cache_data(show_spinner=False)
def get_model_features() -> list[str]:
    model = load_fraud_model()
    return list(model.feature_names_in_)


def prepare_transaction(
    dataframe: pd.DataFrame,
    expected_features: list[str],
) -> pd.DataFrame:

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if len(dataframe) != 1:
        raise ValueError("Exactly one transaction must be selected.")

    missing = [
        feature for feature in expected_features
        if feature not in dataframe.columns
    ]

    if missing:
        raise ValueError(f"Missing model features: {missing}")

    transaction = dataframe.loc[:, expected_features].copy()

    for column in expected_features:
        transaction[column] = pd.to_numeric(
            transaction[column], errors="coerce"
        )

    if transaction.isna().any().any():
        raise ValueError("Transaction contains missing or invalid values.")

    if not np.isfinite(transaction.to_numpy(dtype=float)).all():
        raise ValueError("Transaction contains infinite values.")

    return transaction.reset_index(drop=True)


# ==================================================
# RULE-BASED EXPLANATION
# ==================================================

def generate_explanation(result: dict) -> dict:
    """Generate a human-readable explanation from Agent outputs."""

    ml = result["ml_prediction"]
    risk = result["risk_assessment"]
    analysis = result["transaction_analysis"]
    decision = result["decision"]
    policy = result["policy_configuration"]

    probability = float(ml["fraud_probability"])
    risk_level = risk["risk_level"]
    classification = ml["classification"]
    action = decision["recommended_action"]

    review_threshold = float(policy["review_threshold"])
    escalate_threshold = float(policy["escalate_threshold"])
    analysis_trigger = float(policy["analysis_trigger"])

    # Main interpretation
    if risk_level == "LOW":
        interpretation = (
            "The model estimates a low likelihood of fraudulent activity. "
            "The transaction's estimated fraud probability is below "
            "the configured review threshold."
        )
    elif risk_level == "MEDIUM":
        interpretation = (
            "The transaction has a moderate estimated fraud risk. "
            "It requires additional attention because its fraud "
            "probability has reached the review threshold."
        )
    else:
        interpretation = (
            "The transaction has a high estimated fraud risk. "
            "Its fraud probability has reached or exceeded the "
            "configured escalation threshold."
        )

    # Evidence
    evidence = []

    evidence.append(
        f"The XGBoost model classified this transaction as {classification}."
    )

    evidence.append(
        f"The estimated fraud probability is {probability:.4%}."
    )

    if probability < review_threshold:
        evidence.append(
            f"The probability is below the review threshold "
            f"({review_threshold:.0%})."
        )
    elif probability < escalate_threshold:
        evidence.append(
            f"The probability has reached the review threshold "
            f"({review_threshold:.0%}) but remains below the "
            f"escalation threshold ({escalate_threshold:.0%})."
        )
    else:
        evidence.append(
            f"The probability has reached or exceeded the escalation "
            f"threshold ({escalate_threshold:.0%})."
        )

    if analysis.get("analysis_skipped"):
        evidence.append(
            f"Additional transaction analysis was skipped because "
            f"the probability is below the analysis trigger "
            f"({analysis_trigger:.0%})."
        )
    else:
        indicators = analysis.get("transaction_indicators") or {}

        if indicators:
            active_indicators = [
                str(name).replace("_", " ").title()
                for name, value in indicators.items()
                if bool(value)
            ]

            if active_indicators:
                evidence.append(
                    "Additional transaction indicators: "
                    + ", ".join(active_indicators)
                    + "."
                )
            else:
                evidence.append(
                    "No additional transaction indicators were flagged "
                    "by the current analysis rules."
                )

        if analysis.get("amount_category"):
            evidence.append(
                f"Transaction amount category: "
                f"{analysis['amount_category']}."
            )

        if analysis.get("time_category"):
            evidence.append(
                f"Transaction time category: "
                f"{analysis['time_category']}."
            )

    # Recommendation explanation
    action_explanations = {
        "ALLOW": (
            "The current policy recommends allowing this transaction "
            "because its assessed risk is low."
        ),
        "REVIEW": (
            "The current policy recommends manual review before "
            "a final operational decision is made."
        ),
        "ESCALATE": (
            "The current policy recommends escalating this transaction "
            "for further investigation because of its high assessed risk."
        ),
    }

    return {
        "interpretation": interpretation,
        "evidence": evidence,
        "action_explanation": action_explanations.get(
            action, "No explanation is available for this action."
        ),
        "risk_level": risk_level,
        "recommended_action": action,
    }


# ==================================================
# RESULT RENDERING
# ==================================================

def render_result(result: dict) -> None:

    if result.get("status") != "success":
        st.error("Agent execution failed.")

        st.write("Failed tool:", result.get("failed_tool", "Unknown"))
        st.error(result.get("error", {}).get("message", "Unknown error"))

        with st.expander("Technical Details"):
            st.json(result.get("error", {}))
            st.write(result.get("execution_trace", []))

        return

    ml = result["ml_prediction"]
    threshold = result["threshold_analysis"]
    analysis = result["transaction_analysis"]
    risk = result["risk_assessment"]
    decision = result["decision"]

    probability = float(ml["fraud_probability"])
    classification = ml["classification"]
    risk_level = risk["risk_level"]
    action = decision["recommended_action"]

    explanation = generate_explanation(result)

    st.divider()
    st.subheader("Transaction Analysis Result")

    # Main metrics
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Fraud Probability",
        f"{probability:.4%}",
    )

    col2.metric(
        "ML Classification",
        classification,
    )

    col3.metric(
        "Risk Level",
        risk_level,
    )

    st.progress(
        min(max(probability, 0.0), 1.0),
        text=f"Estimated fraud probability: {probability:.4%}",
    )

    # Human-readable interpretation
    st.divider()
    st.subheader("System Interpretation")

    st.write(explanation["interpretation"])

    # Evidence
    st.markdown("#### Why did the system produce this result?")

    for item in explanation["evidence"]:
        st.markdown(f"- {item}")

    # Recommended action
    st.divider()
    st.subheader("Recommended Action")

    if action == "ALLOW":
        st.success(f"**{action}** — Low estimated risk")
    elif action == "REVIEW":
        st.warning(f"**{action}** — Manual review recommended")
    else:
        st.error(f"**{action}** — Further investigation recommended")

    st.write(explanation["action_explanation"])

    st.caption(
        "This is a system recommendation only. "
        "It does not automatically approve, reject, or block "
        "a real financial transaction."
    )

    # Additional transaction analysis
    st.divider()
    st.subheader("Transaction Evidence")

    if analysis.get("analysis_skipped"):
        st.info(
            "Additional transaction analysis was not executed. "
            + str(analysis.get("reason", ""))
        )
    else:
        evidence_col1, evidence_col2 = st.columns(2)

        evidence_col1.metric(
            "Transaction Amount",
            f"{analysis.get('transaction_amount', 0):,.2f}",
        )

        evidence_col2.metric(
            "Transaction Time",
            f"{analysis.get('transaction_time', 0):,.0f}",
        )

        st.write(
            "**Amount Category:**",
            analysis.get("amount_category"),
        )

        st.write(
            "**Time Category:**",
            analysis.get("time_category"),
        )

        indicators = analysis.get("transaction_indicators") or {}

        if indicators:
            st.markdown("**Transaction Indicators**")

            for name, value in indicators.items():
                st.write(
                    f"- {str(name).replace('_', ' ').title()}: "
                    f"{'Detected' if value else 'Not detected'}"
                )

    # Technical information
    with st.expander("Technical Details", expanded=False):

        st.markdown("#### Threshold Analysis")
        st.json(threshold)

        st.markdown("#### Risk Assessment")
        st.json(risk)

        st.markdown("#### Agent Execution Trace")

        for index, step in enumerate(
            result.get("execution_trace", []), start=1
        ):
            st.write(f"{index}. {step}")

        st.markdown("#### Policy Configuration")
        st.json(result.get("policy_configuration", {}))


# ==================================================
# PAGE HEADER
# ==================================================

st.title("Transaction Monitoring")

st.caption(
    "CreditGuard AI · ML-powered transaction analysis "
    "with conditional Agent orchestration"
)

st.markdown(
    "Analyze an individual transaction using the trained "
    "XGBoost model and CreditGuard AI Agent."
)


# ==================================================
# LOAD MODEL
# ==================================================

try:
    expected_features = get_model_features()

except Exception as error:
    st.error(f"Unable to load the model feature contract: {error}")
    st.stop()


# ==================================================
# LOAD CONFIGURED DATASET
# ==================================================

st.subheader("1. Transaction Selection")

settings = load_dataset_settings()
dataset_path = resolve_dataset_path(settings)

if dataset_path is None:
    st.warning(
        "No active dataset was found. "
        "Please upload and configure a dataset through Settings & Dataset."
    )
    st.stop()

try:
    dataset = load_dataset(dataset_path)

except Exception as error:
    st.error(f"Could not load the active dataset: {error}")
    st.stop()

st.caption(f"Active dataset: {dataset_path.name}")

if dataset.empty:
    st.warning("The configured dataset is empty.")
    st.stop()

missing = [
    feature for feature in expected_features
    if feature not in dataset.columns
]

if missing:
    st.error(f"Dataset is missing model features: {missing}")
    st.stop()

info1, info2, info3 = st.columns(3)

info1.metric("Total Transactions", f"{len(dataset):,}")
info2.metric("Model Features", len(expected_features))
info3.metric("Dataset", dataset_path.name)

selected_row = st.number_input(
    "Select Transaction Row",
    min_value=0,
    max_value=len(dataset) - 1,
    value=0,
    step=1,
)

transaction_data = dataset.iloc[[int(selected_row)]].copy()

transaction_label = f"Dataset row {int(selected_row)}"

with st.expander("Selected Transaction Preview"):
    st.dataframe(
        transaction_data,
        width="stretch",
        hide_index=True,
    )


# ==================================================
# AGENT CONFIGURATION
# ==================================================

st.divider()
st.subheader("2. Agent Configuration")

with st.expander("Advanced Configuration", expanded=False):

    threshold = st.slider(
        "Classification Threshold",
        min_value=0.01,
        max_value=0.99,
        value=DEFAULT_THRESHOLD,
        step=0.01,
    )

    analysis_trigger = st.slider(
        "Transaction Analysis Trigger",
        min_value=0.01,
        max_value=0.99,
        value=DEFAULT_ANALYSIS_TRIGGER,
        step=0.01,
    )

    review_threshold = st.slider(
        "Review Threshold",
        min_value=0.01,
        max_value=0.98,
        value=DEFAULT_REVIEW_THRESHOLD,
        step=0.01,
    )

    escalate_threshold = st.slider(
        "Escalation Threshold",
        min_value=0.02,
        max_value=0.99,
        value=DEFAULT_ESCALATE_THRESHOLD,
        step=0.01,
    )

    if review_threshold >= escalate_threshold:
        st.error(
            "Review Threshold must be lower than Escalation Threshold."
        )


# ==================================================
# RUN AGENT
# ==================================================

st.divider()
st.subheader("3. Run Fraud Analysis")

can_analyze = review_threshold < escalate_threshold

if st.button(
    "Analyze Transaction",
    type="primary",
    width="stretch",
    disabled=not can_analyze,
):

    try:
        prepared_transaction = prepare_transaction(
            transaction_data,
            expected_features,
        )

        with st.spinner(
            "CreditGuard AI Agent is analyzing the transaction..."
        ):
            result = run_agent(
                transaction=prepared_transaction,
                threshold=threshold,
                analysis_trigger=analysis_trigger,
                review_threshold=review_threshold,
                escalate_threshold=escalate_threshold,
            )

        st.session_state["last_agent_result"] = result
        st.session_state["last_transaction_label"] = transaction_label

    except Exception as error:
        st.error(f"Unable to analyze transaction: {error}")


# ==================================================
# DISPLAY RESULT
# ==================================================

if "last_agent_result" in st.session_state:

    st.divider()
    st.subheader("4. Analysis Output")

    st.caption(
        "Transaction: "
        + str(
            st.session_state.get(
                "last_transaction_label", "Unknown"
            )
        )
    )

    render_result(st.session_state["last_agent_result"])
