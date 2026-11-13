
"""Fraud Investigation page for CreditGuard AI."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="CreditGuard AI | Fraud Investigation",
    page_icon="🔎",
    layout="wide",
)


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

APP_ROOT = Path(__file__).resolve().parents[1]

AUDIT_LOG_PATH = (
    APP_ROOT / "data" / "logs" / "agent_audit.jsonl"
)


# --------------------------------------------------
# Load and process audit logs
# --------------------------------------------------

@st.cache_data(show_spinner=False)
def load_audit_records() -> list[dict]:
    """Load valid JSONL audit records."""

    if not AUDIT_LOG_PATH.exists():
        return []

    records = []

    with AUDIT_LOG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                record = json.loads(line)

                if isinstance(record, dict):
                    records.append(record)

            except json.JSONDecodeError:
                continue

    return records


def build_investigation_cases(
    records: list[dict],
) -> list[dict]:
    """Group audit events into Agent analysis cases."""

    grouped = defaultdict(list)

    for record in records:

        run_id = record.get("run_id")

        if not run_id:
            continue

        grouped[run_id].append(record)

    cases = []

    for run_id, events in grouped.items():

        events.sort(
            key=lambda item: item.get("timestamp", "")
        )

        case = {
            "run_id": run_id,
            "timestamp": events[-1].get("timestamp"),
            "ml_prediction": {},
            "risk_assessment": {},
            "decision": {},
            "transaction_analysis": {},
            "events": events,
        }

        for event in events:

            tool_name = event.get("tool_name")
            output = event.get("output", {})

            if not isinstance(output, dict):
                continue

            if event.get("status") == "failed":
                continue

            if tool_name == "ml_prediction":
                case["ml_prediction"] = output

            elif tool_name == "risk_assessment":
                case["risk_assessment"] = output

            elif tool_name == "decision":
                case["decision"] = output

            elif tool_name == "transaction_analysis":
                case["transaction_analysis"] = output

        decision = case["decision"]

        # Only completed decisions enter the investigation queue.
        if not decision:
            continue

        action = decision.get("recommended_action")

        case["recommended_action"] = action
        case["risk_level"] = (
            case["risk_assessment"].get("risk_level", "UNKNOWN")
        )

        case["fraud_probability"] = (
            case["ml_prediction"].get("fraud_probability")
        )

        case["transaction_amount"] = (
            case["risk_assessment"].get("transaction_amount")
        )

        cases.append(case)

    cases.sort(
        key=lambda item: item.get("timestamp") or "",
        reverse=True,
    )

    return cases


def format_probability(value):
    if value is None:
        return "N/A"

    return f"{float(value):.4%}"


def format_amount(value):
    if value is None:
        return "N/A"

    return f"${float(value):,.2f}"


# --------------------------------------------------
# Page header
# --------------------------------------------------

st.title("Fraud Investigation")

st.caption(
    "CreditGuard AI · Investigation Queue and Agent Audit Trail"
)

st.markdown(
    """
    Review transactions recommended for further investigation
    and inspect the sequence of Agent decisions.
    """
)


# --------------------------------------------------
# Load records
# --------------------------------------------------

try:
    audit_records = load_audit_records()

except Exception as error:
    st.error(f"Unable to read audit log: {error}")
    st.stop()


if not audit_records:
    st.info(
        "No audit records found. Run a transaction analysis "
        "from Transaction Monitoring first."
    )
    st.stop()


cases = build_investigation_cases(audit_records)

if not cases:
    st.warning(
        "No completed Agent analyses with valid run_id were found."
    )
    st.stop()



# --------------------------------------------------
# Investigation overview
# --------------------------------------------------

st.subheader("1. Investigation Overview")

# All completed Agent analyses
total_analyses = len(cases)

allow_analyses = sum(
    case.get("recommended_action") == "ALLOW"
    for case in cases
)

review_analyses = sum(
    case.get("recommended_action") == "REVIEW"
    for case in cases
)

escalated_analyses = sum(
    case.get("recommended_action") == "ESCALATE"
    for case in cases
)

# Risk distribution across all completed analyses
low_risk = sum(
    case.get("risk_level") == "LOW"
    for case in cases
)

medium_risk = sum(
    case.get("risk_level") == "MEDIUM"
    for case in cases
)

high_risk = sum(
    case.get("risk_level") == "HIGH"
    for case in cases
)

# Overview metrics
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Analyses", total_analyses)
col2.metric("ALLOW", allow_analyses)
col3.metric("REVIEW", review_analyses)
col4.metric("ESCALATE", escalated_analyses)

st.caption("Risk Distribution")

risk_col1, risk_col2, risk_col3 = st.columns(3)

risk_col1.metric("Low Risk", low_risk)
risk_col2.metric("Medium Risk", medium_risk)
risk_col3.metric("High Risk", high_risk)


# --------------------------------------------------
# Investigation queue
# --------------------------------------------------

st.divider()

st.subheader("2. Investigation Queue")

investigation_cases = [
    case
    for case in cases
    if case.get("recommended_action") in ["REVIEW", "ESCALATE"]
]

total_cases = len(investigation_cases)

review_cases = sum(
    case.get("recommended_action") == "REVIEW"
    for case in investigation_cases
)

escalated_cases = sum(
    case.get("recommended_action") == "ESCALATE"
    for case in investigation_cases
)

queue_col1, queue_col2, queue_col3 = st.columns(3)

queue_col1.metric("Investigation Cases", total_cases)
queue_col2.metric("Review", review_cases)
queue_col3.metric("Escalated", escalated_cases)


# --------------------------------------------------
# Filters
# --------------------------------------------------

st.divider()

filter_col1, filter_col2 = st.columns(2)

with filter_col1:
    action_filter = st.selectbox(
        "Recommended Action",
        ["ALL", "REVIEW", "ESCALATE"],
    )

with filter_col2:
    risk_filter = st.selectbox(
        "Risk Level",
        ["ALL", "LOW", "MEDIUM", "HIGH"],
    )


filtered_cases = investigation_cases.copy()

if action_filter != "ALL":
    filtered_cases = [
        case
        for case in filtered_cases
        if case["recommended_action"] == action_filter
    ]

if risk_filter != "ALL":
    filtered_cases = [
        case
        for case in filtered_cases
        if case["risk_level"] == risk_filter
    ]


# --------------------------------------------------
# Queue table
# --------------------------------------------------

if not filtered_cases:

    st.info("No cases match the selected filters.")

else:

    queue_data = []

    for case in filtered_cases:

        queue_data.append({
            "Analysis ID": case["run_id"][:13] + "...",
            "Timestamp": case.get("timestamp", "N/A"),
            "Fraud Probability": format_probability(
                case.get("fraud_probability")
            ),
            "Risk Level": case["risk_level"],
            "Recommended Action": case["recommended_action"],
            "Amount": format_amount(
                case.get("transaction_amount")
            ),
        })

    queue_df = pd.DataFrame(queue_data)

    st.dataframe(
        queue_df,
        width="stretch",
        hide_index=True,
    )


# --------------------------------------------------
# Investigation detail
# --------------------------------------------------

st.divider()

st.subheader("2. Investigation Detail")

if filtered_cases:

    case_options = {
        (
            f'{case["run_id"][:13]}... | '
            f'{case["recommended_action"]} | '
            f'{format_probability(case.get("fraud_probability"))}'
        ): case
        for case in filtered_cases
    }

    selected_label = st.selectbox(
        "Select an investigation case",
        list(case_options.keys()),
    )

    selected_case = case_options[selected_label]

    ml = selected_case["ml_prediction"]
    risk = selected_case["risk_assessment"]
    decision = selected_case["decision"]
    analysis = selected_case["transaction_analysis"]

    st.caption(
        f'Analysis ID: {selected_case["run_id"]}'
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Fraud Probability",
        format_probability(
            selected_case.get("fraud_probability")
        ),
    )

    col2.metric(
        "Risk Level",
        selected_case["risk_level"],
    )

    col3.metric(
        "Recommended Action",
        selected_case["recommended_action"],
    )

    st.markdown("#### Decision Explanation")

    st.write(
        "**ML Classification:**",
        ml.get("classification", "N/A"),
    )

    st.write(
        "**Risk Reason:**",
        risk.get("risk_reason", "N/A"),
    )

    st.write(
        "**Decision Reason:**",
        decision.get("reason", "N/A"),
    )

    st.write(
        "**Transaction Amount:**",
        format_amount(
            risk.get("transaction_amount")
        ),
    )

    if analysis.get("analysis_skipped"):

        st.info(
            "Transaction analysis was skipped: "
            + str(analysis.get("reason", "No reason recorded."))
        )

    elif analysis:

        st.markdown("#### Transaction Analysis")

        st.write(
            "**Amount Category:**",
            analysis.get("amount_category", "N/A"),
        )

        st.write(
            "**Time Category:**",
            analysis.get("time_category", "N/A"),
        )

        st.write(
            "**Transaction Indicators:**",
            analysis.get("transaction_indicators", "N/A"),
        )

    st.info(
        "This investigation record represents an Agent analysis. "
        "The recommended action is not an actual transaction "
        "approval, rejection, or blocking operation."
    )


# --------------------------------------------------
# Audit trail
# --------------------------------------------------

st.divider()

st.subheader("3. Agent Audit Trail")

if filtered_cases:

    selected_events = selected_case["events"]

    st.caption(
        "Event history for the selected Analysis ID."
    )

    audit_table = []

    for event in selected_events:

        audit_table.append({
            "Timestamp": event.get("timestamp", "N/A"),
            "Event": event.get("event_type", "N/A"),
            "Tool": event.get("tool_name", "N/A"),
            "Status": event.get("status", "N/A"),
        })

    st.dataframe(
        pd.DataFrame(audit_table),
        width="stretch",
        hide_index=True,
    )

    with st.expander("View Raw Audit Events"):

        for index, event in enumerate(
            selected_events,
            start=1,
        ):

            st.markdown(
                f"**Event {index}: {event.get('tool_name')}**"
            )

            st.json(event)

else:

    st.info(
        "Select a case to inspect its audit trail."
    )


# --------------------------------------------------
# Data limitations
# --------------------------------------------------

st.divider()

with st.expander("Investigation Data Notes"):

    st.markdown(
        """
        - Each case is identified by its Agent `run_id`.
        - The queue only includes completed REVIEW and ESCALATE decisions.
        - Historical events without a `run_id` are excluded.
        - The current audit log does not store a permanent transaction ID
          or the complete original transaction features.
        - This page provides investigation visibility, not case assignment
          or case-resolution management.
        """
    )

st.caption(
    f"Audit records loaded: {len(audit_records):,} · "
    f"Completed analyses: {len(cases):,}"
)
