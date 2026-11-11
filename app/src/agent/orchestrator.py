
"""Agent Orchestrator for CreditGuard AI."""

import math
from numbers import Real

from src.agent.tools.ml_prediction import run_ml_prediction
from src.agent.tools.threshold_analysis import run_threshold_analysis
from src.agent.tools.transaction_analysis import run_transaction_analysis
from src.agent.tools.risk_assessment import run_risk_assessment
from src.agent.tools.decision import run_decision
from src.agent.tools.audit_log import log_agent_event


def _validate_probability(value, name):
    """Validate probability-like parameters."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a numeric value.")

    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")

    if not 0 < value < 1:
        raise ValueError(f"{name} must be between 0 and 1.")


def _execute_tool(tool_name, tool_function, execution_trace, **kwargs):
    """Execute one tool and record its result or failure."""

    try:
        result = tool_function(**kwargs)

        if result.get("status") != "success":
            raise RuntimeError(
                result.get("error", "Tool returned an unsuccessful status.")
            )

        execution_trace.append(tool_name)

        log_agent_event(
            event_type="tool_execution",
            tool_name=tool_name,
            input_data=kwargs if tool_name != "ml_prediction" else {
                "threshold": kwargs["threshold"]
            },
            output_data=result,
            status="success",
        )

        return result, None

    except Exception as error:
        execution_trace.append(f"{tool_name}_failed")

        failure_result = {
            "tool": tool_name,
            "status": "failed",
            "error_type": type(error).__name__,
            "message": str(error),
        }

        log_agent_event(
            event_type="tool_failure",
            tool_name=tool_name,
            input_data={"tool": tool_name},
            output_data=failure_result,
            status="failed",
        )

        return None, failure_result


def _failure_response(tool_name, error_result, execution_trace):
    """Build a structured response when a tool fails."""

    return {
        "status": "failed",
        "agent": "CreditGuard AI",
        "failed_tool": tool_name,
        "error": error_result,
        "execution_trace": execution_trace,
        "decision": {
            "status": "not_generated",
            "message": "Decision was not generated because a tool failed.",
        },
    }


def run_agent(
    transaction,
    threshold: float = 0.50,
    analysis_trigger: float = 0.10,
    review_threshold: float = 0.20,
    escalate_threshold: float = 0.70,
) -> dict:
    """
    Orchestrate fraud detection tools using conditional routing.
    """

    execution_trace = []

    # 1. Validate input parameters
    try:
        _validate_probability(threshold, "threshold")
        _validate_probability(analysis_trigger, "analysis_trigger")
        _validate_probability(review_threshold, "review_threshold")
        _validate_probability(escalate_threshold, "escalate_threshold")

        if review_threshold >= escalate_threshold:
            raise ValueError(
                "review_threshold must be lower than escalate_threshold."
            )

    except ValueError as error:
        return {
            "status": "failed",
            "agent": "CreditGuard AI",
            "failed_tool": "input_validation",
            "error": {
                "error_type": type(error).__name__,
                "message": str(error),
            },
            "execution_trace": execution_trace,
            "decision": {
                "status": "not_generated",
                "message": "Decision was not generated because input is invalid.",
            },
        }

    # 2. ML Prediction
    ml_result, error_result = _execute_tool(
        "ml_prediction",
        run_ml_prediction,
        execution_trace,
        transaction=transaction,
        threshold=threshold,
    )

    if error_result:
        return _failure_response(
            "ml_prediction", error_result, execution_trace
        )

    probability = ml_result["fraud_probability"]

    # 3. Threshold Analysis
    threshold_result, error_result = _execute_tool(
        "threshold_analysis",
        run_threshold_analysis,
        execution_trace,
        fraud_probability=probability,
        threshold=threshold,
    )

    if error_result:
        return _failure_response(
            "threshold_analysis", error_result, execution_trace
        )

    # 4. Conditional Transaction Analysis
    if probability >= analysis_trigger:

        transaction_result, error_result = _execute_tool(
            "transaction_analysis",
            run_transaction_analysis,
            execution_trace,
            transaction=transaction,
        )

        if error_result:
            return _failure_response(
                "transaction_analysis", error_result, execution_trace
            )

    else:
        transaction_result = {
            "tool": "transaction_analysis",
            "status": "success",
            "transaction_amount": None,
            "analysis_skipped": True,
            "reason": (
                "Fraud probability is below the analysis trigger."
            ),
        }

        execution_trace.append("transaction_analysis_skipped")

        log_agent_event(
            event_type="tool_skipped",
            tool_name="transaction_analysis",
            input_data={
                "fraud_probability": probability,
                "analysis_trigger": analysis_trigger,
            },
            output_data=transaction_result,
            status="success",
        )

    # 5. Risk Assessment
    risk_result, error_result = _execute_tool(
        "risk_assessment",
        run_risk_assessment,
        execution_trace,
        ml_result=ml_result,
        threshold_result=threshold_result,
        transaction_result=transaction_result,
        review_threshold=review_threshold,
        escalate_threshold=escalate_threshold,
    )

    if error_result:
        return _failure_response(
            "risk_assessment", error_result, execution_trace
        )

    # 6. Decision
    decision_result, error_result = _execute_tool(
        "decision",
        run_decision,
        execution_trace,
        risk_result=risk_result,
    )

    if error_result:
        return _failure_response(
            "decision", error_result, execution_trace
        )

    return {
        "status": "success",
        "agent": "CreditGuard AI",
        "execution_trace": execution_trace,
        "ml_prediction": ml_result,
        "threshold_analysis": threshold_result,
        "transaction_analysis": transaction_result,
        "risk_assessment": risk_result,
        "decision": decision_result,
        "policy_configuration": {
            "classification_threshold": threshold,
            "analysis_trigger": analysis_trigger,
            "review_threshold": review_threshold,
            "escalate_threshold": escalate_threshold,
        },
    }
