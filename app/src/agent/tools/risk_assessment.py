
"""Risk Assessment Tool for CreditGuard AI."""


def run_risk_assessment(
    ml_result: dict,
    threshold_result: dict,
    transaction_result: dict,
    review_threshold: float = 0.20,
    escalate_threshold: float = 0.70,
) -> dict:
    """
    Assess transaction risk using ML prediction and policy thresholds.
    """

    results = [
        ml_result,
        threshold_result,
        transaction_result,
    ]

    for result in results:
        if result.get("status") != "success":
            raise ValueError(
                "All input tools must return successful results."
            )

    if not (
        0 < review_threshold < escalate_threshold <= 1
    ):
        raise ValueError(
            "Thresholds must satisfy "
            "0 < review_threshold < escalate_threshold <= 1."
        )

    probability = ml_result["fraud_probability"]

    if not isinstance(probability, (int, float)):
        raise ValueError("Fraud probability must be numeric.")

    if not 0 <= probability <= 1:
        raise ValueError("Fraud probability must be between 0 and 1.")

    if probability >= escalate_threshold:
        risk_level = "HIGH"
        risk_reason = (
            "Fraud probability reached the escalation threshold."
        )

    elif probability >= review_threshold:
        risk_level = "MEDIUM"
        risk_reason = (
            "Fraud probability reached the review threshold."
        )

    else:
        risk_level = "LOW"
        risk_reason = (
            "Fraud probability is below the review threshold."
        )

    return {
        "tool": "risk_assessment",
        "status": "success",
        "fraud_probability": probability,
        "classification_threshold_exceeded": threshold_result["exceeds_threshold"],
        "review_threshold": review_threshold,
        "escalate_threshold": escalate_threshold,
        "transaction_amount": transaction_result[
            "transaction_amount"
        ],
        "risk_level": risk_level,
        "risk_reason": risk_reason,
    }
