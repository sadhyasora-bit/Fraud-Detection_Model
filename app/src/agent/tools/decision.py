
"""Decision Tool for CreditGuard AI."""


def run_decision(
    risk_result: dict,
) -> dict:
    """
    Generate a recommended action based on risk assessment.
    """

    if risk_result.get("status") != "success":
        raise ValueError(
            "Risk Assessment must complete successfully."
        )

    risk_level = risk_result.get("risk_level")

    decisions = {
        "LOW": {
            "recommended_action": "ALLOW",
            "priority": "LOW",
            "reason": (
                "Fraud probability is below the review threshold."
            ),
        },
        "MEDIUM": {
            "recommended_action": "REVIEW",
            "priority": "MEDIUM",
            "reason": (
                "Transaction requires further investigation."
            ),
        },
        "HIGH": {
            "recommended_action": "ESCALATE",
            "priority": "HIGH",
            "reason": (
                "Transaction meets the escalation criteria."
            ),
        },
    }

    if risk_level not in decisions:
        raise ValueError(
            f"Unknown risk level: {risk_level}"
        )

    decision = decisions[risk_level]

    return {
        "tool": "decision",
        "status": "success",
        "risk_level": risk_level,
        "recommended_action": decision["recommended_action"],
        "priority": decision["priority"],
        "reason": decision["reason"],
        "execution_status": "RECOMMENDATION_ONLY",
    }
