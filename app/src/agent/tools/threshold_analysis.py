
"""Threshold Analysis Tool for CreditGuard AI."""


def run_threshold_analysis(
    fraud_probability: float,
    threshold: float = 0.50,
) -> dict:
    """
    Analyze fraud probability against the configured threshold.
    """

    if not 0 <= fraud_probability <= 1:
        raise ValueError(
            "Fraud probability must be between 0 and 1."
        )

    if not 0 < threshold < 1:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    exceeds_threshold = fraud_probability >= threshold

    return {
        "tool": "threshold_analysis",
        "status": "success",
        "fraud_probability": float(fraud_probability),
        "threshold": float(threshold),
        "exceeds_threshold": bool(exceeds_threshold),
        "threshold_result": (
            "ABOVE_THRESHOLD"
            if exceeds_threshold
            else "BELOW_THRESHOLD"
        ),
        "recommended_priority": (
            "HIGH"
            if exceeds_threshold
            else "STANDARD"
        ),
    }
