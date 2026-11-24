"""Allowlisted, read-only tools available to CreditGuard AI Assistant."""


def build_tool_registry(analysis_result: dict) -> dict:
    """Build a per-turn registry bound to the validated analysis result."""
    if not isinstance(analysis_result, dict) or analysis_result.get("status") != "success":
        raise ValueError("A successful analysis result is required.")

    def get_analysis_summary() -> dict:
        ml = analysis_result.get("ml_prediction", {})
        threshold = analysis_result.get("threshold_analysis", {})
        risk = analysis_result.get("risk_assessment", {})
        decision = analysis_result.get("decision", {})
        transaction = analysis_result.get("transaction_analysis", {})
        return {
            "fraud_probability": ml.get("fraud_probability"),
            "ml_classification": ml.get("classification"),
            "classification_threshold": threshold.get("threshold"),
            "classification_threshold_exceeded": threshold.get("exceeds_threshold"),
            "risk_level": risk.get("risk_level"),
            "risk_reason": risk.get("risk_reason"),
            "recommended_action": decision.get("recommended_action"),
            "decision_reason": decision.get("reason"),
            "transaction_analysis": {
                "analysis_skipped": transaction.get("analysis_skipped"),
                "reason": transaction.get("reason"),
                "amount_category": transaction.get("amount_category"),
                "time_category": transaction.get("time_category"),
                "indicators": transaction.get("transaction_indicators"),
            },
        }

    def get_policy_configuration() -> dict:
        return dict(analysis_result.get("policy_configuration", {}))

    return {
        "get_analysis_summary": get_analysis_summary,
        "get_policy_configuration": get_policy_configuration,
    }


def gemini_function_declarations() -> list[dict]:
    """JSON-schema declarations; converted to SDK types by gemini_client."""
    return [
        {
            "name": "get_analysis_summary",
            "description": "Retrieve the already-computed, read-only fraud analysis summary. Does not run or change the ML model.",
            "parameters": {"type": "OBJECT", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "get_policy_configuration",
            "description": "Retrieve the configured deterministic policy thresholds. This tool cannot modify them.",
            "parameters": {"type": "OBJECT", "properties": {}, "additionalProperties": False},
        },
    ]
