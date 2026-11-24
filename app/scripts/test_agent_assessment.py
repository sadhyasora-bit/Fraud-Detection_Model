"""Smoke test for Gemini Agent Assessment.

Run from the app directory:
    python -m scripts.test_agent_assessment
"""
import time

from src.agent.llm.agent_reasoning import run_agent_assessment


def main():
    # Synthetic example only; this does not represent a real transaction.
    sample_result = {
        "status": "success",
        "agent": "CreditGuard AI",
        "run_id": "LOCAL-AGENT-ASSESSMENT-TEST",
        "ml_prediction": {
            "status": "success",
            "fraud_probability": 0.0005,
            "classification": "NORMAL",
        },
        "threshold_analysis": {
            "status": "success",
            "exceeds_threshold": False,
            "threshold_result": "BELOW_THRESHOLD",
        },
        "transaction_analysis": {
            "status": "success",
            "analysis_skipped": True,
            "reason": "Fraud probability is below the analysis trigger.",
        },
        "risk_assessment": {
            "status": "success",
            "risk_level": "LOW",
        },
        "decision": {
            "status": "success",
            "recommended_action": "ALLOW",
            "execution_status": "RECOMMENDATION_ONLY",
        },
        "policy_configuration": {
            "classification_threshold": 0.50,
            "analysis_trigger": 0.10,
            "review_threshold": 0.20,
            "escalate_threshold": 0.70,
        },
    }

    print("Testing CreditGuard AI Agent Assessment...", flush=True)
    started = time.perf_counter()
    result = run_agent_assessment(sample_result, timeout=40)
    elapsed = time.perf_counter() - started

    print(f"\nElapsed: {elapsed:.2f}s")
    print("Status:", result.get("status"))
    print("Decision modified:", result.get("decision_modified"))

    if result.get("status") == "success":
        print("\nAI AGENT ASSESSMENT:")
        print(result["assessment"])
    else:
        print("\nAI AGENT ASSESSMENT UNAVAILABLE:")
        print(result.get("fallback_message"))
        print("Error:", result.get("error"))
        raise SystemExit(1)

    if result.get("decision_modified") is not False:
        raise SystemExit("FAILED: deterministic decision integrity check.")

    print("\nAGENT ASSESSMENT TEST: SUCCESS")


if __name__ == "__main__":
    main()
