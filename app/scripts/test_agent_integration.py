"""Integration test: deterministic CreditGuard pipeline + Gemini assessment.

Run from the app directory:
    python -m scripts.test_agent_integration

Uses the same S1 validation samples and inverse-scaling procedure as
scripts.test_orchestrator. Gemini assessment is additional and must never
modify the deterministic system decision.
"""
from pathlib import Path
import time

from src.ml_data import load_scenario_splits
from src.model_inference import load_s1_scaler
from src.agent.orchestrator import run_agent_with_assessment


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    # Load the same S1 validation data used by test_orchestrator.py.
    splits = load_scenario_splits(
        root=PROJECT_ROOT / "data" / "preprocessed",
        scenario="S1",
        include_test=False,
    )

    scaler = load_s1_scaler()

    # Select one normal and one fraud sample, using the existing test convention.
    test_indices = {
        "NORMAL": splits.y_val[splits.y_val == 0].index[0],
        "FRAUD": splits.y_val[splits.y_val == 1].index[0],
    }

    for label, index in test_indices.items():
        sample = splits.X_val.loc[[index]].copy()

        # Convert the sample back to original feature scale.
        raw_sample = sample.copy()
        raw_sample.iloc[:, :] = scaler.inverse_transform(sample)

        print("\\n" + "=" * 58, flush=True)
        print(f"ACTUAL TRANSACTION: {label}", flush=True)
        print("=" * 58, flush=True)

        started = time.perf_counter()
        result = run_agent_with_assessment(
            transaction=raw_sample,
            threshold=0.50,
            analysis_trigger=0.10,
            review_threshold=0.20,
            escalate_threshold=0.70,
            assessment_timeout=40,
        )
        elapsed = time.perf_counter() - started

        print(f"\\nTotal elapsed: {elapsed:.2f}s", flush=True)
        print("Integration status:", result.get("status"), flush=True)

        system_result = result.get("system_result")
        if not isinstance(system_result, dict):
            raise AssertionError(
                f"{label}: system_result is missing or invalid: {result}"
            )

        print("\\nDeterministic Execution Trace:", flush=True)
        for step in system_result.get("execution_trace", []):
            print(f"- {step}", flush=True)

        print("\\nML Prediction:", flush=True)
        print(system_result.get("ml_prediction"), flush=True)

        print("\\nRisk Assessment:", flush=True)
        print(system_result.get("risk_assessment"), flush=True)

        print("\\nDeterministic Final Decision:", flush=True)
        print(system_result.get("decision"), flush=True)

        assessment = result.get("agent_assessment")
        print("\\nGemini Agent Assessment:", flush=True)
        if isinstance(assessment, dict) and assessment.get("status") == "success":
            print(assessment.get("assessment"), flush=True)
        else:
            print("Assessment unavailable; deterministic result remains available.", flush=True)
            print(assessment, flush=True)

        # Integrity checks: the Gemini layer must not alter the deterministic decision.
        if result.get("decision_modified") is not False:
            raise AssertionError(f"{label}: decision_modified must be False.")

        if system_result.get("status") != "success":
            raise AssertionError(f"{label}: deterministic pipeline did not succeed.")

        decision = system_result.get("decision", {})
        if decision.get("execution_status") != "RECOMMENDATION_ONLY":
            raise AssertionError(
                f"{label}: unexpected execution status: "
                f"{decision.get('execution_status')}"
            )

        if label == "NORMAL" and decision.get("recommended_action") != "ALLOW":
            raise AssertionError(
                "NORMAL validation sample did not produce the expected ALLOW decision."
            )

        if label == "FRAUD" and decision.get("recommended_action") != "ESCALATE":
            raise AssertionError(
                "FRAUD validation sample did not produce the expected ESCALATE decision."
            )

        print(f"\\n{label} INTEGRATION TEST: PASS", flush=True)

    print("\\nCREDITGUARD AI AGENT INTEGRATION TEST: SUCCESS", flush=True)


if __name__ == "__main__":
    main()
