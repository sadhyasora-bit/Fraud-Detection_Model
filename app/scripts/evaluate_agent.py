
import time
from unittest.mock import patch

import pandas as pd

from src.agent.orchestrator import run_agent


# ==========================================
# 1. TEST DATA
# ==========================================

transaction = pd.DataFrame(
    [[0.0] * 30],
    columns=[
        "Time",
        *[f"V{i}" for i in range(1, 29)],
        "Amount",
    ],
)

scenarios = [
    {
        "name": "Very Low Risk",
        "probability": 0.05,
        "expected_analysis": False,
        "expected_risk": "LOW",
        "expected_action": "ALLOW",
    },
    {
        "name": "Low Risk",
        "probability": 0.30,
        "expected_analysis": True,
        "expected_risk": "LOW",
        "expected_action": "ALLOW",
    },
    {
        "name": "Medium Risk",
        "probability": 0.60,
        "expected_analysis": True,
        "expected_risk": "MEDIUM",
        "expected_action": "REVIEW",
    },
    {
        "name": "High Risk",
        "probability": 0.90,
        "expected_analysis": True,
        "expected_risk": "HIGH",
        "expected_action": "ESCALATE",
    },
]


# ==========================================
# 2. EVALUATION
# ==========================================

results = []

for scenario in scenarios:

    probability = scenario["probability"]

    mock_ml_result = {
        "tool": "ml_prediction",
        "status": "success",
        "prediction": int(probability >= 0.50),
        "fraud_probability": probability,
        "threshold": 0.50,
        "classification": (
            "FRAUD" if probability >= 0.50 else "NORMAL"
        ),
    }

    execution_times = []
    scenario_outputs = []

    # Repeat each scenario 3 times
    for _ in range(3):

        start_time = time.perf_counter()

        with patch(
            "src.agent.orchestrator.run_ml_prediction",
            return_value=mock_ml_result,
        ):
            result = run_agent(transaction)

        elapsed = time.perf_counter() - start_time

        execution_times.append(elapsed)
        scenario_outputs.append(result)

    # Use the first execution for accuracy checks
    result = scenario_outputs[0]

    actual_analysis = (
        "transaction_analysis" in result["execution_trace"]
    )

    tool_results = [
        result.get("ml_prediction"),
        result.get("threshold_analysis"),
        result.get("transaction_analysis"),
        result.get("risk_assessment"),
        result.get("decision"),
    ]

    successful_tools = sum(
        1
        for tool_result in tool_results
        if tool_result is not None
        and tool_result.get("status") == "success"
        and not tool_result.get("analysis_skipped", False)
    )

    executed_tools = sum(
        1
        for tool_name in result["execution_trace"]
        if not tool_name.endswith("_skipped")
    )

    tool_success = (
        successful_tools / executed_tools
        if executed_tools > 0
        else 0
    )

    decision = result.get("decision", {})

    routing_correct = (
        actual_analysis == scenario["expected_analysis"]
    )

    decision_correct = (
        result["status"] == "success"
        and result["risk_assessment"]["risk_level"]
        == scenario["expected_risk"]
        and decision.get("recommended_action")
        == scenario["expected_action"]
    )

    investigation_complete = (
        actual_analysis == scenario["expected_analysis"]
        and result["risk_assessment"]["status"] == "success"
        and decision.get("status") == "success"
    )

    # Compare repeated executions for consistency
    first_signature = (
        result["execution_trace"],
        result["risk_assessment"]["risk_level"],
        decision.get("recommended_action"),
    )

    consistency = all(
        (
            output["execution_trace"],
            output["risk_assessment"]["risk_level"],
            output["decision"].get("recommended_action"),
        ) == first_signature
        for output in scenario_outputs
    )

    results.append({
        "scenario": scenario["name"],
        "routing_correct": routing_correct,
        "tool_success": tool_success,
        "decision_correct": decision_correct,
        "investigation_complete": investigation_complete,
        "consistency": consistency,
        "processing_time": sum(execution_times) / len(execution_times),
    })


# ==========================================
# 3. CALCULATE METRICS
# ==========================================

total_scenarios = len(results)

tool_selection_accuracy = (
    sum(r["routing_correct"] for r in results)
    / total_scenarios
) * 100

tool_execution_success = (
    sum(r["tool_success"] for r in results)
    / total_scenarios
) * 100

decision_accuracy = (
    sum(r["decision_correct"] for r in results)
    / total_scenarios
) * 100

investigation_completeness = (
    sum(r["investigation_complete"] for r in results)
    / total_scenarios
) * 100

response_consistency = (
    sum(r["consistency"] for r in results)
    / total_scenarios
) * 100

average_processing_time = (
    sum(r["processing_time"] for r in results)
    / total_scenarios
)


# ==========================================
# 4. DISPLAY RESULTS
# ==========================================

print("\n" + "=" * 65)
print("CREDITGUARD AI - AGENT EVALUATION")
print("=" * 65)

for result in results:

    print(f"\nScenario: {result['scenario']}")
    print("Tool Selection:", result["routing_correct"])
    print("Tool Execution Success:", f"{result['tool_success'] * 100:.2f}%")
    print("Decision Accuracy:", result["decision_correct"])
    print("Investigation Complete:", result["investigation_complete"])
    print("Response Consistency:", result["consistency"])
    print(
        "Average Processing Time:",
        f"{result['processing_time'] * 1000:.3f} ms",
    )


print("\n" + "=" * 65)
print("OVERALL EVALUATION METRICS")
print("=" * 65)

print(f"Tool Selection Accuracy: {tool_selection_accuracy:.2f}%")
print(f"Tool Execution Success: {tool_execution_success:.2f}%")
print(f"Decision Accuracy: {decision_accuracy:.2f}%")
print(f"Investigation Completeness: {investigation_completeness:.2f}%")
print(f"Response Consistency: {response_consistency:.2f}%")
print(f"Average Processing Time: {average_processing_time * 1000:.3f} ms")

print("\n" + "=" * 65)
print("EVALUATION COMPLETED")
print("=" * 65)
