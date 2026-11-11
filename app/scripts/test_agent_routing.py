
from unittest.mock import patch

from src.agent.orchestrator import run_agent


# Simulated fraud probabilities
# Simulated fraud probabilities
test_cases = [
    ("LOW", 0.05, False, "LOW", "ALLOW"),
    ("REVIEW_BOUNDARY", 0.20, True, "MEDIUM", "REVIEW"),
    ("MEDIUM", 0.30, True, "MEDIUM", "REVIEW"),
    ("HIGH_BOUNDARY", 0.70, True, "HIGH", "ESCALATE"),
    ("VERY_HIGH", 0.90, True, "HIGH", "ESCALATE"),
]

# Dummy transaction with 30 features
import pandas as pd

transaction = pd.DataFrame(
    [[0.0] * 30],
    columns=[
        "Time",
        *[f"V{i}" for i in range(1, 29)],
        "Amount",
    ],
)


for name, probability, expected_analysis, expected_risk, expected_action in test_cases:

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

    mock_transaction_result = {
        "tool": "transaction_analysis",
        "status": "success",
        "transaction_amount": 100.0,
        "transaction_time": 1000.0,
        "amount_category": "STANDARD_AMOUNT",
        "available_features": 30,
    }

    with patch(
        "src.agent.orchestrator.run_ml_prediction",
        return_value=mock_ml_result,
    ), patch(
        "src.agent.orchestrator.run_transaction_analysis",
        return_value=mock_transaction_result,
    ) as mock_analysis:

        result = run_agent(transaction)

    analysis_executed = mock_analysis.called

    print(f"\n===== {name} =====")
    print("Fraud Probability:", probability)
    print("Transaction Analysis Executed:", analysis_executed)
    print("Risk Level:", result["risk_assessment"]["risk_level"])
    print("Decision:", result["decision"]["recommended_action"])

    assert analysis_executed == expected_analysis
    assert result["risk_assessment"]["risk_level"] == expected_risk
    assert result["decision"]["recommended_action"] == expected_action

    print("TEST RESULT: PASSED")

print("\nALL ROUTING TESTS PASSED")
