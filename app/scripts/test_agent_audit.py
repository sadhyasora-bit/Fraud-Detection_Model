
import json
from unittest.mock import patch

import pandas as pd

from src.agent.orchestrator import run_agent
from src.agent.tools.audit_log import AUDIT_LOG_PATH


transaction = pd.DataFrame(
    [[0.0] * 30],
    columns=[
        "Time",
        *[f"V{i}" for i in range(1, 29)],
        "Amount",
    ],
)

test_cases = [
    ("LOW", 0.05, False),
    ("HIGH", 0.90, True),
]

for name, probability, expected_analysis in test_cases:

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

    # Record current audit log length
    previous_count = (
        len(AUDIT_LOG_PATH.read_text(encoding="utf-8").splitlines())
        if AUDIT_LOG_PATH.exists()
        else 0
    )

    with patch(
        "src.agent.orchestrator.run_ml_prediction",
        return_value=mock_ml_result,
    ), patch(
        "src.agent.orchestrator.run_transaction_analysis",
        return_value=mock_transaction_result,
    ) as mock_analysis:

        result = run_agent(transaction)

    # Read newly generated audit records
    with AUDIT_LOG_PATH.open("r", encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    new_records = records[previous_count:]

    event_types = [
        record["event_type"]
        for record in new_records
    ]

    skipped_events = [
        record for record in new_records
        if record["event_type"] == "tool_skipped"
    ]

    print(f"\n===== {name} AUDIT TEST =====")
    print("Transaction Analysis Executed:", mock_analysis.called)
    print("Execution Trace:", result["execution_trace"])
    print("Audit Events:", event_types)
    print("New Audit Records:", len(new_records))

    assert mock_analysis.called == expected_analysis

    if expected_analysis:
        assert len(skipped_events) == 0
        assert "transaction_analysis" in result["execution_trace"]
    else:
        assert len(skipped_events) == 1
        assert skipped_events[0]["tool_name"] == "transaction_analysis"
        assert "transaction_analysis_skipped" in result["execution_trace"]

    assert len(new_records) == 5

    print("TEST RESULT: PASSED")

print("\nALL AUDIT INTEGRATION TESTS PASSED")
