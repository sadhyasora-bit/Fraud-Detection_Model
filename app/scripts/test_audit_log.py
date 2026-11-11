import json

from src.agent.tools.audit_log import (
    log_agent_event,
    AUDIT_LOG_PATH,
)

# Simulate an ML Prediction event
result = log_agent_event(
    event_type="tool_execution",
    tool_name="ml_prediction",
    input_data={
        "transaction_id": "TEST-001",
        "threshold": 0.50,
    },
    output_data={
        "prediction": 1,
        "fraud_probability": 0.99978,
        "classification": "FRAUD",
    },
)

print("\n===== AUDIT LOG RESULT =====")

for key, value in result.items():
    print(f"{key}: {value}")

# Verify saved record
print("\n===== SAVED AUDIT RECORD =====")

with AUDIT_LOG_PATH.open("r", encoding="utf-8") as file:
    records = file.readlines()

latest_record = json.loads(records[-1])

for key, value in latest_record.items():
    print(f"{key}: {value}")

print("\nTotal audit records:", len(records))