
from pathlib import Path
from unittest.mock import patch

from src.ml_data import load_scenario_splits
from src.model_inference import load_s1_scaler
from src.agent.orchestrator import run_agent


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load S1 validation data
splits = load_scenario_splits(
    root=PROJECT_ROOT / "data" / "preprocessed",
    scenario="S1",
    include_test=False,
)

scaler = load_s1_scaler()

# Select one normal sample
normal_index = splits.y_val[splits.y_val == 0].index[0]

# Get validation sample
sample = splits.X_val.loc[[normal_index]].copy()

# Convert to original feature scale
raw_sample = sample.copy()
raw_sample.iloc[:, :] = scaler.inverse_transform(sample)

# Simulate audit log failure
with patch(
    "src.agent.orchestrator.log_agent_event",
    side_effect=OSError("Simulated audit log failure"),
):
    result = run_agent(
        transaction=raw_sample,
        threshold=0.50,
        analysis_trigger=0.10,
    )

print("\n===== AUDIT FAILURE TEST =====")

print("\nAgent Status:")
print(result["status"])

print("\nExecution Trace:")
for step in result["execution_trace"]:
    print(f"- {step}")

print("\nFinal Decision:")
print(result["decision"])

# Assertions
assert result["status"] == "success", (
    "Agent should remain successful despite audit failure."
)

assert result["decision"]["status"] == "success", (
    "Decision should still be generated."
)

assert any(
    step.startswith("audit_log_failed:")
    for step in result["execution_trace"]
), "Audit failure should be recorded in execution trace."

print("\nTEST RESULT: PASSED")
