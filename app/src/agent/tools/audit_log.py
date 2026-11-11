"""Audit Log Tool for CreditGuard AI."""

import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

AUDIT_LOG_PATH = (
    PROJECT_ROOT / "data" / "logs" / "agent_audit.jsonl"
)


def log_agent_event(
    event_type: str,
    tool_name: str,
    input_data: dict,
    output_data: dict,
    status: str = "success",
) -> dict:
    """Record an agent event into a JSONL audit log."""

    if not all([
        isinstance(event_type, str) and event_type.strip(),
        isinstance(tool_name, str) and tool_name.strip(),
        isinstance(input_data, dict),
        isinstance(output_data, dict),
    ]):
        raise ValueError("Invalid audit log input.")

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "tool_name": tool_name,
        "status": status,
        "input": input_data,
        "output": output_data,
    }

    AUDIT_LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(record, ensure_ascii=False, default=str) + "\n"
        )

    return {
        "tool": "audit_log",
        "status": "success",
        "log_file": str(AUDIT_LOG_PATH),
        "recorded_event": event_type,
        "recorded_tool": tool_name,
    }