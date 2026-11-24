"""Privacy-aware audit logging for CreditGuard AI."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUDIT_LOG_PATH = PROJECT_ROOT / "data" / "logs" / "agent_audit.jsonl"
_SENSITIVE_KEYS = {"transaction", "transaction_amount", "amount", "time", "card_number",
                   "cardnumber", "account_number", "accountnumber", "pan", "cvv", "cvc",
                   "customer_name", "email", "phone", "address", "features"}


def _sanitize(value, key=None):
    """Remove raw transaction/PII values before writing audit records."""
    if key and key.lower() in _SENSITIVE_KEYS:
        # Preserve only non-sensitive metadata for tabular inputs.
        if hasattr(value, "columns") and hasattr(value, "__len__"):
            return {"redacted": True, "row_count": len(value),
                    "columns": [str(col) for col in value.columns]}
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): _sanitize(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if hasattr(value, "columns") and hasattr(value, "__len__"):
        return {"type": type(value).__name__, "redacted": True,
                "row_count": len(value), "columns": [str(col) for col in value.columns]}
    return value


def log_agent_event(event_type: str, tool_name: str, input_data: dict,
                    output_data: dict, status: str = "success",
                    run_id: Optional[str] = None) -> dict:
    """Append one sanitized event to the JSONL audit log."""
    if not all([
        isinstance(event_type, str) and event_type.strip(),
        isinstance(tool_name, str) and tool_name.strip(),
        isinstance(input_data, dict), isinstance(output_data, dict),
        isinstance(status, str) and status.strip(),
    ]):
        raise ValueError("Invalid audit log input.")
    if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
        raise ValueError("run_id must be a non-empty string or None.")

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "event_type": event_type,
        "tool_name": tool_name,
        "status": status,
        "input": _sanitize(input_data),
        "output": _sanitize(output_data),
    }
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return {"tool": "audit_log", "status": "success", "log_file": str(AUDIT_LOG_PATH),
            "recorded_event": event_type, "recorded_tool": tool_name, "run_id": run_id}
