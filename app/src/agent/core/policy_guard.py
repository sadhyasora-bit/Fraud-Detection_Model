"""Policy guard for CreditGuard AI tool execution."""

from typing import Any

ALLOWED_ASSISTANT_TOOLS = {
    "get_analysis_summary",
    "get_policy_configuration",
}
MAX_TOOL_CALLS_PER_TURN = 3


def validate_tool_call(name: str, arguments: Any, allowed_tools=None) -> tuple[bool, str]:
    """Allow only registered, read-only assistant tools with object arguments."""
    allowed = set(allowed_tools or ALLOWED_ASSISTANT_TOOLS)
    if not isinstance(name, str) or name not in allowed:
        return False, "Tool is not allowlisted for this assistant context."
    if not isinstance(arguments, dict):
        return False, "Tool arguments must be a JSON object."
    if arguments:
        return False, "This read-only tool does not accept arguments."
    return True, "Allowed by policy."
