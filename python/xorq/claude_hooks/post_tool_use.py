#!/usr/bin/env python3
"""PostToolUse hook - logs tool failures to a JSONL file."""

import json
import sys
import time
from pathlib import Path


# Patterns in tool_response that indicate a failure
ERROR_INDICATORS = [
    "error",
    "Error",
    "ERROR",
    "Traceback",
    "Exception",
    "FAILED",
    "exit code",
    "command not found",
    "No such file",
    "Permission denied",
]


def is_error_response(tool_name, tool_response):
    """Check if a tool_response indicates a failure."""
    if tool_response is None:
        return False

    # Bash tool: check for non-zero exit code
    if tool_name == "Bash" and isinstance(tool_response, dict):
        if tool_response.get("exitCode", 0) != 0:
            return True

    # For string responses, check for error indicators
    response_str = (
        json.dumps(tool_response)
        if not isinstance(tool_response, str)
        else tool_response
    )
    return any(indicator in response_str for indicator in ERROR_INDICATORS)


def extract_error_summary(tool_name, tool_response):
    """Extract a concise error summary from the tool response."""
    if isinstance(tool_response, dict):
        # Bash tool: use stderr or stdout
        for key in ("stderr", "stdout", "error", "message"):
            if key in tool_response and tool_response[key]:
                return str(tool_response[key])[:500]
        return json.dumps(tool_response)[:500]

    return str(tool_response)[:500]


def main():
    """PostToolUse hook handler."""
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        return 0

    session_id = hook_input.get("session_id", "unknown")
    transcript_path = hook_input.get("transcript_path")
    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})
    tool_response = hook_input.get("tool_response")

    if not transcript_path:
        return 0

    if not is_error_response(tool_name, tool_response):
        return 0

    # Log the failure
    try:
        sessions_dir = Path(transcript_path).parent
        failures_path = sessions_dir / f"{session_id}.failures.jsonl"

        entry = {
            "timestamp": int(time.time()),
            "tool_name": tool_name,
            "tool_input": tool_input,
            "error_summary": extract_error_summary(tool_name, tool_response),
        }

        with open(failures_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"post_tool_use: failed to log failure: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
