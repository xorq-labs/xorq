#!/usr/bin/env python3
"""PostToolUseFailure hook - logs failures to a JSONL file for the stop hook to analyze."""

import json
import sys
import time
from pathlib import Path


def log_failure(session_id, transcript_path, tool_name, tool_input, error):
    """Append a failure entry to {session_id}.failures.jsonl alongside the transcript."""
    if not transcript_path:
        return

    sessions_dir = Path(transcript_path).parent
    failures_path = sessions_dir / f"{session_id}.failures.jsonl"

    entry = {
        "timestamp": int(time.time()),
        "tool_name": tool_name,
        "tool_input": tool_input,
        "error_summary": (error or "")[:500],
    }

    with open(failures_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    """PostToolUseFailure hook handler."""
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        hook_input = {}

    session_id = hook_input.get("session_id", "unknown")
    transcript_path = hook_input.get("transcript_path")
    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})
    error = hook_input.get("error", "")

    # Log the failure for the stop hook to discover
    try:
        log_failure(session_id, transcript_path, tool_name, tool_input, error)
    except Exception as e:
        print(f"memelord: failed to log failure: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
