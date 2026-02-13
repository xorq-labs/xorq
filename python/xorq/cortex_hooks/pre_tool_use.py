#!/usr/bin/env python3
"""PreToolUse hook for Cortex Code - blocks cortex CLI calls, redirects to xorq."""

import json
import sys


def main():
    """PreToolUse hook handler - blocks Bash commands that use the cortex CLI."""
    try:
        hook_input = json.load(sys.stdin)
    except Exception:
        return 0

    tool_name = hook_input.get("tool_name", "")
    tool_input = hook_input.get("tool_input", {})

    if tool_name != "Bash":
        return 0

    command = tool_input.get("command", "")

    # Check if the command invokes the cortex CLI
    # Match "cortex" as a standalone token (not as part of another word like cortex_hooks)
    tokens = command.split()
    if any(token == "cortex" for token in tokens):
        print(
            "BLOCKED: The `cortex` CLI is not used in this project. "
            "Use the `xorq` CLI instead. For example: "
            "`xorq build`, `xorq catalog`, `xorq run`, `xorq agents`.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
