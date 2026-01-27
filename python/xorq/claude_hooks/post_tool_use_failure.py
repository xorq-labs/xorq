#!/usr/bin/env python3
"""PostToolUseFailure hook - triggered after a tool use fails."""

import sys
import os
import json


def detect_xorq_error(stderr, stdout, tool_name):
    """Detect if the failure is xorq-related and provide troubleshooting guidance."""

    combined_output = f"{stderr}\n{stdout}".lower()

    # Common xorq error patterns and their troubleshooting guides
    error_patterns = {
        "failed to coerce arguments": {
            "title": "Type Coercion Error - Use into_backend()",
            "guide": """
🔧 Type coercion error during planning

**The Fix:**
Add `.into_backend()` to your data expression before using it in ML pipelines.

**Example:**
```python
data = (
    source
    .group_by(["playerID", "yearID"])
    .agg(AB=ibis._.AB.sum(), H=ibis._.H.sum())
    .filter(ibis._.AB >= 100)
    .mutate(...)
).into_backend()  # <-- Add this!

train = data.filter(ibis._.yearID < 2010)
test = data.filter(ibis._.yearID >= 2010)
```

**Why this happens:**
Type inference issues between ibis expressions and xorq's ML pipeline.
`.into_backend()` materializes the schema and resolves type mismatches.

**When to use:**
- Before passing data to Pipeline.fit()
- When seeing "coercion from Float64, Float64, Int64 to..." errors
- When signature matching fails in ML pipelines
"""
        },
        "error during planning": {
            "title": "Type Coercion Error - Use into_backend()",
            "guide": """
🔧 Type coercion error during planning

**The Fix:**
Add `.into_backend()` to your data expression before using it in ML pipelines.

**Example:**
```python
data = (
    source
    .group_by(["playerID", "yearID"])
    .agg(AB=ibis._.AB.sum(), H=ibis._.H.sum())
    .filter(ibis._.AB >= 100)
    .mutate(...)
).into_backend()  # <-- Add this!

train = data.filter(ibis._.yearID < 2010)
test = data.filter(ibis._.yearID >= 2010)
```
"""
        }
    }

    # Check for xorq-related errors
    for pattern, info in error_patterns.items():
        if pattern in combined_output:
            return info

    # Check if xorq command was involved
    if "xorq" in combined_output or tool_name == "Bash" and "xorq" in os.environ.get("LAST_COMMAND", ""):
        return {
            "title": "Xorq Operation Failed",
            "guide": """
❌ Xorq operation failed

**General troubleshooting:**
1. Check command syntax: `xorq --help`
2. Get workflow guidance: `xorq agents prime`
3. Review error message above for specific details

**Common commands:**
- `xorq catalog ls` - List expressions
- `xorq build <file>.py -e <expr>` - Build expression
- `xorq run <alias> -f arrow` - Run expression
- `xorq agents vignette list` - See patterns

**Need comprehensive help?**
- If you have the Expression Builder skill available, it provides complete workflow guidance and API reference
- `xorq agents prime` - Get current context
- Check CLAUDE.md in project root
"""
        }

    return None


def main():
    """PostToolUseFailure hook handler."""
    # Get tool failure information from environment or stdin
    # Claude Code provides this information to the hook

    # Read from environment variables if available
    tool_name = os.environ.get("TOOL_NAME", "")
    stderr = os.environ.get("STDERR", "")
    stdout = os.environ.get("STDOUT", "")

    # If not in environment, try reading from stdin
    if not stderr and not stdout:
        try:
            input_data = sys.stdin.read()
            if input_data:
                data = json.loads(input_data)
                tool_name = data.get("tool_name", "")
                stderr = data.get("stderr", "")
                stdout = data.get("stdout", "")
        except (json.JSONDecodeError, Exception):
            # If we can't read input, just exit gracefully
            return 0

    # Detect if this is a xorq-related error
    error_info = detect_xorq_error(stderr, stdout, tool_name)

    if error_info:
        # Print troubleshooting guide
        print(f"\n{'='*60}")
        print(f"🔧 {error_info['title']}")
        print(f"{'='*60}")
        print(error_info['guide'])
        print(f"{'='*60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
