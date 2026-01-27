#!/usr/bin/env python3
"""SessionStart hook - triggered when a Claude Code session begins."""

import json
import subprocess
import sys


def main():
    """SessionStart hook handler - runs xorq agents onboard."""
    try:
        # Run xorq agents onboard to get workflow context
        result = subprocess.run(
            ["xorq", "agents", "onboard"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0 and result.stdout.strip():
            # Inject onboarding content as system message
            payload = {
                "suppressOutput": False,
                "systemMessage": result.stdout.strip()
            }
            print(json.dumps(payload))
        else:
            # No output or error - don't add context
            print(json.dumps({"suppressOutput": False}))
    except Exception:
        # If xorq command fails, don't block session start
        print(json.dumps({"suppressOutput": False}))

    return 0


if __name__ == "__main__":
    sys.exit(main())
