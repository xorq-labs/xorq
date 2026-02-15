#!/usr/bin/env python3
"""PostToolUse hook - injects xorq onboarding instructions after every tool use."""

import subprocess
import sys


def main():
    """PostToolUse hook handler.

    Runs `xorq agents onboard` and prints the output to stderr so that
    the agent receives workflow context after each tool invocation.
    """
    try:
        result = subprocess.run(
            ["xorq", "agents", "onboard"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            print(result.stdout.strip(), file=sys.stderr)
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
