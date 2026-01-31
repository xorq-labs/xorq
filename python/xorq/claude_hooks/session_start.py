#!/usr/bin/env python3
"""SessionStart hook - prints xorq agents onboard to stdout."""

import subprocess
import sys


def main():
    """SessionStart hook handler - runs xorq agents onboard and prints to stdout."""
    try:
        # Run xorq agents onboard to get workflow context
        result = subprocess.run(
            ["xorq", "agents", "onboard"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0 and result.stdout.strip():
            # Print onboarding content to stdout
            print(result.stdout)

    except Exception:
        # If xorq command fails, don't block session start
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
