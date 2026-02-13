#!/usr/bin/env python3
"""UserPromptSubmit hook - runs xorq agents onboard on every prompt."""

import subprocess
import sys


def main():
    """UserPromptSubmit hook handler - runs xorq agents onboard and prints to stdout."""
    try:
        result = subprocess.run(
            ["xorq", "agents", "onboard"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0 and result.stdout.strip():
            print(result.stdout)

    except Exception:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
