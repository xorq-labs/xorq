#!/usr/bin/env python3
"""SessionStart hook for Cortex Code - prints xorq agents onboard to stdout."""

import os
import subprocess
import sys
from pathlib import Path


def main():
    """SessionStart hook handler - runs xorq agents onboard and prints to stdout."""
    try:
        # Get the working directory from environment (set by Cortex Code)
        # or fall back to current directory
        work_dir = os.environ.get("CORTEX_WORKDIR", os.getcwd())
        work_path = Path(work_dir)

        # Check if this is an xorq project by looking for .xorq directory or catalog.yaml
        if not (work_path / ".xorq").exists() and not (work_path / ".xorq" / "catalog.yaml").exists():
            # Not an xorq project, skip silently
            return 0

        # Run xorq agents onboard to get workflow context
        result = subprocess.run(
            ["xorq", "agents", "onboard"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=work_dir
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
