#!/usr/bin/env python3
"""Stop hook for Cortex Code - checks for uncataloged builds."""

import json
import os
import subprocess
import sys
from pathlib import Path


def get_uncataloged_builds(work_dir):
    """Check for uncataloged builds in .xorq/builds/."""
    builds_dir = Path(work_dir) / ".xorq" / "builds"

    if not builds_dir.exists():
        return []

    # Get all build hashes
    all_builds = [d.name for d in builds_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

    if not all_builds:
        return []

    try:
        # Get cataloged builds
        result = subprocess.run(
            ["xorq", "catalog", "ls"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=work_dir
        )

        if result.returncode != 0:
            return []

        # Parse catalog output to get build hashes
        cataloged_hashes = set()
        for line in result.stdout.strip().split("\n")[2:]:  # Skip header lines
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 3:
                cataloged_hashes.add(parts[2])  # Build hash is 3rd column

        # Find uncataloged builds
        uncataloged = [b for b in all_builds if b not in cataloged_hashes]
        return uncataloged[:5]  # Limit to first 5

    except Exception:
        return []


def main():
    """Stop hook handler - checks for uncataloged builds."""
    try:
        # Get working directory from environment
        work_dir = os.environ.get("CORTEX_WORKDIR", os.getcwd())

        # Check if this is an xorq project
        if not (Path(work_dir) / ".xorq").exists():
            # Not an xorq project, allow stop
            print(json.dumps({"suppressOutput": False}))
            return 0

        uncataloged = get_uncataloged_builds(work_dir)

        if uncataloged:
            # Block stopping until builds are cataloged
            message = "⚠️  UNCATALOGED BUILDS DETECTED\n\n"
            message += f"Found {len(uncataloged)} uncataloged build(s) in .xorq/builds/:\n"
            for build_hash in uncataloged:
                message += f"  • {build_hash}\n"
            message += "\n📝 REQUIRED STEPS TO COMPLETE YOUR WORK:\n\n"
            message += "1. Catalog the builds:\n"
            message += "   xorq catalog add .xorq/builds/<hash> --alias <name>\n\n"
            message += "2. Commit to git:\n"
            message += "   git add .xorq/builds/ .xorq/catalog.yaml\n"
            message += "   git commit -m \"feat: add <name> expression\"\n\n"
            message += "⚠️  WORK IS NOT DONE UNTIL BUILDS ARE CATALOGED AND COMMITTED!\n\n"
            message += "💡 Use 'xorq catalog ls' to verify cataloged builds"

            payload = {
                "suppressOutput": False,
                "decision": "block",
                "reason": message
            }
            print(json.dumps(payload))
        else:
            # No uncataloged builds
            print(json.dumps({"suppressOutput": False}))

    except Exception:
        # Don't block stop on errors
        print(json.dumps({"suppressOutput": False}))

    return 0


if __name__ == "__main__":
    sys.exit(main())
