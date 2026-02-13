#!/usr/bin/env python3
"""Stop hook - checks for uncataloged builds."""

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

    all_builds = [d.name for d in builds_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

    if not all_builds:
        return []

    try:
        result = subprocess.run(
            ["xorq", "catalog", "ls"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=work_dir,
        )

        if result.returncode != 0:
            return []

        cataloged_hashes = set()
        for line in result.stdout.strip().split("\n")[2:]:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 3:
                cataloged_hashes.add(parts[2])

        uncataloged = [b for b in all_builds if b not in cataloged_hashes]
        return uncataloged[:5]

    except Exception:
        return []


def main():
    """Stop hook handler - checks for uncataloged builds."""
    try:
        work_dir = os.getcwd()

        if not (Path(work_dir) / ".xorq").exists():
            return 0

        uncataloged = get_uncataloged_builds(work_dir)

        if uncataloged:
            message = "UNCATALOGED BUILDS DETECTED\n\n"
            message += f"Found {len(uncataloged)} uncataloged build(s) in .xorq/builds/:\n"
            for build_hash in uncataloged:
                message += f"  - {build_hash}\n"
            message += "\nREQUIRED STEPS:\n"
            message += "1. Catalog: xorq catalog add .xorq/builds/<hash> --alias <name>\n"
            message += "2. Commit: git add .xorq/builds/ .xorq/catalog.yaml && git commit\n"
            print(message)

    except Exception:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
