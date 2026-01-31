#!/usr/bin/env python3
"""PostToolUseFailure hook - appends TROUBLESHOOTING.md to stderr on failure."""

import sys
import os
from pathlib import Path


def main():
    """PostToolUseFailure hook handler."""
    # Find TROUBLESHOOTING.md in package resources
    try:
        import xorq
        xorq_package_dir = Path(xorq.__file__).parent
        troubleshooting_path = xorq_package_dir / "agent" / "resources" / "expression-builder" / "resources" / "TROUBLESHOOTING.md"

        if troubleshooting_path.exists():
            troubleshooting_content = troubleshooting_path.read_text()

            # Append to stderr
            print("\n" + "="*60, file=sys.stderr)
            print("📚 XORQ TROUBLESHOOTING GUIDE", file=sys.stderr)
            print("="*60, file=sys.stderr)
            print(troubleshooting_content, file=sys.stderr)
            print("="*60 + "\n", file=sys.stderr)
    except Exception:
        # If we can't find or read the file, just exit gracefully
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
