#!/usr/bin/env python3
"""PostToolUseFailure hook - appends TROUBLESHOOTING.md to stderr on failure."""

import sys
import os
from pathlib import Path


def main():
    """PostToolUseFailure hook handler."""
    try:
        # Simple troubleshooting reminder
        print("\n" + "="*60, file=sys.stderr)
        print("📚 XORQ TROUBLESHOOTING", file=sys.stderr)
        print("="*60, file=sys.stderr)
        print("\nCommon fixes:", file=sys.stderr)
        print("  • Schema errors: Check table.schema() before building", file=sys.stderr)
        print("  • Import errors: Use 'from xorq.vendor import ibis'", file=sys.stderr)
        print("  • Build errors: Verify variable name matches -e argument", file=sys.stderr)
        print("\nQuick commands:", file=sys.stderr)
        print("  xorq agents prompt list --tier reliability", file=sys.stderr)
        print("  xorq agents prompt show fix_schema_errors", file=sys.stderr)
        print("  xorq build <file>.py -e expr --pdb", file=sys.stderr)
        print("\nGet help:", file=sys.stderr)
        print("  • Use expression-builder skill if available in Claude Code", file=sys.stderr)
        print("  • Full guide: skills/expression-builder/resources/TROUBLESHOOTING.md", file=sys.stderr)
        print("="*60 + "\n", file=sys.stderr)
    except Exception:
        # If we can't find or read the file, just exit gracefully
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
