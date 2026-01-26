#!/usr/bin/env python3
"""Prepare commit message hook for xorq projects.

Adds xorq-specific context to commit messages:
- Lists modified expressions
- Adds catalog changes summary
- Suggests conventional commit format
"""

import sys
import subprocess
from pathlib import Path


def get_modified_expressions():
    """Get list of modified expression files."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True,
        text=True,
        check=True
    )
    files = result.stdout.strip().split("\n") if result.stdout.strip() else []

    expressions = []
    for f in files:
        if f.endswith(".py") and "expr" in f:
            expressions.append(Path(f).stem)

    return expressions


def get_catalog_changes():
    """Check if catalog.yaml was modified."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "catalog.yaml"],
        capture_output=True,
        text=True,
        check=False
    )
    return bool(result.stdout.strip())


def enhance_commit_message(commit_msg_file, source_type):
    """Enhance the commit message with xorq context."""
    # Read current message
    with open(commit_msg_file, "r") as f:
        current_msg = f.read()

    # Don't modify if it's a merge or has content already
    if source_type == "merge" or (current_msg.strip() and not current_msg.startswith("#")):
        return

    # Gather xorq context
    expressions = get_modified_expressions()
    catalog_changed = get_catalog_changes()

    # Build enhanced message
    lines = []

    # Add template if empty
    if not current_msg.strip() or current_msg.startswith("#"):
        lines.append("# feat(xorq): <description>")
        lines.append("#")
        lines.append("# Conventional commit types for xorq:")
        lines.append("# - feat(expr): new expression or pipeline")
        lines.append("# - fix(expr): fix expression logic")
        lines.append("# - perf(expr): optimize expression performance")
        lines.append("# - refactor(expr): refactor expression code")
        lines.append("# - docs: update documentation")
        lines.append("# - test: add or update tests")
        lines.append("#")

    if expressions:
        lines.append("# Modified expressions:")
        for expr in expressions:
            lines.append(f"#   - {expr}")
        lines.append("#")

    if catalog_changed:
        lines.append("# ✓ Catalog updated")
        lines.append("#")

    # Add the original content
    lines.append(current_msg)

    # Write back
    with open(commit_msg_file, "w") as f:
        f.write("\n".join(lines))


def main():
    """Main prepare-commit-msg hook logic."""
    if len(sys.argv) < 2:
        return 0

    commit_msg_file = sys.argv[1]
    source_type = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        enhance_commit_message(commit_msg_file, source_type)
    except Exception as e:
        # Don't fail the commit if our enhancement fails
        print(f"xorq prepare-commit-msg: Warning: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())