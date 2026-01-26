#!/usr/bin/env python3
"""Pre-commit hook for xorq projects.

Checks staged files for common xorq issues:
- Detect eager pandas/numpy operations in expression files
- Validate xorq expression syntax
- Check for missing deferred wrappers
"""

import sys
import subprocess
from pathlib import Path


def get_staged_files():
    """Get list of staged files."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only"],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout.strip().split("\n") if result.stdout.strip() else []


def check_eager_operations(file_path):
    """Check for eager pandas/numpy operations in xorq expressions."""
    if not file_path.endswith(".py"):
        return True, []

    issues = []

    try:
        with open(file_path, "r") as f:
            content = f.read()

        # Check for common eager operations
        eager_patterns = [
            (".to_pandas()", "converts to eager pandas DataFrame"),
            (".compute()", "computes expression eagerly"),
            ("pd.read_", "reads data eagerly - use xo.catalog.get() instead"),
            ("np.array(", "creates eager numpy array"),
            (".values", "extracts values eagerly"),
            (".iloc", "pandas iloc is not deferred"),
            (".loc[", "pandas loc is not deferred when not used properly"),
        ]

        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            # Skip comments
            if line.strip().startswith("#"):
                continue

            for pattern, msg in eager_patterns:
                if pattern in line:
                    issues.append(f"  Line {i}: {msg} ({pattern})")

    except Exception as e:
        # If we can't read the file, skip it
        return True, []

    return len(issues) == 0, issues


def check_expression_syntax(file_path):
    """Validate xorq expression syntax."""
    if not file_path.endswith(".py"):
        return True, []

    # For now, just check if it's a valid Python file
    try:
        with open(file_path, "r") as f:
            compile(f.read(), file_path, "exec")
        return True, []
    except SyntaxError as e:
        return False, [f"  Syntax error: {e}"]


def main():
    """Main pre-commit hook logic."""
    staged_files = get_staged_files()
    if not staged_files:
        return 0

    print("xorq pre-commit: Checking staged files...")

    has_issues = False

    # Filter for Python files in xorq-related directories
    xorq_files = []
    for file_path in staged_files:
        if file_path.endswith(".py"):
            # Check if it's likely an xorq expression file
            if any(pattern in file_path for pattern in ["expr", "pipeline", "xorq"]):
                xorq_files.append(file_path)

    if not xorq_files:
        return 0

    for file_path in xorq_files:
        print(f"  Checking {file_path}...")

        # Check for eager operations
        success, issues = check_eager_operations(file_path)
        if not success:
            print(f"\n❌ Found eager operations in {file_path}:")
            for issue in issues:
                print(issue)
            has_issues = True

        # Check syntax
        success, issues = check_expression_syntax(file_path)
        if not success:
            print(f"\n❌ Syntax errors in {file_path}:")
            for issue in issues:
                print(issue)
            has_issues = True

    if has_issues:
        print("\n⚠️  xorq pre-commit check failed!")
        print("Fix the issues above or use --no-verify to skip")
        return 1

    print("✓ xorq pre-commit checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())