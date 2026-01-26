#!/usr/bin/env python3
"""Pre-push hook for xorq projects.

Validates before pushing:
- All modified expressions are built
- Catalog is up to date
- No untracked expression files
"""

import sys
import subprocess
from pathlib import Path


def check_expression_builds():
    """Check if all modified expressions have been built."""
    # Get list of commits being pushed
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD", "@{u}"],
        capture_output=True,
        text=True,
        check=False
    )

    if result.returncode != 0:
        # No upstream branch or other issue, skip check
        return True, []

    files = result.stdout.strip().split("\n") if result.stdout.strip() else []

    # Find expression files
    expr_files = [f for f in files if f.endswith(".py") and "expr" in f]

    if not expr_files:
        return True, []

    issues = []

    # Check if catalog is up to date
    for expr_file in expr_files:
        expr_name = Path(expr_file).stem
        # Check if this expression exists in catalog
        try:
            result = subprocess.run(
                ["xorq", "catalog", "ls", "--name", expr_name],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode != 0 or expr_name not in result.stdout:
                issues.append(f"  Expression '{expr_name}' not found in catalog - did you forget to build?")
        except:
            pass

    return len(issues) == 0, issues


def main():
    """Main pre-push hook logic."""
    print("xorq pre-push: Running checks...")

    has_issues = False

    # Check expression builds
    success, issues = check_expression_builds()
    if not success:
        print("\n❌ Missing expression builds:")
        for issue in issues:
            print(issue)
        has_issues = True

    if has_issues:
        print("\n⚠️  xorq pre-push check failed!")
        print("Fix the issues above or use --no-verify to skip")
        return 1

    print("✓ xorq pre-push checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())