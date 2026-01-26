"""Git hooks management for xorq.

This module provides functionality to install, uninstall, and run git hooks
for xorq projects, similar to other dev tools like bd.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from textwrap import dedent
from typing import List, Optional

# Git hooks that xorq supports
SUPPORTED_HOOKS = [
    "pre-commit",
    "post-commit",
    "pre-push",
    "post-checkout",
    "post-merge",
    "prepare-commit-msg",
]

# Thin shim template for git hooks
HOOK_SHIM_TEMPLATE = '''#!/usr/bin/env sh
# xorq-shim v1
# xorq-hooks-version: {version}
#
# xorq git {hook_name} hook - thin shim
#
# This shim delegates to 'xorq hooks run {hook_name}' which contains
# the actual hook logic. This pattern ensures hook behavior is always
# in sync with the installed xorq version - no manual updates needed.

# Check if xorq is available
if ! command -v xorq >/dev/null 2>&1; then
    echo "Warning: xorq command not found in PATH, skipping {hook_name} hook" >&2
    echo "  Install xorq: pip install xorq" >&2
    echo "  Or add xorq to your PATH" >&2
    exit 0
fi

exec xorq hooks run {hook_name} "$@"
'''


def get_git_dir() -> Optional[Path]:
    """Get the .git directory for the current repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(result.stdout.strip())
    except subprocess.CalledProcessError:
        return None


def get_hooks_dir() -> Optional[Path]:
    """Get the git hooks directory."""
    git_dir = get_git_dir()
    if git_dir:
        return git_dir / "hooks"
    return None


def get_xorq_hooks_dir() -> Path:
    """Get the directory where xorq hooks logic is stored."""
    return Path(__file__).parent / "git_hooks"


def backup_existing_hook(hook_path: Path) -> bool:
    """Backup an existing hook if it's not an xorq hook."""
    if not hook_path.exists():
        return True

    content = hook_path.read_text()

    # Check if it's already an xorq hook
    if "xorq-shim" in content:
        return True

    # Check if it's a bd hook (we can coexist)
    if "bd-shim" in content:
        return True

    # Backup the existing hook
    backup_path = hook_path.with_suffix(".backup")
    counter = 1
    while backup_path.exists():
        backup_path = hook_path.with_suffix(f".backup.{counter}")
        counter += 1

    shutil.copy2(hook_path, backup_path)
    print(f"  Backed up existing {hook_path.name} to {backup_path.name}")
    return True


def install_hooks(force: bool = False) -> int:
    """Install xorq git hooks."""
    hooks_dir = get_hooks_dir()
    if not hooks_dir:
        print("Error: Not in a git repository")
        return 1

    if not hooks_dir.exists():
        hooks_dir.mkdir(parents=True, exist_ok=True)

    installed = []
    skipped = []
    chained = []

    for hook_name in SUPPORTED_HOOKS:
        hook_path = hooks_dir / hook_name

        # Check if hook already exists
        if hook_path.exists():
            content = hook_path.read_text()
            if "xorq-shim" in content and not force:
                skipped.append(hook_name)
                continue

            # If it's another tool's hook (like bd), we need to chain them
            if "bd-shim" in content:
                # Create a wrapper that calls both bd and xorq
                chain_content = f'''#!/usr/bin/env sh
# xorq+bd chain hook v1
#
# This hook chains both bd and xorq hooks

# Run bd hook first
if command -v bd >/dev/null 2>&1; then
    bd hooks run {hook_name} "$@"
    BD_EXIT=$?
    if [ $BD_EXIT -ne 0 ]; then
        exit $BD_EXIT
    fi
fi

# Run xorq hook
if command -v xorq >/dev/null 2>&1; then
    xorq hooks run {hook_name} "$@"
    XORQ_EXIT=$?
    if [ $XORQ_EXIT -ne 0 ]; then
        exit $XORQ_EXIT
    fi
fi

exit 0
'''
                # Backup the original bd hook
                backup_existing_hook(hook_path)
                hook_path.write_text(chain_content)
                hook_path.chmod(0o755)
                chained.append(hook_name)
                continue

        # Write the hook shim
        hook_content = HOOK_SHIM_TEMPLATE.format(
            version="0.1.0",
            hook_name=hook_name
        )

        hook_path.write_text(hook_content)
        hook_path.chmod(0o755)
        installed.append(hook_name)

    if installed or chained:
        print("✓ Git hooks installed successfully")

    if installed:
        print("\nInstalled hooks:")
        for hook in installed:
            print(f"  - {hook}")

    if chained:
        print("\nChained with bd hooks:")
        for hook in chained:
            print(f"  - {hook} (bd + xorq)")

    if skipped:
        print("\nAlready installed (skipped):")
        for hook in skipped:
            print(f"  - {hook}")
        print("\nUse --force to reinstall")

    return 0


def uninstall_hooks() -> int:
    """Uninstall xorq git hooks."""
    hooks_dir = get_hooks_dir()
    if not hooks_dir:
        print("Error: Not in a git repository")
        return 1

    removed = []

    for hook_name in SUPPORTED_HOOKS:
        hook_path = hooks_dir / hook_name

        if hook_path.exists():
            content = hook_path.read_text()
            if "xorq-shim" in content:
                # Check if there's a backup to restore
                backup_path = hook_path.with_suffix(".backup")
                if backup_path.exists():
                    shutil.move(backup_path, hook_path)
                    print(f"  Restored original {hook_name} from backup")
                else:
                    hook_path.unlink()
                removed.append(hook_name)

    if removed:
        print("✓ Git hooks uninstalled successfully")
        print("\nRemoved hooks:")
        for hook in removed:
            print(f"  - {hook}")
    else:
        print("No xorq git hooks found to uninstall")

    return 0


def run_hook(hook_name: str, args: List[str]) -> int:
    """Run a specific git hook.

    This is called by the thin shim installed in .git/hooks/
    """
    # Get the hook logic directory
    hooks_logic_dir = get_xorq_hooks_dir()
    hook_script = hooks_logic_dir / f"{hook_name}.py"

    # Check if we have logic for this hook
    if not hook_script.exists():
        # No specific logic for this hook, that's OK
        return 0

    # Run the hook script
    try:
        result = subprocess.run(
            [sys.executable, str(hook_script)] + args,
            check=False
        )
        return result.returncode
    except Exception as e:
        print(f"Error running {hook_name} hook: {e}", file=sys.stderr)
        return 1


def list_hooks() -> int:
    """List the status of all git hooks."""
    hooks_dir = get_hooks_dir()
    if not hooks_dir:
        print("Error: Not in a git repository")
        return 1

    print("Git hooks status:")
    print()

    for hook_name in SUPPORTED_HOOKS:
        hook_path = hooks_dir / hook_name

        if hook_path.exists():
            content = hook_path.read_text()
            if "xorq+bd chain" in content:
                status = "⚡ xorq + bd"
            elif "xorq-shim" in content:
                status = "✓ xorq"
            elif "bd-shim" in content:
                status = "○ bd"
            else:
                status = "○ other"
        else:
            status = "- not installed"

        print(f"  {hook_name:20} {status}")

    print()
    print("Legend: ✓ = xorq, ○ = other tool, ⚡ = chained, - = not installed")

    return 0