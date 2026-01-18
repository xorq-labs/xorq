"""Landing checklist for xorq agent sessions.

Shows what needs to be done before closing a session or committing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from xorq.agent.prime import get_catalog_entries, get_recent_builds


def check_catalog_uncommitted() -> bool:
    """Check if .xorq/catalog.yaml has uncommitted changes.

    This checks for BOTH unstaged and staged changes compared to HEAD.
    Returns True if the file is different from the last commit.
    """
    catalog_path = Path(".xorq/catalog.yaml")
    if not catalog_path.exists():
        return False

    try:
        # Check for ANY changes compared to HEAD (staged OR unstaged)
        result = subprocess.run(
            ["git", "diff", "HEAD", "--name-only", str(catalog_path)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # If there's output, file has uncommitted changes (either staged or unstaged)
        return bool(result.stdout.strip())
    except Exception:
        return False


def check_builds_uncommitted() -> list[str]:
    """Check if there are uncommitted builds.

    Returns list of uncommitted build files (both staged and unstaged changes).
    """
    builds_dir = Path(".xorq/builds")
    if not builds_dir.exists():
        return []

    try:
        # Check for changes compared to HEAD (includes staged and unstaged)
        result = subprocess.run(
            ["git", "diff", "HEAD", "--name-only", str(builds_dir)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        uncommitted = [
            line.strip() for line in result.stdout.strip().split("\n") if line.strip()
        ]

        # Also check for untracked files in builds dir
        result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", str(builds_dir)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        untracked = [
            line.strip() for line in result.stdout.strip().split("\n") if line.strip()
        ]

        return uncommitted + untracked
    except Exception:
        return []


def check_git_status() -> dict:
    """Get overall git status."""
    try:
        # Check if there are uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        has_uncommitted = bool(result.stdout.strip())

        # Check if we're ahead of remote
        result = subprocess.run(
            ["git", "rev-list", "--count", "@{u}..HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        commits_ahead = int(result.stdout.strip()) if result.returncode == 0 else 0

        return {
            "has_uncommitted": has_uncommitted,
            "commits_ahead": commits_ahead,
        }
    except Exception:
        return {"has_uncommitted": False, "commits_ahead": 0}


def render_landing_checklist(limit: int = 5) -> str:
    """Render the landing checklist with current project state."""

    # Get project state
    recent_builds = get_recent_builds(limit=limit)
    catalog_entries = get_catalog_entries(limit=limit)
    catalog_uncommitted = check_catalog_uncommitted()
    builds_uncommitted = check_builds_uncommitted()
    git_status = check_git_status()

    sections = [
        "# 🚨 Landing Checklist",
        "",
        "Before closing this session, ensure all work is committed and pushed.",
        "",
    ]

    # Show recent activity
    if recent_builds or catalog_entries:
        sections.append("## Recent Activity")
        sections.append("")

        if recent_builds:
            sections.append("**Recent Builds:**")
            for build_hash, time_str in recent_builds:
                sections.append(f"- `{build_hash[:12]}...` ({time_str})")
            sections.append("")

        if catalog_entries:
            sections.append("**Catalog Entries:**")
            for entry in catalog_entries:
                alias = entry.get("alias", "")
                revision = entry.get("revision", "")
                root_tag = entry.get("root_tag", "")
                if root_tag:
                    sections.append(f"- `{alias}` @ {revision} → {root_tag}")
                else:
                    sections.append(f"- `{alias}` @ {revision}")
            sections.append("")

    # Critical checklist
    sections.append("## Required Steps")
    sections.append("")

    # Check 1: Catalog committed
    if catalog_uncommitted:
        sections.append(
            "- [ ] ❌ **Commit catalog**: `.xorq/catalog.yaml` has uncommitted changes"
        )
        sections.append("  ```bash")
        sections.append("  git add .xorq/catalog.yaml")
        sections.append("  ```")
    else:
        sections.append("- [x] ✅ Catalog committed")
    sections.append("")

    # Check 2: Builds committed
    if builds_uncommitted:
        sections.append(
            f"- [ ] ❌ **Commit builds**: {len(builds_uncommitted)} uncommitted builds"
        )
        sections.append("  ```bash")
        sections.append("  git add builds/")
        sections.append("  ```")
    else:
        sections.append("- [x] ✅ Builds committed")
    sections.append("")

    # Check 3: All changes committed
    if git_status["has_uncommitted"]:
        sections.append("- [ ] ❌ **Commit all changes**")
        sections.append("  ```bash")
        sections.append('  git commit -m "Update catalog and builds"')
        sections.append("  ```")
    else:
        sections.append("- [x] ✅ All changes committed")
    sections.append("")

    # Check 4: Pushed to remote
    if git_status["commits_ahead"] > 0:
        sections.append(
            f"- [ ] ❌ **Push to remote**: {git_status['commits_ahead']} commits ahead"
        )
        sections.append("  ```bash")
        sections.append("  git push")
        sections.append("  ```")
    else:
        sections.append("- [x] ✅ Pushed to remote")
    sections.append("")

    # Check 5: Builds validated
    sections.append("- [ ] **Validate builds** (recommended)")
    sections.append("  ```bash")
    sections.append("  xorq run <alias> --limit 10")
    sections.append("  ```")
    sections.append("")

    # Final status
    all_good = (
        not catalog_uncommitted
        and not builds_uncommitted
        and not git_status["has_uncommitted"]
        and git_status["commits_ahead"] == 0
    )

    if all_good:
        sections.extend(
            [
                "---",
                "",
                "✅ **All checks passed!** Session is ready to close.",
                "",
            ]
        )
    else:
        sections.extend(
            [
                "---",
                "",
                "⚠️  **Action required** - Complete the checklist above before closing.",
                "",
            ]
        )

    return "\n".join(sections).strip() + "\n"


def agent_land_command(args, limit: int = 10) -> None:
    """Execute the land command - show landing checklist.

    Args:
        args: CLI arguments
        limit: Maximum number of builds/catalog entries to show
    """
    checklist = render_landing_checklist(limit=limit)
    print(checklist, end="")

    # Check what's being staged for commit
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        staged_files = (
            set(result.stdout.strip().split("\n")) if result.stdout.strip() else set()
        )
    except Exception:
        staged_files = set()

    # Get status
    catalog_uncommitted = check_catalog_uncommitted()
    builds_uncommitted = check_builds_uncommitted()

    # If catalog has changes and is NOT staged, warn and fail
    if catalog_uncommitted and ".xorq/catalog.yaml" not in staged_files:
        print(
            "\n⚠️  WARNING: .xorq/catalog.yaml has uncommitted changes but is not staged!"
        )
        print("   Add it to this commit: git add .xorq/catalog.yaml")
        exit(1)

    # If builds have changes but none are staged, warn
    if builds_uncommitted:
        builds_staged = any(
            "builds/" in f or ".xorq/builds/" in f for f in staged_files
        )
        if not builds_staged:
            print("\n⚠️  WARNING: Builds have uncommitted changes but are not staged!")
            print("   Add them to this commit: git add builds/")
            # Don't fail here, as builds might be intermediate work
            # exit(1)

    # All good
    exit(0)


if __name__ == "__main__":
    # Allow running as standalone script
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    agent_land_command(args, limit=args.limit)
