from __future__ import annotations

import os
import shutil
import stat
import subprocess
import time
from pathlib import Path
from textwrap import dedent


def find_builds_dir() -> Path | None:
    cwd = Path.cwd()
    xorq_builds = cwd / ".xorq" / "builds"
    if xorq_builds.exists() and xorq_builds.is_dir():
        return xorq_builds

    builds = cwd / "builds"
    return builds if builds.exists() and builds.is_dir() else None


def get_recent_builds(limit: int = 5) -> list[tuple[str, str]]:
    builds_dir = find_builds_dir()
    if not builds_dir:
        return []

    build_dirs = [
        d for d in builds_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]
    build_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)

    results = []
    for d in build_dirs[:limit]:
        try:
            mtime = d.stat().st_mtime
            seconds_ago = time.time() - mtime
            if seconds_ago < 60:
                time_str = "just now"
            elif seconds_ago < 3600:
                time_str = f"{int(seconds_ago / 60)}m ago"
            elif seconds_ago < 86400:
                time_str = f"{int(seconds_ago / 3600)}h ago"
            else:
                time_str = f"{int(seconds_ago / 86400)}d ago"
            results.append((d.name, time_str))
        except Exception:
            results.append((d.name, "unknown"))

    return results


def get_catalog_entries(limit: int = 10) -> list[dict]:
    try:
        result = subprocess.run(
            ["xorq", "catalog", "ls"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            return []

        # Parse the output (skip header lines)
        lines = result.stdout.strip().split("\n")
        entries = []

        for line in lines[2:]:  # Skip header and separator
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 3:
                alias = parts[0]
                revision = parts[1]
                build_hash = parts[2] if len(parts) > 2 else ""
                root_tag = parts[3] if len(parts) > 3 else ""
                entries.append(
                    {
                        "alias": alias,
                        "revision": revision,
                        "hash": build_hash,
                        "root_tag": root_tag,
                    }
                )

        return entries[:limit]
    except Exception:
        return []




def register_claude_skill() -> Path | None:
    """Register the expression-builder skill with Claude Code.

    Installs to .claude/skills/expression-builder in the current project directory.

    Returns the path where the skill was registered, or None if source not found.
    """
    # Find the skill source directory (no generation needed, direct copy)
    import xorq

    xorq_package_dir = Path(xorq.__file__).parent

    # Use bundled package resources
    skill_source = xorq_package_dir / "agent" / "resources" / "expression-builder"
    if not skill_source.exists():
        print(f"⚠️  Could not find expression-builder skill source")
        return None

    # Install to project-local .claude/skills directory
    project_root = Path.cwd()
    claude_skills_dir = project_root / ".claude" / "skills"
    skill_dest = claude_skills_dir / "expression-builder"

    # Create Claude skills directory if needed
    claude_skills_dir.mkdir(parents=True, exist_ok=True)

    # Copy skill if it doesn't exist or update it
    if skill_dest.exists():
        # Skill already registered, optionally update it (handle read-only files)
        def handle_remove_readonly(func, path, exc):
            """Error handler for shutil.rmtree to handle read-only files."""
            if isinstance(exc[1], PermissionError):
                # Make the file writable and try again
                os.chmod(path, stat.S_IWRITE)
                func(path)
            else:
                raise

        shutil.rmtree(skill_dest, onerror=handle_remove_readonly)

    shutil.copytree(skill_source, skill_dest)

    # Setup skill-rules.json for auto-activation
    _setup_skill_rules(claude_skills_dir, skill_source)

    return skill_dest




def _setup_skill_rules(claude_skills_dir: Path, skill_source: Path) -> None:
    """Setup or update skill-rules.json with xorq skill triggers."""
    import json

    skill_rules_path = claude_skills_dir / "skill-rules.json"
    skill_rules_source = skill_source / "skill-rules.json"

    # If source skill-rules.json doesn't exist, skip
    if not skill_rules_source.exists():
        return

    # Load the expression-builder skill rules template
    skill_rules = json.loads(skill_rules_source.read_text())

    # Check if skill-rules.json already exists
    if skill_rules_path.exists():
        # Merge with existing rules
        existing_rules = json.loads(skill_rules_path.read_text())

        # Update only the expression-builder skill entry, preserve others
        if "skills" not in existing_rules:
            existing_rules["skills"] = {}

        # Update expression-builder skill entry
        if "skills" in skill_rules and "expression-builder" in skill_rules["skills"]:
            existing_rules["skills"]["expression-builder"] = skill_rules["skills"]["expression-builder"]

        # Ensure version and description exist
        if "version" not in existing_rules:
            existing_rules["version"] = skill_rules.get("version", "1.0")
        if "description" not in existing_rules:
            existing_rules["description"] = "Skill activation triggers for Claude Code"

        # Write merged rules
        skill_rules_path.write_text(json.dumps(existing_rules, indent=4))
    else:
        # No existing rules, copy the template
        shutil.copy(skill_rules_source, skill_rules_path)


def bootstrap_agent_docs(
    project_root: str | Path, agents: list[str] | None = None, max_lines: int = 60
) -> list[Path]:
    """Create agent-specific documentation files and register skills.

    Parameters
    ----------
    project_root : str | Path
        Project root directory
    agents : list[str] | None
        List of agents to setup. Valid values: ["claude"].
        If None, defaults to ["claude"] for backwards compatibility.
    max_lines : int
        Maximum lines for agent doc content

    Returns
    -------
    list[Path]
        List of created files
    """
    root = Path(project_root)
    root.mkdir(parents=True, exist_ok=True)

    # Default to Claude for backwards compatibility
    if agents is None:
        agents = ["claude"]

    created: list[Path] = []

    # Setup Claude
    if "claude" in agents:
        # Note: CLAUDE.md and AGENTS.md are no longer created here
        # Use 'xorq agents skill install' to install the skill instead
        print("ℹ️  To install the xorq skill for Claude Code, run: xorq agents skill install")
        print("ℹ️  To see lean onboarding instructions, run: xorq agents onboard")

    return created


def render_lean_onboarding() -> str:
    """Render a minimal onboarding message for AGENTS.md, bd-style."""
    return dedent("""\
        # xorq Onboarding
Track ALL expression in xorq catalog
All data work gets cataloged as expression for reuse, reproducibility and composability
Always commit the builds and catalog
Run `xorq agents prime` for dynamic workflow context, or install hooks (`xorq agents hooks install`) for auto-injection.

## Catalog-First Workflow
1. `xorq catalog ls` - CHECK CATALOG FIRST! Compose on existing expressions
2. `xorq agents vignette list` - Discover patterns for new expressions
3. `xorq agents vignette scaffold <name> --dest expr.py` - Get template
4. Follow the scaffolded pattern exactly
5. `xorq build expr.py -e expr` - Build expression
6. `xorq catalog add .xorq/builds/<hash> --alias <name>` - Register to catalog

**Key principle:** Compose on existing catalog entries before creating new ones!

For full workflow details: `xorq agents prime`
        """).strip()


def render_onboarding_summary(step: str | None = None) -> str:
    """Render dynamic workflow context (the 'prime' command)."""
    # Get current project state
    recent_builds = get_recent_builds(limit=3)
    catalog_entries = get_catalog_entries(limit=10)

    sections = ["# xorq Workflow Context", ""]

    # Show project state
    if catalog_entries:
        sections.append("## Active Catalog")
        for entry in catalog_entries[:8]:
            alias = entry.get("alias", "")
            root_tag = entry.get("root_tag", "")
            tag_str = f" → {root_tag}" if root_tag else ""
            sections.append(f"- `{alias}`{tag_str}")
        if len(catalog_entries) > 8:
            sections.append(f"- ...and {len(catalog_entries) - 8} more")
        sections.append("")

    if recent_builds and not catalog_entries:
        sections.append("## Recent Builds (uncataloged)")
        for build_hash, time_str in recent_builds[:3]:
            sections.append(f"- `{build_hash[:12]}...` ({time_str})")
        sections.append("💡 Catalog these: `xorq catalog add builds/<hash> --alias name`")
        sections.append("")

    # Context-aware guidance
    sections.append("## Next Steps")

    if not catalog_entries:
        sections.append("")
        sections.append("📋 **No catalog entries yet!**")
        sections.append("1. Explore vignettes: `xorq agents vignette list`")
        sections.append("2. Scaffold a pattern: `xorq agents vignette scaffold <name> --dest expr.py`")
        sections.append("3. Create your first expression and catalog it")
    elif len(catalog_entries) < 3:
        sections.append("")
        sections.append("🚀 **Growing your catalog:**")
        sections.append("1. Check if you can compose on existing entries")
        sections.append("2. Explore: `xo.catalog.get('{}').schema()`".format(catalog_entries[0]['alias']))
        sections.append("3. Add more reusable data sources and transformations")
    else:
        sections.append("")
        sections.append("✨ **Catalog is established! Compose first, create second:**")
        sections.append("1. Always check: `xorq catalog ls` before creating new datasets")
        sections.append("2. Compose: `result = xo.catalog.get('existing').filter(...)`")
        sections.append("3. Only create new catalog entries when composing isn't enough")

    sections.append("")
    sections.append("## ⚡ Before Writing Code")
    sections.append("")
    sections.append("**Follow this sequence:**")
    sections.append("")
    sections.append("1. **Explore xorq catalog** - Check what already exists")
    sections.append("   ```bash")
    sections.append("   xorq catalog ls")
    sections.append("   ```")
    sections.append("")
    sections.append("2. **Explore expression-builder skill** - Get patterns and guidance")
    sections.append("   - Invoke the skill before writing code")
    sections.append("   - Get UDF/UDAF patterns, caching strategies, optimization tips")
    sections.append("")
    sections.append("3. **Write your expression** - Follow patterns from skill/vignettes")
    sections.append("")
    sections.append("")
    sections.append("## Core Commands")
    sections.append("```bash")
    sections.append("xorq catalog ls                     # List expressions (START HERE!)")
    sections.append("xorq build expr.py -e expr          # Build expression")
    sections.append("xorq catalog add builds/<h> --alias # Catalog build")
    sections.append("xorq run <alias> -f arrow           # Run expression")
    sections.append("xorq agents vignette list           # List available vignettes")
    sections.append("```")
    sections.append("")

    # Add Critical Errors section
    sections.append("## ⚠️  Critical Errors (Hard to Debug)")
    sections.append("")
    sections.append("**Type coercion errors in UDF/UDAF pipelines:**")
    sections.append("   - ❌ Passing expressions directly to UDAFs without type coercion")
    sections.append("   - ❌ Getting \"Failed to coerce arguments\" errors (Decimal128→Int64, Utf8View→Utf8)")
    sections.append("   - ❌ Calling `.into_backend()` without a connection parameter")
    sections.append("   - ✅ Use `.into_backend(con=xo.connect())` before passing data to UDF/UDAF operations")
    sections.append("   - Example:")
    sections.append("     ```python")
    sections.append("     import xorq as xo")
    sections.append("     ")
    sections.append("     # Prepare data with explicit type coercion")
    sections.append("     con = xo.connect()")
    sections.append("     data = (")
    sections.append("         source")
    sections.append("         .filter(conditions)")
    sections.append("         .select([col.cast('int64'), col2.cast('float64')])")
    sections.append("     ).into_backend(con=con)  # <-- Resolves type inference issues")
    sections.append("     ")
    sections.append("     # Now safe to use in UDAF")
    sections.append("     result = data.aggregate(my_udaf.on_expr(data))")
    sections.append("     ```")
    sections.append("")
    sections.append("**Remember:** Run `xorq agents prime` for dynamic workflow context. Keep everything deferred!")
    sections.append("")

    return "\n".join(sections).strip() + "\n"
