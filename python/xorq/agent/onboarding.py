from __future__ import annotations

import os
import shutil
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent


def find_builds_dir() -> Path | None:
    """Find the builds/ directory in current project."""
    cwd = Path.cwd()
    # Check both .xorq/builds and builds/
    xorq_builds = cwd / ".xorq" / "builds"
    if xorq_builds.exists() and xorq_builds.is_dir():
        return xorq_builds

    builds = cwd / "builds"
    return builds if builds.exists() and builds.is_dir() else None


def get_recent_builds(limit: int = 5) -> list[tuple[str, str]]:
    """Get most recently modified build directories with timestamps.

    Returns: List of (hash, relative_time) tuples
    """
    builds_dir = find_builds_dir()
    if not builds_dir:
        return []

    # Get all build directories sorted by modification time
    build_dirs = [
        d for d in builds_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]
    build_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)

    results = []
    for d in build_dirs[:limit]:
        # Get relative time (e.g., "2 hours ago")
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
    """Get recent catalog entries using xorq catalog ls.

    Returns: List of dicts with alias, revision, hash, root_tag
    """
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


def _format_builds_status(
    recent_builds: list[tuple[str, str]], catalog_entries: list[dict]
) -> str:
    """Format the current builds status section."""
    if not recent_builds and not catalog_entries:
        return dedent("""\
            ## Current Project State

            - No builds or catalog entries found
            - Run `xorq build expr.py -e expr` to create your first build
            - Then: `xorq catalog add builds/<hash> --alias my-pipeline`
            """).strip()

    output = "## Current Project State\n\n"

    # Recent builds
    if recent_builds:
        output += "**Recent Builds:**\n"
        for build_hash, time_str in recent_builds[:5]:
            output += f"- `{build_hash[:12]}...` ({time_str})\n"
        output += "\n"

    # Catalog entries
    if catalog_entries:
        output += "**Cataloged Pipelines:**\n"
        for entry in catalog_entries[:10]:
            alias = entry.get("alias", "")
            revision = entry.get("revision", "")
            root_tag = entry.get("root_tag", "")
            if root_tag:
                output += f"- `{alias}` @ {revision} → {root_tag}\n"
            else:
                output += f"- `{alias}` @ {revision}\n"
        output += "\n"
        output += "Run `xorq catalog ls` for full list.\n"
    else:
        output += "**No catalog entries yet.**\n"
        output += "- Catalog builds: `xorq catalog add builds/<hash> --alias name`\n"

    return output.strip()


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


@dataclass
class OnboardingStep:
    """A step in the onboarding workflow."""
    key: str
    title: str
    checklist: list[str]
    commands: list[str]
    prompt_refs: list[str]


ONBOARDING_STEPS: tuple[OnboardingStep, ...] = (
    OnboardingStep(
        "init",
        "Initialize project",
        [
            "Run `xorq init` to setup a new project, then `xorq agents init` to setup agent guides",
        ],
        [
            "xorq init",
            "xorq agents init",
        ],
        [],
    ),
    OnboardingStep(
        "templates",
        "Learn from templates",
        [
            "Use `xorq agent templates list` to see available templates",
            "Scaffold templates to your project: `xorq agent templates scaffold <name>`",
            "Templates show how to implement pipelines, ML workflows, and feature engineering",
            "Examples include: pipeline_example, sklearn classifiers, diamonds price prediction",
        ],
        [
            "xorq agent templates list",
            "xorq agent templates scaffold pipeline_example --dest my_pipeline.py",
            "xorq agent templates show <name>",
        ],
        [],
    ),
    OnboardingStep(
        "build",
        "Build deferred expressions",
        [
            "Check schema first: `print(table.schema())`",
            "ALL work must be deferred xorq/ibis expressions",
            "Use xorq skills for deferred pandas/sklearn patterns",
            "Build expressions from Python files",
            "Reference templates for common patterns",
        ],
        [
            "xorq build expr.py -e expr",
        ],
        [],
    ),
    OnboardingStep(
        "catalog",
        "Catalog builds",
        [
            "Register builds with aliases for reuse",
        ],
        [
            "xorq catalog add builds/<hash> --alias <name>",
            "xorq catalog ls",
        ],
        [],
    ),
    OnboardingStep(
        "explore",
        "Explore with DuckDB CLI",
        [
            "Stream expressions to DuckDB for interactive SQL exploration",
            "Use Arrow IPC streaming for ad-hoc analytics",
            "Find source nodes with `xorq catalog sources` for composition",
            "Verify schema compatibility with `xorq catalog schema`",
        ],
        [
            "# Simple: Stream source to DuckDB",
            "xorq run <alias> -f arrow -o /dev/stdout 2>/dev/null | duckdb -c \"LOAD arrow; SELECT * FROM read_arrow('/dev/stdin') LIMIT 10\"",
            "",
            "# Advanced: Compose pipeline and explore interactively",
            "xorq run source -f arrow -o /dev/stdout 2>/dev/null | xorq run-unbound transform --to_unbind_hash <hash> --typ xorq.expr.relations.Read -f arrow -o /dev/stdout 2>/dev/null | duckdb -c \"LOAD arrow; SELECT * FROM read_arrow('/dev/stdin')\"",
            "",
            "# Find source nodes for composition",
            "xorq catalog sources <alias>",
            "xorq catalog schema <alias>",
        ],
        [],
    ),
    OnboardingStep(
        "compose",
        "Compose with memtable pattern",
        [
            "Build transforms with memtable placeholders for independent development",
            "Compose source and transform via Arrow IPC streaming",
            "Test transforms with sample data before applying to real sources",
            "Chain multiple transforms together dynamically",
        ],
        [
            "# Build transform with memtable placeholder",
            "# In transform.py: source = xo.memtable({\"col1\": [1, 2]})",
            "xorq build transform.py -e expr",
            "xorq catalog add builds/<hash> --alias my-transform",
            "",
            "# Find memtable node hash",
            "xorq catalog sources my-transform",
            "",
            "# Compose source + transform via Arrow IPC",
            "xorq run source -f arrow -o /dev/stdout 2>/dev/null | xorq run-unbound transform --to_unbind_hash <hash> --typ xorq.expr.relations.Read -o output.parquet",
            "",
            "# View lineage",
            "xorq lineage <alias>",
        ],
        [],
    ),
    OnboardingStep(
        "land",
        "Commit and push",
        [
            "Commit catalog.yaml and builds/ directory",
        ],
        [
            "git add .xorq/catalog.yaml builds/",
            "git commit -m 'Update catalog and builds'",
            "git push",
        ],
        [],
    ),
)


def register_claude_skill() -> Path | None:
    """Register the expression-builder skill with Claude Code.

    Installs to .claude/skills/expression-builder in the current project directory.

    Returns the path where the skill was registered, or None if source not found.
    """

    # Generate fresh skill from shared content
    try:
        from xorq.agent.resources.common.generate_skills import generate_all_skills

        generate_all_skills()
    except Exception as e:
        print(f"⚠️  Could not regenerate skills: {e}")

    # Find the skill source directory (should be in the package)
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


def register_codex_skill(project_root: Path) -> Path | None:
    """Register the xorq skill with OpenAI Codex.

    This copies the Codex skill from package resources to the project's .xorq/codex directory
    and appends bootstrap content to ~/.codex/AGENTS.md

    Returns the path where the skill was registered, or None if source not found.
    """
    import xorq

    # Generate fresh skill from shared content
    try:
        from xorq.agent.resources.common.generate_skills import generate_all_skills

        generate_all_skills()
    except Exception as e:
        print(f"⚠️  Could not regenerate skills: {e}")

    project_root = Path(project_root)
    skill_dest = project_root / ".xorq" / "codex"

    # Find the skill source in package resources
    xorq_package_dir = Path(xorq.__file__).parent
    skill_source = xorq_package_dir / "agent" / "resources" / "codex"

    # Check if skill source exists in package
    if not skill_source.exists():
        return None

    # Copy skill to project .xorq directory if it doesn't exist or update it
    skill_dest.parent.mkdir(parents=True, exist_ok=True)

    if skill_dest.exists():
        # Update existing skill files
        shutil.rmtree(skill_dest)

    shutil.copytree(skill_source, skill_dest)

    # Add bootstrap to ~/.codex/AGENTS.md
    codex_agents_file = Path.home() / ".codex" / "AGENTS.md"
    codex_agents_file.parent.mkdir(parents=True, exist_ok=True)

    bootstrap_file = skill_dest / "bootstrap.md"
    if bootstrap_file.exists():
        bootstrap_content = bootstrap_file.read_text()

        # Check if bootstrap already exists
        if codex_agents_file.exists():
            existing_content = codex_agents_file.read_text()
            if "Xorq Superpowers for Codex" not in existing_content:
                # Append bootstrap
                with codex_agents_file.open("a") as f:
                    f.write("\n\n")
                    f.write(bootstrap_content)
        else:
            # Create file with bootstrap
            codex_agents_file.write_text(bootstrap_content)

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
        List of agents to setup. Valid values: ["claude", "codex"].
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
        content = _render_agent_doc(max_lines=max_lines)
        for filename in ("AGENTS.md", "CLAUDE.md"):
            dest = root / filename
            if dest.exists():
                continue
            dest.write_text(content, encoding="utf-8")
            created.append(dest)

        # Register the xorq skill with Claude Code
        skill_path = register_claude_skill()
        if skill_path:
            print(f"✅ Registered xorq skill with Claude Code at {skill_path}")
            print("✅ Setup skill auto-activation in ~/.claude/skills/skill-rules.json")

    # Setup Codex
    if "codex" in agents:
        skill_path = register_codex_skill(root)
        if skill_path:
            print(f"✅ Registered xorq skill for Codex at {skill_path}")
            print("✅ Added bootstrap to ~/.codex/AGENTS.md")
            created.append(skill_path)
        else:
            print("⚠️  Could not find Codex skill source")

    return created


def render_lean_onboarding() -> str:
    """Render a minimal onboarding message for AGENTS.md, bd-style."""
    return dedent("""\
        # xorq Onboarding
Track ALL expression in xorq catalog
All data work gets cataloged as expression for reuse, reproducibility and composability
Always commit the builds and catalog
Install hooks (`xorq agents hooks install`) for auto-injection of workflow context.

## Catalog-First Workflow
1. `xorq catalog ls` - CHECK CATALOG FIRST! Compose on existing expressions
2. `xorq agents vignette list` - Discover patterns for new expressions
3. `xorq agents vignette scaffold <name> --dest expr.py` - Get template
4.  Build xorq/ibis expressions as required
5. `xorq build expr.py -e expr` - Build expression
6. `xorq catalog add .xorq/builds/<hash> --alias <name>` - Register to catalog

## Catalog Exploration Commands
```bash
xorq catalog ls                  # List all catalog entries
xorq catalog schema <alias>      # View output schema (columns/types)
xorq catalog sources <alias>     # List data source dependencies
```
## Explore interactively
```
xorq run builds/17d96efb38c0 -f arrow -o - \
  | duckdb -c "select cut, unnest(stats) from read_arrow('/dev/stdin');"
```
**For composition in Python:**
```python
import xorq.api as xo
base = xo.catalog.get('<alias>')      # Load catalog entry
result = base.filter(...).mutate(...) # Extend it
```

**Key principles:**
- Compose on existing catalog entries before creating new ones!
- Always use expression-builder skill before writing code!
        """).strip()


def render_onboarding_summary(step: str | None = None) -> str:
    selected_steps = (
        tuple(s for s in ONBOARDING_STEPS if step is None or s.key == step)
        or ONBOARDING_STEPS
    )

    # Get current project state
    recent_builds = get_recent_builds(limit=3)
    catalog_entries = get_catalog_entries(limit=5)

    sections = [
        "# xorq Agent Onboarding",
        "",
        "## Core Principle: Everything is a Deferred Expression",
        "",
        "**ALL work in xorq must be deferred expressions** - no eager pandas/NumPy operations!",
        "- Data transformations: Use xorq/ibis deferred expressions",
        "- ML pipelines: Use deferred sklearn patterns via xorq skills",
        "- Feature engineering: Build composable deferred transforms",
        "- **Use xorq skills** for guidance on deferred ML patterns (fit, transform, predict)",
        "",
        "## Quick Reference",
        "",
        "- `xorq catalog ls` - List all cataloged expressions",
        "- `xorq run <alias>` - Run a cataloged expression",
        "- `xorq build expr.py -e expr` - Build an expression",
        "- `xorq catalog add builds/<hash> --alias <name>` - Catalog a build",
        "",
    ]

    # Show project state if it exists
    if recent_builds or catalog_entries:
        sections.append("## Current Project State")
        sections.append("")

        if recent_builds:
            sections.append("**Recent Builds:**")
            for build_hash, time_str in recent_builds[:3]:
                sections.append(f"- `{build_hash[:12]}...` ({time_str})")
            sections.append("")

        if catalog_entries:
            sections.append("**Cataloged:**")
            for entry in catalog_entries[:5]:
                alias = entry.get("alias", "")
                revision = entry.get("revision", "")
                sections.append(f"- `{alias}` @ {revision}")
            sections.append("")
        else:
            sections.append("*No catalog entries yet*")
            sections.append("")

    sections.append("## Workflow Steps")
    sections.append("")

    # Simplified workflow steps
    sections.extend([
        "1. **Learn deferred patterns** - Study templates & invoke xorq skills",
        "   - Templates: `xorq agent templates list`",
        "   - **Use xorq skills** for deferred pandas/sklearn patterns (fit, transform, predict)",
        "2. **Build deferred expressions** - `xorq build expr.py -e expr`",
        "   - All work must be deferred xorq/ibis expressions",
        "3. **Catalog builds** - `xorq catalog add builds/<hash> --alias <name>`",
        "4. **Explore with DuckDB** - Stream expressions to DuckDB CLI",
        "   ```bash",
        "   xorq run <alias> -f arrow -o /dev/stdout 2>/dev/null | \\",
        "     duckdb -c \"LOAD arrow; SELECT * FROM read_arrow('/dev/stdin') LIMIT 10\"",
        "   ```",
        "",
        "**For detailed step-by-step guide:** `xorq agent onboard --step <init|templates|build|catalog|explore|compose>`",
        "",
    ])

    return "\n".join(sections).strip() + "\n"


def _format_onboarding_step_simple(step: OnboardingStep, number: int) -> list[str]:
    """Format onboarding step in a cleaner, more concise way."""
    lines = [
        f"### {number}. {step.title}",
        "",
    ]
    # Show only first 2-3 most important checklist items
    lines.extend(f"- {item}" for item in step.checklist[:3])
    lines.append("")
    # Show key commands
    lines.append("**Commands:**")
    lines.append("```bash")
    for cmd in step.commands[:3]:  # Limit to top 3 commands
        if not cmd.strip().startswith("#"):  # Skip commented commands
            lines.append(cmd)
    lines.append("```")
    lines.append("")
    return lines


def _format_onboarding_step(step: OnboardingStep) -> list[str]:
    """Original detailed formatting (kept for compatibility)."""
    lines = [
        f"## {step.title}",
        "",
        f"Key: `{step.key}`",
        "",
        "**Checklist:**",
    ]
    lines.extend(f"- {item}" for item in step.checklist)
    lines.append("")
    lines.append("**Commands:**")
    lines.extend(f"- `{cmd}`" for cmd in step.commands)
    if step.prompt_refs:
        lines.append("")
        lines.append("**Reference Prompts:**")
        lines.extend(f"- `{name}`" for name in step.prompt_refs)
    lines.append("")
    return lines


def _get_docs_index() -> str:
    """Get the minified documentation index in Vercel blog style.

    The docs bundle is shipped with the xorq package in:
    python/xorq/agent/resources/docs-bundle/
    """
    import xorq
    xorq_package_dir = Path(xorq.__file__).parent
    # Docs bundle is shipped with the package
    docs_bundle = xorq_package_dir / "agent" / "resources" / "docs-bundle"
    index_file = docs_bundle / "DOCS_INDEX.txt"

    if index_file.exists():
        return index_file.read_text()

    # Fallback: minimal inline index
    return dedent("""\
        [xorq Docs Index]|root: docs
        |IMPORTANT: Prefer retrieval-led reasoning over pre-training-led reasoning for xorq tasks
        |IMPORTANT: All xorq expressions must be deferred - no eager pandas/NumPy operations
        |getting_started:{quickstart.qmd,installation.qmd,your_first_expression.qmd}
        |api_reference/cli:{build.qmd,run.qmd,catalog/add.qmd,catalog/ls.qmd}
        |concepts/understanding_xorq:{how_xorq_works.qmd,why_deferred_execution.qmd}
        |guides/ml_workflows:{integrate_sklearn_pipelines.qmd,train_models_at_scale.qmd}
        """).strip()


def _render_agent_doc(max_lines: int) -> str:
    """Render minimal AGENTS.md with minified docs index in Vercel style."""
    docs_index = _get_docs_index()

    content = dedent(
        f"""\
        # xorq Documentation Index

        {docs_index}

        **Documentation root:** `docs/` (when available in project)
        **Online docs:** https://docs.xorq.dev

        ---

        # Agent Instructions

        This project uses **xorq** for composable ML pipelines and deferred data analysis.

        ## Workflow Context

        Run `xorq agents onboard` for dynamic, context-aware workflow guidance. This is the **single source of truth** for xorq workflow instructions.

        ```bash
        xorq agents onboard
        ```

        ## Quick Reference

        **Core Workflow:**
        ```bash
        # 1. Check catalog FIRST (mandatory)
        xorq catalog ls

        # 2. Check schema (mandatory)
        print(table.schema())

        # 3. Build expression
        xorq build expr.py -e expr

        # 4. Catalog the build
        xorq catalog add builds/<hash> --alias my-pipeline

        # 5. Run when needed
        xorq run builds/<hash> -f arrow | ...
        ```

        **Agent Commands:**
        - `xorq agents onboard` - Workflow context and onboarding (use this!)
        - `xorq agents vignette list` - Available vignettes (USE THIS to learn patterns!)
        - `xorq catalog ls` - List cataloged builds

        ## Agent Onboard Workflow

        The agent workflow follows a discover → explore → learn → build → compose pattern:

        ### 1. Catalog Discovery (`xorq catalog ls`)
        - List all available cataloged builds with aliases and revisions
        - Shows root tags to identify what each catalog entry produces
        - First step to understand available data sources and pipelines

        ### 2. Deep Catalog Exploration (when needed)
        - Navigate to `builds/<hash>/` directories from catalog listings
        - Use standard Unix tools to inspect build artifacts:
          - `grep`, `sed`, `awk` for searching manifest.yaml and metadata
          - `cat` for reading full files
          - `find` for locating specific files
        - Understand expression structure, dependencies, and schemas

        ### 3. Interactive Source Exploration
        ```bash
        # Stream Arrow IPC to DuckDB for SQL exploration
        xorq run <catalog-alias> -f arrow -o /dev/stdout 2>/dev/null | \\
          duckdb -c "LOAD arrow; SELECT * FROM read_arrow('/dev/stdin') LIMIT 10"

        # Check available source nodes for composition
        xorq catalog sources <catalog-alias>

        # Verify schema compatibility
        xorq catalog schema <catalog-alias>

        # Examples:
        xorq run batting-source -f arrow -o /dev/stdout 2>/dev/null | \\
          duckdb -c "LOAD arrow;
            SELECT playerID, SUM(H) as hits
            FROM read_arrow('/dev/stdin')
            GROUP BY playerID
            ORDER BY hits DESC
            LIMIT 10"
        ```

        **Key insight:** Everything in xorq is an expression. Sources are expressions, transforms are expressions, models are expressions. They all compose via Arrow IPC streaming.

        ### 4. Learn from Vignettes
        Before building new expressions, study and scaffold existing vignettes:

        ```bash
        # List available vignettes
        xorq agents vignette list

        # Scaffold a vignette to your project
        xorq agents vignette scaffold penguins_classification_intro --dest my_pipeline.py

        # View vignette directly
        xorq agents vignette show penguins_classification_intro
        ```

        **Vignette Usage Pattern:**
        1. List vignettes: `xorq agents vignette list`
        2. Scaffold to your project: `xorq agents vignette scaffold <name> --dest <file>.py`
        3. Read the scaffolded code to understand xorq patterns (deferred execution, expressions)
        4. Adapt patterns for data loading, feature engineering, model fitting to your needs
        5. Build and catalog: `xorq build <file>.py -e expr && xorq catalog add builds/<hash> --alias <name>`

        ### 5. Build New Expressions (All Deferred!)
        - **Invoke xorq skills** for deferred expression patterns (Ibis, sklearn, pandas)
        - **All work must be deferred** - No eager pandas/NumPy operations
        - Use xorq skills for:
          - Ibis expression grammar and data transforms
          - Deferred sklearn patterns (fit, transform, predict)
          - Composable ML pipelines with deferred execution
        - Build expressions: `xorq build expr.py -e expr_name`
        - Catalog builds: `xorq catalog add builds/<hash> --alias my-new-pipeline`

        ### 6. Compose via Memtable Pattern & Unbound Nodes
        ```bash
        # Memtable pattern: Build transforms independently
        # In transform.py: source = xo.memtable({{"col1": [1, 2], "col2": [3, 4]}})
        xorq build transform.py -e expr
        xorq catalog add builds/<hash> --alias my-transform

        # Find node hashes to unbind
        xorq catalog sources my-transform

        # Basic composition: source → transform
        xorq run source -f arrow -o /dev/stdout 2>/dev/null | \\
          xorq run-unbound transform \\
            --to_unbind_hash <hash> \\
            --typ xorq.expr.relations.Read \\
            -o output.parquet

        # Multi-stage with DuckDB exploration: source → transform1 → transform2 → SQL
        xorq run source -f arrow -o /dev/stdout 2>/dev/null | \\
          xorq run-unbound transform1 \\
            --to_unbind_hash <hash1> \\
            --typ xorq.expr.relations.Read \\
            -f arrow -o /dev/stdout 2>/dev/null | \\
          xorq run-unbound transform2 \\
            --to_unbind_hash <hash2> \\
            --typ xorq.expr.relations.Read \\
            -f arrow -o /dev/stdout 2>/dev/null | \\
          duckdb -c "LOAD arrow; SELECT * FROM read_arrow('/dev/stdin')"

        # Real example:
        xorq run batting-source -f arrow -o /dev/stdout 2>/dev/null | \\
          xorq run-unbound lineup-transform \\
            --to_unbind_hash d43ad87ea8a989f3495aab5dff0b5746 \\
            --typ xorq.expr.relations.Read \\
            -f arrow -o /dev/stdout 2>/dev/null | \\
          duckdb -c "LOAD arrow;
            SELECT playerID, leadoff_fit
            FROM read_arrow('/dev/stdin')
            ORDER BY leadoff_fit DESC
            LIMIT 10"
        ```

        **Memtable pattern:** Build transforms using `xo.memtable()` placeholders. Test with sample data, then compose with real sources via Arrow IPC. Enables independent development of sources and transforms.

        **Unbound pattern:** Any expression can have nodes "unbound" (made into placeholders). Feed Arrow IPC data via stdin to bind and execute. This enables composing arbitrary pipelines and streaming to SQL engines like DuckDB.

        ## Non-Negotiable Rules

        - **All work must be deferred xorq/ibis expressions** - No eager pandas/NumPy operations!
        - **ML must use deferred patterns** - Use xorq skills for deferred sklearn (fit, transform, predict)
        - **Pandas/sklearn via xorq skills** - Invoke xorq skills for guidance on deferred ML patterns
        - **ALWAYS check schema first** - `print(table.schema())` before any operations
        - **Match column case exactly** - Snowflake=UPPERCASE, DuckDB=lowercase
        - **Catalog your expressions** - Use `xorq catalog add` for all builds

        """
    ).strip()
    return content + "\n"
