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


@dataclass(frozen=True)
class OnboardingStep:
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
        "Compose cataloged expressions",
        [
            "Get placeholders with xo.catalog.get_placeholder('alias', tag='tag') (preferred)",
            "Or load from build directory for debugging: xo.catalog.load_expr('builds/<hash>')",
            "Build transforms using placeholders, then catalog",
            "Compose via CLI using run-unbound with tags",
        ],
        [
            "# Python composition with placeholders (RECOMMENDED)",
            "# source = xo.catalog.get_placeholder('my-source', tag='src')",
            "# transform = source.filter(xo._.value > 100).select('id', 'value')",
            "# Save transform.py with: expr = transform",
            "# xorq build transform.py -e expr && xorq catalog add builds/<hash> --alias my-transform",
            "",
            "# CLI composition via run-unbound",
            "xorq run source -o arrow | xorq run-unbound transform --to_unbind_tag src",
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
    """Register the xorq skill with Claude Code.

    Returns the path where the skill was registered, or None if already exists.
    """

    # Generate fresh skill from shared content
    try:
        from xorq.agent.resources.common.generate_skills import generate_all_skills

        generate_all_skills()
    except Exception as e:
        print(f"⚠️  Could not regenerate skills: {e}")

    # Find the skill source directory (should be in the package)
    import xorq

    xorq_package_dir = Path(xorq.__file__).parent.parent.parent
    skill_source = xorq_package_dir / "skills" / "xorq"

    # Claude Code skills directory
    claude_skills_dir = Path.home() / ".claude" / "skills"
    skill_dest = claude_skills_dir / "xorq"

    # Check if skill source exists
    if not skill_source.exists():
        # If not found, try relative to project root
        import sys

        for path in sys.path:
            candidate = Path(path) / "skills" / "xorq"
            if candidate.exists():
                skill_source = candidate
                break
        else:
            # Can't find skill source
            return None

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
        # Update existing skill files (handle read-only files)
        def handle_remove_readonly(func, path, exc):
            """Error handler for shutil.rmtree to handle read-only files."""
            import stat
            if isinstance(exc[1], PermissionError):
                # Make the file writable and try again
                os.chmod(path, stat.S_IWRITE)
                func(path)
            else:
                raise

        shutil.rmtree(skill_dest, onerror=handle_remove_readonly)

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

    # Load the xorq skill rules template
    xorq_rules = json.loads(skill_rules_source.read_text())

    # Check if skill-rules.json already exists
    if skill_rules_path.exists():
        # Merge with existing rules
        existing_rules = json.loads(skill_rules_path.read_text())

        # Update only the xorq skill entry, preserve others
        if "skills" not in existing_rules:
            existing_rules["skills"] = {}

        # Update xorq skill entry
        if "skills" in xorq_rules and "xorq" in xorq_rules["skills"]:
            existing_rules["skills"]["xorq"] = xorq_rules["skills"]["xorq"]

        # Ensure version and description exist
        if "version" not in existing_rules:
            existing_rules["version"] = xorq_rules.get("version", "1.0")
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
        "5. **Commit and push** - `xorq agents land` for checklist",
        "",
        "**For detailed step-by-step guide:** `xorq agent onboard --step <init|templates|build|catalog|explore|compose|land>`",
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


def _render_agent_doc(max_lines: int) -> str:
    """Render minimal AGENTS.md pointing to xorq agent commands as source of truth."""
    content = dedent(
        """\
        # Agent Instructions

        This project uses **xorq** for composable ML pipelines and deferred data analysis.

        ## Workflow Context

        Run `xorq agent onboard` for dynamic, context-aware workflow guidance. This is the **single source of truth** for xorq workflow instructions.

        ```bash
        xorq agent onboard
        ```

        ## Quick Reference

        **Core Workflow:**
        ```bash
        # 1. Check schema FIRST (mandatory)
        print(table.schema())

        # 2. Build expression
        xorq build expr.py -e expr

        # 3. Catalog the build
        xorq catalog add builds/<hash> --alias my-pipeline

        # 4. Run when needed
        xorq run builds/<hash> -f arrow | ...
        ```

        **Agent Commands:**
        - `xorq agent onboard` - Workflow context and onboarding (use this!)
        - `xorq agent land` - Session close checklist (MANDATORY before completion)
        - `xorq agent templates list` - Available templates (USE THIS to learn patterns!)
        - `xorq catalog ls` - List cataloged builds

        ## Agent Onboard/Land Workflow

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
        xorq run <catalog-alias> -f arrow -o /dev/stdout 2>/dev/null | \
          duckdb -c "LOAD arrow; SELECT * FROM read_arrow('/dev/stdin') LIMIT 10"

        # Check available source nodes for composition
        xorq catalog sources <catalog-alias>

        # Verify schema compatibility
        xorq catalog schema <catalog-alias>

        # Examples:
        xorq run batting-source -f arrow -o /dev/stdout 2>/dev/null | \
          duckdb -c "LOAD arrow;
            SELECT playerID, SUM(H) as hits
            FROM read_arrow('/dev/stdin')
            GROUP BY playerID
            ORDER BY hits DESC
            LIMIT 10"
        ```

        **Key insight:** Everything in xorq is an expression. Sources are expressions, transforms are expressions, models are expressions. They all compose via Arrow IPC streaming.

        ### 4. Learn from Templates
        Before building new expressions, study and scaffold existing templates:

        ```bash
        # List available templates
        xorq agent templates list

        # Scaffold a template to your project
        xorq agent templates scaffold pipeline_example --dest my_pipeline.py
        xorq agent templates scaffold diamonds_price_prediction --dest my_model.py

        # See template details
        xorq agent templates show pipeline_example

        # Or read directly from examples/
        cat examples/pipeline_example.py
        cat examples/diamonds_price_prediction.py
        ```

        **Available Templates:**
        - **pipeline_example**: sklearn pipelines with StandardScaler + KNeighborsClassifier on iris
        - **diamonds_price_prediction**: Feature engineering, train/test splits, LinearRegression
        - **sklearn_classifier_comparison**: Compare multiple classifiers on same dataset
        - **deferred_fit_transform_predict**: Complete deferred ML workflow pattern
        - **penguins_demo**: Minimal multi-engine example, good starting point (basic scaffold)

        **Template Usage Pattern:**
        1. List templates: `xorq agent templates list`
        2. Scaffold to your project: `xorq agent templates scaffold <name> --dest <file>.py`
        3. Read the scaffolded code to understand xorq patterns (deferred execution, expressions)
        4. Adapt patterns for data loading, feature engineering, model fitting to your needs
        5. Build and catalog: `xorq build <file>.py -e expr && xorq catalog add builds/<hash> --alias <name>`

        ### 5. Import Patterns for Build Scripts

        **✅ Correct imports (always use these):**
        ```python
        import xorq.api as xo           # Main xorq API
        from xorq.caching import ParquetCache  # For caching

        # Catalog functions
        placeholder = xo.catalog.get_placeholder("my-alias", tag="tag")  # Get placeholder (for transforms)
        expr = xo.catalog.load_expr("builds/...")   # Load from build dir (for debugging)
        ```

        **❌ Don't use these (not available in build context):**
        ```python
        from xorq.common.utils.ibis_utils import from_ibis  # Will fail!
        from xorq.vendor import ibis  # Use xo._ instead
        ```

        **Key principle:** Use `xo.*` for all operations. The `xo` namespace provides everything you need.

        ### 6. Build New Expressions (All Deferred!)
        - **Invoke xorq skills** for deferred expression patterns (Ibis, sklearn, pandas)
        - **All work must be deferred** - No eager pandas/NumPy operations
        - Use xorq skills for:
          - Ibis expression grammar and data transforms
          - Deferred sklearn patterns (fit, transform, predict)
          - Composable ML pipelines with deferred execution
        - Build expressions: `xorq build expr.py -e expr_name`
        - Catalog builds: `xorq catalog add builds/<hash> --alias my-new-pipeline`

        ### 7. Build Transforms Using Catalog Placeholders
        ```python
        # Get placeholder with schema from catalog (Python API)
        import xorq.api as xo

        # Get placeholder - only loads schema, not full expression
        source_placeholder = xo.catalog.get_placeholder("batting-source")

        # Build transform using placeholder
        lineup_transform = (
            source_placeholder
            .select("playerID", "H", "AB")
            .mutate(batting_avg=xo._.H / xo._.AB)
        )

        # Build and catalog
        # xorq build transform.py -e lineup_transform
        # xorq catalog add builds/<hash> --alias lineup-transform
        ```

        ```bash
        # CLI composition via Arrow IPC (when needed)
        # Find node hashes to unbind
        xorq catalog sources my-transform

        # Basic composition: source → transform
        xorq run source -f arrow -o /dev/stdout 2>/dev/null | \
          xorq run-unbound transform \
            --to_unbind_hash <hash> \
            --typ xorq.expr.relations.Read \
            -o output.parquet

        # Multi-stage with DuckDB exploration
        xorq run batting-source -f arrow -o /dev/stdout 2>/dev/null | \
          xorq run-unbound lineup-transform \
            --to_unbind_hash d43ad87ea8a989f3495aab5dff0b5746 \
            --typ xorq.expr.relations.Read \
            -f arrow -o /dev/stdout 2>/dev/null | \
          duckdb -c "LOAD arrow;
            SELECT playerID, leadoff_fit
            FROM read_arrow('/dev/stdin')
            ORDER BY leadoff_fit DESC
            LIMIT 10"
        ```

        **Composition patterns:** Use `xo.catalog.get_placeholder()` for Python-native composition (preferred), or Arrow IPC streaming for CLI pipelines. The catalog is the single source of truth for all expressions.

        ### 8. Land the Plane (`xorq agent land`)
        **MANDATORY before session completion:**
        - Validates all builds are cataloged
        - Checks git status (catalog.yaml and builds/ must be committed)
        - Ensures remote is up to date
        - Provides validation commands

        **Never skip this step.** Work is not done until pushed and validated.

        ## Non-Negotiable Rules

        - **All work must be deferred xorq/ibis expressions** - No eager pandas/NumPy operations!
        - **ML must use deferred patterns** - Use xorq skills for deferred sklearn (fit, transform, predict)
        - **Pandas/sklearn via xorq skills** - Invoke xorq skills for guidance on deferred ML patterns
        - **ALWAYS check schema first** - `print(table.schema())` before any operations
        - **Match column case exactly** - Snowflake=UPPERCASE, DuckDB=lowercase
        - **Catalog your expressions** - Use `xorq catalog add` for all builds
        - **Session close protocol** - Run `xorq agent land` to see mandatory steps

        """
    ).strip()
    return content + "\n"
