from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent


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
        "Initialize project and install dependencies",
        [
            "Ensure `xorq` CLI is installed in the active environment.",
            "Run `xorq init --agent` to fetch templates plus AGENTS/CLAUDE guides.",
            "Install helper CLIs referenced in AGENTS.md (bd, uv, etc.).",
        ],
        [
            "xorq init --agent",
            "bd onboard  # if bd is installed",
        ],
        [
            "planning_phase",
            "context_blocks/xorq_core",
        ],
    ),
    OnboardingStep(
        "build",
        "Author expressions using existing sources",
        [
            "Identify registered sources (DuckDB, Snowflake) and inspect schemas.",
            "Create or edit expression scripts using xorq templates/skills.",
            "Build artifacts locally to verify manifests/caches.",
        ],
        [
            "xorq agent prime",
            "xorq build expr.py -e expr",
        ],
        [
            "xorq_vendor_ibis",
            "context_blocks/critical_rules",
        ],
    ),
    OnboardingStep(
        "catalog",
        "Publish artifacts into the xorq catalog",
        [
            "Inspect the build output (manifest, metadata, cache).",
            "Assign aliases and register artifacts for reuse.",
            "View catalog entries with root tags via `xorq catalog ls`.",
            "Link catalog entries to bd issues or template skills.",
        ],
        [
            "xorq catalog add builds/<hash> --alias <name>",
            "xorq catalog ls  # Shows aliases, revisions, and root tags",
        ],
        [
            "context_blocks/expression_deliverables",
        ],
    ),
    OnboardingStep(
        "test",
        "Validate builds and reliability gates",
        [
            "Execute expressions using catalog aliases or revisions.",
            "Run template-specific tests/linters documented in AGENTS.md.",
            "Capture lineage/inspection outputs for review.",
        ],
        [
            "xorq run <alias> -o /tmp/out.parquet  # Run by alias",
            "xorq run <alias>@r2 -o /tmp/out.parquet  # Run specific revision",
            "xorq lineage <alias>  # Show column lineage with root tag",
        ],
        [
            "context_blocks/phase_data_preparation_completion_check",
            "context_blocks/phase_modeling_completion_check",
        ],
    ),
    OnboardingStep(
        "land",
        "Land the plane (hand-off checklist)",
        [
            "Update/close bd issues, commit manifests, and document results.",
            "Run `bd sync`, `git push`, and verify clean status.",
            "Surface relevant prompts for the next agent session.",
        ],
        [
            "bd sync",
            "git push",
            "xorq agent prime",
        ],
        [
            "context_blocks/phase_checks",
        ],
    ),
)


def register_claude_skill() -> Path | None:
    """Register the xorq skill with Claude Code.

    Returns the path where the skill was registered, or None if already exists.
    """
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
        # Skill already registered, optionally update it
        shutil.rmtree(skill_dest)

    shutil.copytree(skill_source, skill_dest)
    return skill_dest


def bootstrap_agent_docs(project_root: str | Path, max_lines: int = 60) -> list[Path]:
    """Create AGENTS.md and CLAUDE.md populated with core prompt excerpts."""
    root = Path(project_root)
    root.mkdir(parents=True, exist_ok=True)
    content = _render_agent_doc(max_lines=max_lines)
    created: list[Path] = []
    for filename in ("AGENTS.md", "CLAUDE.md"):
        dest = root / filename
        if dest.exists():
            continue
        dest.write_text(content, encoding="utf-8")
        created.append(dest)

    # Register the xorq skill with Claude Code
    skill_path = register_claude_skill()
    if skill_path:
        print(f"Registered xorq skill with Claude Code at {skill_path}")

    return created


def render_onboarding_summary(step: str | None = None) -> str:
    from xorq.agent.prime import (
        _format_builds_status,
        get_catalog_entries,
        get_recent_builds,
    )

    bd_ok = bd_cli_available()
    selected_steps = (
        tuple(s for s in ONBOARDING_STEPS if step is None or s.key == step)
        or ONBOARDING_STEPS
    )

    # Get current project state
    recent_builds = get_recent_builds(limit=3)
    catalog_entries = get_catalog_entries(limit=5)
    project_state = ""
    if recent_builds or catalog_entries:
        project_state = _format_builds_status(recent_builds, catalog_entries)
        project_state = "\n## Your Project Status\n\n" + project_state + "\n"

    sections = [
        "# 🚀 xorq Agent Onboarding",
        "",
        "**Welcome!** Follow this guide to build composable ML pipelines with xorq.",
        "",
        "💡 **Quick Start**: Run `xorq agent prime` for dynamic workflow context.",
        "",
    ]

    if project_state:
        sections.append(project_state)

    sections.append("## Workflow Steps\n")

    for idx, onboarding_step in enumerate(selected_steps, 1):
        sections.extend(_format_onboarding_step_simple(onboarding_step, idx))

    sections.extend(
        [
            "",
            "---",
            "",
            "**Need Help?**",
            "- 📖 Run `xorq agent prime` - Dynamic workflow context",
            "- 📝 Run `xorq agent templates list` - View available templates",
            "- 📊 Run `xorq catalog ls` - See your cataloged pipelines",
            "",
        ]
    )

    if bd_ok:
        sections.append(
            "✅ `bd` CLI detected - you can use `bd sync` for session management"
        )
    else:
        sections.append("ℹ️  Optional: Install `bd` CLI for enhanced session management")

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
    """Render minimal AGENTS.md pointing to `xorq agent prime` as source of truth."""
    content = dedent(
        """\
        # Agent Instructions

        This project uses **xorq** for composable ML pipelines and deferred data analysis.

        ## Workflow Context

        Run `xorq agent prime` for dynamic, context-aware workflow guidance. This is the **single source of truth** for xorq workflow instructions.

        ```bash
        xorq agent prime
        ```

        The `prime` command provides:
        - Current project state (recent builds, catalog status)
        - Session close protocol (critical git workflow)
        - Core xorq rules (schema checks, deferred execution, no pandas)
        - Essential command reference

        ## Quick Reference

        **Core Workflow:**
        ```bash
        # 1. Check schema FIRST (mandatory)
        print(table.schema())

        # 2. Build expression
        xorq build expr.py -e expr

        # 3. Catalog the build
        xorq catalog add builds/<hash> --alias my-pipeline

        # 4. View cataloged entries (shows root tags)
        xorq catalog ls

        # 5. Run by alias or revision
        xorq run my-pipeline -o output.parquet
        xorq run my-pipeline@r2 -o output.parquet  # Specific revision
        ```

        **Agent Commands:**
        - `xorq agent prime` - Workflow context (use this!)
        - `xorq agent templates list` - Available templates
        - `xorq agent onboard` - Onboarding guide
        - `xorq catalog ls` - List cataloged builds

        ## Non-Negotiable Rules

        - **All work must be deferred xorq/ibis expressions** - No pandas/NumPy scripts
        - **ALWAYS check schema first** - `print(table.schema())` before any operations
        - **Match column case exactly** - Snowflake=UPPERCASE, DuckDB=lowercase
        - **Catalog your expressions** - Use `xorq catalog add` for all builds
        - **Session close protocol** - Run `xorq agent prime` to see mandatory steps

        ## Customization

        Create `.xorq/PRIME.md` to customize workflow guidance for this project.

        ---

        **For full workflow context, run `xorq agent prime` at session start.**
        """
    ).strip()
    return content + "\n"


def bd_cli_available() -> bool:
    return shutil.which("bd") is not None
