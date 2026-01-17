from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

from .prompts import (
    CORE_PROMPT_TIER,
    PROMPT_TIER_DESCRIPTIONS,
    iter_prompt_specs,
    load_prompt_text,
    prompt_excerpt,
    prompt_title,
)


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
            "xorq agent prompt list --tier core",
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
            "Link catalog entries to bd issues or template skills.",
        ],
        [
            "xorq catalog add builds/<hash> --alias <name>",
            "xorq catalog ls",
        ],
        [
            "context_blocks/expression_deliverables",
        ],
    ),
    OnboardingStep(
        "test",
        "Validate builds and reliability gates",
        [
            "Execute expressions or unbound variants locally.",
            "Run template-specific tests/linters documented in AGENTS.md.",
            "Capture lineage/inspection outputs for review.",
        ],
        [
            "xorq run builds/<hash> -o /tmp/out.parquet",
            "xorq lineage <alias>",
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
            "xorq agent prompt list --tier reliability",
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
    bd_ok = bd_cli_available()
    selected_steps = (
        tuple(s for s in ONBOARDING_STEPS if step is None or s.key == step)
        or ONBOARDING_STEPS
    )
    sections = [
        "# xorq Agent Onboarding",
        "",
        "Use this guide to progress from project initialization to landing the plane. "
        "Reference `xorq agent prompt list` to pull full guidance for each step.",
        "",
    ]
    if not bd_ok:
        sections.append(
            "⚠️ `bd` CLI not detected on PATH. Install via "
            "`brew install steveyegge/beads/bd` or "
            "`npm install -g @beads/bd`, then rerun `bd onboard`."
        )
        sections.append("")
    for onboarding_step in selected_steps:
        sections.extend(_format_onboarding_step(onboarding_step))
    sections.append(
        "\nPrompts live under `docs/agent_prompts/`; use "
        "`xorq agent prompt show <name>` for detailed context."
    )
    return "\n".join(sections).strip() + "\n"


def _format_onboarding_step(step: OnboardingStep) -> list[str]:
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
    header = dedent(
        """\
        # Agent Instructions

        This project was initialized with `xorq init --agent`. You now have access to
        the bundled xorq prompt library. Use `xorq agent prompt list` or
        `xorq agent prompt show <name>` to explore the full catalog.

        The sections below contain curated excerpts from the core prompts so new agents
        see the most important information without leaving the repo.
        """
    ).strip()
    sections: list[str] = [header, ""]
    for spec in iter_prompt_specs(tier=CORE_PROMPT_TIER):
        text = load_prompt_text(spec)
        title = prompt_title(text, spec.name)
        excerpt = prompt_excerpt(text, max_lines=max_lines)
        sections.append(f"## {title} (`{spec.rel_path}`)")
        sections.append("")
        sections.append("```markdown")
        sections.append(excerpt.strip())
        sections.append("```")
        sections.append("")
    other_tiers = [
        f"- `{tier}` — {PROMPT_TIER_DESCRIPTIONS[tier]}"
        for tier in PROMPT_TIER_DESCRIPTIONS
        if tier != CORE_PROMPT_TIER
    ]
    sections.append("## Explore the Remaining Prompt Bundles")
    sections.append("")
    sections.append(
        "Run `xorq agent prompt list --tier <tier>` to inspect additional guidance:"
    )
    sections.extend(other_tiers)
    sections.append("")
    sections.append(
        "All prompt files live under `docs/agent_prompts/` for manual browsing."
    )
    sections.append("")
    sections.append(
        "When onboarding new agents, link to this file and encourage referencing the "
        "full prompt list before starting work."
    )
    return "\n".join(sections).rstrip() + "\n"


def bd_cli_available() -> bool:
    return shutil.which("bd") is not None
