"""Generate agent-specific SKILL.md files from shared content."""

from __future__ import annotations

from pathlib import Path


def generate_skill(
    agent: str,
    version: str = "0.2.0",
    output_path: Path | None = None,
) -> str:
    """Generate agent-specific SKILL.md from shared core and wrapper.

    Parameters
    ----------
    agent : str
        Agent type: "claude" or "codex"
    version : str
        Version string for the skill
    output_path : Path | None
        If provided, write output to this path

    Returns
    -------
    str
        Generated SKILL.md content
    """
    resources_dir = Path(__file__).parent

    # Read shared core content
    core_content = (resources_dir / "skill_core.md").read_text()

    # Read agent-specific wrapper
    wrapper_file = resources_dir / f"{agent}_wrapper.md"
    if not wrapper_file.exists():
        raise ValueError(f"Unknown agent type: {agent}")

    wrapper_content = wrapper_file.read_text()

    # Substitute placeholders
    generated = wrapper_content.replace("{{VERSION}}", version)
    generated = generated.replace("{{CORE_CONTENT}}", core_content)

    # Write to output path if specified
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(generated)

    return generated


def generate_all_skills(version: str = "0.2.0") -> dict[str, Path]:
    """Generate all agent-specific skills and write to their destinations.

    Parameters
    ----------
    version : str
        Version string for the skills

    Returns
    -------
    dict[str, Path]
        Mapping of agent name to output path
    """
    resources_dir = Path(__file__).parent.parent
    generated = {}

    # Generate Codex skill to resources/codex/SKILL.md
    codex_output = resources_dir / "codex" / "SKILL.md"
    generate_skill("codex", version=version, output_path=codex_output)
    generated["codex"] = codex_output

    # Generate Claude skill to skills/xorq/SKILL.md (project root)
    project_root = resources_dir.parent.parent.parent.parent
    claude_output = project_root / "skills" / "xorq" / "SKILL.md"
    generate_skill("claude", version=version, output_path=claude_output)
    generated["claude"] = claude_output

    return generated


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        agent = sys.argv[1]
        version = sys.argv[2] if len(sys.argv) > 2 else "0.2.0"
        output = generate_skill(agent, version=version)
        print(output)
    else:
        # Generate all skills
        generated = generate_all_skills()
        for agent, path in generated.items():
            print(f"✅ Generated {agent} skill: {path}")
