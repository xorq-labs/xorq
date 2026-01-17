from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


def _resolve_prompt_root() -> Path:
    this_file = Path(__file__).resolve()
    for parent in this_file.parents:
        candidate = parent / "docs" / "agent_prompts"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Unable to locate docs/agent_prompts relative to "
        f"{this_file}. Ensure prompts are bundled with the package."
    )


PROMPT_ROOT = _resolve_prompt_root()

CORE_PROMPT_TIER = "core"
RELIABILITY_PROMPT_TIER = "reliability"
ADVANCED_PROMPT_TIER = "advanced"

PROMPT_TIER_DESCRIPTIONS = {
    CORE_PROMPT_TIER: "Critical onboarding prompts surfaced automatically",
    RELIABILITY_PROMPT_TIER: "Troubleshooting guides and reliability patterns",
    ADVANCED_PROMPT_TIER: "Advanced skills, completion checks, and ML guidance",
}


@dataclass(frozen=True)
class PromptSpec:
    name: str
    rel_path: str
    tier: str
    description: str

    @property
    def path(self) -> Path:
        return PROMPT_ROOT / self.rel_path


PROMPT_SPECS: tuple[PromptSpec, ...] = (
    # Core prompts
    PromptSpec(
        "planning_phase",
        "planning_phase.md",
        CORE_PROMPT_TIER,
        "Planning ritual and struct prediction rules",
    ),
    PromptSpec(
        "sequential_execution",
        "sequential_execution.md",
        CORE_PROMPT_TIER,
        "Step-by-step execution guidance",
    ),
    PromptSpec(
        "context_blocks/xorq_core",
        "context_blocks/xorq_core.md",
        CORE_PROMPT_TIER,
        "Deferred execution core rules",
    ),
    # Reliability prompts
    PromptSpec(
        "context_blocks/critical_rules",
        "context_blocks/critical_rules.md",
        RELIABILITY_PROMPT_TIER,
        "Critical rules: schema checking, column case, no pandas, vendor ibis, data sources",
    ),
    PromptSpec(
        "context_blocks/xorq_patterns",
        "context_blocks/xorq_patterns.md",
        RELIABILITY_PROMPT_TIER,
        "Working patterns and common pitfalls",
    ),
    PromptSpec(
        "context_blocks/udf_patterns",
        "context_blocks/udf_patterns.md",
        RELIABILITY_PROMPT_TIER,
        "UDF patterns for deferred computation",
    ),
    PromptSpec(
        "context_blocks/backend_troubleshooting",
        "context_blocks/backend_troubleshooting.md",
        RELIABILITY_PROMPT_TIER,
        "Backend and UDF troubleshooting",
    ),
    PromptSpec(
        "context_blocks/quick_reference",
        "context_blocks/quick_reference.md",
        RELIABILITY_PROMPT_TIER,
        "Quick fixes and common patterns",
    ),
    PromptSpec(
        "context_blocks/xorq_connection",
        "context_blocks/xorq_connection.md",
        RELIABILITY_PROMPT_TIER,
        "Connection troubleshooting",
    ),
    PromptSpec(
        "context_blocks/post_aggregation_column_access",
        "context_blocks/post_aggregation_column_access.md",
        RELIABILITY_PROMPT_TIER,
        "Column access after aggregations",
    ),
    # Advanced prompts
    PromptSpec(
        "llm_object_guide",
        "llm_object_guide.md",
        ADVANCED_PROMPT_TIER,
        "LLM object usage guide",
    ),
    PromptSpec(
        "package_exploration_guide",
        "package_exploration_guide.md",
        ADVANCED_PROMPT_TIER,
        "Package exploration workflow",
    ),
    PromptSpec(
        "context_blocks/xorq_ml_core",
        "context_blocks/xorq_ml_core.md",
        ADVANCED_PROMPT_TIER,
        "Core ML workflow and critical rules",
    ),
    PromptSpec(
        "context_blocks/transformation_patterns",
        "context_blocks/transformation_patterns.md",
        ADVANCED_PROMPT_TIER,
        "Common transformation templates",
    ),
    PromptSpec(
        "context_blocks/expression_deliverables",
        "context_blocks/expression_deliverables.md",
        ADVANCED_PROMPT_TIER,
        "Deliverable expectations",
    ),
    PromptSpec(
        "context_blocks/phase_checks",
        "context_blocks/phase_checks.md",
        ADVANCED_PROMPT_TIER,
        "Phase completion checks",
    ),
    PromptSpec(
        "context_blocks/repl_exploration",
        "context_blocks/repl_exploration.md",
        ADVANCED_PROMPT_TIER,
        "REPL exploration techniques",
    ),
)

PROMPT_INDEX = {spec.name: spec for spec in PROMPT_SPECS}


def list_prompt_names() -> tuple[str, ...]:
    return tuple(PROMPT_INDEX)


def iter_prompt_specs(tier: str | None = None) -> Iterator[PromptSpec]:
    for spec in PROMPT_SPECS:
        if tier and spec.tier != tier:
            continue
        yield spec


def get_prompt_spec(name: str) -> PromptSpec:
    try:
        return PROMPT_INDEX[name]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt: {name}") from exc


def load_prompt_text(name: str | PromptSpec) -> str:
    spec = name if isinstance(name, PromptSpec) else get_prompt_spec(name)
    return spec.path.read_text(encoding="utf-8")


def prompt_title(text: str, default: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        return stripped.lstrip("# ").strip()
    return default


def prompt_excerpt(text: str, max_lines: int = 80) -> str:
    lines = text.strip().splitlines()
    if max_lines and len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + "\n..."
    return "\n".join(lines)
