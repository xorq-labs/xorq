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
        "xorq_vendor_ibis",
        "xorq_vendor_ibis.md",
        CORE_PROMPT_TIER,
        "Vendor ibis import and ordering rules",
    ),
    PromptSpec(
        "context_blocks/xorq_core",
        "context_blocks/xorq_core.md",
        CORE_PROMPT_TIER,
        "Deferred execution core rules",
    ),
    # Reliability prompts
    PromptSpec(
        "context_blocks/data_source_rules",
        "context_blocks/data_source_rules.md",
        RELIABILITY_PROMPT_TIER,
        "Source selection and policy guardrails",
    ),
    PromptSpec(
        "context_blocks/must_check_schema",
        "context_blocks/must_check_schema.md",
        RELIABILITY_PROMPT_TIER,
        "Schema inspection requirements",
    ),
    PromptSpec(
        "context_blocks/column_case_rules",
        "context_blocks/column_case_rules.md",
        RELIABILITY_PROMPT_TIER,
        "Column casing norms per backend",
    ),
    PromptSpec(
        "context_blocks/avoid_pandas",
        "context_blocks/avoid_pandas.md",
        RELIABILITY_PROMPT_TIER,
        "Why pandas usage must be contained",
    ),
    PromptSpec(
        "context_blocks/pandas_udf_patterns",
        "context_blocks/pandas_udf_patterns.md",
        RELIABILITY_PROMPT_TIER,
        "Deferred pandas UDF recipes",
    ),
    PromptSpec(
        "context_blocks/udaf_aggregation_patterns",
        "context_blocks/udaf_aggregation_patterns.md",
        RELIABILITY_PROMPT_TIER,
        "Aggregation/UDAF helper patterns",
    ),
    PromptSpec(
        "context_blocks/backend_operation_workarounds",
        "context_blocks/backend_operation_workarounds.md",
        RELIABILITY_PROMPT_TIER,
        "Workarounds for backend quirks",
    ),
    PromptSpec(
        "context_blocks/fix_schema_errors",
        "context_blocks/fix_schema_errors.md",
        RELIABILITY_PROMPT_TIER,
        "Schema mismatch fixes",
    ),
    PromptSpec(
        "context_blocks/fix_attribute_errors",
        "context_blocks/fix_attribute_errors.md",
        RELIABILITY_PROMPT_TIER,
        "Attribute error fixes",
    ),
    PromptSpec(
        "context_blocks/fix_data_errors",
        "context_blocks/fix_data_errors.md",
        RELIABILITY_PROMPT_TIER,
        "Data integrity error playbook",
    ),
    PromptSpec(
        "context_blocks/fix_import_errors",
        "context_blocks/fix_import_errors.md",
        RELIABILITY_PROMPT_TIER,
        "Import failure fixes",
    ),
    PromptSpec(
        "context_blocks/fix_udf_backend_errors",
        "context_blocks/fix_udf_backend_errors.md",
        RELIABILITY_PROMPT_TIER,
        "Backend-related UDF fixes",
    ),
    PromptSpec(
        "context_blocks/xorq_connection",
        "context_blocks/xorq_connection.md",
        RELIABILITY_PROMPT_TIER,
        "Connection troubleshooting",
    ),
    PromptSpec(
        "context_blocks/xorq_practical_patterns",
        "context_blocks/xorq_practical_patterns.md",
        RELIABILITY_PROMPT_TIER,
        "Field-tested xorq recipes",
    ),
    PromptSpec(
        "context_blocks/xorq_technical_patterns",
        "context_blocks/xorq_technical_patterns.md",
        RELIABILITY_PROMPT_TIER,
        "Technical guardrails and caveats",
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
        "context_blocks/xorq_ml_complete",
        "context_blocks/xorq_ml_complete.md",
        ADVANCED_PROMPT_TIER,
        "Complete ML reference",
    ),
    PromptSpec(
        "context_blocks/optimization_patterns",
        "context_blocks/optimization_patterns.md",
        ADVANCED_PROMPT_TIER,
        "Optimization/solver patterns",
    ),
    PromptSpec(
        "context_blocks/plotting_patterns",
        "context_blocks/plotting_patterns.md",
        ADVANCED_PROMPT_TIER,
        "Visualization reminders",
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
        "context_blocks/task_understanding",
        "context_blocks/task_understanding.md",
        ADVANCED_PROMPT_TIER,
        "Task comprehension prompts",
    ),
    PromptSpec(
        "context_blocks/repl_exploration",
        "context_blocks/repl_exploration.md",
        ADVANCED_PROMPT_TIER,
        "REPL exploration techniques",
    ),
    PromptSpec(
        "context_blocks/summary_patterns",
        "context_blocks/summary_patterns.md",
        ADVANCED_PROMPT_TIER,
        "Summary write-up patterns",
    ),
    PromptSpec(
        "context_blocks/phase_initialization_completion_check",
        "context_blocks/phase_initialization_completion_check.md",
        ADVANCED_PROMPT_TIER,
        "Initialization completion checklist",
    ),
    PromptSpec(
        "context_blocks/phase_data_preparation_completion_check",
        "context_blocks/phase_data_preparation_completion_check.md",
        ADVANCED_PROMPT_TIER,
        "Data preparation completion checklist",
    ),
    PromptSpec(
        "context_blocks/phase_data_transform_completion_check",
        "context_blocks/phase_data_transform_completion_check.md",
        ADVANCED_PROMPT_TIER,
        "Transformation completion checklist",
    ),
    PromptSpec(
        "context_blocks/phase_modeling_completion_check",
        "context_blocks/phase_modeling_completion_check.md",
        ADVANCED_PROMPT_TIER,
        "Modeling completion checklist",
    ),
    PromptSpec(
        "context_blocks/phase_communication_completion_check",
        "context_blocks/phase_communication_completion_check.md",
        ADVANCED_PROMPT_TIER,
        "Communication completion checklist",
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
