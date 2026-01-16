from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

from xorq.init_templates import InitTemplates


@dataclass(frozen=True)
class AgentSkill:
    name: str
    description: str
    template: InitTemplates
    prompts: tuple[str, ...]
    catalog_hint: str
    default_table: str

    def default_script(self) -> str:
        prompt_lines = "\n".join(f"# - {prompt}" for prompt in self.prompts)
        return (
            dedent(
                f"""\
            \"\"\"Skill scaffold: {self.name}
            {self.description}

            Referenced prompts:
            {prompt_lines}

            After editing, build with `xorq build {self.name}.py -e expr`
            and catalog via `xorq catalog add builds/<hash> --alias {self.catalog_hint}`.
            \"\"\"

            import xorq.api as xo
            from xorq.api import _
            from xorq.vendor import ibis


            con = xo.connect()
            table = con.table("{self.default_table}")
            print(table.schema())  # align with context_blocks/must_check_schema

            expr = (
                table
                # TODO: customize filters/aggregations
            )
            """
            ).strip()
            + "\n"
        )


SKILLS: tuple[AgentSkill, ...] = (
    AgentSkill(
        "cached_fetcher",
        "Start from the cached-fetcher template to hydrate upstream tables and cache results.",
        InitTemplates.cached_fetcher,
        ("planning_phase", "context_blocks/xorq_core"),
        catalog_hint="cached-fetcher-base",
        default_table="SOURCE_TABLE",
    ),
    AgentSkill(
        "sklearn_pipeline",
        "Deferred sklearn pipeline with train/predict separation.",
        InitTemplates.sklearn,
        ("planning_phase", "context_blocks/xorq_ml_complete"),
        catalog_hint="sklearn-pipeline",
        default_table="TRAINING_DATA",
    ),
    AgentSkill(
        "penguins_demo",
        "Minimal multi-engine example (penguins template) for demonstrations.",
        InitTemplates.penguins,
        ("context_blocks/xorq_core", "context_blocks/expression_deliverables"),
        catalog_hint="penguins-agg",
        default_table="PENGUINS",
    ),
)

SKILL_INDEX = {skill.name: skill for skill in SKILLS}


def iter_skills():
    return iter(SKILLS)


def list_skill_names() -> tuple[str, ...]:
    return tuple(SKILL_INDEX)


def get_skill(name: str) -> AgentSkill:
    try:
        return SKILL_INDEX[name]
    except KeyError as exc:
        raise ValueError(f"Unknown skill: {name}") from exc


def scaffold_skill(skill: AgentSkill, dest: Path, overwrite: bool = False) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        raise FileExistsError(f"{dest} already exists (use --overwrite to replace)")
    dest.write_text(skill.default_script(), encoding="utf-8")
    return dest
