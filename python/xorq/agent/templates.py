from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

from xorq.init_templates import InitTemplates


@dataclass(frozen=True)
class AgentTemplate:
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
            \"\"\"Template scaffold: {self.name}
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
            print(table.schema())  # ALWAYS check schema first

            expr = (
                table
                # TODO: customize filters/aggregations
            )
            """
            ).strip()
            + "\n"
        )


TEMPLATES: tuple[AgentTemplate, ...] = (
    AgentTemplate(
        "cached_fetcher",
        "Start from the cached-fetcher template to hydrate upstream tables and cache results.",
        InitTemplates.cached_fetcher,
        (),
        catalog_hint="cached-fetcher-base",
        default_table="SOURCE_TABLE",
    ),
    AgentTemplate(
        "sklearn_pipeline",
        "Deferred sklearn pipeline with train/predict separation.",
        InitTemplates.sklearn,
        (),
        catalog_hint="sklearn-pipeline",
        default_table="TRAINING_DATA",
    ),
    AgentTemplate(
        "penguins_demo",
        "Minimal multi-engine example (penguins template) for demonstrations.",
        InitTemplates.penguins,
        (),
        catalog_hint="penguins-agg",
        default_table="PENGUINS",
    ),
)

TEMPLATE_INDEX = {template.name: template for template in TEMPLATES}


def iter_templates():
    return iter(TEMPLATES)


def list_template_names() -> tuple[str, ...]:
    return tuple(TEMPLATE_INDEX)


def get_template(name: str) -> AgentTemplate:
    try:
        return TEMPLATE_INDEX[name]
    except KeyError as exc:
        raise ValueError(f"Unknown template: {name}") from exc


def scaffold_template(
    template: AgentTemplate, dest: Path, overwrite: bool = False
) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        raise FileExistsError(f"{dest} already exists (use --overwrite to replace)")
    dest.write_text(template.default_script(), encoding="utf-8")
    return dest
