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
    example_file: str | None = None  # Path to example file relative to examples/

    def default_script(self) -> str:
        # If there's an example file, try to read it
        if self.example_file:
            try:
                import xorq

                xorq_root = Path(xorq.__file__).parent.parent.parent
                example_path = xorq_root / "examples" / self.example_file
                if example_path.exists():
                    return example_path.read_text()
            except Exception:
                pass  # Fall back to generated scaffold

        # Generate a basic scaffold if no example file
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
        "pipeline_example",
        "sklearn pipeline with StandardScaler + KNeighborsClassifier on iris dataset.",
        InitTemplates.sklearn,
        (),
        catalog_hint="sklearn-pipeline",
        default_table="iris",
        example_file="pipeline_example.py",
    ),
    AgentTemplate(
        "diamonds_price_prediction",
        "Feature engineering, train/test splits, and LinearRegression for diamond pricing.",
        InitTemplates.sklearn,
        (),
        catalog_hint="diamonds-price-model",
        default_table="DIAMONDS",
        example_file="diamonds_price_prediction.py",
    ),
    AgentTemplate(
        "sklearn_classifier_comparison",
        "Compare multiple sklearn classifiers (SVM, RandomForest, etc) on same dataset.",
        InitTemplates.sklearn,
        (),
        catalog_hint="classifier-comparison",
        default_table="TRAINING_DATA",
        example_file="sklearn_classifier_comparison.py",
    ),
    AgentTemplate(
        "deferred_fit_transform_predict",
        "Complete deferred ML workflow with fit, transform, and predict stages.",
        InitTemplates.sklearn,
        (),
        catalog_hint="deferred-ml-pipeline",
        default_table="TRAINING_DATA",
        example_file="deferred_fit_transform_predict_example.py",
    ),
    AgentTemplate(
        "penguins_demo",
        "Minimal multi-engine example using penguins dataset - good starting point.",
        InitTemplates.penguins,
        (),
        catalog_hint="penguins-agg",
        default_table="PENGUINS",
        example_file=None,  # Keep as basic scaffold
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
