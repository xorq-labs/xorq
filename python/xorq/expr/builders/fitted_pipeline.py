"""FittedPipelineSpec — builder for fitted ML pipelines."""

from __future__ import annotations

import json
from pathlib import Path

import cloudpickle
from attr import field, frozen
from attr.validators import instance_of

from xorq.expr.builders import BUILDER_META_FILENAME, BuilderKind, BuilderSpec
from xorq.vendor.ibis import Expr


PIPELINE_PICKLE_FILENAME = "fitted_pipeline.pkl"


@frozen
class FittedPipelineSpec(BuilderSpec):
    """Builder that wraps a FittedPipeline.

    Produces expressions via predict, transform, predict_proba, etc.
    """

    tag_name: str = field(default=str(BuilderKind.FittedPipeline), init=False)
    fitted_pipeline: object = field(validator=instance_of(object))

    @property
    def steps(self) -> tuple[dict, ...]:
        return tuple(
            {"name": fs.step.name, "estimator": type(fs.step.instance).__name__}
            for fs in self.fitted_pipeline.fitted_steps
        )

    @property
    def is_predict(self) -> bool:
        return self.fitted_pipeline.is_predict

    def build_expr(self, *, data: Expr, method: str = "predict") -> Expr:
        """Produce an expression by applying the fitted pipeline to data."""
        fp = self.fitted_pipeline
        match method:
            case "predict":
                return fp.predict(data)
            case "transform":
                return fp.transform(data)
            case "predict_proba":
                return fp.predict_proba(data)
            case "decision_function":
                return fp.decision_function(data)
            case "feature_importances":
                return fp.feature_importances(data)
            case _:
                raise ValueError(
                    f"Unknown method {method!r}; expected one of "
                    f"predict, transform, predict_proba, decision_function, feature_importances"
                )

    @classmethod
    def from_tagged(cls, tag_node) -> dict:
        """Extract provenance description from ML tags on an expression.

        Returns a dict (not a full FittedPipelineSpec) because tags do not
        contain fitted weights — only pipeline structure metadata.
        """
        from xorq.expr.ml.enums import FittedPipelineTagKey  # noqa: PLC0415

        tag_key = tag_node.metadata.get("tag")
        steps_info = tuple(
            {"name": d["name"], "estimator": d["typ"].__name__}
            for step_items in tag_node.metadata.get(FittedPipelineTagKey.ALL_STEPS, ())
            for d in (dict(step_items),)
        )
        return {
            "type": str(BuilderKind.FittedPipeline),
            "description": f"{tag_key}, {len(steps_info)} steps",
            "is_predict": tag_key
            in (
                str(FittedPipelineTagKey.PREDICT),
                str(FittedPipelineTagKey.PREDICT_PROBA),
                str(FittedPipelineTagKey.DECISION_FUNCTION),
            ),
            "steps": steps_info,
        }

    @classmethod
    def from_build_dir(cls, path: Path) -> FittedPipelineSpec:
        """Reconstruct from a catalog build directory."""
        pkl_path = Path(path) / PIPELINE_PICKLE_FILENAME
        if not pkl_path.exists():
            raise ValueError(f"No {PIPELINE_PICKLE_FILENAME} found in {path}")
        with open(pkl_path, "rb") as f:
            fitted_pipeline = cloudpickle.load(f)  # noqa: S301
        return cls(fitted_pipeline=fitted_pipeline)

    def to_build_dir(self, path: Path) -> None:
        """Serialize the fitted pipeline to a build directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / PIPELINE_PICKLE_FILENAME, "wb") as f:
            cloudpickle.dump(self.fitted_pipeline, f)

        meta = {
            "type": str(BuilderKind.FittedPipeline),
            "description": " -> ".join(s["estimator"] for s in self.steps)
            + " (fitted)",
            "steps": self.steps,
            "is_predict": self.is_predict,
        }
        (path / BUILDER_META_FILENAME).write_text(json.dumps(meta, indent=2))
