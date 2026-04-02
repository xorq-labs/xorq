"""FittedPipelineBuilder — builder for fitted ML pipelines."""

from __future__ import annotations

import json
from pathlib import Path

import cloudpickle
from attr import field, frozen
from attr.validators import instance_of

from xorq.expr.builders import BUILDER_META_FILENAME, BuilderKind, Builder
from xorq.vendor.ibis import Expr


PIPELINE_PICKLE_FILENAME = "fitted_pipeline.pkl"


@frozen
class FittedPipelineBuilder(Builder):
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
        """Select a method and yield an expression on *data*.

        The builder is a selector — pick predict, transform, predict_proba,
        etc.  Training data is fixed in the fit subgraph; *data* is the
        inference input.
        """
        fp = self.fitted_pipeline
        fn = getattr(fp, method, None)
        if fn is None:
            raise ValueError(f"FittedPipeline has no method {method!r}")
        return fn(data)

    @classmethod
    def from_tagged(cls, tag_node) -> dict:
        """Extract provenance description from ML tags on an expression.

        Returns a dict (not a full FittedPipelineBuilder) because tags do not
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
    def from_build_dir(cls, path: Path) -> FittedPipelineBuilder:
        """Reconstruct from a catalog build directory.

        Model weights are preserved via the cached ``model`` property on each
        ``FittedStep``, which was eagerly populated during ``to_build_dir``.
        """
        pkl_path = Path(path) / PIPELINE_PICKLE_FILENAME
        if not pkl_path.exists():
            raise ValueError(f"No {PIPELINE_PICKLE_FILENAME} found in {path}")
        with open(pkl_path, "rb") as f:
            fitted_pipeline = cloudpickle.load(f)  # noqa: S301
        return cls(fitted_pipeline=fitted_pipeline)

    def rebind(self, train_data) -> FittedPipelineBuilder:
        """Return a new builder with training data references rebound.

        Replaces stale ``DatabaseTable`` nodes in the fitted pipeline's
        expression trees with the provided live *train_data* expression,
        so deferred fit UDFs can execute against the current backend.
        """
        import attr  # noqa: PLC0415

        from xorq.common.utils.graph_utils import replace_nodes, walk_nodes  # noqa: PLC0415
        from xorq.vendor.ibis.expr.operations import DatabaseTable  # noqa: PLC0415

        fp = self.fitted_pipeline
        db_tables = tuple(walk_nodes(DatabaseTable, fp.expr))
        if not db_tables:
            return self

        new_op = train_data.op()

        def _replacer(node, kwargs):
            if isinstance(node, DatabaseTable) and node.name == db_tables[0].name:
                return new_op
            return node.__recreate__(kwargs) if kwargs else node

        new_steps = tuple(
            attr.evolve(fs, expr=replace_nodes(_replacer, fs.expr).to_expr())
            for fs in fp.fitted_steps
        )
        new_expr = replace_nodes(_replacer, fp.expr).to_expr()
        new_fp = attr.evolve(fp, fitted_steps=new_steps, expr=new_expr)
        return FittedPipelineBuilder(fitted_pipeline=new_fp)

    def to_build_dir(self, path: Path) -> Path:
        """Serialize the fitted pipeline to a build directory.

        Eagerly materializes fitted model weights for each step so they
        survive cloudpickle roundtrip without needing training data.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Populate each step's cached model property before pickling
        tuple(fs.model for fs in self.fitted_pipeline.fitted_steps)

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
        return path
