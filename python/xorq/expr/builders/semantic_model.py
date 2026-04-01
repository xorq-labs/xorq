"""SemanticModelSpec — builder for BSL semantic models.

TODO: This is a reference implementation in xorq. It will be moved to
boring_semantic_layer and registered via entry point in BSL's pyproject.toml.
Once BSL ships its own SemanticModelSpec, this module can be removed.
"""

from __future__ import annotations

import json
from pathlib import Path

from attr import field, frozen
from attr.validators import instance_of

from xorq.expr.builders import BUILDER_META_FILENAME, BuilderKind, BuilderSpec
from xorq.vendor.ibis import Expr


@frozen
class SemanticModelSpec(BuilderSpec):
    """Builder that wraps a BSL SemanticModel.

    Produces expressions by calling ``.query(dimensions=..., measures=...)``.
    """

    tag_name: str = field(default=str(BuilderKind.SemanticModel), init=False)
    model: object = field(validator=instance_of(object))

    @property
    def available_dimensions(self) -> tuple[str, ...]:
        return tuple(self.model.dimensions.keys())

    @property
    def available_measures(self) -> tuple[str, ...]:
        return tuple(self.model.measures.keys())

    def build_expr(
        self, *, dimensions: tuple[str, ...], measures: tuple[str, ...]
    ) -> Expr:
        """Produce an expression by querying the semantic model."""
        return self.model.query(dimensions=dimensions, measures=measures)

    @classmethod
    def from_tagged(cls, tag_node) -> SemanticModelSpec:
        """Recover SemanticModel from a BSL HashingTag on an expression."""
        from boring_semantic_layer import SemanticTableOp  # noqa: PLC0415

        from xorq.common.utils.graph_utils import walk_nodes  # noqa: PLC0415

        semantic_ops = walk_nodes(SemanticTableOp, tag_node)
        if not semantic_ops:
            raise ValueError("No SemanticTableOp found in expression tree")
        semantic_op = semantic_ops[0]
        model = semantic_op.to_expr()
        return cls(model=model)

    @classmethod
    def from_build_dir(cls, path: Path) -> SemanticModelSpec:
        """Reconstruct from a catalog build directory.

        Reads the serialized expression and recovers the SemanticModel.
        """
        from xorq.ibis_yaml.compiler import load_expr  # noqa: PLC0415

        expr_path = Path(path) / "expr.yaml"
        if not expr_path.exists():
            raise ValueError(f"No expr.yaml found in {path}")
        expr = load_expr(expr_path)
        return cls._from_expr(expr)

    @classmethod
    def _from_expr(cls, expr):
        """Recover a SemanticModelSpec from an expression containing BSL tags."""
        from boring_semantic_layer import SemanticTableOp  # noqa: PLC0415

        from xorq.common.utils.graph_utils import walk_nodes  # noqa: PLC0415

        semantic_ops = walk_nodes(SemanticTableOp, expr)
        if not semantic_ops:
            raise ValueError("No SemanticTableOp found in expression tree")
        model = semantic_ops[0].to_expr()
        return cls(model=model)

    def to_build_dir(self, path: Path) -> None:
        """Write the semantic model to a build directory."""
        from xorq.ibis_yaml.compiler import build_expr  # noqa: PLC0415

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Write the base table expression (with BSL tags) as expr.yaml
        base_expr = self.model.to_expr() if hasattr(self.model, "to_expr") else None
        if base_expr is not None:
            build_expr(base_expr, build_dir=path)

        meta = {
            "type": str(BuilderKind.SemanticModel),
            "description": (
                f"{len(self.available_dimensions)} dims, "
                f"{len(self.available_measures)} measures"
            ),
            "available_dimensions": self.available_dimensions,
            "available_measures": self.available_measures,
        }
        (path / BUILDER_META_FILENAME).write_text(json.dumps(meta, indent=2))
