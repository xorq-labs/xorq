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
        return tuple(self.model.dimensions)

    @property
    def available_measures(self) -> tuple[str, ...]:
        return tuple(self.model.measures)

    def build_expr(
        self, *, dimensions: tuple[str, ...], measures: tuple[str, ...]
    ) -> Expr:
        """Produce an expression by querying the semantic model."""
        return self.model.query(dimensions=dimensions, measures=measures)

    @classmethod
    def from_tagged(cls, tag_node) -> SemanticModelSpec:
        """Recover SemanticModel from a BSL Tag on an expression."""
        from boring_semantic_layer import from_tagged  # noqa: PLC0415

        model = from_tagged(tag_node.to_expr())
        return cls(model=model)

    @classmethod
    def from_build_dir(cls, path: Path) -> SemanticModelSpec:
        """Reconstruct from a catalog build directory.

        Reads the serialized expression and recovers the SemanticModel.
        Resolves database_table Read nodes to parquet files in the build dir
        before BSL recovery so the model has live data.
        """
        from boring_semantic_layer import from_tagged  # noqa: PLC0415

        from xorq.common.utils.graph_utils import replace_nodes, walk_nodes  # noqa: PLC0415
        from xorq.expr.relations import Read, Tag  # noqa: PLC0415
        from xorq.ibis_yaml.compiler import load_expr  # noqa: PLC0415
        from xorq.ibis_yaml.enums import MemtableTypes  # noqa: PLC0415
        from xorq.vendor import ibis  # noqa: PLC0415

        path = Path(path)
        if not (path / "expr.yaml").exists():
            raise ValueError(f"No expr.yaml found in {path}")
        expr = load_expr(path)

        # load_expr handles memtables but not database_tables — resolve those
        # by reading the archived parquet files back into memtables so the
        # recovered BSL model has live data
        db_reads = tuple(
            dr
            for dr in walk_nodes(Read, expr)
            if "database_tables/" in str(dict(dr.read_kwargs).get("path", ""))
        )
        if db_reads:
            import pyarrow.parquet as pq  # noqa: PLC0415

            def _resolve(dr):
                read_kwargs = dict(dr.read_kwargs)
                original_path = Path(read_kwargs["path"])
                # resolve relative to our build dir: use database_tables/<filename>
                parquet_path = path / "database_tables" / original_path.name
                df = pq.read_table(parquet_path).to_pandas()
                return ibis.memtable(df, schema=dr.schema, name=dr.name).op()

            replacements = {dr: _resolve(dr) for dr in db_reads}

            def _replacer(node, kwargs):
                match node:
                    case _ if node in replacements:
                        return replacements[node]
                    case _:
                        return node.__recreate__(kwargs) if kwargs else node

            expr = replace_nodes(_replacer, expr).to_expr()

        # Recover BSL model from the (now data-resolved) expression
        bsl_tags = tuple(
            t for t in walk_nodes(Tag, expr) if t.metadata.get("tag") == "bsl"
        )
        if not bsl_tags:
            raise ValueError("No BSL tags found in expression tree")
        model = from_tagged(bsl_tags[0].to_expr())
        return cls(model=model)

    @classmethod
    def _from_expr(cls, expr):
        """Recover a SemanticModelSpec from an expression containing BSL tags."""
        from boring_semantic_layer import from_tagged  # noqa: PLC0415

        from xorq.common.utils.graph_utils import walk_nodes  # noqa: PLC0415
        from xorq.expr.relations import Tag  # noqa: PLC0415

        bsl_tags = tuple(
            t for t in walk_nodes(Tag, expr) if t.metadata.get("tag") == "bsl"
        )
        if not bsl_tags:
            raise ValueError("No BSL tags found in expression tree")
        model = from_tagged(bsl_tags[0].to_expr())
        return cls(model=model)

    def rebind(self, table) -> SemanticModelSpec:
        """Return a new spec with the same dims/measures but a different base table."""
        from boring_semantic_layer.expr import SemanticModel  # noqa: PLC0415

        return SemanticModelSpec(
            model=SemanticModel(
                table=table,
                dimensions=self.model.get_dimensions(),
                measures=self.model.get_measures(),
                calc_measures=self.model.get_calculated_measures(),
                name=self.model.name,
                description=self.model.description,
            )
        )

    def to_build_dir(self, path: Path) -> Path:
        """Write the semantic model under ``path/<expr_hash>/``.

        Returns the hash-named subdirectory containing the build artifacts.
        """
        from xorq.ibis_yaml.compiler import build_expr  # noqa: PLC0415

        path = Path(path)
        tagged_expr = self.model.to_tagged()
        # build_expr writes to path/<hash>/ and returns that path
        expr_build_path = build_expr(tagged_expr, builds_dir=path)

        meta = {
            "type": str(BuilderKind.SemanticModel),
            "description": (
                f"{len(self.available_dimensions)} dims, "
                f"{len(self.available_measures)} measures"
            ),
            "available_dimensions": self.available_dimensions,
            "available_measures": self.available_measures,
        }
        (expr_build_path / BUILDER_META_FILENAME).write_text(
            json.dumps(meta, indent=2)
        )
        return expr_build_path
