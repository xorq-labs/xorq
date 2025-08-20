from __future__ import annotations

from typing import Any, Callable, Iterable

from xorq.vendor.ibis.common.collections import FrozenDict, FrozenOrderedDict
from xorq.vendor.ibis.expr.operations.relations import Relation
from xorq.vendor.ibis.expr.schema import Schema


# Notes on design:
# - .values must map column name -> Value ops that reference *parent* relations.
# - .schema must come from those values' dtypes, so Field(dtype) can resolve from rel.schema.
# - We store user lambdas (dimensions/measures) and only evaluate them against the base ibis table
#   when computing .values / .schema. This preserves immutability and keeps IR nodes hashable.


class SemanticModel(Relation):
    """Wrap a base Ibis table with semantic definitions (dimensions + measures)."""

    table: Any  # Relation | ir.Table is fine; Relation.__coerce__ will handle Expr
    dimensions: Any  # FrozenDict[str, Callable[[ir.Table], ir.Value]]
    measures: Any  # FrozenDict[str, Callable[[ir.Table], ir.Value]]

    def __init__(
        self,
        table: Any,
        dimensions: dict[str, Callable] | None = None,
        measures: dict[str, Callable] | None = None,
    ) -> None:
        # Ensure immutable collections for hashability
        dims = FrozenDict(dimensions or {})
        meas = FrozenDict(measures or {})
        # Coerce table/expr into a Relation op
        base_rel = Relation.__coerce__(table.op() if hasattr(table, "op") else table)
        super().__init__(table=base_rel, dimensions=dims, measures=meas)

    @property
    def values(self) -> FrozenOrderedDict[str, Any]:
        """Expose semantic fields as expressions referencing the base relation."""
        # Use the base Table expr so lambdas return ir.Value; then capture .op()
        base_tbl = self.base.to_expr()
        out: dict[str, Any] = {}
        # Dimensions
        for name, fn in self.dimensions.items():
            expr = fn(base_tbl)
            out[name] = expr.op()
        # Measures (may be aggregations; this is fine as a "value definition")
        for name, fn in self.measures.items():
            expr = fn(base_tbl)
            out[name] = expr.op()
        return FrozenOrderedDict(out)

    @property
    def schema(self) -> Schema:
        return Schema({name: v.dtype for name, v in self.values.items()})


class SemanticFilter(Relation):
    """Semantic-level filter; predicate can reference base columns and named dimensions."""

    source: Any
    predicate: Callable

    def __init__(self, source: Any, predicate: Callable) -> None:
        super().__init__(source=Relation.__coerce__(source), predicate=predicate)

    @property
    def values(self) -> FrozenOrderedDict[str, Any]:
        return self.source.values

    @property
    def schema(self) -> Schema:
        return self.source.schema


class SemanticProject(Relation):
    """Semantic-level projection: choose named dimensions/measures."""

    source: Any
    fields: tuple[str, ...]

    def __init__(self, source: Any, fields: Iterable[str]) -> None:
        super().__init__(source=Relation.__coerce__(source), fields=tuple(fields))

    @property
    def values(self) -> FrozenOrderedDict[str, Any]:
        src_vals = self.source.values
        # Keep only requested names that exist in the semantic model
        return FrozenOrderedDict(
            {k: v for k, v in src_vals.items() if k in self.fields}
        )

    @property
    def schema(self) -> Schema:
        return Schema({k: v.dtype for k, v in self.values.items()})


class SemanticGroupBy(Relation):
    """Marker relation carrying grouping keys by name."""

    source: Any
    keys: tuple[str, ...]

    def __init__(self, source: Any, keys: Iterable[str]) -> None:
        super().__init__(source=Relation.__coerce__(source), keys=tuple(keys))

    @property
    def values(self) -> FrozenOrderedDict[str, Any]:
        # No shape change until aggregation; hand through upstream values.
        return self.source.values

    @property
    def schema(self) -> Schema:
        return self.source.schema


class SemanticAggregate(Relation):
    """Aggregation over grouping keys + measure lambdas."""

    source: Any
    keys: tuple[str, ...]
    aggs: Any  # FrozenDict[str, Callable]

    def __init__(
        self,
        source: Any,
        keys: Iterable[str],
        aggs: dict[str, Callable] | None,
    ) -> None:
        frozen_aggs = FrozenDict(aggs or {})
        super().__init__(source=Relation.__coerce__(source), keys=tuple(keys), aggs=frozen_aggs)

    @property
    def values(self) -> FrozenOrderedDict[str, Any]:
        # Conceptually, this defines output columns; use the *base* table to define dtypes.
        # The actual grouping/aggregation happens in lowering.
        # We include both group keys and measure outputs.
        # Resolve group keys by name from underlying SemanticModel dims or fallback to base columns.
        root = _find_root_model(self.source)
        base_tbl = root.table.to_expr() if root else self.source.to_expr()
        vals: dict[str, Any] = {}
        # Group keys
        for k in self.keys:
            if root and k in root.dimensions:
                vals[k] = root.dimensions[k](base_tbl).op()
            else:
                vals[k] = base_tbl[k].op()
        # Measures
        for name, fn in self.aggs.items():
            vals[name] = fn(base_tbl).op()
        return FrozenOrderedDict(vals)

    @property
    def schema(self) -> Schema:
        return Schema({n: v.dtype for n, v in self.values.items()})


class SemanticMutate(Relation):
    """Post-aggregation mutations (expressions over aggregated outputs)."""

    source: Any
    post: Any  # FrozenDict[str, Callable]

    def __init__(self, source: Any, post: dict[str, Callable] | None) -> None:
        frozen_post = FrozenDict(post or {})
        super().__init__(source=Relation.__coerce__(source), post=frozen_post)

    @property
    def values(self) -> FrozenOrderedDict[str, Any]:
        # Mutations define new columns but don't affect upstream dereference rules.
        return self.source.values

    @property
    def schema(self) -> Schema:
        # Actual dtypes materialize after lowering; preserving upstream schema here is fine.
        return self.source.schema


def _find_root_model(node: Any) -> SemanticModel | None:
    """Walk upstream until we find the first SemanticModel."""
    cur = node
    while cur is not None:
        if isinstance(cur, SemanticModel):
            return cur
        parent = getattr(cur, "source", None)
        cur = parent
    return None
