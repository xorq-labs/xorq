from __future__ import annotations

from typing import Callable

from xorq.semantic.ops import (
    SemanticAggregate,
    SemanticFilter,
    SemanticGroupBy,
    SemanticModel,
    SemanticMutate,
    SemanticProject,
)
from xorq.vendor.ibis.expr.api import Table as IbisTable


def to_semantic_table(table: IbisTable) -> IbisTable:
    """Initialize an empty SemanticModel over an Ibis table."""
    return SemanticModel(table=table, dimensions={}, measures={}).to_expr()


def with_dimensions(table: IbisTable, **dimensions: Callable) -> IbisTable:
    """Attach or extend dimension lambdas (name -> fn(table) -> column)."""
    node = table.op()
    if not isinstance(node, SemanticModel):
        node = SemanticModel(table=table, dimensions={}, measures={})
    new_dims = {**getattr(node, "dimensions", {}), **dimensions}
    return SemanticModel(
        table=node.table.to_expr(),
        dimensions=new_dims,
        measures=getattr(node, "measures", {}),
    ).to_expr()


def with_measures(table: IbisTable, **measures: Callable) -> IbisTable:
    """Attach or extend measure lambdas (name -> fn(table) -> scalar/agg)."""
    node = table.op()
    if not isinstance(node, SemanticModel):
        node = SemanticModel(table=table, dimensions={}, measures={})
    new_meas = {**getattr(node, "measures", {}), **measures}
    return SemanticModel(
        table=node.table.to_expr(),
        dimensions=getattr(node, "dimensions", {}),
        measures=new_meas,
    ).to_expr()


def where_(table: IbisTable, predicate: Callable) -> IbisTable:
    """Add a semantic filter node to the AST."""
    return SemanticFilter(source=table.op(), predicate=predicate).to_expr()


def select_(table: IbisTable, *fields: str) -> IbisTable:
    """Add a semantic projection of named dimensions/measures."""
    return SemanticProject(source=table.op(), fields=fields).to_expr()


def group_by_(table: IbisTable, *keys: str) -> IbisTable:
    """Add a semantic GROUP BY marker."""
    return SemanticGroupBy(source=table.op(), keys=keys).to_expr()


def aggregate_(table: IbisTable, **measures: Callable) -> IbisTable:
    """Add a semantic AGGREGATE node."""
    node = table.op()
    keys = getattr(node, "keys", ())  # inherit keys if called after group_by_
    return SemanticAggregate(source=node, keys=keys, aggs=measures).to_expr()


def mutate_(table: IbisTable, **post_aggs: Callable) -> IbisTable:
    """Add a post-aggregation semantic MUTATE node."""
    return SemanticMutate(source=table.op(), post=post_aggs).to_expr()
