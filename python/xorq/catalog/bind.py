from functools import reduce

from xorq.common.utils.graph_utils import replace_unbound
from xorq.expr.relations import CatalogSource, RemoteTable, gen_name
from xorq.vendor.ibis.expr.types.core import ExprMetadata


def _validate_schema(source_schema, transform_schema, source_name, transform_name):
    """Validate that source schema is a superset of transform's input schema."""
    missing = tuple(
        (col, typ) for col, typ in transform_schema.items() if col not in source_schema
    )
    type_mismatches = tuple(
        (col, source_schema[col], typ)
        for col, typ in transform_schema.items()
        if col in source_schema and source_schema[col] != typ
    )
    if not missing and not type_mismatches:
        return
    lines = (
        f"Schema mismatch between source {source_name!r} and transform {transform_name!r}:",
        *(
            (f"  missing columns: {', '.join(col for col, _ in missing)}",)
            if missing
            else ()
        ),
        *(
            f"  type mismatch: {col} (source: {src_t}, transform expects: {trn_t})"
            for col, src_t, trn_t in type_mismatches
        ),
    )
    raise ValueError("\n".join(lines))


def _resolve_alias(alias, entry):
    """Resolve the alias for a bound entry."""
    match alias:
        case str():
            return alias
        case None if aliases := getattr(entry, "aliases", ()):
            return aliases[0].alias
        case _:
            return None


def _resolve_source(source, con, alias):
    """Resolve source to (node, con)."""
    from xorq.catalog.catalog import CatalogEntry  # noqa: PLC0415
    from xorq.vendor.ibis.expr.types.core import Expr  # noqa: PLC0415

    match source:
        case CatalogEntry():
            resolved_con = con if con is not None else source.expr._find_backend()
            return (
                CatalogSource.from_entry(
                    source,
                    resolved_con,
                    alias=_resolve_alias(alias, source),
                ),
                resolved_con,
            )
        case Expr() if isinstance(source.op(), RemoteTable):
            resolved_con = con if con is not None else source._find_backend()
            return source.op(), resolved_con
        case Expr():
            resolved_con = con if con is not None else source._find_backend()
            return RemoteTable.from_expr(resolved_con, source), resolved_con
        case _:
            raise TypeError(
                f"source must be a CatalogEntry or Expr, got {type(source)}"
            )


def _bind_one(transform_entry, current_expr, con):
    """Bind a single transform entry onto current_expr."""
    transform_expr = transform_entry.expr
    transform_meta = ExprMetadata(transform_expr)

    if transform_meta._unbound_node is None:
        raise ValueError(
            f"{transform_entry.name!r} has no UnboundTable (kind: {transform_meta.kind}). "
            f"Only unbound_expr entries can be used as transforms."
        )

    _validate_schema(
        current_expr.as_table().schema(),
        transform_meta.schema_in,
        "(current)",
        transform_entry.name,
    )

    # CatalogSource is a RemoteTable subclass, so this arm catches both.
    source_node = (
        current_expr.op()
        if isinstance(current_expr.op(), RemoteTable)
        else RemoteTable.from_expr(con, current_expr)
    )

    composed_expr = replace_unbound(transform_expr, source_node)

    return CatalogSource(
        name=gen_name(),
        schema=composed_expr.as_table().schema(),
        source=con,
        remote_expr=composed_expr,
        catalog_name=getattr(transform_entry.catalog, "name", None),
        catalog_path=str(transform_entry.catalog.repo_path),
        entry_name=transform_entry.name,
        alias=_resolve_alias(None, transform_entry),
        kind=str(transform_entry.kind),
    ).to_expr()


def bind(source, *transforms, con=None, alias=None):
    """Bind a source through one or more unbound transform entries.

    Parameters
    ----------
    source : CatalogEntry or Expr
        The data source. CatalogEntry gets wrapped in CatalogSource.
    *transforms : CatalogEntry
        One or more catalog entries with UnboundTable, applied in order.
    con : Backend, optional
        Override the backend connection.
    alias : str, optional
        Override the source alias.
    """
    if not transforms:
        raise ValueError("At least one transform entry is required.")

    source_node, resolved_con = _resolve_source(source, con, alias)

    return reduce(
        lambda expr, t: _bind_one(t, expr, resolved_con),
        transforms,
        source_node.to_expr(),
    )
