from functools import reduce

from toolz import curry

from xorq.common.utils.graph_utils import replace_unbound
from xorq.expr.relations import CatalogSource, RemoteTable, gen_name
from xorq.vendor.ibis.expr.types.core import ExprMetadata


def _get_transform_schema_issues(source_schema, transform_schema):
    bads = {
        col: (source_typ, transform_typ)
        for col, transform_typ in transform_schema.items()
        if (source_typ := source_schema.get(col)) != transform_typ
    }
    missing = {
        col: transform_typ
        for col, (source_typ, transform_typ) in bads.items()
        if source_typ is None
    }
    mismatch = {
        col: (source_typ, transform_typ)
        for col, (source_typ, transform_typ) in bads.items()
        if source_typ is not None
    }
    return missing, mismatch


def _validate_schema(source_schema, transform_schema, source_name, transform_name):
    """Validate that source schema is a superset of transform's input schema."""
    missing, mismatch = _get_transform_schema_issues(source_schema, transform_schema)
    if missing or mismatch:
        errors = (
            *(f"  missing: {col}" for col in missing),
            *(
                f"  type mismatch: {col} (source: {source_typ}, transform: {transform_typ})"
                for (col, (source_typ, transform_typ)) in mismatch.items()
            ),
        )
        raise ValueError(
            "\n".join(
                (
                    f"Schema mismatch between source {source_name!r} and transform {transform_name!r}:",
                    *errors,
                )
            )
        )


def _resolve_alias(alias, entry):
    """Resolve the alias for a bound entry."""
    match alias:
        case str():
            return alias
        case None if aliases := getattr(entry, "aliases", ()):
            return aliases[0].alias
        case _:
            return None


def _resolve_con(source, con):
    """Return *con* when given, otherwise discover it from *source*."""
    return con if con is not None else source._find_backend()


def _ensure_remote(node, con, expr):
    """Return *node* as-is if already a RemoteTable, otherwise wrap *expr*."""
    return node if isinstance(node, RemoteTable) else RemoteTable.from_expr(con, expr)


def _resolve_source(source, con, alias):
    """Resolve *source* to a ``(node, backend)`` pair."""
    from xorq.catalog.catalog import CatalogEntry  # noqa: PLC0415
    from xorq.vendor.ibis.expr.types.core import Expr  # noqa: PLC0415

    match source:
        case CatalogEntry():
            resolved_con = _resolve_con(source.expr, con)
            node = CatalogSource.from_entry(
                source,
                resolved_con,
                alias=_resolve_alias(alias, source),
            )
            return node, resolved_con
        case Expr():
            resolved_con = _resolve_con(source, con)
            return _ensure_remote(source.op(), resolved_con, source), resolved_con
        case _:
            raise TypeError(
                f"source must be a CatalogEntry or Expr, got {type(source)}"
            )


@curry
def _bind_one(current_expr, transform_entry, con):
    """Bind a single transform entry onto *current_expr*."""
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

    source_node = _ensure_remote(current_expr.op(), con, current_expr)
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
        lambda expr, transform: _bind_one(expr, transform, resolved_con),
        transforms,
        source_node.to_expr(),
    )
