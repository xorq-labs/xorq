from enum import StrEnum
from functools import reduce

from xorq.common.utils.graph_utils import replace_unbound
from xorq.expr.relations import RemoteTable, gen_name
from xorq.ibis_yaml.enums import ExprKind


class CatalogTag(StrEnum):
    SOURCE = "catalog-source"
    TRANSFORM = "catalog-transform"
    CODE = "catalog-code"


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


def _validate_chain(source_schema, transforms):
    """Pre-validate the full transform chain before building expressions.

    Checks that every transform has an UnboundTable and that schemas are
    compatible through the chain: source → transform[0] → transform[1] → …
    """
    from xorq.catalog.catalog import CatalogEntry  # noqa: PLC0415

    metas = []
    for i, entry in enumerate(transforms):
        if not isinstance(entry, CatalogEntry):
            raise TypeError(
                f"transforms[{i}] must be a CatalogEntry, got {type(entry)}"
            )

        meta = entry.expr.ls.metadata
        if meta.kind != ExprKind.UnboundExpr:
            raise ValueError(
                f"transforms[{i}] ({entry.name!r}) has no UnboundTable "
                f"(kind: {meta.kind}). Only unbound_expr entries can be used as transforms."
            )
        metas.append((entry.name, meta))

    current_schema = source_schema
    for name, meta in metas:
        _validate_schema(current_schema, meta.schema_in, "(current)", name)
        current_schema = meta.schema_out

    return tuple(metas)


def _ensure_remote(node, con, expr):
    """Return *node* as-is if already a RemoteTable, otherwise wrap *expr*."""
    return node if isinstance(node, RemoteTable) else RemoteTable.from_expr(con, expr)


def _make_source_tag(expr, entry, alias):
    """Wrap *expr* in a HashingTag recording the catalog source provenance."""
    resolved_alias = (
        alias
        if isinstance(alias, str)
        else next((a.alias for a in getattr(entry, "aliases", ())), None)
    )
    return expr.hashing_tag(
        CatalogTag.SOURCE,
        entry_name=entry.name,
        alias=resolved_alias,
        kind=str(entry.kind),
    )


def _resolve_source(source, con, alias):
    """Resolve *source* to a ``(tagged_expr, backend)`` pair."""
    from xorq.catalog.catalog import CatalogEntry  # noqa: PLC0415
    from xorq.vendor.ibis.expr.types.core import Expr  # noqa: PLC0415

    match source:
        case CatalogEntry():
            resolved_con = con if con is not None else source.expr._find_backend()
            node = RemoteTable.from_expr(resolved_con, source.expr)
            tagged = _make_source_tag(node.to_expr(), source, alias)
            return tagged, resolved_con
        case Expr():
            resolved_con = con if con is not None else source._find_backend()
            node = _ensure_remote(source.op(), resolved_con, source)
            return node.to_expr(), resolved_con
        case _:
            raise TypeError(
                f"source must be a CatalogEntry or Expr, got {type(source)}"
            )


def _bind_one(current_expr, transform_entry, con):
    """Bind a single transform entry onto *current_expr*, tagging the result."""
    transform_expr = transform_entry.expr
    source_node = _ensure_remote(current_expr.op(), con, current_expr)
    composed_expr = replace_unbound(transform_expr, source_node)

    result = RemoteTable(
        name=gen_name(),
        schema=composed_expr.as_table().schema(),
        source=con,
        remote_expr=composed_expr,
    ).to_expr()

    return result.hashing_tag(
        CatalogTag.TRANSFORM,
        entry_name=transform_entry.name,
        kind=str(transform_entry.kind),
    )


def _validate_one_catalog(source, transforms):
    """Assert all CatalogEntry arguments belong to the same catalog."""
    from xorq.catalog.catalog import CatalogEntry  # noqa: PLC0415

    catalog, *others = tuple(
        {
            entry.catalog
            for entry in (source, *transforms)
            if isinstance(entry, CatalogEntry)
        }
    )
    if others:
        repo_paths = (catalog.repo_path, *(other.repo_path for other in others))
        raise ValueError(f"Got multiple catalogs: {', '.join(map(str, repo_paths))}")


def bind(source, *transforms, con=None, alias=None):
    """Bind a source through one or more unbound transform entries.

    Parameters
    ----------
    source : CatalogEntry or Expr
        The data source. CatalogEntry is wrapped in RemoteTable + HashingTag.
    *transforms : CatalogEntry
        One or more catalog entries with UnboundTable, applied in order.
        Each step is tagged with a ``HashingTag(CatalogTag.TRANSFORM)``.
    con : Backend, optional
        Override the backend connection.
    alias : str, optional
        Override the source alias.
    """
    if not transforms:
        raise ValueError("At least one transform entry is required.")

    _validate_one_catalog(source, transforms)

    source_expr, resolved_con = _resolve_source(source, con, alias)
    source_schema = source_expr.as_table().schema()
    _validate_chain(source_schema, transforms)

    return reduce(
        lambda expr, transform: _bind_one(expr, transform, resolved_con),
        transforms,
        source_expr,
    )


def _eval_code(code, source):
    """Evaluate inline Ibis code with a restricted namespace.

    Only xorq, vendored ibis, and the bound ``source`` expression are
    available.  The expression is AST-whitelisted to prevent object
    introspection escapes.
    """
    import xorq.api as xo  # noqa: PLC0415
    from xorq.common.utils.eval_utils import safe_eval  # noqa: PLC0415
    from xorq.vendor import ibis  # noqa: PLC0415

    namespace = {"__builtins__": {}, "xo": xo, "ibis": ibis, "source": source}
    return safe_eval(code, namespace)


def _make_source_expr(source, con=None, alias=None):
    """Wrap a CatalogEntry as a RemoteTable + HashingTag without transforms."""
    source_expr, _ = _resolve_source(source, con, alias)
    return source_expr
