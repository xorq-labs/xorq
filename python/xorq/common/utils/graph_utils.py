from collections import defaultdict
from typing import Any, OrderedDict, Tuple

import xorq.expr.relations as rel
import xorq.expr.udf as udf
import xorq.vendor.ibis.expr.operations as ops
from xorq.expr.operations import NamedScalarParameter
from xorq.vendor.ibis import Expr
from xorq.vendor.ibis.expr.operations.core import Node


opaque_ops = (
    rel.Read,
    rel.CachedNode,
    rel.RemoteTable,
    rel.FlightUDXF,
    rel.FlightExpr,
    udf.ExprScalarUDF,
)


def to_node(maybe_expr: Any) -> Node:
    match maybe_expr:
        case Node():
            return maybe_expr
        case Expr():
            return maybe_expr.op()
        case _:
            raise ValueError(f"Don't know how to handle type {type(maybe_expr)}")


def gen_children_of(node: Node) -> Tuple[Node, ...]:
    match node:
        case ops.Field():
            rel_node = node.rel
            gen = () if rel_node is None else (to_node(rel_node),)

        case rel.RemoteTable():
            gen = (to_node(node.remote_expr),)

        case rel.CachedNode():
            gen = (to_node(node.parent),)

        case rel.FlightExpr() | rel.FlightUDXF():
            gen = (to_node(node.input_expr),)

        case udf.ExprScalarUDF():
            gen = (to_node(node.computed_kwargs_expr),)

        case rel.Read():
            gen = ()
        case _:
            raw_children = getattr(node, "__children__", ())
            gen = map(to_node, raw_children)
    yield from filter(None, gen)


def bfs(node):
    from collections import deque  # noqa: PLC0415

    from xorq.vendor.ibis.common.graph import Graph  # noqa: PLC0415

    queue = deque((to_node(node),))
    dct = {}
    while queue:
        if (node := queue.popleft()) not in dct:
            children = tuple(gen_children_of(node))
            dct[node] = children
            queue.extend(children)
    return Graph(dct)


def walk_nodes(node_types, expr):
    # TODO should this function use an ordered set
    visited = set()
    to_visit = [to_node(expr)]
    result = ()

    while to_visit:
        node = to_visit.pop()
        if node in visited:
            continue
        visited.add(node)
        if isinstance(node, node_types):
            result += (node,)

        to_visit += (
            child
            for child in OrderedDict.fromkeys(gen_children_of(node))
            if child not in visited
        )

    return result


def replace_nodes(replacer, expr):
    # Cache results of opaque sub-expression traversals by their root node.
    # Sub-expression roots are often shared across multiple opaque nodes (e.g.
    # each pipeline step's ExprScalarUDF references accumulated sub-expressions
    # that overlap heavily), so without this memo each shared root gets
    # re-traversed once per reference — O(n²) for a depth-n pipeline.
    sub_expr_memo = {}

    def do_recreate(op, _kwargs, **kwargs):
        kwargs = dict(zip(op.__argnames__, op.__args__)) | (_kwargs or {}) | kwargs
        return op.__recreate__(kwargs)

    def _replace_sub(sub_op):
        if sub_op not in sub_expr_memo:
            sub_expr_memo[sub_op] = sub_op.replace(process_node).to_expr()
        return sub_expr_memo[sub_op]

    def process_node(op, _kwargs):
        op = replacer(op, _kwargs)
        match op:
            case rel.RemoteTable():
                remote_expr = _replace_sub(op.remote_expr.op())
                return do_recreate(op, _kwargs, remote_expr=remote_expr)
            case rel.CachedNode():
                parent = _replace_sub(op.parent.op())
                return do_recreate(op, _kwargs, parent=parent)
            case rel.FlightExpr() | rel.FlightUDXF():
                input_expr = _replace_sub(op.input_expr.op())
                return do_recreate(op, _kwargs, input_expr=input_expr)
            case udf.ExprScalarUDF():
                computed_kwargs_expr = _replace_sub(op.computed_kwargs_expr.op())
                with_cke = op.with_computed_kwargs_expr(computed_kwargs_expr)
                return do_recreate(with_cke, _kwargs)
            case rel.Read():
                return op
            case _:
                if isinstance(op, opaque_ops):
                    raise ValueError(f"unhandled opaque op {type(op)}")
                return op

    initial_op = expr.op() if hasattr(expr, "op") else expr
    op = initial_op.replace(process_node)
    return op


def replace_sources(source_mapping, expr, *, transfer_tables=False):
    """Rewrite an expression graph, replacing backend sources.

    Every node that carries a ``source`` attribute (DatabaseTable, Read,
    RemoteTable, CachedNode, FlightExpr, FlightUDXF, SQLQueryResult, …) is
    recreated with the mapped replacement when its current source is found
    in *source_mapping*.  The mapping is keyed by backend identity (``id``).

    Sub-expressions reachable only through opaque fields (``remote_expr``,
    ``parent``, ``input_expr``, ``computed_kwargs_expr``) are rewritten
    recursively via the existing ``replace_nodes`` infrastructure.

    Parameters
    ----------
    source_mapping : dict[int, Any]
        ``{id(old_backend): new_backend, ...}``
    expr : Expr | Node
        The expression to rewrite.
    transfer_tables : bool, default False
        If True, materialize and register ``DatabaseTable`` data on the new
        backend so the rewritten expression is immediately executable.
        If False (the default) and the rewrite would produce
        ``DatabaseTable`` nodes whose data only exists on the old backend,
        raise ``ValueError``.

    Returns
    -------
    Expr
        A new expression with sources replaced.

    Raises
    ------
    ValueError
        When *transfer_tables* is False and the expression contains
        ``DatabaseTable`` nodes that require data transfer.
    """

    def _maybe_replace_cache(cache):
        """Rebuild a Cache object if its storage.source is in the mapping."""
        from attr import evolve  # noqa: PLC0415

        storage = getattr(cache, "storage", None)
        if storage is None:
            return cache
        source = getattr(storage, "source", None)
        if source is None or id(source) not in source_mapping:
            return cache
        new_storage = evolve(storage, source=source_mapping[id(source)])
        return evolve(cache, storage=new_storage)

    # Track DatabaseTable nodes that need data transferred: (old_backend, new_backend, table_name)
    tables_to_transfer = []

    def replacer(node, kwargs):
        overrides = {}

        source = getattr(node, "source", None)
        if source is not None and id(source) in source_mapping:
            new_source = source_mapping[id(source)]
            overrides["source"] = new_source

            # DatabaseTable (but not its subclasses like CachedNode, RemoteTable,
            # Read) needs its data transferred to the new backend.
            if type(node) is ops.DatabaseTable:
                tables_to_transfer.append((source, new_source, node.name))

        cache = getattr(node, "cache", None)
        if cache is not None:
            new_cache = _maybe_replace_cache(cache)
            if new_cache is not cache:
                overrides["cache"] = new_cache

        if overrides or kwargs:
            merged = dict(zip(node.__argnames__, node.__args__))
            if kwargs:
                merged |= kwargs
            merged |= overrides
            return node.__recreate__(merged)
        return node

    result = replace_nodes(replacer, expr).to_expr()

    if tables_to_transfer:
        # Filter out tables that already exist on the target backend
        # (e.g. cloned backends that share the same underlying connection).
        missing = _find_missing_tables(tables_to_transfer)
        if missing:
            if not transfer_tables:
                names = sorted({name for _, _, name in missing})
                raise ValueError(
                    f"Expression contains DatabaseTable nodes {names} whose "
                    f"data would need to be materialized and transferred to "
                    f"the new backend. Use deferred reads (e.g. "
                    f"deferred_read_parquet) to avoid this, or pass "
                    f"transfer_tables=True to materialize."
                )
            _transfer_tables(missing)

    return result


def _find_missing_tables(tables_to_transfer):
    """Return the subset of tables that don't exist on the target backend."""
    missing = []
    seen = set()
    for old_backend, new_backend, table_name in tables_to_transfer:
        key = (id(new_backend), table_name)
        if key in seen:
            continue
        seen.add(key)
        try:
            if table_name in new_backend.list_tables():
                continue
        except Exception:
            pass
        missing.append((old_backend, new_backend, table_name))
    return missing


def _transfer_tables(tables_to_transfer):
    """Materialize and register table data on new backends."""
    for old_backend, new_backend, table_name in tables_to_transfer:
        table = old_backend.table(table_name).to_pyarrow()
        new_backend.create_table(table_name, table)


def replace_unbound(expr, replacement, *, target=None):
    """Replace a single UnboundTable in *expr* with *replacement*.

    When *target* is ``None`` the expression is searched for UnboundTable
    nodes; if exactly one is found it is used as the target, otherwise a
    ``ValueError`` is raised.  Pass *target* explicitly to skip the
    search and replace only that specific node.
    """
    replacement = to_node(replacement)

    if target is None:
        found = walk_nodes(ops.UnboundTable, expr)
        if not found:
            raise ValueError("no UnboundTable found in expression")
        if len(found) > 1:
            raise ValueError(
                f"expression contains {len(found)} UnboundTable nodes; "
                f"pass target explicitly"
            )
        target = found[0]

    def replacer(node, kwargs):
        if node is target:
            return replacement
        elif kwargs:
            return node.__recreate__(kwargs)
        else:
            return node

    return replace_nodes(replacer, expr).to_expr()


def rename_params(expr, rename_map: dict[str, str]):
    """Rename NamedScalarParameter labels in an expression.

    Parameters
    ----------
    expr : Expr
        The expression to rewrite.
    rename_map : dict[str, str]
        ``{old_label: new_label, ...}``

    Returns
    -------
    Expr
        A new expression with matching parameter labels renamed.
    """

    def replacer(node, kwargs):
        if kwargs:
            node = node.__recreate__(kwargs)
        if isinstance(node, NamedScalarParameter) and node.label in rename_map:
            return NamedScalarParameter(
                dtype=node.dtype,
                label=rename_map[node.label],
                default=node.default,
            )
        return node

    return replace_nodes(replacer, expr).to_expr()


def validate_params(expr):
    """Raise TypeError if two NamedScalarParameter nodes share a label but have different dtypes."""

    dtypes_by_label = defaultdict(set)
    for node in walk_nodes(NamedScalarParameter, expr):
        dtypes_by_label[node.label].add(node.dtype)
    conflicts = {
        label: dtypes for label, dtypes in dtypes_by_label.items() if len(dtypes) > 1
    }
    if conflicts:
        messages = tuple(
            f"Parameter label {label!r} used with conflicting dtypes: "
            + ", ".join(str(d) for d in dtypes)
            for label, dtypes in conflicts.items()
        )
        raise TypeError("\n".join(messages))


def get_ordered_unique_sources(nodes):
    # Use id() for deduplication because backend __hash__ collides for
    # same-class instances and __eq__ only differs by session-local idx.
    sources, seen = (), set()
    for source in (node.source for node in nodes):
        if id(source) not in seen:
            seen.add(id(source))
            sources += (source,)
    return sources


def find_all_sources(expr):
    import xorq.vendor.ibis.expr.operations as ops  # noqa: PLC0415

    node_types = (
        ops.DatabaseTable,
        ops.SQLQueryResult,
        rel.CachedNode,
        rel.Read,
        rel.RemoteTable,
        rel.FlightUDXF,
        rel.FlightExpr,
        # ExprScalarUDF has an expr we need to get to
        # FlightOperator has a dynamically generated connection: it should be passed a Profile instead
    )
    nodes = walk_nodes(node_types, expr)
    sources = get_ordered_unique_sources(nodes)
    return sources
