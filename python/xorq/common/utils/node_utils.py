import importlib

import dask
import toolz

import xorq.expr.relations as rel
import xorq.vendor.ibis.expr.operations as ops
from xorq.common.utils.dask_normalize.dask_normalize_utils import (
    patch_normalize_op_caching,
)
from xorq.common.utils.graph_utils import (
    bfs,
    replace_nodes,
    walk_nodes,
)


replace_typs = (
    ops.PhysicalTable,
    rel.CachedNode,
    rel.Read,
)


@toolz.curry
def do_replace_dct(node, kwargs, *, replace_dct):
    if (replaced := replace_dct.get(node)) is not None:
        return replaced.__recreate__(kwargs) if kwargs else replaced
    elif kwargs:
        return node.__recreate__(kwargs)
    else:
        return node


def get_typs(maybe_typs):
    match maybe_typs:
        case None:
            typs = replace_typs
        case tuple():
            typs = maybe_typs
        case str():
            (module, attr) = maybe_typs.rsplit(".", 1)
            typs = (getattr(importlib.import_module(module), attr),)
    return typs


def find_by_expr_hash(expr, to_replace_hash, typs=replace_typs):
    """
    Find a node in the expression whose tokenize hash prefix matches the given hash.
    Raises if no matches or multiple matches.
    """
    typs = get_typs(typs)
    # Determine prefix length from provided hash
    hash_len = len(to_replace_hash)
    with patch_normalize_op_caching():
        matches = []
        for node in walk_nodes(typs, expr):
            tok = dask.base.tokenize(node.to_expr())
            if tok[:hash_len] == to_replace_hash:
                matches.append(node)
    if not matches:
        raise ValueError(f"No node found matching hash prefix: {to_replace_hash}")
    if len(matches) > 1:
        raise ValueError(f"Multiple nodes found matching hash prefix: {to_replace_hash}")
    return matches[0]


def replace_by_expr_hash(expr, to_replace_hash, replace_with, typs=replace_typs):
    typs = get_typs(typs)
    to_replace = find_by_expr_hash(expr, to_replace_hash, typs=typs)
    replaced = replace_nodes(
        do_replace_dct(
            replace_dct={to_replace: replace_with},
        ),
        expr,
    )
    return replaced.to_expr()


def unbind_expr_hash(expr, to_replace_hash, typs=replace_typs):
    typs = get_typs(typs)
    to_replace = find_by_expr_hash(expr, to_replace_hash, typs=typs)
    replace_with = ops.UnboundTable("unbound", to_replace.schema)
    unbound_expr = replace_nodes(
        do_replace_dct(
            replace_dct={to_replace: replace_with},
        ),
        expr,
    )
    return unbound_expr.to_expr()


def gen_downstream(expr, downstream_of):
    gi = bfs(expr).invert()

    def inner(op):
        for child in gi[op]:
            yield child
            yield from inner(child)

    yield from inner(downstream_of)


def elide_downstream_cached_node(expr, downstream_of):
    cns = set(
        el
        for el in gen_downstream(expr, downstream_of)
        if isinstance(el, rel.CachedNode)
    )

    def elide_cached_node(node, kwargs):
        # Remove debug printing of cached nodes
        if node in cns:
            while isinstance(node, rel.CachedNode):
                node = node.parent.op()
            node = replace_nodes(elide_cached_node, node.to_expr())
        if kwargs:
            node = node.__recreate__(kwargs)
        return node

    return elide_cached_node
