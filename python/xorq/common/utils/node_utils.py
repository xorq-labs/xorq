import dask
import toolz

import xorq.expr.relations as rel
import xorq.vendor.ibis.expr.operations as ops
from xorq.common.utils.graph_utils import (
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
    if replaced := replace_dct.get(node):
        return replaced.__recreate__(kwargs) if kwargs else replaced
    elif kwargs:
        return node.__recreate__(kwargs)
    else:
        return node


def find_by_expr_hash(expr, to_replace_hash, typs=replace_typs):
    (to_replace, *rest) = (
        node
        for node in walk_nodes(typs, expr)
        if dask.base.tokenize(node.to_expr()) == to_replace_hash
    )
    if rest:
        raise ValueError
    return to_replace


def replace_by_expr_hash(expr, to_replace_hash, replace_with, typs=replace_typs):
    to_replace = find_by_expr_hash(expr, to_replace_hash, typs=typs)
    replaced = replace_nodes(
        do_replace_dct(
            replace_dct={to_replace: replace_with},
        ),
        expr,
    )
    return replaced.to_expr()


def unbind_expr_hash(expr, to_replace_hash, typs=replace_typs):
    to_replace = find_by_expr_hash(expr, to_replace_hash, typs=typs)
    replace_with = ops.UnboundTable("unbound", to_replace.schema)
    unbound_expr = replace_nodes(
        do_replace_dct(
            replace_dct={to_replace: replace_with},
        ),
        expr,
    )
    return unbound_expr.to_expr()
