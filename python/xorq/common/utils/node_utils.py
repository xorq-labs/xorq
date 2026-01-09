import importlib

import dask
import toolz

import xorq.expr.relations as rel
import xorq.vendor.ibis.expr.operations as ops
from xorq.common.utils.graph_utils import (
    bfs,
    find_all_sources,
    replace_nodes,
    walk_nodes,
)
from xorq.vendor.ibis.expr.operations import UnboundTable


replace_typs = (
    ops.PhysicalTable,
    rel.CachedNode,
    rel.Read,
    # can't use Tag for anything involving hashes: hash value is the same as parent
    # rel.Tag,
)


def compute_expr_hash(expr, strategy=None):
    """
    Compute hash for an expression with optional normalization strategy.

    Args:
        expr: Expression to hash (can be Expr or node, will convert to expr if needed)
        strategy: Optional normalization strategy for consistent hashing

    Returns:
        Hash string for the expression
    """
    # Convert to expr if it's a node
    if hasattr(expr, "to_expr"):
        expr = expr.to_expr()

    if strategy is None:
        return expr.ls.tokenized

    with strategy.normalization_context(expr):
        return dask.base.tokenize(expr.ls.untagged)


@toolz.curry
def do_replace_dct(node, kwargs, *, replace_dct):
    if (replaced := replace_dct.get(node)) is not None:
        return replaced
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
        case _:
            raise ValueError
    return typs


def find_by_expr_hash(expr, to_replace_hash, typs=None, strategy=None):
    def matches_hash(node):
        return compute_expr_hash(node, strategy) == to_replace_hash

    typs = get_typs(typs)
    (to_replace, *rest) = filter(
        matches_hash,
        walk_nodes(typs, expr),
    )
    if rest:
        raise ValueError
    return to_replace


def find_by_expr_tag(expr, tag):
    yield from (node for node in walk_nodes(rel.Tag, expr) if node.tag == tag)


def find_node(expr, hash, tag, typs=None, strategy=None):
    match [hash, tag]:
        case [None, None]:
            raise ValueError
        case [_, None]:
            if isinstance(typs, tuple) and rel.Tag in typs:
                raise ValueError
            return find_by_expr_hash(expr, hash, typs=typs, strategy=strategy)
        case [None, _]:
            (node, *rest) = find_by_expr_tag(expr, tag)
            if rest:
                raise ValueError
            else:
                return node
        case _:
            raise ValueError


def replace_by_expr_hash(expr, to_replace_hash, replace_with, typs=None, strategy=None):
    typs = get_typs(typs)
    to_replace = find_by_expr_hash(expr, to_replace_hash, typs=typs, strategy=strategy)
    replaced = replace_nodes(
        do_replace_dct(
            replace_dct={to_replace: replace_with},
        ),
        expr,
    )
    return replaced.to_expr()


def unbind_expr_hash(expr, to_replace_hash, typs=None, strategy=None):
    typs = get_typs(typs)
    to_replace = find_by_expr_hash(expr, to_replace_hash, typs=typs, strategy=strategy)
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
        if node in cns:
            while isinstance(node, rel.CachedNode):
                node = node.parent.op()
            node = replace_nodes(elide_cached_node, node.to_expr())
        if kwargs:
            node = node.__recreate__(kwargs)
        return node

    return elide_cached_node


def expr_to_unbound(expr, hash, tag, typs, strategy=None):
    """create an unbound expr that only needs to have a source of record batches fed in"""

    found = find_node(expr, hash=hash, tag=tag, typs=typs, strategy=strategy)
    found_expr = found.to_expr()
    to_unbind_hash = hash if hash else compute_expr_hash(found_expr, strategy)
    match find_all_sources(found_expr):
        case []:
            raise ValueError("found no connections")
        case [found_con]:
            pass
        case _:
            found_con = found_expr._find_backend()
            assert found_con
    unbound_table = UnboundTable("unbound", found.schema)
    replace_with = unbound_table.to_expr().into_backend(found_con).op()
    replaced = replace_by_expr_hash(
        expr, to_unbind_hash, replace_with, typs=(type(found),), strategy=strategy
    )
    (found,) = walk_nodes(UnboundTable, replaced)
    elided = replace_nodes(elide_downstream_cached_node(replaced, found), replaced)
    return elided
