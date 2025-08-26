import importlib

import toolz

import xorq.expr.relations as rel
import xorq.vendor.ibis.expr.operations as ops
from xorq.common.utils.graph_utils import (
    bfs,
    replace_nodes,
    walk_nodes,
)


replace_typs = (
    ops.PhysicalTable,
    rel.CachedNode,
    rel.Read,
    # can't use Tag for anything involving hashes: hash value is the same as parent
    # rel.Tag,
)


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


def find_by_expr_hash(expr, to_replace_hash, typs=None):
    typs = get_typs(typs)
    (to_replace, *rest) = (
        node
        for node in walk_nodes(typs, expr)
        if node.to_expr().ls.tokenized == to_replace_hash
    )
    if rest:
        raise ValueError
    return to_replace


def find_by_expr_tag(expr, tag):
    yield from (node for node in walk_nodes(rel.Tag, expr) if node.tag == tag)


def find_node(expr, hash, tag, typs=None):
    match [hash, tag]:
        case [None, None]:
            raise ValueError
        case [_, None]:
            if isinstance(typs, tuple) and rel.Tag in typs:
                raise ValueError
            return find_by_expr_hash(expr, hash, typs=typs)
        case [None, _]:
            (node, *rest) = find_by_expr_tag(expr, tag)
            if rest:
                raise ValueError
            else:
                return node
        case _:
            raise ValueError


def replace_by_expr_hash(expr, to_replace_hash, replace_with, typs=None):
    typs = get_typs(typs)
    to_replace = find_by_expr_hash(expr, to_replace_hash, typs=typs)
    replaced = replace_nodes(
        do_replace_dct(
            replace_dct={to_replace: replace_with},
        ),
        expr,
    )
    return replaced.to_expr()


def unbind_expr_hash(expr, to_replace_hash, typs=None):
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
        if node in cns:
            while isinstance(node, rel.CachedNode):
                node = node.parent.op()
            node = replace_nodes(elide_cached_node, node.to_expr())
        if kwargs:
            node = node.__recreate__(kwargs)
        return node

    return elide_cached_node
