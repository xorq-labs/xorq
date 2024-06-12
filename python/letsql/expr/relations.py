from typing import Any

from ibis import Schema
from ibis.common.collections import FrozenDict
from ibis.expr import operations as ops


def replace_cache_table(node, _, **kwargs):
    if isinstance(node, CachedNode):
        return kwargs["parent"]
    else:
        return node.__recreate__(kwargs)


def replace_source_factory(source: Any):
    def replace_source(node, _, **kwargs):
        if "source" in kwargs:
            kwargs["source"] = source
        return node.__recreate__(kwargs)

    return replace_source


def make_native_op(node):
    # FIXME: how to reference let.Backend.name?
    if node.source.name != "let":
        raise ValueError
    sources = node.source._sources
    native_source = sources.get_backend(node)
    if native_source.name == "let":
        raise ValueError

    def replace_table(_node, _, **_kwargs):
        return sources.get_table_or_op(_node, _node.__recreate__(_kwargs))

    return node.replace(replace_table).to_expr()


class CachedNode(ops.Relation):
    schema: Schema
    parent: Any
    source: Any
    storage: Any
    values = FrozenDict()
