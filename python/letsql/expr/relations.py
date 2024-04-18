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


class CachedNode(ops.Relation):
    schema: Schema
    parent: Any
    source: Any
    storage: Any
    values = FrozenDict()
