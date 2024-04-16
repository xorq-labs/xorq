from typing import Any

from ibis import Schema
from ibis.common.collections import FrozenDict
from ibis.expr import operations as ops


def replace_cache_table(node, _, **kwargs):
    if isinstance(node, CachedNode):
        return kwargs["parent"]
    else:
        return node.__recreate__(kwargs)


class CachedNode(ops.Relation):
    schema: Schema
    parent: Any
    source: Any
    storage: Any
    values = FrozenDict()
