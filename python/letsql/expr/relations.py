from typing import Any

from ibis import Schema
from ibis.common.collections import FrozenDict
from ibis.expr import operations as ops


def contract_cache_table(node, _, **kwargs):
    if hasattr(node, "parent"):
        parent = kwargs.get("parent", node.parent)
        node = node.replace({node.parent: parent})
    if isinstance(node, CachedNode):
        node = node.replace({node: node.parent})
    return node


class CachedNode(ops.Relation):
    schema: Schema
    parent: Any
    source: Any
    storage: Any
    values = FrozenDict()
