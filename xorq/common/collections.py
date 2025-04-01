from __future__ import annotations

from typing import Dict

from xorq.expr.relations import (
    CachedNode,
    Read,
)
from xorq.vendor.ibis.expr.operations import relations as ops


def _find_backend(value):
    node_types = (
        ops.UnboundTable,
        ops.DatabaseTable,
        ops.SQLQueryResult,
        CachedNode,
        Read,
    )
    (backend, *rest) = set(table.source for table in value.find(node_types))
    if len(rest) > 1:
        raise ValueError("Multiple backends found for this expression")
    return backend


class SourceDict:
    def __init__(self):
        self.sources: Dict[ops.DatabaseTable, ops.DatabaseTable] = {}

    def __setitem__(self, key, value):
        if not isinstance(value, ops.Relation):
            raise ValueError
        self.sources[key] = value

    def __getitem__(self, key):
        return self.sources[key]

    def __contains__(self, key):
        return self.sources.__contains__(key)

    def get_backend(self, key, default=None):
        if key in self.sources:
            value = self.sources[key]
            return getattr(value, "source", _find_backend(value))
        return default

    def get_table_or_op(self, key, default=None):
        if key in self.sources:
            return self.sources[key]

        return default
