from __future__ import annotations

from typing import Dict

from ibis.expr.operations import relations as ops


def _find_backend(value):
    backends = set()
    node_types = (ops.UnboundTable, ops.DatabaseTable, ops.SQLQueryResult)
    for table in value.find(node_types):
        backends.add(table.source)

    if len(backends) > 1:
        raise ValueError("Multiple backends found for this expression")

    return backends.pop()


class SourceDict:
    def __init__(self):
        self.sources: Dict[ops.DatabaseTable, ops.DatabaseTable] = {}

    def __setitem__(self, key, value):
        if not isinstance(value, ops.Relation):
            raise ValueError
        self.sources[key] = value

    def __getitem__(self, key):
        return self.sources[key]

    def get_backend(self, key, default=None):
        if key in self.sources:
            value = self.sources[key]
            return getattr(value, "source", _find_backend(value))
        return default

    def get_table_or_op(self, key, default=None):
        if key in self.sources:
            return self.sources[key]

        return default
