from __future__ import annotations

from typing import Dict

from ibis.expr.operations import relations as ops


class SourceDict:
    def __init__(self):
        self.sources: Dict[ops.DatabaseTable, ops.DatabaseTable] = {}

    def __setitem__(self, key, value):
        if not isinstance(value, ops.DatabaseTable):
            raise ValueError
        self.sources[key] = value

    def __getitem__(self, key):
        return self.sources[key]

    def get_backend(self, key, default=None):
        if key in self.sources:
            return self.sources[key].source

        return default

    def get_table(self, key, default=None):
        if key in self.sources:
            return self.sources[key]

        return default
