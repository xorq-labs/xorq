import letsql as ls
from letsql.examples.core import (
    get_name_to_suffix,
    get_table_from_name,
    whitelist,
)


class Example:
    def __init__(self, name):
        self.name = name

    def fetch(self, table_name=None, backend=None):
        if backend is None:
            backend = ls.connect()
        return get_table_from_name(self.name, backend, table_name or self.name)


def __dir__():
    return (
        "get_table_from_name",
        *whitelist,
    )


def __getattr__(name):
    from letsql.vendor.ibis import examples as ibex

    lookup = get_name_to_suffix()

    if name not in lookup:
        return getattr(ibex, name)

    return Example(name)


__all__ = (
    "get_table_from_name",
    *whitelist,
)
