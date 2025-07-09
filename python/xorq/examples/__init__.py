import xorq as xo
from xorq.examples.core import (
    get_name_to_suffix,
    get_table_from_name,
    whitelist,
)


class Example:
    def __init__(self, name):
        self.name = name

    def fetch(self, backend=None, table_name=None, deferred=True, **kwargs):
        if backend is None:
            backend = xo.connect()

        return get_table_from_name(
            self.name,
            backend,
            table_name or self.name,
            deferred=deferred,
            **kwargs,
        )


def __dir__():
    return (
        "get_table_from_name",
        *whitelist,
    )


def __getattr__(name):
    from xorq.vendor.ibis import examples as ibex

    lookup = get_name_to_suffix()

    if name not in lookup:
        return getattr(ibex, name)

    return Example(name)


__all__ = (
    "get_table_from_name",
    *whitelist,
)
