import pyarrow as pa

import xorq as xo
from xorq.examples.core import (
    get_name_to_suffix,
    get_table_from_name,
    whitelist,
)


schemas = {
    "penguins": pa.schema(
        [
            pa.field("species", pa.string()),
            pa.field("island", pa.string()),
            pa.field("bill_length_mm", pa.float64()),
            pa.field("bill_depth_mm", pa.float64()),
            pa.field("flipper_length_mm", pa.int64()),
            pa.field("body_mass_g", pa.int64()),
            pa.field("sex", pa.string()),
            pa.field("year", pa.int64()),
        ]
    )
}


class Example:
    def __init__(self, name):
        self.name = name

    def fetch(self, backend=None, table_name=None, deferred=True, **kwargs):
        if backend is None:
            backend = xo.connect()

        if self.name in schemas:
            kwargs.setdefault("schema", schemas[self.name])

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
