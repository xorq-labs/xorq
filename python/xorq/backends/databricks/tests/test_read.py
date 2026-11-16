from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

from xorq.vendor.ibis import util


if TYPE_CHECKING:
    from xorq.backends.databricks import Backend


def test_read_record_batches_into_other_schema(con: Backend) -> None:
    catalog = con.current_catalog
    schema = util.gen_name("xorq_read_schema")
    table_name = util.gen_name("xorq_read_table")
    data = pa.table({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    con.create_database(schema, catalog=catalog)
    try:
        t = con.read_record_batches(
            data.to_reader(),
            table_name=table_name,
            catalog_name=catalog,
            db_schema_name=schema,
        )
        result = t.order_by("a").to_pyarrow()
        assert result.to_pydict() == data.to_pydict()
    finally:
        con.drop_table(table_name, database=(catalog, schema), force=True)
        con.drop_database(schema, catalog=catalog, force=True)
