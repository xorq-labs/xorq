from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xorq.vendor.ibis import util
from xorq.vendor.ibis.backends.bigquery import Backend as IbisBigQueryBackend


if TYPE_CHECKING:
    import pyarrow as pa

    from xorq.vendor.ibis.expr import types as ir


class Backend(IbisBigQueryBackend):
    def read_record_batches(
        self,
        record_batches: pa.RecordBatchReader | pa.Table,
        table_name: str | None = None,
        **kwargs: Any,
    ) -> ir.Table:
        """Load an Arrow batch source into a BigQuery table.

        The batches are materialized to an in-memory Parquet buffer and loaded
        through a BigQuery load job (the ADBC driver has no bulk-ingest path),
        landing the data in the connection's current dataset.

        Parameters
        ----------
        record_batches
            A `pa.RecordBatchReader` or `pa.Table` to load.
        table_name
            Optional name for the created table; a name is generated if omitted.
        kwargs
            Additional keyword arguments forwarded to
            `google.cloud.bigquery.LoadJobConfig`.

        Returns
        -------
        Table
            An Ibis table expression backed by the loaded data.
        """
        from xorq.common.utils.bigquery_utils import BigQueryLoader  # noqa: PLC0415

        table_name = table_name or util.gen_name("bigquery_record_batches")
        BigQueryLoader(self).load_record_batches(table_name, record_batches, **kwargs)
        return self.table(table_name)


def connect(*args: Any, **kwargs: Any) -> Backend:
    con = Backend()
    return con.connect(*args, **kwargs)
