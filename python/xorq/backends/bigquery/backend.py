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
        mode: str = "create",
        **kwargs: Any,
    ) -> ir.Table:
        """Ingest an Arrow batch source into a BigQuery table via ADBC.

        Uses the BigQuery ADBC driver (installed out-of-band with
        ``dbc install bigquery``), mirroring the snowflake and databricks
        backends. The table lands in the connection's current dataset.

        Parameters
        ----------
        record_batches
            A `pa.RecordBatchReader` or `pa.Table` to ingest.
        table_name
            Optional name for the created table; a name is generated if omitted.
        mode
            ADBC ingest mode (e.g. `"create"`, `"append"`, `"replace"`).
        kwargs
            Additional keyword arguments forwarded to `adbc_ingest`.

        Returns
        -------
        Table
            An Ibis table expression backed by the ingested data.
        """
        from xorq.common.utils.bigquery_utils import BigQueryADBC  # noqa: PLC0415

        table_name = table_name or util.gen_name("bigquery_record_batches")
        BigQueryADBC(self).adbc_ingest(table_name, record_batches, mode=mode, **kwargs)
        return self.table(table_name)


def connect(*args: Any, **kwargs: Any) -> Backend:
    con = Backend()
    return con.connect(*args, **kwargs)
