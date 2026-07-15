from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

from attr import field, frozen
from attr.validators import instance_of

from xorq.backends.bigquery import Backend as BigQueryBackend


if TYPE_CHECKING:
    import pyarrow as pa


@frozen
class BigQueryLoader:
    """Load Arrow batch sources into BigQuery via a Parquet load job.

    The BigQuery ADBC driver does not implement bulk ingest, so ingestion
    goes through `google.cloud.bigquery.Client.load_table_from_file`, the
    same mechanism backing `read_parquet`.
    """

    con = field(validator=instance_of(BigQueryBackend))

    def load_record_batches(
        self,
        table_name: str,
        record_batches: pa.RecordBatchReader | pa.Table,
        **kwargs: Any,
    ) -> None:
        import pyarrow.parquet as pq  # noqa: PLC0415
        from google.cloud import bigquery as bq  # noqa: PLC0415

        from xorq.common.utils.rbr_utils import coerce_to_arrow_table  # noqa: PLC0415

        con = self.con
        arrow_table = coerce_to_arrow_table(record_batches)

        dataset_ref = bq.DatasetReference(con.data_project, con.current_database)
        table_ref = dataset_ref.table(table_name)

        buffer = io.BytesIO()
        pq.write_table(arrow_table, buffer)
        buffer.seek(0)

        job_config = bq.LoadJobConfig(
            source_format=bq.SourceFormat.PARQUET,
            write_disposition=bq.WriteDisposition.WRITE_TRUNCATE,
            **kwargs,
        )
        load_job = con.client.load_table_from_file(
            buffer, table_ref, job_config=job_config
        )
        load_job.result()
