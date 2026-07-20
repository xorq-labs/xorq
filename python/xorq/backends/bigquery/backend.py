from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xorq.common.utils.enums import IngestMode
from xorq.vendor.ibis import util
from xorq.vendor.ibis.backends.bigquery import Backend as IbisBigQueryBackend
from xorq.vendor.ibis.backends.profiles import Profile


if TYPE_CHECKING:
    import pyarrow as pa

    from xorq.vendor.ibis.expr import types as ir


class Backend(IbisBigQueryBackend):
    # live runtime objects (secrets / prebuilt clients) that can't be serialized
    # to YAML and must not be baked into a connection profile / build artifact;
    # reconnection re-derives credentials from Application Default Credentials
    _profile_exclude_kwargs = ("credentials", "client", "storage_client")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # the live connection keeps these via _con_kwargs; only the profile
        # (used for hashing and build serialization) is stripped
        profile = getattr(self, "_profile", None)
        if profile is None:
            return
        kept = {
            key: value
            for key, value in profile.kwargs_dict.items()
            if key not in self._profile_exclude_kwargs
        }
        if len(kept) != len(profile.kwargs_dict):
            self._profile = Profile(
                con_name=profile.con_name,
                kwargs_tuple=tuple(kept.items()),
                idx=profile.idx,
            )

    def __hash__(self) -> int:
        # the vendored __hash__ hashes db_identity, which embeds the credential
        # object still kept in _con_kwargs; __eq__ compares the credential-free
        # _profile.hash_name. Key the hash off the same stripped profile so the
        # hash/eq contract holds (equal backends hash equal)
        profile = getattr(self, "_profile", None)
        if profile is None:
            return super().__hash__()
        return hash(profile.hash_name)

    def read_record_batches(
        self,
        record_batches: pa.RecordBatchReader | pa.Table,
        table_name: str | None = None,
        mode: IngestMode | str = IngestMode.CREATE,
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
            Ingest mode; an `IngestMode` or its string value (`"create"`,
            `"append"`, `"replace"`, `"create_append"`). An unsupported value
            raises `ValueError` before the driver is touched.
        kwargs
            Additional keyword arguments forwarded to `adbc_ingest`.

        Returns
        -------
        Table
            An Ibis table expression backed by the ingested data.
        """
        mode = IngestMode(mode)
        from xorq.common.utils.bigquery_utils import BigQueryADBC  # noqa: PLC0415

        table_name = table_name or util.gen_name("bigquery_record_batches")
        BigQueryADBC(self).adbc_ingest(table_name, record_batches, mode=mode, **kwargs)
        return self.table(table_name)


def connect(*args: Any, **kwargs: Any) -> Backend:
    con = Backend()
    return con.connect(*args, **kwargs)
