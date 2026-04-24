from xorq.vendor.ibis.backends.trino import Backend as IbisTrinoBackend


class Backend(IbisTrinoBackend):
    def tokenize_table(self, dt):
        from xorq.common.utils.dask_normalize.dask_normalize_expr import (  # noqa: PLC0415
            normalize_seq_with_caller,
        )

        return normalize_seq_with_caller(
            dt.name,
            dt.schema,
            dt.source,
            dt.namespace,
            caller="normalize_remote_databasetable",
        )

    def __dask_tokenize__(self):
        from xorq.common.utils.dask_normalize.dask_normalize_utils import (  # noqa: PLC0415
            normalize_seq_with_caller,
        )

        return normalize_seq_with_caller(self.name, self.con.host)
