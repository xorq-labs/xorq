from typing import MutableMapping

import pandas as pd
import pyarrow as pa

import xorq.common.exceptions as com
from xorq.vendor.ibis.backends.pandas import Backend as IbisPandasBackend


class Backend(IbisPandasBackend):
    def do_connect(
        self,
        dictionary: MutableMapping[str, pd.DataFrame] | None = None,
    ) -> None:
        """Construct a client from a dictionary of pandas DataFrames.

        Parameters
        ----------
        dictionary
            An optional mapping of string table names to pandas DataFrames.

        Examples
        --------
        >>> import xorq as xo
        >>> ls.pandas.connect({"t": pd.DataFrame({"a": [1, 2, 3]})})  # doctest: +ELLIPSIS
        <ibis.backends.pandas.Backend object at 0x...>
        """
        self.dictionary = dictionary or {}
        self.schemas = {}

    def drop_table(self, name: str, *, force: bool = False) -> None:
        if not force and name in self.dictionary:
            raise com.XorqError(
                "Cannot drop existing table. Call drop_table with force=True to drop existing table."
            )
        del self.dictionary[name]
        del self.schemas[name]

    def read_record_batches(
        self,
        record_batches: pa.RecordBatchReader,
        table_name: str | None = None,
    ):
        self.dictionary[table_name] = record_batches.read_pandas()
        return self.table(table_name)
