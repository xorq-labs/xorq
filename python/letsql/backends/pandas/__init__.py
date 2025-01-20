from typing import MutableMapping

import pandas as pd
from ibis.backends.pandas import Backend as IbisPandasBackend


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
        >>> import letsql as ls
        >>> ls.pandas.connect({"t": pd.DataFrame({"a": [1, 2, 3]})})  # doctest: +ELLIPSIS
        <ibis.backends.pandas.Backend object at 0x...>
        """
        self.dictionary = dictionary or {}
        self.schemas = {}
