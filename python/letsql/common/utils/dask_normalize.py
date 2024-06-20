from contextlib import contextmanager
from unittest.mock import (
    Mock,
    patch,
)

import dask
import toolz

import letsql.common.utils.dask_normalize_expr  # noqa: F401
import letsql.common.utils.dask_normalize_function  # noqa: F401
import letsql.common.utils.dask_normalize_other  # noqa: F401


dask.config.set({"tokenize.ensure-deterministic": True})


@contextmanager
def patch_normalize_token(*typs, f=toolz.functoolz.return_none):
    with patch.dict(
        dask.base.normalize_token._lookup,
        {typ: Mock(side_effect=f) for typ in typs},
    ) as dct:
        mocks = {typ: dct[typ] for typ in typs}
        yield mocks
