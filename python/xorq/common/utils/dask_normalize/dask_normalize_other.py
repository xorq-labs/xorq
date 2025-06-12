import dask
import numpy as np
import pandas as pd
import pyarrow as pa

from xorq.common.utils.dask_normalize.dask_normalize_utils import (
    normalize_seq_with_caller,
)


# preemptively cause registration of numpy, pandas
dask.base.normalize_token.dispatch(np.dtype)
dask.base.normalize_token.dispatch(pd.DataFrame)


@dask.base.normalize_token.register(dict)
def normalize_dict(dct):
    return normalize_seq_with_caller(*sorted(dct.items()))


@dask.base.normalize_token.register(pd._libs.interval.Interval)
def normalize_interval(interval):
    objs = (interval.left, interval.right, interval.closed)
    return normalize_seq_with_caller(*objs)


@dask.base.normalize_token.register(pd._libs.tslibs.timestamps.Timestamp)
def normalize_timestamp(timestamp):
    objs = (str(timestamp),)
    return normalize_seq_with_caller(*objs)


@dask.base.normalize_token.register(np.random.RandomState)
def normalize_random_state(random_state):
    return normalize_seq_with_caller(random_state.get_state())


@dask.base.normalize_token.register(type)
def normalize_type(typ):
    return normalize_seq_with_caller(typ.__name__, typ.__module__)


@dask.base.normalize_token.register(pa.Table)
def normalize_pyarrow_table(table: pa.Table):
    return normalize_seq_with_caller(
        tuple(
            dask.base.tokenize(el.serialize().to_pybytes()) for el in table.to_batches()
        ),
        caller="normalize_pyarrow_table",
    )


@dask.base.normalize_token.register(pa.Schema)
def normalize_pyarrow_schema(schema: pa.Schema):
    from xorq import Schema

    return normalize_seq_with_caller(
        Schema.from_pyarrow(schema).to_pandas(),
        caller="normalize_pyarrow_schema",
    )
