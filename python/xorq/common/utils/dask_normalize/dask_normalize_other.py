import dask
import numpy as np
import pandas as pd

from xorq.common.utils.dask_normalize.dask_normalize_utils import (
    normalize_seq_with_caller,
)


# preemptively cause registration of numpy, pandas
dask.base.normalize_token.dispatch(np.dtype)
dask.base.normalize_token.dispatch(pd.DataFrame)


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
