import dask
import numpy as np
import pandas as pd


# pre-emptively cause registration of numpy, pandas
dask.base.normalize_token.dispatch(np.dtype)
dask.base.normalize_token.dispatch(pd.DataFrame)


@dask.base.normalize_token.register(pd._libs.interval.Interval)
def normalize_interval(interval):
    objs = (interval.left, interval.right, interval.closed)
    return objs


@dask.base.normalize_token.register(pd._libs.tslibs.timestamps.Timestamp)
def normalize_timestamp(timestamp):
    objs = (str(timestamp),)
    return objs


@dask.base.normalize_token.register(np.random.RandomState)
def normalize_random_state(random_state):
    return random_state.get_state()


@dask.base.normalize_token.register(type)
def normalize_type(typ):
    return (typ.__name__, typ.__module__)
