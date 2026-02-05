import dask
import numpy as np
import pyarrow as pa

from xorq.common.utils.dask_normalize.dask_normalize_utils import (
    normalize_seq_with_caller,
)


def lazy_register_pandas():
    import pandas as pd

    @dask.base.normalize_token.register(pd._libs.interval.Interval)
    def normalize_interval(interval):
        objs = (interval.left, interval.right, interval.closed)
        return normalize_seq_with_caller(*objs)

    @dask.base.normalize_token.register(pd._libs.tslibs.timestamps.Timestamp)
    def normalize_timestamp(timestamp):
        objs = (str(timestamp),)
        return normalize_seq_with_caller(*objs)


def safe_lazy_register(toplevel, function):
    if existing := dask.base.normalize_token._lazy.get(toplevel):

        def do_both():
            existing()
            function()

        to_register = do_both
    else:
        to_register = function
    dask.base.normalize_token.register_lazy(toplevel, to_register)


safe_lazy_register("pandas", lazy_register_pandas)


def lazy_register_sklearn():
    from sklearn.base import BaseEstimator

    from xorq.ibis_yaml.utils import freeze

    @dask.base.normalize_token.register(BaseEstimator)
    def normalize_sklearn_estimator(estimator):
        params = freeze(tuple(estimator.get_params(deep=True).items()))
        return normalize_seq_with_caller(
            type(estimator).__name__,
            type(estimator).__module__,
            params,
            caller="normalize_sklearn_estimator",
        )


safe_lazy_register("sklearn", lazy_register_sklearn)


@dask.base.normalize_token.register(dict)
def normalize_dict(dct):
    return normalize_seq_with_caller(*sorted(dct.items()))


@dask.base.normalize_token.register(np.random.RandomState)
def normalize_random_state(random_state):
    return normalize_seq_with_caller(random_state.get_state())


@dask.base.normalize_token.register(type(np.mean))
def normalize_numpy_array_function_dispatcher(func):
    """Normalize numpy array function dispatchers (e.g., np.mean, np.sum).

    Used by sklearn estimators like FeatureAgglomeration that have pooling_func=np.mean.
    """
    return normalize_seq_with_caller(
        func.__name__,
        func.__module__,
        caller="normalize_numpy_array_function_dispatcher",
    )


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
    from xorq.vendor.ibis import Schema

    return normalize_seq_with_caller(
        Schema.from_pyarrow(schema).to_pandas(),
        caller="normalize_pyarrow_schema",
    )
