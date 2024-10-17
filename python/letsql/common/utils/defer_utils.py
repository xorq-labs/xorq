import ibis
import ibis.backends.sql.compiler
import pandas as pd
import pyarrow as pa
import toolz


DEFAULT_CHUNKSIZE = 10_000


def read_csv_rbr(*args, schema=None, chunksize=DEFAULT_CHUNKSIZE, dtype=None, **kwargs):
    """Deferred and streaming csv reading via pandas"""
    if dtype is not None:
        raise Exception("pass `dtype` as pyarrow `schema`")
    if chunksize is None:
        raise ValueError("chunksize must not be `None`")
    if schema is not None:
        dtype = {
            col: typ.to_pandas()
            for col, typ in ibis.Schema.from_pyarrow(schema).items()
        }
    # schema is always nullable (this is good)
    gen = map(
        pa.RecordBatch.from_pandas,
        pd.read_csv(
            *args,
            dtype=dtype,
            chunksize=chunksize,
            **kwargs,
        ),
    )
    if schema is None:
        (el, gen) = toolz.peek(gen)
        schema = el.schema
    rbr = pa.RecordBatchReader.from_batches(
        schema,
        gen,
    )
    return rbr
