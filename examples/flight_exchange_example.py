import functools
import pickle

import pandas as pd
import pyarrow as pa

import xorq as xo
from xorq.common.utils.rbr_utils import (
    instrument_reader,
    streaming_split_exchange,
)
from xorq.flight import FlightServer
from xorq.flight.action import AddExchangeAction
from xorq.flight.exchanger import AbstractExchanger


SPLIT_KEY = "split"
MODEL_BINARY_KEY = "model_binary"


value = 0


def train_batch_df(df):
    global value
    value += len(df)
    return value


class IterativeSplitTrainExchanger(AbstractExchanger):
    @classmethod
    @property
    def exchange_f(cls):
        def train_batch(split_reader):
            df = split_reader.read_pandas()
            (split, *rest) = df[SPLIT_KEY].unique()
            assert not rest
            value = train_batch_df(df)
            batch = pa.RecordBatch.from_pydict(
                {
                    MODEL_BINARY_KEY: [pickle.dumps(value)],
                    SPLIT_KEY: [split],
                }
            )
            return batch

        return functools.partial(streaming_split_exchange, SPLIT_KEY, train_batch)

    @classmethod
    @property
    def schema_in_required(cls):
        return None

    @classmethod
    @property
    def schema_in_condition(cls):
        def condition(schema_in):
            return any(field.name == SPLIT_KEY for field in schema_in)

        return condition

    @classmethod
    @property
    def calc_schema_out(cls):
        def f(schema_in):
            split_field = next(field for field in schema_in if field.name == SPLIT_KEY)
            model_binary_field = pa.field(MODEL_BINARY_KEY, pa.binary())
            return pa.schema(
                (
                    model_binary_field,
                    split_field,
                )
            )

        return f

    @classmethod
    @property
    def description(cls):
        return "iteratively train model on data ordered by `split`"

    @classmethod
    @property
    def command(cls):
        return "iterative-split-train"


def train_test_split_union(expr, name=SPLIT_KEY, *args, **kwargs):
    splits = xo.expr.ml.train_test_splits(expr, *args, **kwargs)
    return xo.union(
        *(
            split.mutate(**{name: xo.literal(i, "int64")})
            for i, split in enumerate(splits)
        )
    )


con = xo.connect()
N = 10_000
df = pd.DataFrame({"a": range(N), "b": range(N, 2 * N)})
t = con.register(df, "t")
expr = train_test_split_union(
    t, unique_key="a", test_sizes=(0.2, 0.3, 0.5), random_seed=0
)


rbr_in = instrument_reader(xo.to_pyarrow_batches(expr), prefix="input ::")
with FlightServer() as server:
    client = server.client
    client.do_action(
        AddExchangeAction.name, IterativeSplitTrainExchanger, options=client._options
    )
    (fut, rbr_out) = client.do_exchange(IterativeSplitTrainExchanger.command, rbr_in)
    df_out = instrument_reader(rbr_out, prefix="output ::").read_pandas()
    print(fut.result())
    print(df_out.assign(model=df_out.model_binary.map(pickle.loads)))


pytest_examples_passed = True
