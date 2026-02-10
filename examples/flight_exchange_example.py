"""Iterative split-train exchanger using Flight with streaming exchange.

Traditional approach: You would build a custom Flight server that receives
batched data, splits it by partition key, trains models per split, and streams
results back. All the partitioning, streaming, and protocol code must be
written by hand.

With xorq: Define an AbstractExchanger subclass with split logic and a training
function, then use streaming_split_exchange to handle partitioned streaming
automatically. The Flight framework manages the exchange protocol while you
focus on the model training logic.
"""

import functools
import pickle

import pandas as pd
import pyarrow as pa

import xorq.api as xo
import xorq.expr.datatypes as dt
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
    @property
    def exchange_f(self):
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

    @property
    def schema_in_required(self):
        return None

    @property
    def schema_in_condition(self):
        def condition(schema_in):
            return any(name == SPLIT_KEY for name in schema_in)

        return condition

    @property
    def calc_schema_out(self):
        def f(schema_in):
            return xo.schema(
                {
                    MODEL_BINARY_KEY: dt.binary,
                    SPLIT_KEY: schema_in[SPLIT_KEY],
                }
            )

        return f

    @property
    def description(self):
        return "iteratively train model on data ordered by `split`"

    @property
    def command(self):
        return "iterative-split-train"

    @property
    def query_result(self):
        return {
            "schema-in-required": self.schema_in_required,
            "schema-in-condition": self.schema_in_condition,
            "calc-schema-out": self.calc_schema_out,
            "description": self.description,
            "command": self.command,
        }


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


if __name__ == "__pytest_main__":
    rbr_in = instrument_reader(xo.to_pyarrow_batches(expr), prefix="input ::")
    exchanger = IterativeSplitTrainExchanger()
    with FlightServer() as server:
        client = server.client
        client.do_action(
            AddExchangeAction.name,
            exchanger,
            options=client._options,
        )
        (fut, rbr_out) = client.do_exchange_batches(
            exchanger.command, rbr_in
        )
        df_out = instrument_reader(rbr_out, prefix="output ::").read_pandas()
        print(fut.result())
        print(df_out.assign(model=df_out.model_binary.map(pickle.loads)))

    pytest_examples_passed = True
