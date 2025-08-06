import decimal

import pyarrow as pa
import pytest

import xorq as xo
from xorq.tests.util import assert_frame_equal
from xorq.vendor.ibis.expr import datatypes as dt


@pytest.fixture(scope="session")
def batch():
    return pa.RecordBatch.from_arrays(
        [
            pa.array([0, 1.5, 2.3, 3, 4, 5, 6]),
            pa.array([7, 4, 3, 8, 9, 1, 6]),
            pa.array(["A", "A", "A", "A", "B", "B", "B"]),
        ],
        names=["a", "b", "c"],
    )


@pytest.fixture(scope="session")
def con(batch):
    conn = xo.pandas.connect()
    conn.read_record_batches([batch], table_name="test")
    return conn


@pytest.fixture(scope="session")
def df(batch):
    return batch.to_pandas()


def test_multiply(con, df):
    t = con.table("test")
    expr = t.mutate(my_mul=2 * t.a)

    df = df.assign(my_mul=df.a * 2)
    assert_frame_equal(expr.execute(), df)


def test_hash(con, df):
    t = con.table("test")
    expr = t.mutate(my_hash=t.c.hash())

    df = df.assign(my_hash=df.c.map(hash))
    assert_frame_equal(expr.execute(), df)


def test_decimal_multiply(con):
    expr = xo.literal(decimal.Decimal("1.1"), type=dt.Decimal(38, 9)) * 2
    assert con.execute(expr) is not None
