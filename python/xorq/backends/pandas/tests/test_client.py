from __future__ import annotations

from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from pytest import param

import xorq.api as xo
import xorq.common.exceptions as com
import xorq.vendor.ibis.expr.operations as ops
from xorq.tests.util import assert_frame_equal


@pytest.fixture
def client():
    return xo.pandas.connect(
        {
            "df": pd.DataFrame({"a": [1, 2, 3], "b": list("abc")}),
            "df_unknown": pd.DataFrame({"array_of_strings": [["a", "b"], [], ["c"]]}),
        }
    )


@pytest.fixture
def table(client):
    return client.table("df")


@pytest.fixture
def test_data():
    return pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": list("abcde")})


@pytest.fixture(scope="function")
def pandas_con():
    return xo.pandas.connect()


ZERO_PADDED_CSV = (
    "id,region,seen_at\n001,north,2024-01-01 00:00:00\n002,south,2024-01-02 00:00:00\n"
)
ZERO_PADDED_SCHEMA = {"id": "string", "region": "string", "seen_at": "timestamp"}


@pytest.fixture
def csv_file(tmp_path):
    def write(text, name="data.csv"):
        path = tmp_path / name
        path.write_text(text)
        return path

    return write


@pytest.fixture
def zero_padded_csv(csv_file):
    # ids that type inference turns into integers, and a timestamp column
    # pandas cannot parse through dtype=
    return csv_file(ZERO_PADDED_CSV), xo.schema(ZERO_PADDED_SCHEMA)


def test_connect_no_args():
    con = xo.pandas.connect()
    assert dict(con.tables) == {}


def test_client_table(table):
    assert isinstance(table.op(), ops.DatabaseTable)


@pytest.mark.parametrize(
    "lamduh",
    [(lambda df: df), (lambda df: pa.Table.from_pandas(df))],
    ids=["dataframe", "pyarrow table"],
)
def test_create_table(client, test_data, lamduh):
    test_data = lamduh(test_data)
    client.create_table("testing", obj=test_data)
    assert "testing" in client.list_tables()
    client.create_table("testingschema", schema=client.get_schema("testing"))
    assert "testingschema" in client.list_tables()


def test_create_table_from_memtable(client, test_data):
    client.create_table("from_memtable", obj=xo.memtable(test_data))
    assert_frame_equal(client.table("from_memtable").execute(), test_data)


def test_create_table_rejects_non_memtable_expr(client, table):
    with pytest.raises(com.BackendConversionError):
        client.create_table("from_expr", obj=table.mutate(c=table.a))


def test_create_table_overwrite_replaces_schema(client):
    client.create_table("t", pd.DataFrame({"a": [1]}))
    assert client.table("t").columns == ["a"]
    replaced = client.create_table("t", pd.DataFrame({"b": [2.5]}), overwrite=True)
    assert replaced.columns == ["b"]
    assert replaced.execute().b.tolist() == [2.5]


def test_literal(client):
    lit = xo.literal(1)
    result = client.execute(lit)
    assert result == 1


def test_list_tables(client):
    assert client.list_tables(like="df_unknown")
    assert not client.list_tables(like="not_in_the_database")
    assert client.list_tables()


def test_drop_table_before_schema_is_read():
    con = xo.pandas.connect({"t": pd.DataFrame({"a": [1]})})
    con.drop_table("t", force=True)
    assert con.list_tables() == []
    con.create_table("t", pd.DataFrame({"b": [2]}))
    assert con.table("t").columns == ["b"]


def test_drop(table):
    table = table.mutate(c=table.a)
    expr = table.drop("a")
    result = expr.execute()
    expected = table[["b", "c"]].execute()
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "unit",
    [
        "Y",
        "M",
        "D",
        "h",
        "m",
        "s",
        "ms",
        "us",
        "ns",
        param("ps", marks=pytest.mark.xfail),
        param("fs", marks=pytest.mark.xfail),
        param("as", marks=pytest.mark.xfail),
    ],
)
def test_datetime64_infer(client, unit):
    value = np.datetime64("2018-01-02", unit)
    expr = xo.literal(value, type="timestamp")
    result = client.execute(expr)
    assert result == pd.Timestamp(value).to_pydatetime()


@pytest.mark.parametrize(
    ("text", "schema", "expected"),
    [
        param(
            ZERO_PADDED_CSV,
            ZERO_PADDED_SCHEMA,
            {
                "id": ["001", "002"],
                "seen_at": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
            },
            id="zero-padded-timestamp",
        ),
        # numpy int and bool dtypes refuse missing values; pandas' nullable
        # ones read them, and the declared schema is kept, not re-inferred
        param(
            "n,b,note\n1,true,\n,,\n3,false,\n",
            {"n": "int64", "b": "boolean", "note": "string"},
            {"n": [1, None, 3], "b": [True, None, False], "note": [None] * 3},
            id="missing-values",
        ),
        param(
            "k,gap\na,1 days\nb,\nc,2 days 12:00:00\n",
            {"k": "string", "gap": "interval('s')"},
            {"gap": [pd.Timedelta(days=1), None, pd.Timedelta(days=2, hours=12)]},
            id="interval-text",
        ),
        # bare numbers are counted in the schema's unit, not as nanoseconds
        param(
            "k,gap\na,1\nb,\nc,90\n",
            {"k": "string", "gap": "interval('m')"},
            {"gap": [pd.Timedelta(minutes=1), None, pd.Timedelta(minutes=90)]},
            id="interval-numbers",
        ),
        # numbers and text may share a column: 60 is 60 seconds here, not
        # 60 nanoseconds because another row is text
        param(
            "k,gap\na,60\nb,00:01:00\nc,\n",
            {"k": "string", "gap": "interval('s')"},
            {"gap": [pd.Timedelta(seconds=60), pd.Timedelta(minutes=1), None]},
            id="interval-mixed",
        ),
        # a fraction elsewhere in the column must not either (half a
        # nanosecond itself rounds to zero)
        param(
            "k,gap\na,9007199254740993\nb,0.5\nc,\n",
            {"k": "string", "gap": "interval('ns')"},
            {"gap": [pd.Timedelta(9007199254740993), pd.Timedelta(0), None]},
            id="interval-fraction",
        ),
        # nor may padding demote an integer to a fraction
        param(
            "k,gap\na, 9007199254740993 \nb,\n",
            {"k": "string", "gap": "interval('ns')"},
            {"gap": [pd.Timedelta(9007199254740993), None]},
            id="interval-padded",
        ),
        # only ascii digits are integers; other unicode digits are text and
        # fail as such, not silently null
        param(
            "k,gap\na,\u0663\nb,1\n",
            {"k": "string", "gap": "interval('ns')"},
            ValueError,
            id="interval-unicode-digit",
        ),
        # an integer past int64 must not wrap: pandas reads it as unsigned,
        # and to_timedelta would count it as a negative nanosecond
        param(
            "k,gap\na,18446744073709551615\nb,0.5\n",
            {"k": "string", "gap": "interval('ns')"},
            OverflowError,
            id="interval-overflow",
        ),
        # a missing value must not push the integers through float64
        param(
            "k,gap\na,9007199254740993\nb,\n",
            {"k": "string", "gap": "interval('ns')"},
            {"gap": [pd.Timedelta(9007199254740993), None]},
            id="interval-precision",
        ),
        # dates and decimals are read as text, or pandas guesses numbers:
        # 20240101 is not seconds since the epoch and 1.00 is not a float
        param(
            "d,amount\n20240101,1.00\n20240102,2.50\n",
            {"d": "date", "amount": "decimal(10, 2)"},
            {
                "d": [date(2024, 1, 1), date(2024, 1, 2)],
                "amount": [Decimal("1.00"), Decimal("2.50")],
            },
            id="date-decimal",
        ),
    ],
)
def test_read_csv_schema(csv_file, pandas_con, text, schema, expected):
    schema = xo.schema(schema)
    if isinstance(expected, type):
        with pytest.raises(expected):
            pandas_con.read_csv(csv_file(text), schema=schema)
        return
    t = pandas_con.read_csv(csv_file(text), schema=schema)
    assert t.schema() == schema
    df = t.execute()
    for name, values in expected.items():
        assert [None if pd.isna(v) else v for v in df[name]] == values


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        param({"dtype": {"id": str}}, "dtype", id="dtype"),
        param({"converters": {"id": str.strip}}, "converters", id="converters"),
        param({"parse_dates": ["seen_at"]}, "parse_dates", id="parse-dates"),
        param({"engine": "pyarrow"}, "pyarrow", id="pyarrow-engine"),
    ],
)
def test_read_csv_schema_rejects_parse_overrides(
    zero_padded_csv, pandas_con, kwargs, match
):
    # the schema decides how each column is parsed; pandas ignores dtype for
    # converted and date-parsed columns and applies it after inference under
    # the pyarrow engine, and none of that could be reconciled after the read
    path, schema = zero_padded_csv
    with pytest.raises(TypeError, match=match):
        pandas_con.read_csv(path, schema=schema, **kwargs)


def test_read_csv_schema_rejects_nested(csv_file, pandas_con):
    path = csv_file("xs\n[1,2]\n")
    with pytest.raises(TypeError, match="nested"):
        pandas_con.read_csv(path, schema=xo.schema({"xs": "array<int64>"}))


def test_read_csv_without_schema_reinfers(zero_padded_csv, csv_file, pandas_con):
    # a reread under the same name without a schema drops the cached one
    path, schema = zero_padded_csv
    other = csv_file("x,y\n1,2\n", name="other.csv")
    pandas_con.read_csv(path, "t", schema=schema)
    t = pandas_con.read_csv(other, "t")
    assert t.schema() == xo.schema({"x": "int64", "y": "int64"})
    assert t.execute().to_dict("list") == {"x": [1], "y": [2]}


def test_read_csv_failed_reread_keeps_table(csv_file, pandas_con):
    # a reread that fails validation leaves the existing table, and its
    # cached schema, as they were
    path = csv_file("id,note\n001,\n")
    schema = xo.schema({"id": "string", "note": "string"})
    pandas_con.read_csv(path, "t", schema=schema)
    with pytest.raises(ValueError, match="missing from schema"):
        pandas_con.read_csv(path, "t", schema=xo.schema({"id": "string"}))
    assert pandas_con.table("t").schema() == schema
    assert pandas_con.table("t").execute()["id"].tolist() == ["001"]


def test_read_csv_schema_replaces_cached_schema(zero_padded_csv, pandas_con):
    path, schema = zero_padded_csv
    assert pandas_con.read_csv(path, "padded").schema()["id"].is_integer()
    t = pandas_con.read_csv(path, "padded", schema=schema)
    assert t.schema() == schema
    assert t.execute()["id"].tolist() == ["001", "002"]


def test_read_csv_schema_usecols(zero_padded_csv, pandas_con):
    path, schema = zero_padded_csv
    t = pandas_con.read_csv(path, schema=schema, usecols=["region"])
    assert t.schema() == xo.schema({"region": "string"})
    with pytest.raises(ValueError, match="missing from schema"):
        pandas_con.read_csv(path, schema=xo.schema({"id": "string"}))


def test_deferred_read_csv_schema(zero_padded_csv, pandas_con):
    path, schema = zero_padded_csv
    expr = xo.deferred_read_csv(
        path, con=pandas_con, schema=schema, table_name="padded"
    )
    assert expr.execute()["id"].tolist() == ["001", "002"]
    assert pandas_con.table("padded").schema() == schema


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        param(ops.StringLength, True, id="kernel-op"),
        param(ops.Levenshtein, False, id="unhandled-op"),
    ],
)
def test_has_operation(op, expected):
    assert xo.pandas.has_operation(op) is expected
