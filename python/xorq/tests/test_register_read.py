from __future__ import annotations

import gzip
from urllib.request import Request, urlopen

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pytest

import xorq.api as xo
from xorq.caching import ParquetCache
from xorq.common.utils.defer_utils import infer_csv_schema_pandas
from xorq.ibis_yaml.enums import ExprKind
from xorq.tests.util import assert_frame_equal
from xorq.vendor.ibis import Schema
from xorq.vendor.ibis.common.collections import FrozenDict
from xorq.vendor.ibis.expr.types.core import ExprMetadata


@pytest.fixture
def gzip_csv(data_dir, tmp_path):
    basename = "diamonds.csv"
    f = tmp_path.joinpath(f"{basename}.gz")
    data = data_dir.joinpath("csv", basename).read_bytes()
    f.write_bytes(gzip.compress(data))
    return str(f.absolute())


@pytest.fixture
def ctx(con):
    return con.con


def test_register_csv(con, data_dir):
    fname = "diamonds.csv"
    table_name = "diamonds"
    table = con.read_csv(data_dir / "csv" / fname, table_name=table_name)
    assert any(table_name in t for t in con.list_tables())
    assert table.count().execute() > 0


@pytest.mark.xfail
def test_register_csv_gz(con, data_dir, gzip_csv):
    # TODO review why there is a regression
    table = con.read_csv(gzip_csv, table_name="diamonds")
    assert table.count().execute() > 0


def test_register_with_dotted_name(con, data_dir, tmp_path):
    basename = "foo.bar.baz/diamonds.csv"
    f = tmp_path.joinpath(basename)
    f.parent.mkdir()
    data = data_dir.joinpath("csv", "diamonds.csv").read_bytes()
    f.write_bytes(data)
    table = con.read_csv(str(f.absolute()), table_name="diamonds")
    assert table.count().execute() > 0


def test_register_parquet(con, data_dir):
    fname = "functional_alltypes.parquet"
    table_name = "funk_all"
    table = con.read_parquet(data_dir / "parquet" / fname, table_name=table_name)

    assert any(table_name in t for t in con.list_tables())
    assert table.count().execute() > 0


def test_read_csv(con, data_dir):
    t = con.read_csv(data_dir / "csv" / "functional_alltypes.csv")
    assert t.count().execute()


def test_read_csv_from_url(con):
    t = con.read_csv(
        "https://raw.githubusercontent.com/ibis-project/testing-data/refs/heads/master/csv/astronauts.csv"
    )
    assert t.head().execute() is not None


@pytest.mark.s3
def test_read_csv_from_s3(con):
    schema = pa.schema(
        [
            pa.field("question", pa.string()),
            pa.field("product_description", pa.string()),
            pa.field("image_url", pa.string()),
            pa.field("label", pa.uint8(), nullable=True),
        ]
    )

    t = con.read_csv(
        "s3://humor-detection-pds/Humorous.csv",
        schema=schema,
        storage_options={
            "aws.region": "us-west-2",
        },
    )
    assert t.head().execute() is not None


@pytest.mark.s3
def test_kind_deferred_read_is_source():
    expr = xo.deferred_read_parquet(
        "s3://bucket/data.parquet",
        schema={"a": "int64", "b": "string"},
    )
    assert ExprMetadata.from_expr(expr).kind == ExprKind.Source


def test_read_parquet(con, data_dir):
    t = con.read_parquet(data_dir / "parquet" / "functional_alltypes.parquet")
    assert t.count().execute()


def test_read_parquet_from_url(con):
    t = con.read_parquet(
        "https://nasa-avionics-data-ml.s3.us-east-2.amazonaws.com/Tail_652_1_parquet/652200101120916.16p0.parquet",
        table_name="nasa_table",
    )

    assert "nasa_table" in con.list_tables()
    assert t.op().name == "nasa_table"
    assert t.head().execute() is not None


def test_register_table(con):
    tab = pa.table({"x": [1, 2, 3]})
    con.create_table("my_table", tab)
    assert con.table("my_table").x.sum().execute() == 6


def test_register_pandas(con):
    df = pd.DataFrame({"x": [1, 2, 3]})
    con.create_table("my_pandas_table", df)
    assert con.table("my_pandas_table").x.sum().execute() == 6


def test_register_batches(con):
    batch = pa.record_batch([pa.array([1, 2, 3])], names=["x"])
    con.create_table("my_batch_table", batch)
    assert con.table("my_batch_table").x.sum().execute() == 6


def test_register_dataset(con):
    tab = pa.table({"x": [1, 2, 3]})
    dataset = ds.InMemoryDataset(tab)
    t = xo.memtable(dataset, name="my_table")
    assert con.execute(t.x.sum()) == 6


def test_register_memtable(con):
    data = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 3, 4, 5, 6]})
    t = xo.memtable(data)
    assert con.execute(t.a.sum()) == 15


@pytest.mark.gcs
def test_deferred_read_parquet_from_gcs(tmp_path):
    con = xo.connect()
    path = "gs://cloud-samples-data/bigquery/us-states/us-states.parquet"
    expr = (
        xo.deferred_read_parquet(path, con)
        .cache(
            cache=ParquetCache.from_kwargs(
                source=xo.duckdb.connect(), relative_path=tmp_path
            )
        )
        .limit(10)
    )

    assert not expr.execute().empty
    assert any(tmp_path.iterdir())


@pytest.mark.s3
def test_read_csv_from_s3_and_cache(tmp_path):
    con = xo.connect()

    # by providing the schema we avoid calling pandas
    schema = Schema.from_pyarrow(
        pa.schema(
            [
                pa.field("question", pa.string()),
                pa.field("product_description", pa.string()),
                pa.field("image_url", pa.string()),
                pa.field("label", pa.uint8(), nullable=True),
            ]
        )
    )

    path = "s3://humor-detection-pds/Humorous.csv"
    t = xo.deferred_read_csv(
        path,
        con,
        schema=schema,
        storage_options=FrozenDict(
            {
                "aws.region": "us-west-2",
            }
        ),
    )

    expr = t.cache(
        cache=ParquetCache.from_kwargs(
            source=xo.duckdb.connect(), relative_path=tmp_path
        )
    ).limit(10)

    assert not expr.execute().empty
    assert any(tmp_path.iterdir())


def test_get_object_metadata_local_filesystem(data_dir):
    url = data_dir / "parquet" / "functional_alltypes.parquet"
    metadata = xo.get_object_metadata(str(url.resolve()))

    assert isinstance(metadata, dict)


def test_get_object_metadata_https(ctx):
    url = "https://raw.githubusercontent.com/ibis-project/testing-data/refs/heads/master/csv/astronauts.csv"

    metadata = xo.get_object_metadata(url)
    assert isinstance(metadata, dict)

    request = Request(url, method="GET")
    with urlopen(request) as response:
        etag = response.headers.get("ETag")
        assert etag == metadata["e_tag"]


@pytest.mark.s3
def test_get_object_metadata_s3():
    metadata = xo.get_object_metadata(
        "s3://humor-detection-pds/Humorous.csv",
        storage_options={
            "aws.region": "us-west-2",
        },
    )

    assert isinstance(metadata, dict)


@pytest.mark.gcs
def test_get_object_metadata_gcs():
    metadata = xo.get_object_metadata(
        "gs://cloud-samples-data/bigquery/us-states/us-states.parquet",
    )
    assert isinstance(metadata, dict)


@pytest.mark.parametrize(
    "get_con",
    [
        lambda: xo.connect(),
        lambda: xo.datafusion.connect(),
        lambda: xo.duckdb.connect(),
        pytest.param(lambda: xo.postgres.connect_env(), marks=pytest.mark.postgres),
    ],
)
def test_expr_over_same_table_multiple_times(parquet_dir, get_con):
    ls_con = xo.connect()
    other_con = get_con()
    astronauts_path = parquet_dir.joinpath("astronauts.parquet")
    table_name = "astronauts"
    col = "id"

    astronauts_name = f"{ls_con.name}_{table_name}"
    astronauts = ls_con.read_parquet(astronauts_path, table_name=astronauts_name)

    if other_con.name == "postgres":
        t = other_con.table(table_name)
    else:
        t = other_con.read_parquet(astronauts_path, table_name=table_name)

    ls_table_name = f"{other_con.name}_{table_name}"
    other = t.into_backend(ls_con, ls_table_name)

    expr = astronauts[[col]].distinct().join(other[[col]].distinct(), col)

    assert (first := expr.execute()) is not None
    assert (second := expr.execute()) is not None
    assert_frame_equal(first.sort_values(col), second.sort_values(col))


@pytest.mark.parametrize(
    "get_con,read_kwargs",
    [
        pytest.param(lambda: xo.duckdb.connect(), {"sep": ";"}, id="duckdb"),
        pytest.param(lambda: xo.connect(), {"delimiter": ";"}, id="datafusion"),
        pytest.param(lambda: xo.pandas.connect(), {"sep": ";"}, id="pandas"),
        pytest.param(
            lambda: xo.postgres.connect_env(),
            {"sep": ";"},
            id="postgres",
            marks=pytest.mark.postgres,
        ),
    ],
)
def test_deferred_read_csv_kwargs(get_con, read_kwargs, tmp_path):
    """The reader kwargs reach schema inference, in the backend's own spelling."""
    path = tmp_path / "semicolon.csv"
    path.write_text("a;b;c\n1;x;2.5\n2;y;3.5\n")
    expr = xo.deferred_read_csv(path, get_con(), **read_kwargs)
    assert expr.schema() == xo.schema({"a": "int64", "b": "string", "c": "float64"})
    assert_frame_equal(
        expr.execute(),
        pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [2.5, 3.5]}),
    )


@pytest.mark.parametrize(
    "get_con,read_kwargs",
    [
        pytest.param(
            lambda: xo.duckdb.connect(), {"delim": "|", "header": False}, id="duckdb"
        ),
        pytest.param(
            lambda: xo.connect(),
            {"delimiter": "|", "has_header": False},
            id="datafusion",
        ),
    ],
)
def test_deferred_read_csv_headerless(get_con, read_kwargs, tmp_path):
    """The issue's file: one pipe-delimited row and no header row."""
    path = tmp_path / "Acquisition_2000Q1.txt"
    path.write_text("100007365142|R|JPMORGAN CHASE BANK|8|75000")
    df = xo.deferred_read_csv(path, get_con(), **read_kwargs).execute()
    assert df.values.tolist() == [[100007365142, "R", "JPMORGAN CHASE BANK", 8, 75000]]


def test_deferred_read_csv_custom_inference_is_called_with_the_path(tmp_path):
    """A custom ``infer_schema=`` reads the file its own way; the reader kwargs
    still reach the backend."""
    path = tmp_path / "semicolon.csv"
    path.write_text("a;b\n1;x\n2;y\n")
    declared = xo.schema({"a": "int64", "b": "string"})
    expr = xo.deferred_read_csv(
        path, xo.duckdb.connect(), infer_schema=lambda path: declared, sep=";"
    )
    assert expr.schema() == declared
    assert_frame_equal(expr.execute(), pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))


def test_infer_csv_schema_pandas_takes_pandas_kwargs(tmp_path):
    path = tmp_path / "comma.csv"
    path.write_text("a,b\n1,x\n")
    assert infer_csv_schema_pandas(path, usecols=["b"]) == xo.schema({"b": "string"})


def test_deferred_read_csv_names(tmp_path):
    """``names`` reaches the inference with the header option it goes with."""
    path = tmp_path / "headerless.csv"
    path.write_text("1,x\n2,y\n")
    expr = xo.deferred_read_csv(
        path, xo.pandas.connect(), header=None, names=("a", "b")
    )
    assert_frame_equal(expr.execute(), pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}))
