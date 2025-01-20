from __future__ import annotations

import gzip

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pytest

import letsql as ls


@pytest.fixture
def gzip_csv(data_dir, tmp_path):
    basename = "diamonds.csv"
    f = tmp_path.joinpath(f"{basename}.gz")
    data = data_dir.joinpath("csv", basename).read_bytes()
    f.write_bytes(gzip.compress(data))
    return str(f.absolute())


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
        "https://opendata-downloads.s3.amazonaws.com/opa_properties_public.csv"
    )
    assert t.head().execute() is not None


@pytest.mark.s3
def test_read_csv_from_s3(con):
    t = con.read_csv(
        "s3://opendata-downloads/opa_properties_public.csv",
    )
    assert t.head().execute() is not None


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
    t = ls.memtable(dataset, name="my_table")
    assert con.execute(t.x.sum()) == 6


def test_register_memtable(con):
    data = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 3, 4, 5, 6]})
    t = ls.memtable(data)
    assert con.execute(t.a.sum()) == 15
