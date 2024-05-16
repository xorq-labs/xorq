from __future__ import annotations

import gzip

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pytest


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
    table = con.register(data_dir / "csv" / fname, table_name=table_name)
    assert any(table_name in t for t in con.list_tables())
    assert table.count().execute() > 0


@pytest.mark.xfail
def test_register_csv_gz(con, data_dir, gzip_csv):
    # TODO review why there is a regression
    table = con.register(gzip_csv, table_name="diamonds")
    assert table.count().execute() > 0


def test_register_with_dotted_name(con, data_dir, tmp_path):
    basename = "foo.bar.baz/diamonds.csv"
    f = tmp_path.joinpath(basename)
    f.parent.mkdir()
    data = data_dir.joinpath("csv", "diamonds.csv").read_bytes()
    f.write_bytes(data)
    table = con.register(str(f.absolute()), table_name="diamonds")
    assert table.count().execute() > 0


def test_register_parquet(con, data_dir):
    fname = "functional_alltypes.parquet"
    table_name = "funk_all"
    table = con.register(data_dir / "parquet" / fname, table_name=table_name)

    assert any(table_name in t for t in con.list_tables())
    assert table.count().execute() > 0


def test_read_csv(con, data_dir):
    t = con.read_csv(data_dir / "csv" / "functional_alltypes.csv")
    assert t.count().execute()


def test_read_parquet(con, data_dir):
    t = con.read_parquet(data_dir / "parquet" / "functional_alltypes.parquet")
    assert t.count().execute()


def test_register_table(con):
    tab = pa.table({"x": [1, 2, 3]})
    con.register(tab, "my_table")
    assert con.table("my_table").x.sum().execute() == 6


def test_register_pandas(con):
    df = pd.DataFrame({"x": [1, 2, 3]})
    con.register(df, "my_table")
    assert con.table("my_table").x.sum().execute() == 6


def test_register_batches(con):
    batch = pa.record_batch([pa.array([1, 2, 3])], names=["x"])
    con.register(batch, "my_table")
    assert con.table("my_table").x.sum().execute() == 6


def test_register_dataset(con):
    tab = pa.table({"x": [1, 2, 3]})
    dataset = ds.InMemoryDataset(tab)
    con.register(dataset, "my_table")
    assert con.table("my_table").x.sum().execute() == 6
