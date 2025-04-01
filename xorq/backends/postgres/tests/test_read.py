import pandas as pd
import pytest

import xorq as xo


def test_read_csv(con):
    name = "iris"
    table_name = f"testing-{name}"
    path = xo.options.pins.get_path(name)
    assert table_name not in con.tables
    t = con.read_csv(path, table_name)
    assert table_name in con.tables
    assert xo.execute(t).equals(pd.read_csv(path))


def test_read_csv_raises(con):
    name = "iris"
    table_name = f"testing-{name}"
    path = xo.options.pins.get_path(name)
    assert table_name not in con.tables
    with pytest.raises(
        ValueError, match="If `table_name` is not provided, `temporary` must be True"
    ):
        con.read_csv(path)
    assert table_name not in con.tables


def test_read_csv_temporary(con):
    name = "iris"
    table_name = f"testing-{name}"
    path = xo.options.pins.get_path(name)
    assert table_name not in con.tables
    t = con.read_csv(path, temporary=True)
    assert t.op().name in con.tables
    assert xo.execute(t).equals(pd.read_csv(path))


def test_read_csv_named_temporary(con):
    name = "iris"
    table_name = f"testing-{name}"
    path = xo.options.pins.get_path(name)
    assert table_name not in con.tables
    t = con.read_csv(path, table_name, temporary=True)
    assert table_name == t.op().name
    assert table_name in con.tables
    assert xo.execute(t).equals(pd.read_csv(path))


def test_read_parquet(con):
    name = "astronauts"
    table_name = f"testing-{name}"
    path = xo.options.pins.get_path(name)
    assert table_name not in con.tables
    t = con.read_parquet(path, table_name)
    assert table_name in con.tables
    assert xo.execute(t).equals(pd.read_parquet(path))


def test_read_parquet_raises(con):
    name = "astronauts"
    table_name = f"testing-{name}"
    path = xo.options.pins.get_path(name)
    assert table_name not in con.tables
    with pytest.raises(
        ValueError, match="If `table_name` is not provided, `temporary` must be True"
    ):
        con.read_parquet(path)
    assert table_name not in con.tables


def test_read_parquet_temporary(con):
    name = "astronauts"
    table_name = f"testing-{name}"
    path = xo.options.pins.get_path(name)
    assert table_name not in con.tables
    t = con.read_parquet(path, temporary=True)
    assert t.op().name in con.tables
    assert xo.execute(t).equals(pd.read_parquet(path))


def test_read_parquet_named_temporary(con):
    name = "astronauts"
    table_name = f"testing-{name}"
    path = xo.options.pins.get_path(name)
    assert table_name not in con.tables
    t = con.read_parquet(path, table_name, temporary=True)
    assert table_name == t.op().name
    assert table_name in con.tables
    assert xo.execute(t).equals(pd.read_parquet(path))
