import pandas as pd
import pytest
from adbc_driver_manager._lib import OperationalError

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


def test_read_csv_multiple_paths(con):
    name = "iris"
    table_name = f"testing-{name}"
    path = xo.options.pins.get_path(name)
    assert table_name not in con.tables
    t = con.read_csv([path, path], table_name)
    assert table_name in con.tables
    assert len(xo.execute(t)) == 2 * len(pd.read_csv(path))


@pytest.mark.parametrize(
    "second",
    [
        pd.DataFrame({"C": [7, 8, 9], "D": [10, 11, 12]}),
        pd.DataFrame({"A": [7, 8, 9], "B": ["a", "b", "c"]}),
    ],
)
def test_read_csv_incompatible_schemas_raises(con, tmp_path, second):
    first_file_path = tmp_path / "first.csv"
    second_file_path = tmp_path / "second.csv"

    first = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    first.to_csv(first_file_path, index=False)

    second.to_csv(second_file_path, index=False)

    table_name = "test_incompatible_schemas"

    with pytest.raises(OperationalError):
        con.read_csv([str(first_file_path), str(second_file_path)], table_name)
