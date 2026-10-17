import pandas as pd
import pytest
from adbc_driver_manager._lib import OperationalError

import xorq.api as xo


def test_read_csv(pg, iris_path):
    name = "iris"
    table_name = f"testing-{name}"
    assert table_name not in pg.tables
    t = pg.read_csv(iris_path, table_name)
    assert table_name in pg.tables
    assert xo.execute(t).equals(pd.read_csv(iris_path))


def test_read_csv_raises(pg, iris_path):
    name = "iris"
    table_name = f"testing-{name}"
    assert table_name not in pg.tables
    with pytest.raises(
        ValueError, match="If `table_name` is not provided, `temporary` must be True"
    ):
        pg.read_csv(iris_path)
    assert table_name not in pg.tables


def test_read_csv_temporary(pg, iris_path):
    name = "iris"
    table_name = f"testing-{name}"
    assert table_name not in pg.tables
    t = pg.read_csv(iris_path, temporary=True)
    assert t.op().name in pg.tables
    assert xo.execute(t).equals(pd.read_csv(iris_path))


def test_read_csv_named_temporary(pg, iris_path):
    name = "iris"
    table_name = f"testing-{name}"
    assert table_name not in pg.tables
    t = pg.read_csv(iris_path, table_name, temporary=True)
    assert table_name == t.op().name
    assert table_name in pg.tables
    assert xo.execute(t).equals(pd.read_csv(iris_path))


def test_read_parquet(pg, astronauts_parquet_path, astronauts_df):
    table_name = "testing-astronauts"
    assert table_name not in pg.tables
    t = pg.read_parquet(astronauts_parquet_path, table_name)
    assert table_name in pg.tables
    assert xo.execute(t).equals(astronauts_df)


def test_read_parquet_raises(pg, astronauts_parquet_path):
    table_name = "testing-astronauts"
    assert table_name not in pg.tables
    with pytest.raises(
        ValueError, match="If `table_name` is not provided, `temporary` must be True"
    ):
        pg.read_parquet(astronauts_parquet_path)
    assert table_name not in pg.tables


def test_read_parquet_temporary(pg, astronauts_parquet_path, astronauts_df):
    table_name = "testing-astronauts"
    assert table_name not in pg.tables
    t = pg.read_parquet(astronauts_parquet_path, temporary=True)
    assert t.op().name in pg.tables
    assert xo.execute(t).equals(astronauts_df)


def test_read_parquet_named_temporary(pg, astronauts_parquet_path, astronauts_df):
    table_name = "testing-astronauts"
    assert table_name not in pg.tables
    t = pg.read_parquet(astronauts_parquet_path, table_name, temporary=True)
    assert table_name == t.op().name
    assert table_name in pg.tables
    assert xo.execute(t).equals(astronauts_df)


def test_read_csv_multiple_paths(pg, iris_path):
    name = "iris"
    table_name = f"testing-{name}"
    assert table_name not in pg.tables
    t = pg.read_csv([iris_path, iris_path], table_name)
    assert table_name in pg.tables
    assert len(xo.execute(t)) == 2 * len(pd.read_csv(iris_path))


@pytest.mark.parametrize(
    "second",
    [
        pd.DataFrame({"C": [7, 8, 9], "D": [10, 11, 12]}),
        pd.DataFrame({"A": [7, 8, 9], "B": ["a", "b", "c"]}),
    ],
)
def test_read_csv_incompatible_schemas_raises(pg, tmp_path, second):
    first_file_path = tmp_path / "first.csv"
    second_file_path = tmp_path / "second.csv"

    first = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    first.to_csv(first_file_path, index=False)

    second.to_csv(second_file_path, index=False)

    table_name = "test_incompatible_schemas"

    with pytest.raises(OperationalError):
        pg.read_csv([str(first_file_path), str(second_file_path)], table_name)


def test_read_csv_reader_kwarg(pg, tmp_path):
    # a pandas reader option must reach the reader and only the reader
    path = tmp_path / "semicolon.csv"
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})
    df.to_csv(path, index=False, sep=";")
    table_name = "testing-semicolon"
    assert table_name not in pg.tables
    t = pg.read_csv(path, table_name, sep=";")
    assert table_name in pg.tables
    assert xo.execute(t).equals(df)


def test_read_csv_chunksize_reaches_reader(pg, tmp_path):
    path = tmp_path / "five.csv"
    df = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": list("abcde")})
    df.to_csv(path, index=False)
    table_name = "testing-chunked"
    assert table_name not in pg.tables
    # the reader rejects a chunksize below one; before, the value never got there
    with pytest.raises(ValueError, match="chunksize"):
        pg.read_csv(path, table_name, chunksize=0)
    assert table_name not in pg.tables
    t = pg.read_csv(path, table_name, chunksize=2)
    assert xo.execute(t).equals(df)


def test_deferred_read_csv_reader_kwarg(pg, tmp_path):
    # the deferred entry point packs reader options next to its own
    # ingest arguments; they must still reach only the reader
    path = tmp_path / "semicolon-deferred.csv"
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})
    df.to_csv(path, index=False, sep=";")
    table_name = "testing-semicolon-deferred"
    assert table_name not in pg.tables
    # schema inference does not see reader options, so declare the schema
    t = xo.deferred_read_csv(
        path,
        con=pg,
        table_name=table_name,
        schema=xo.schema({"A": "int64", "B": "string"}),
        sep=";",
    )
    assert xo.execute(t).equals(df)
    assert table_name in pg.tables


@pytest.fixture
def ingest_schema(pg):
    name = "testing_ingest"
    pg.create_database(name, force=True)
    yield name
    pg.drop_database(name, force=True, cascade=True)


def write_frame(df, path, method):
    if method == "read_csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path)
    return path


# the same ADBC ingest option must reach the ingest from either reader; before,
# read_parquet took it through **kwargs and read_csv sent it to the csv reader
@pytest.mark.parametrize("method", ["read_csv", "read_parquet"])
def test_ingest_kwargs_reach_the_ingest(pg, ingest_schema, tmp_path, method):
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})
    path = write_frame(df, tmp_path / "ingest", method)
    table_name = "testing-ingest"
    t = getattr(pg, method)(
        path, table_name, ingest_kwargs={"db_schema_name": ingest_schema}
    )
    assert table_name in pg.list_tables(database=ingest_schema)
    assert table_name not in pg.tables
    assert xo.execute(t).equals(df)


# the readers once accepted a password and ran the ingest as the backend's own
# role regardless; the argument is now refused instead of ignored
@pytest.mark.parametrize("method", ["read_csv", "read_parquet"])
def test_readers_reject_password(pg, tmp_path, method):
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})
    path = write_frame(df, tmp_path / "ingest", method)
    table_name = "testing-password"
    with pytest.raises(TypeError, match="password"):
        getattr(pg, method)(path, table_name, password="other")
    assert table_name not in pg.tables
