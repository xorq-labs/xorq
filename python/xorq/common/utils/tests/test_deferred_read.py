import functools
import itertools
import pathlib

import dask
import duckdb
import pandas as pd
import pytest
from adbc_driver_manager import ProgrammingError
from attr import (
    field,
    frozen,
)
from attr.validators import (
    in_,
    optional,
)

import xorq as xo
from xorq.caching import (
    ParquetStorage,
)
from xorq.common.utils.defer_utils import (
    deferred_read_csv,
    deferred_read_parquet,
)
from xorq.common.utils.inspect_utils import (
    get_partial_arguments,
)
from xorq.tests.util import assert_frame_equal


@frozen
class PinsResource:
    name = field(validator=in_(xo.options.pins.get_board().pin_list()))
    suffix = field(validator=optional(in_((".csv", ".parquet"))), default=None)

    def __attrs_post_init__(self):
        if self.suffix is None:
            object.__setattr__(self, "suffix", self.path.suffix)
        if self.path.suffix != self.suffix:
            raise ValueError

    @property
    def table_name(self):
        return f"test-{self.name}"

    @property
    @functools.cache
    def path(self):
        return pathlib.Path(xo.options.pins.get_path(self.name))

    def get_underlying_method(self, con):
        return getattr(con, self.deferred_reader.method_name)

    @property
    def deferred_reader(self):
        match self.suffix:
            case ".parquet":
                return deferred_read_parquet
            case ".csv":
                return deferred_read_csv
            case _:
                raise ValueError

    @property
    def immediate_reader(self):
        match self.suffix:
            case ".parquet":
                return pd.read_parquet
            case ".csv":
                return pd.read_csv
            case _:
                raise ValueError

    @property
    @functools.cache
    def df(self):
        return self.immediate_reader(self.path)


@pytest.fixture(scope="session")
def iris_csv():
    return PinsResource(name="iris", suffix=".csv")


@pytest.fixture(scope="session")
def astronauts_parquet():
    return PinsResource(name="astronauts", suffix=".parquet")


def filter_sepal_length(t):
    return t.sepal_length > 5


def filter_field21(t):
    return t.field21 > 2


def ensure_tmp_csv(csv_name, tmp_path):
    source_path = pathlib.Path(xo.options.pins.get_path(csv_name))
    target_path = tmp_path.joinpath(source_path.name)
    if not target_path.exists():
        target_path.write_text(source_path.read_text())
    return target_path


def mutate_csv(path, line=None):
    if line is None:
        line = path.read_text().strip().rsplit("\n", 1)[-1]
    with path.open("at") as fh:
        fh.writelines([line])


@pytest.mark.parametrize("pins_resource", ("iris_csv", "astronauts_parquet"))
def test_deferred_read_cache_key_check(con, tmp_path, pins_resource, request):
    # check that we don't invoke read when we calc key
    pins_resource = request.getfixturevalue(pins_resource)
    storage = ParquetStorage(source=xo.connect(), relative_path=tmp_path)

    assert pins_resource.table_name not in con.tables
    t = pins_resource.deferred_reader(con, pins_resource.path, pins_resource.table_name)
    storage.get_key(t)
    assert pins_resource.table_name not in con.tables


@pytest.mark.parametrize("pins_resource", ("iris_csv", "astronauts_parquet"))
def test_deferred_read_to_sql(con, pins_resource, request):
    # check that we don't invoke read when we convert to sql
    pins_resource = request.getfixturevalue(pins_resource)
    assert pins_resource.table_name not in con.tables
    t = pins_resource.deferred_reader(con, pins_resource.path, pins_resource.table_name)
    xo.to_sql(t)
    assert pins_resource.table_name not in con.tables


@pytest.mark.parametrize(
    "con,pins_resource",
    itertools.product(
        (xo.pandas.connect(), xo.postgres.connect_env()),
        ("iris_csv", "astronauts_parquet"),
    ),
)
def test_deferred_read(con, pins_resource, request):
    pins_resource = request.getfixturevalue(pins_resource)
    assert pins_resource.table_name not in con.tables
    t = pins_resource.deferred_reader(con, pins_resource.path, pins_resource.table_name)
    assert xo.execute(t).equals(pins_resource.df)
    assert pins_resource.table_name in con.tables
    # is this a test of mode for postgres?
    if con.name != "pandas":
        # verify that we can't execute again (pandas happily clobbers)
        with pytest.raises(
            ProgrammingError,
            match=f'relation "{pins_resource.table_name}" already exists',
        ):
            assert xo.execute(t).equals(pins_resource.df)
    con.drop_table(pins_resource.table_name, force=True)
    assert pins_resource.table_name not in tuple(con.tables)


@pytest.mark.parametrize(
    "con,pins_resource",
    itertools.product(
        (xo.postgres.connect_env(),),
        ("iris_csv", "astronauts_parquet"),
    ),
)
def test_deferred_read_temporary(con, pins_resource, request):
    pins_resource = request.getfixturevalue(pins_resource)
    t = pins_resource.deferred_reader(con, pins_resource.path, None, temporary=True)
    table_name = t.op().name
    assert xo.execute(t).equals(pins_resource.df)
    assert table_name in con.tables
    con.drop_table(table_name)
    assert table_name not in con.tables


@pytest.mark.parametrize(
    "con,pins_resource,filter_",
    (
        (con, pins_resource, filter_)
        for con in (xo.pandas.connect(), xo.postgres.connect_env(), xo.duckdb.connect())
        for (pins_resource, filter_) in (
            ("iris_csv", filter_sepal_length),
            ("astronauts_parquet", filter_field21),
        )
    ),
)
def test_cached_deferred_read(con, pins_resource, filter_, request, tmp_path):
    pins_resource = request.getfixturevalue(pins_resource)
    storage = ParquetStorage(source=xo.connect(), relative_path=tmp_path)

    df = pins_resource.df[filter_].reset_index(drop=True)
    t = pins_resource.deferred_reader(con, pins_resource.path, pins_resource.table_name)
    expr = t[filter_].cache(storage=storage)

    # no work is done yet
    assert pins_resource.table_name not in con.tables
    assert not storage.exists(expr)

    # something exists in both con and storage
    assert xo.execute(expr).equals(df)
    assert pins_resource.table_name in con.tables
    assert storage.exists(expr)

    # we read from cache even if the table disappears
    try:
        con.drop_table(t.op().name, force=True)
    except duckdb.duckdb.CatalogException:
        con.drop_view(t.op().name)

    assert xo.execute(expr).equals(df)
    assert pins_resource.table_name not in con.tables

    # we repopulate the cache
    storage.drop(expr)
    assert xo.execute(expr).equals(df)
    assert pins_resource.table_name in con.tables
    assert storage.exists(expr)

    if con.name == "postgres":
        # we are mode="create" by default, which means losing cache creates collision
        mode = get_partial_arguments(pins_resource.get_underlying_method(con))["mode"]
        assert mode == "create"
        storage.drop(expr)
        with pytest.raises(
            ProgrammingError,
            match=f'relation "{pins_resource.table_name}" already exists',
        ):
            xo.execute(expr)

        # with mode="replace" we can clobber
        t = pins_resource.deferred_reader(
            con, pins_resource.path, pins_resource.table_name, mode="replace"
        )
        expr = t[filter_].cache(storage=storage)
        assert xo.execute(expr).equals(df)
        assert storage.exists(expr)
        assert pins_resource.table_name in con.tables
        # this fails above, but works here because of mode="replace"
        storage.drop(expr)
        assert xo.execute(expr).equals(df)


@pytest.mark.parametrize(
    "con",
    (xo.pandas.connect(), xo.postgres.connect_env()),
)
def test_cached_csv_mutate(con, iris_csv, tmp_path):
    target_path = ensure_tmp_csv(iris_csv.name, tmp_path)
    storage = ParquetStorage(source=xo.connect(), relative_path=tmp_path)
    # make sure the con is "clean"
    if iris_csv.table_name in con.tables:
        con.drop_table(iris_csv.table_name, force=True)

    df = iris_csv.df
    kwargs = {"mode": "replace"} if con.name == "postgres" else {}
    t = iris_csv.deferred_reader(con, target_path, iris_csv.table_name, **kwargs)
    expr = t.cache(storage=storage)

    # nothing exists yet
    assert iris_csv.table_name not in con.tables
    assert not storage.exists(expr)

    # initial cache population
    assert xo.execute(expr).equals(df)
    assert iris_csv.table_name in con.tables
    assert storage.exists(expr)

    # mutate
    mutate_csv(target_path)
    df = iris_csv.immediate_reader(target_path)
    assert not storage.exists(expr)
    assert xo.execute(expr).equals(df)
    assert storage.exists(expr)


@pytest.mark.parametrize(
    "method_name,path",
    [
        (
            "deferred_read_csv",
            "https://raw.githubusercontent.com/ibis-project/testing-data/refs/heads/master/csv/astronauts.csv",
        ),
        (
            "deferred_read_parquet",
            "https://nasa-avionics-data-ml.s3.us-east-2.amazonaws.com/Tail_652_1_parquet/652200101120916.16p0.parquet",
        ),
    ],
)
@pytest.mark.parametrize(
    "remote",
    [True, False],
)
def test_deferred_read_cache(con, tmp_path, method_name, path, remote):
    storage = ParquetStorage(source=xo.connect(), relative_path=tmp_path)
    read_method = getattr(xo, method_name)
    connection = con if remote else xo.duckdb.connect()

    t = read_method(connection, path)
    uncached = t.head(10)
    assert storage.get_key(uncached) is not None

    expr = uncached.cache(storage=storage)
    assert not expr.execute().empty


def test_deferred_read_kwargs(pg):
    name = "iris"
    read0, read1 = (
        xo.examples.get_table_from_name(name, pg, mode=mode)
        for mode in ("create", "replace")
    )
    hash0, hash1 = (dask.base.tokenize(expr) for expr in (read0, read1))
    assert hash0 != hash1


def test_deferred_read_parquet_multiple_paths():
    path = xo.config.options.pins.get_path("lending-club")
    expr = deferred_read_parquet(xo.connect(), (path, path))
    assert not expr.execute().empty


def test_deferred_read_csv_multiple_paths():
    path = xo.config.options.pins.get_path("iris")
    con = xo.connect()

    t = con.read_csv(path, table_name="iris")

    expr = deferred_read_csv(con, (path, path), schema=t.schema())

    assert not expr.execute().empty


@pytest.fixture(scope="function")
def backend(request, con):
    lookup = {"duckdb": xo.duckdb.connect(), "postgres": con, "xorq": xo.connect()}

    return lookup.get(request.param, con)


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("duckdb", id="duckdb"),
        pytest.param("postgres", id="postgres"),
        pytest.param("xorq", id="xorq"),
    ],
    indirect=True,
)
def test_register_csv_with_glob_string(data_dir, backend):
    table_name = f"{backend.name}_astronauts"
    glob_pattern = str(data_dir / "csv" / "*astronauts.csv")
    expected = backend.read_csv(
        glob_pattern, table_name=f"{table_name}_expected"
    ).execute()

    read = xo.deferred_read_csv(backend, glob_pattern, table_name=table_name)
    actual = read.execute()  # triggers the table creation

    assert any(table_name in t for t in backend.list_tables())
    assert_frame_equal(expected, actual)


def test_register_empty_glob_pattern_fails(data_dir, con):
    glob_pattern = str(data_dir / "csv" / "*foo.csv")

    with pytest.raises(ValueError, match="At least one path is required"):
        xo.deferred_read_csv(con, glob_pattern, table_name="foo")
