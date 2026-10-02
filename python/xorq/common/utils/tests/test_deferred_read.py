import functools
import itertools
import pathlib

import pandas as pd
import pytest
from attr import (
    field,
    frozen,
)
from attr.validators import (
    in_,
    optional,
)

import xorq.api as xo
from xorq.caching import (
    ParquetCache,
)
from xorq.common.utils.dasher import tokenize
from xorq.common.utils.defer_utils import (
    deferred_read_csv,
    deferred_read_parquet,
)
from xorq.common.utils.inspect_utils import (
    get_partial_arguments,
)
from xorq.ibis_yaml.compiler import (
    build_expr,
    load_expr,
)
from xorq.tests.util import assert_frame_equal


duckdb = pytest.importorskip("duckdb")
ProgrammingError = pytest.importorskip("adbc_driver_manager").ProgrammingError


@frozen
class PinsResource:
    name = field()
    suffix = field(validator=optional(in_((".csv", ".parquet"))), default=None)

    def __attrs_post_init__(self):
        if self.suffix is None:
            object.__setattr__(self, "suffix", self.path.suffix)
        if self.path.suffix != self.suffix:
            raise ValueError(
                f"path suffix {self.path.suffix!r} does not match expected suffix {self.suffix!r}"
            )
        if self.name not in xo.options.pins.get_board().pin_list():
            raise ValueError(f"name {self.name!r} not found in pins board")

    @property
    def table_name(self):
        return f"test-{self.name}"

    @functools.cached_property
    def path(self):
        return pathlib.Path(xo.options.pins.get_path(self.name))

    @property
    def method_name(self):
        match self.suffix:
            case ".parquet":
                return "read_parquet"
            case ".csv":
                return "read_csv"
            case _:
                raise ValueError(f"unsupported suffix {self.suffix!r}")

    def get_underlying_method(self, con):
        return getattr(con, self.method_name)

    @property
    def deferred_reader(self):
        match self.suffix:
            case ".parquet":
                return deferred_read_parquet
            case ".csv":
                return deferred_read_csv
            case _:
                raise ValueError(f"unsupported suffix {self.suffix!r}")

    @property
    def immediate_reader(self):
        match self.suffix:
            case ".parquet":
                return pd.read_parquet
            case ".csv":
                return pd.read_csv
            case _:
                raise ValueError(f"unsupported suffix {self.suffix!r}")

    @functools.cached_property
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
    cache = ParquetCache.from_kwargs(source=xo.connect(), relative_path=tmp_path)

    assert pins_resource.table_name not in con.tables
    t = pins_resource.deferred_reader(pins_resource.path, con, pins_resource.table_name)
    cache.strategy.calc_key(t)
    assert pins_resource.table_name not in con.tables


@pytest.mark.parametrize("pins_resource", ("iris_csv", "astronauts_parquet"))
def test_deferred_read_to_sql(con, pins_resource, request):
    # check that we don't invoke read when we convert to sql
    pins_resource = request.getfixturevalue(pins_resource)
    assert pins_resource.table_name not in con.tables
    t = pins_resource.deferred_reader(pins_resource.path, con, pins_resource.table_name)
    xo.to_sql(t)
    assert pins_resource.table_name not in con.tables


@pytest.mark.parametrize(
    "get_con,pins_resource",
    [
        pytest.param(con, pins_resource, marks=marks)
        for con, marks in [
            (lambda: xo.pandas.connect(), []),
            (lambda: xo.postgres.connect_env(), [pytest.mark.postgres]),
        ]
        for pins_resource in ("iris_csv", "astronauts_parquet")
    ],
)
def test_deferred_read(get_con, pins_resource, request):
    con = get_con()
    pins_resource = request.getfixturevalue(pins_resource)
    assert pins_resource.table_name not in con.tables
    kwargs = {"mode": "create"} if con.name != "pandas" else {}
    t = pins_resource.deferred_reader(
        pins_resource.path, con, pins_resource.table_name, **kwargs
    )
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


@pytest.mark.postgres
@pytest.mark.parametrize(
    "get_con,pins_resource",
    itertools.product(
        (lambda: xo.postgres.connect_env(),),
        ("iris_csv", "astronauts_parquet"),
    ),
)
def test_deferred_read_temporary(get_con, pins_resource, request):
    con = get_con()
    pins_resource = request.getfixturevalue(pins_resource)
    t = pins_resource.deferred_reader(pins_resource.path, con, None, temporary=True)
    table_name = t.op().name
    assert xo.execute(t).equals(pins_resource.df)
    assert table_name in con.tables
    con.drop_table(table_name)
    assert table_name not in con.tables


@pytest.mark.parametrize(
    "get_con,pins_resource,filter_",
    [
        pytest.param(con, pins_resource, filter_, marks=marks)
        for con, marks in [
            (lambda: xo.pandas.connect(), []),
            (lambda: xo.postgres.connect_env(), [pytest.mark.postgres]),
            (lambda: xo.duckdb.connect(), []),
        ]
        for (pins_resource, filter_) in (
            ("iris_csv", filter_sepal_length),
            ("astronauts_parquet", filter_field21),
        )
    ],
)
def test_cached_deferred_read(get_con, pins_resource, filter_, request, tmp_path):
    con = get_con()
    pins_resource = request.getfixturevalue(pins_resource)
    cache = ParquetCache.from_kwargs(source=xo.connect(), relative_path=tmp_path)

    df = pins_resource.df[filter_].reset_index(drop=True)
    kwargs = {"mode": "create"} if con.name == "postgres" else {}
    t = pins_resource.deferred_reader(
        pins_resource.path, con, pins_resource.table_name, **kwargs
    )
    expr = t[filter_].cache(cache=cache)

    # no work is done yet
    assert pins_resource.table_name not in con.tables
    assert not cache.exists(expr)

    # something exists in both con and cache
    assert xo.execute(expr).equals(df)
    assert pins_resource.table_name in con.tables
    assert cache.exists(expr)

    # we read from cache even if the table disappears
    try:
        con.drop_table(t.op().name, force=True)
    except duckdb.CatalogException:
        con.drop_view(t.op().name)

    assert xo.execute(expr).equals(df)
    assert pins_resource.table_name not in con.tables

    # we repopulate the cache
    cache.drop(expr)
    assert xo.execute(expr).equals(df)
    assert pins_resource.table_name in con.tables
    assert cache.exists(expr)

    if con.name == "postgres":
        # we are mode="create" by default, which means losing cache creates collision
        mode = get_partial_arguments(pins_resource.get_underlying_method(con))["mode"]
        assert mode == "create"
        cache.drop(expr)
        with pytest.raises(
            ProgrammingError,
            match=f'relation "{pins_resource.table_name}" already exists',
        ):
            xo.execute(expr)

        # with mode="replace" we can clobber
        t = pins_resource.deferred_reader(
            pins_resource.path, con, pins_resource.table_name, mode="replace"
        )
        expr = t[filter_].cache(cache=cache)
        assert xo.execute(expr).equals(df)
        assert cache.exists(expr)
        assert pins_resource.table_name in con.tables
        # this fails above, but works here because of mode="replace"
        cache.drop(expr)
        assert xo.execute(expr).equals(df)


@pytest.mark.parametrize(
    "get_con",
    [
        lambda: xo.pandas.connect(),
        pytest.param(lambda: xo.postgres.connect_env(), marks=pytest.mark.postgres),
    ],
)
def test_cached_csv_mutate(get_con, iris_csv, tmp_path):
    con = get_con()
    target_path = ensure_tmp_csv(iris_csv.name, tmp_path)
    cache = ParquetCache.from_kwargs(source=xo.connect(), relative_path=tmp_path)
    # make sure the con is "clean"
    if iris_csv.table_name in con.tables:
        con.drop_table(iris_csv.table_name, force=True)

    df = iris_csv.df
    kwargs = {"mode": "replace"} if con.name == "postgres" else {}
    t = iris_csv.deferred_reader(target_path, con, iris_csv.table_name, **kwargs)
    expr = t.cache(cache=cache)

    # nothing exists yet
    assert iris_csv.table_name not in con.tables
    assert not cache.exists(expr)

    # initial cache population
    assert xo.execute(expr).equals(df)
    assert iris_csv.table_name in con.tables
    assert cache.exists(expr)

    # mutate
    mutate_csv(target_path)
    df = iris_csv.immediate_reader(target_path)
    assert not cache.exists(expr)
    assert xo.execute(expr).equals(df)
    assert cache.exists(expr)


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
    cache = ParquetCache.from_kwargs(source=xo.connect(), relative_path=tmp_path)
    read_method = getattr(xo, method_name)
    connection = con if remote else xo.duckdb.connect()

    t = read_method(path, connection)
    uncached = t.head(10)
    assert cache.strategy.calc_key(uncached) is not None

    expr = uncached.cache(cache=cache)
    assert not expr.execute().empty


@pytest.mark.postgres
def test_deferred_read_kwargs(pg):
    name = "iris"
    read0, read1 = (
        xo.examples.get_table_from_name(name, pg, mode=mode)
        for mode in ("create", "replace")
    )
    hash0, hash1 = (tokenize(expr) for expr in (read0, read1))
    assert hash0 != hash1


def test_deferred_read_parquet_multiple_paths(parquet_dir):
    path = str(parquet_dir / "astronauts.parquet")
    expr = deferred_read_parquet((path, path), xo.connect())
    assert not expr.execute().empty


def test_deferred_read_csv_multiple_paths(csv_dir):
    path = str(csv_dir / "astronauts.csv")
    con = xo.connect()

    t = con.read_csv(path)

    expr = deferred_read_csv((path, path), con, schema=t.schema())

    assert not expr.execute().empty


deferred_read_params = (
    pytest.param(deferred_read_parquet, "parquet", id="parquet"),
    pytest.param(deferred_read_csv, "csv", id="csv"),
)


@pytest.mark.parametrize("deferred_read,suffix", deferred_read_params)
@pytest.mark.parametrize(
    "to_paths",
    (pytest.param(list, id="list"), pytest.param(tuple, id="tuple")),
)
def test_deferred_read_sequence_of_paths(
    data_dir, tmp_path, deferred_read, suffix, to_paths
):
    path = data_dir / suffix / f"astronauts.{suffix}"
    paths = (path, str(path))
    expr = deferred_read(to_paths(paths), xo.connect(), table_name="t")
    single = deferred_read(path, xo.connect()).execute()
    expected = pd.concat((single, single), ignore_index=True)

    assert tokenize(expr) == tokenize(
        deferred_read(paths, xo.connect(), table_name="t")
    )
    assert_frame_equal(expr.execute(), expected)
    build_path = build_expr(expr, builds_dir=tmp_path / "builds")
    assert_frame_equal(load_expr(build_path).execute(), expected)


@pytest.mark.parametrize("deferred_read,suffix", deferred_read_params)
def test_deferred_read_relocatable_rejects_sequence_of_paths(
    data_dir, deferred_read, suffix
):
    path = data_dir / suffix / f"astronauts.{suffix}"
    with pytest.raises(ValueError, match="needs a single path"):
        deferred_read([path, path], xo.connect(), relocatable=True)


@pytest.mark.parametrize("deferred_read,suffix", deferred_read_params)
@pytest.mark.parametrize("relocatable", (False, True))
@pytest.mark.parametrize(
    "to_paths",
    (pytest.param(list, id="list"), pytest.param(tuple, id="tuple")),
)
def test_deferred_read_rejects_empty_paths_with_schema(
    data_dir, deferred_read, suffix, relocatable, to_paths
):
    path = data_dir / suffix / f"astronauts.{suffix}"
    schema = deferred_read(path, xo.connect()).schema()
    with pytest.raises(ValueError, match="At least one path is required"):
        deferred_read(
            to_paths(()), xo.connect(), schema=schema, relocatable=relocatable
        )


@pytest.mark.parametrize("deferred_read,suffix", deferred_read_params)
@pytest.mark.parametrize("backend_name", ("pandas", "datafusion"))
def test_deferred_read_single_path_backend_rejects_sequence_of_paths(
    data_dir, deferred_read, suffix, backend_name
):
    path = data_dir / suffix / f"astronauts.{suffix}"
    con = getattr(xo, backend_name).connect()
    with pytest.raises(
        ValueError, match=f"the {backend_name} backend reads a single path, got 2"
    ):
        deferred_read([path, path], con, table_name="t")


@pytest.mark.parametrize("deferred_read,suffix", deferred_read_params)
def test_deferred_read_pandas_one_element_sequence_is_single_path(
    data_dir, deferred_read, suffix
):
    path = data_dir / suffix / f"astronauts.{suffix}"
    expr = deferred_read([path], xo.pandas.connect())
    assert_frame_equal(
        expr.execute(), deferred_read(path, xo.pandas.connect()).execute()
    )


@pytest.mark.parametrize("deferred_read,suffix", deferred_read_params)
def test_deferred_read_sequence_of_paths_build_warning(
    data_dir, tmp_path, deferred_read, suffix
):
    path = data_dir / suffix / f"astronauts.{suffix}"
    expr = deferred_read([path, path], xo.connect(), table_name="t")
    with pytest.warns(UserWarning, match="Multi-path reads cannot be relocated") as rec:
        build_expr(expr, builds_dir=tmp_path / "builds")
    (message,) = (str(w.message) for w in rec if "Multi-path reads" in str(w.message))
    assert "machine-local paths" in message
    assert not any("relocatable=True" in str(w.message) for w in rec)


@pytest.mark.parametrize("deferred_read,suffix", deferred_read_params)
@pytest.mark.parametrize(
    "to_paths",
    (pytest.param(list, id="list"), pytest.param(tuple, id="tuple")),
)
def test_deferred_read_one_element_sequence_is_single_path(
    data_dir, tmp_path, deferred_read, suffix, to_paths
):
    path = data_dir / suffix / f"astronauts.{suffix}"
    expr = deferred_read(to_paths((path,)), xo.connect(), "t", relocatable=True)

    assert tokenize(expr) == tokenize(
        deferred_read(path, xo.connect(), "t", relocatable=True)
    )
    build_path = build_expr(expr, builds_dir=tmp_path / "builds")
    assert_frame_equal(
        load_expr(build_path).execute(), deferred_read(path, xo.connect()).execute()
    )


@pytest.fixture(scope="function")
def backend(request, con):
    match request.param:
        case "duckdb":
            return xo.duckdb.connect()
        case "postgres":
            return con
        case "xorq_datafusion":
            return xo.connect()
        case _:
            return con


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("duckdb", id="duckdb"),
        pytest.param("postgres", id="postgres"),
        pytest.param("xorq_datafusion", id="xorq_datafusion"),
    ],
    indirect=True,
)
def test_register_csv_with_glob_string(data_dir, backend):
    table_name = f"{backend.name}_astronauts"
    glob_pattern = str(data_dir / "csv" / "*astronauts.csv")
    expected = backend.read_csv(
        glob_pattern, table_name=f"{table_name}_expected"
    ).execute()

    read = xo.deferred_read_csv(glob_pattern, backend, table_name=table_name)
    actual = read.execute()  # triggers the table creation

    assert any(table_name in t for t in backend.list_tables())
    assert_frame_equal(expected, actual)


def test_register_empty_glob_pattern_fails(data_dir, con):
    glob_pattern = str(data_dir / "csv" / "*foo.csv")

    with pytest.raises(ValueError, match="At least one path is required"):
        xo.deferred_read_csv(glob_pattern, con, table_name="foo")
