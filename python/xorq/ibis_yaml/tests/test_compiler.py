import os
import pathlib

import dask
import pandas as pd
import pytest
import yaml

import xorq as xo
import xorq.vendor.ibis as ibis
from xorq.caching import ParquetStorage
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.ibis_yaml.compiler import ArtifactStore, BuildManager
from xorq.ibis_yaml.config import config
from xorq.ibis_yaml.sql import find_relations
from xorq.tests.util import assert_frame_equal
from xorq.vendor.ibis.common.collections import FrozenOrderedDict


@pytest.mark.snapshot_check
def test_build_manager_expr_hash(t, build_dir, snapshot):
    build_manager = ArtifactStore(build_dir)
    actual = build_manager.get_expr_hash(t)
    snapshot.assert_match(actual, "build_manager_expr_hash.txt")


@pytest.fixture(scope="session")
def users_df():
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5],
            "age": [25, 32, 28, 45, 31],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        }
    )


def test_build_manager_roundtrip(t, build_dir):
    build_manager = ArtifactStore(build_dir)
    yaml_dict = {"a": "string"}
    expr_hash = "dummy-value"
    build_manager.save_yaml(yaml_dict, expr_hash, "expr.yaml")

    with open(build_dir / expr_hash / "expr.yaml") as f:
        out = f.read()
    assert out == "a: string\n"
    result = build_manager.load_yaml(expr_hash, "expr.yaml")
    assert result == yaml_dict


def test_build_manager_paths(t, build_dir):
    new_path = build_dir / "new_path"

    assert not os.path.exists(new_path)
    build_manager = ArtifactStore(new_path)
    assert os.path.exists(new_path)

    build_manager.get_build_path("hash")
    assert os.path.exists(new_path / "hash")


def test_clean_frozen_dict_yaml(build_dir):
    build_manager = ArtifactStore(build_dir)
    data = FrozenOrderedDict(
        {"string": "text", "integer": 42, "float": 3.14, "boolean": True, "none": None}
    )

    expected_yaml = """string: text
integer: 42
float: 3.14
boolean: true
none: null
"""
    out_path = build_manager.save_yaml(data, "hash", "expr.yaml")
    result = out_path.read_text()

    assert expected_yaml == result


def test_ibis_compiler(t, build_dir):
    t = xo.memtable({"a": [0, 1], "b": [0, 1]})
    expr = t.filter(t.a == 1).drop("b")
    compiler = BuildManager(build_dir)
    expr_hash = compiler.compile_expr(expr)

    roundtrip_expr = compiler.load_expr(expr_hash)

    assert expr.execute().equals(roundtrip_expr.execute())


def test_ibis_compiler_parquet_reader(build_dir):
    backend = xo.duckdb.connect()
    parquet_path = xo.config.options.pins.get_path("awards_players")
    awards_players = deferred_read_parquet(
        backend, parquet_path, table_name="award_players"
    )
    expr = awards_players.filter(awards_players.lgID == "NL").drop("yearID", "lgID")
    compiler = BuildManager(build_dir)
    expr_hash = compiler.compile_expr(expr)
    roundtrip_expr = compiler.load_expr(expr_hash)

    assert expr.execute().equals(roundtrip_expr.execute())


def test_compiler_sql(build_dir, parquet_dir):
    backend = xo.datafusion.connect()
    awards_players = deferred_read_parquet(
        backend,
        parquet_dir / "awards_players.parquet",
        table_name="awards_players",
    )
    expr = awards_players.filter(awards_players.lgID == "NL").drop("yearID", "lgID")

    compiler = BuildManager(build_dir)
    expr_hash = compiler.compile_expr(expr)
    _roundtrip_expr = compiler.load_expr(expr_hash)
    expected_relation = find_relations(awards_players)[0]
    expted_sql_hash = dask.base.tokenize(str(ibis.to_sql(expr)))[: config.hash_length]

    assert os.path.exists(build_dir / expr_hash / "sql.yaml")
    assert os.path.exists(build_dir / expr_hash / "metadata.json")
    metadata = compiler.artifact_store.read_json(build_dir, expr_hash, "metadata.json")

    assert "current_library_version" in metadata
    sql_text = pathlib.Path(build_dir / expr_hash / "sql.yaml").read_text()
    expected_result = (
        "queries:\n"
        "  main:\n"
        "    engine: datafusion\n"
        f"    profile_name: {expr._find_backend()._profile.hash_name}\n"
        "    relations:\n"
        f"    - {expected_relation}\n"
        "    options: {}\n"
        f"    sql_file: {expted_sql_hash}.sql\n"
    )
    assert sql_text == expected_result


def test_deferred_reads_yaml(build_dir):
    backend = xo.datafusion.connect()
    # Factor out the config path
    config_path = xo.config.options.pins.get_path("awards_players")
    awards_players = deferred_read_parquet(
        backend,
        config_path,
        table_name="awards_players",
    )
    expr = awards_players.filter(awards_players.lgID == "NL").drop("yearID", "lgID")

    # Get the dynamic relation and profile hash
    expected_relation = find_relations(awards_players)[0]
    expected_profile = backend._profile.hash_name

    compiler = BuildManager(build_dir)
    expr_hash = compiler.compile_expr(expr)
    _roundtrip_expr = compiler.load_expr(expr_hash)

    yaml_path = build_dir / expr_hash / "deferred_reads.yaml"
    assert os.path.exists(yaml_path)
    sql_text = pathlib.Path(yaml_path).read_text()

    sql_str = str(ibis.to_sql(awards_players))
    expected_sql_file = dask.base.tokenize(sql_str)[: config.hash_length] + ".sql"

    expected_read_path = str(config_path)

    expected_result = (
        "reads:\n"
        f"  {expected_relation}:\n"
        "    engine: datafusion\n"
        f"    profile_name: {expected_profile}\n"
        "    relations:\n"
        f"    - {expected_relation}\n"
        "    options:\n"
        "      method_name: read_parquet\n"
        "      name: awards_players\n"
        "      read_kwargs:\n"
        f"      - path: {expected_read_path}\n"
        "      - table_name: awards_players\n"
        f"    sql_file: {expected_sql_file}\n"
    )

    assert sql_text == expected_result


def test_ibis_compiler_expr_schema_ref(t, build_dir):
    t = xo.memtable({"a": [0, 1], "b": [0, 1]})
    expr = t.filter(t.a == 1).drop("b")
    compiler = BuildManager(build_dir)
    expr_hash = compiler.compile_expr(expr)

    with open(build_dir / expr_hash / "expr.yaml") as f:
        yaml_dict = yaml.safe_load(f)

    assert yaml_dict["expression"]["schema_ref"]


def test_multi_engine_deferred_reads(build_dir):
    con0 = xo.connect()
    con1 = xo.connect()
    con2 = xo.duckdb.connect()
    con3 = xo.connect()

    awards_players = xo.examples.awards_players.fetch(con0).into_backend(con1)
    batting = xo.examples.batting.fetch(con2).into_backend(con1)
    expr = (
        awards_players.join(batting, predicates=["playerID", "yearID", "lgID"])
        .into_backend(con3)
        .filter(xo._.G == 1)
    )
    compiler = BuildManager(build_dir)
    expr_hash = compiler.compile_expr(expr)

    roundtrip_expr = compiler.load_expr(expr_hash)

    assert expr.execute().equals(roundtrip_expr.execute())


def test_multi_engine_with_caching(build_dir):
    con0 = xo.connect()
    con1 = xo.connect()
    con2 = xo.duckdb.connect()
    con3 = xo.connect()

    awards_players = xo.examples.awards_players.fetch(con0).into_backend(con1).cache()
    batting = xo.examples.batting.fetch(con2).into_backend(con1).cache()
    expr = (
        awards_players.join(batting, predicates=["playerID", "yearID", "lgID"])
        .into_backend(con3)
        .filter(xo._.G == 1)
    )
    compiler = BuildManager(build_dir)
    expr_hash = compiler.compile_expr(expr)

    roundtrip_expr = compiler.load_expr(expr_hash)

    assert expr.execute().equals(roundtrip_expr.execute())


@pytest.mark.parametrize(
    "environment_factory",
    (
        pytest.param(None, id="no_env"),
        pytest.param(lambda p: p / "env_cache", id="with_env"),
    ),
)
@pytest.mark.parametrize(
    "cli_factory",
    (
        pytest.param(None, id="no_cli"),
        pytest.param(lambda p: p / "cli_cache", id="with_cli"),
    ),
)
def test_multi_engine_with_caching_with_parquet(
    build_dir, tmp_path, environment_factory, cli_factory, monkeypatch
):
    expected_cache_dir = tmp_path
    if environment_factory is not None:
        cache_dir = environment_factory(tmp_path)
        monkeypatch.setenv("XORQ_CACHE_DIR", str(cache_dir))
        expected_cache_dir = cache_dir.joinpath(tmp_path)

    con0 = xo.connect()
    con1 = xo.connect()
    con2 = xo.duckdb.connect()
    con3 = xo.connect()

    storage = ParquetStorage(source=con1, relative_path=tmp_path)

    awards_players = (
        xo.examples.awards_players.fetch(con0).into_backend(con1).cache(storage=storage)
    )
    batting = xo.examples.batting.fetch(con1).into_backend(con2).cache(storage=storage)
    expr = (
        awards_players.join(batting, predicates=["playerID", "yearID", "lgID"])
        .into_backend(con3)
        .filter(xo._.G == 1)
    )

    if cli_factory is not None:
        cli_cache_dir = cli_factory(tmp_path)
        compiler = BuildManager(build_dir, cache_dir=cli_cache_dir)
        expected_cache_dir = cli_cache_dir.joinpath(tmp_path)
    else:
        compiler = BuildManager(build_dir)

    expr_hash = compiler.compile_expr(expr)

    roundtrip_expr = compiler.load_expr(expr_hash)

    assert expr.execute().equals(roundtrip_expr.execute())
    assert expected_cache_dir.exists()


@pytest.mark.parametrize(
    "table_from_df",
    (
        pytest.param(lambda _, df: xo.memtable(df, name="users"), id="memtable"),
        pytest.param(
            lambda con, df: con.register(df, table_name="users"), id="database_table"
        ),
    ),
)
def test_roundtrip_database_table(build_dir, users_df, table_from_df):
    original = xo.connect()

    t = table_from_df(original, users_df)
    expr = t.filter(t.age > 30).select(t.user_id, t.name)

    compiler = BuildManager(build_dir)
    expr_hash = compiler.compile_expr(expr)
    roundtrip_expr = compiler.load_expr(expr_hash)

    assert_frame_equal(xo.execute(expr), roundtrip_expr.execute())


@pytest.mark.parametrize(
    "table_from_df",
    (
        pytest.param(lambda _, df: xo.memtable(df, name="users"), id="memtable"),
        pytest.param(
            lambda con, df: con.register(df, table_name="users"), id="database_table"
        ),
    ),
)
def test_roundtrip_database_table_cached(build_dir, tmp_path, users_df, table_from_df):
    original = xo.connect()
    ddb = xo.duckdb.connect()

    storage = ParquetStorage(source=ddb, relative_path=tmp_path)

    t = table_from_df(original, users_df)
    expr = (
        t.filter(t.age > 30).select(t.user_id, t.name, t.age * 2).cache(storage=storage)
    )

    compiler = BuildManager(build_dir)
    expr_hash = compiler.compile_expr(expr)

    roundtrip_expr = compiler.load_expr(expr_hash)

    assert_frame_equal(xo.execute(expr), roundtrip_expr.execute())


def test_build_pandas_backend(build_dir, users_df):
    xo_con = xo.connect()
    pandas_con = xo.pandas.connect()
    t = xo_con.register(users_df, table_name="users")

    expected = (
        t.filter(t.age > 30)
        .select(t.user_id, t.name, t.age * 2)
        .into_backend(pandas_con, name="pandas_users")
    )

    compiler = BuildManager(build_dir)
    expr_hash = compiler.compile_expr(expected)
    actual = compiler.load_expr(expr_hash)

    assert_frame_equal(xo.execute(expected), actual.execute())
