import datetime
import hashlib
import itertools
import json
import os
import pathlib
import tempfile
import warnings

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import toolz
import yaml12
from toolz import identity

import xorq.api as xo
import xorq.expr.datatypes as dt
import xorq.vendor.ibis as ibis
import xorq.vendor.ibis.expr.operations.relations as rel
from xorq import udf
from xorq.caching import (
    ParquetCache,
    ParquetSnapshotCache,
    ParquetTTLSnapshotCache,
    SourceCache,
    SourceSnapshotCache,
)
from xorq.caching.strategy import SnapshotStrategy, snapshot_normalize_read
from xorq.catalog.backend import GitBackend
from xorq.catalog.catalog import Catalog
from xorq.common.constants import READ_IDENTITY_KEYS
from xorq.common.utils.dasher import tokenize
from xorq.common.utils.defer_utils import (
    deferred_read_csv,
    deferred_read_parquet,
)
from xorq.common.utils.file_utils import normalize_read_path_md5sum
from xorq.common.utils.graph_utils import find_all_sources, walk_nodes
from xorq.common.utils.name_utils import get_uid_prefix
from xorq.conftest import array_types_df
from xorq.expr.relations import CachedNode, CacheTag, Read, RemoteTable
from xorq.expr.udf import ExprScalarUDF
from xorq.ibis_yaml.compiler import (
    ArtifactStore,
    DumpFiles,
    ExprKind,
    RefEnum,
    _extract_sql_queries,
    _is_relocatable_candidate,
    _mark_reads_relocatable,
    _sanitize_generated_names,
    build_expr,
    load_expr,
)
from xorq.ibis_yaml.config import config
from xorq.ibis_yaml.sql import find_relations
from xorq.ibis_yaml.translate import warn_on_local_path
from xorq.tests.util import assert_frame_equal
from xorq.vendor.ibis.common.collections import FrozenOrderedDict


do_roundtrip_expr = toolz.compose(load_expr, build_expr)


def get_local_path(parquet_dir, name):
    pins_path = parquet_dir.joinpath(f"{name}.parquet")
    local_path = pathlib.Path(pins_path.name)
    local_path.write_bytes(pins_path.read_bytes())
    return local_path


@pytest.mark.snapshot_check
def test_artifact_store_expr_hash(t, builds_dir, snapshot):
    artifact_store = ArtifactStore.from_path_and_expr(builds_dir, t)
    actual = artifact_store.root_path.name
    snapshot.assert_match(actual, "artifact_store_expr_hash.txt")


@pytest.fixture(scope="session")
def users_df():
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4, 5],
            "age": [25, 32, 28, 45, 31],
            "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        }
    )


def test_build_manager_roundtrip(t, builds_dir):
    artifact_store = ArtifactStore.from_path_and_expr(builds_dir, t)
    yaml_dict = {"a": "string"}
    artifact_store.save_yaml(yaml_dict, DumpFiles.expr)

    out = artifact_store.root_path.joinpath(DumpFiles.expr).read_text()
    assert out == "---\na: string\n...\n"
    result = artifact_store.load_yaml(DumpFiles.expr)
    assert result == yaml_dict


def test_build_manager_paths(t, builds_dir):
    new_path = builds_dir / "new_path"

    assert not new_path.exists()
    ArtifactStore.from_path_and_expr(new_path, t)
    assert new_path.exists()


def test_clean_frozen_dict_yaml(builds_dir):
    artifact_store = ArtifactStore(builds_dir)
    data = FrozenOrderedDict(
        {"string": "text", "integer": 42, "float": 3.14, "boolean": True, "none": None}
    )

    expected_yaml = """---
string: text
integer: 42
float: 3.14
boolean: true
none: ~
...
"""
    out_path = artifact_store.save_yaml(data, DumpFiles.expr)
    result = out_path.read_text()

    assert expected_yaml == result


def test_ibis_compiler(t, builds_dir):
    t = xo.memtable({"a": [0, 1], "b": [0, 1]})
    expr = t.filter(t.a == 1).drop("b")
    roundtrip_expr = do_roundtrip_expr(expr, builds_dir=builds_dir)
    assert expr.execute().equals(roundtrip_expr.execute())


def test_memtable_yaml_stable_across_builds(tmp_path_factory):
    """Building the same memtable expr in separate dirs must produce identical expr.yaml.

    Without content-based normalization (normalize_read_path_md5sum) the Read
    nodes created from InMemoryTable would hash mtime/inode, which differ
    across build directories, making the YAML non-deterministic.
    """
    t = xo.memtable({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    expr = t.filter(t.x > 1)

    dirs = [tmp_path_factory.mktemp(f"builds{i}") for i in range(2)]
    yamls = []
    for d in dirs:
        build_path = build_expr(expr, builds_dir=d)
        artifact_store = ArtifactStore(build_path)
        yamls.append(artifact_store.load_yaml(DumpFiles.expr))

    assert yamls[0] == yamls[1]


def test_ibis_compiler_parquet_reader(builds_dir, parquet_dir):
    backend = xo.duckdb.connect()
    parquet_path = parquet_dir / "awards_players.parquet"
    awards_players = deferred_read_parquet(
        parquet_path, backend, table_name="award_players"
    )
    expr = awards_players.filter(awards_players.lgID == "NL").drop("yearID", "lgID")
    roundtrip_expr = do_roundtrip_expr(expr, builds_dir=builds_dir)
    assert expr.execute().equals(roundtrip_expr.execute())


def test_compiler_sql(builds_dir, parquet_dir):
    backend = xo.datafusion.connect()
    awards_players = deferred_read_parquet(
        parquet_dir / "awards_players.parquet",
        backend,
        table_name="awards_players",
    )
    expr = awards_players.filter(awards_players.lgID == "NL").drop("yearID", "lgID")

    build_path = build_expr(expr, builds_dir=builds_dir, debug=True)
    # make sure we can load
    load_expr(build_path)
    expected_relation = find_relations(awards_players)[0]
    expted_sql_hash = tokenize(str(ibis.to_sql(expr)))[: config.hash_length]

    assert build_path.joinpath(DumpFiles.sql).exists()
    assert build_path.joinpath(DumpFiles.build_metadata).exists()
    metadata = json.loads(build_path.joinpath(DumpFiles.build_metadata).read_text())

    assert "current_library_version" in metadata
    sql_text = build_path.joinpath(DumpFiles.sql).read_text()
    # build_expr normalizes profile idx, so read the canonical name from the
    # built profiles.yaml rather than from the original (un-rewritten) backend
    built_profiles = yaml12.parse_yaml(
        build_path.joinpath(DumpFiles.profiles).read_text()
    )
    (profile_name,) = built_profiles.keys()
    expected_result = (
        "---\n"
        "queries:\n"
        "  main:\n"
        "    engine: datafusion\n"
        f"    profile_name: {profile_name}\n"
        "    relations:\n"
        f"      - {expected_relation}\n"
        "    options: {}\n"
        f"    sql_file: {expted_sql_hash}.sql\n"
        "...\n"
    )
    assert sql_text == expected_result


def test_deferred_reads_yaml(builds_dir, parquet_dir):
    backend = xo.datafusion.connect()
    # Factor out the config path
    config_path = parquet_dir / "awards_players.parquet"
    awards_players = deferred_read_parquet(
        config_path,
        backend,
        table_name="awards_players",
    )
    expr = awards_players.filter(awards_players.lgID == "NL").drop("yearID", "lgID")

    expected_relation = find_relations(awards_players)[0]

    build_path = build_expr(expr, builds_dir=builds_dir, debug=True)
    # read canonical profile name from the built profiles.yaml
    built_profiles = yaml12.parse_yaml(
        build_path.joinpath(DumpFiles.profiles).read_text()
    )
    (expected_profile,) = built_profiles.keys()
    yaml_path = build_path.joinpath(DumpFiles.deferred_reads)
    assert yaml_path.exists()
    sql_text = yaml_path.read_text()

    sql_str = str(ibis.to_sql(awards_players))
    expected_sql_file = tokenize(sql_str)[: config.hash_length] + ".sql"

    expected_read_path = str(config_path)

    expected_result = (
        "---\n"
        "reads:\n"
        f"  {expected_relation}:\n"
        "    engine: datafusion\n"
        f"    profile_name: {expected_profile}\n"
        "    relations:\n"
        f"      - {expected_relation}\n"
        "    options:\n"
        "      method_name: read_parquet\n"
        "      name: awards_players\n"
        "      read_kwargs:\n"
        f"        - hash_path: {expected_read_path}\n"
        "        - table_name: awards_players\n"
        f"    sql_file: {expected_sql_file}\n"
        "...\n"
    )

    assert sql_text == expected_result


def test_ibis_compiler_expr_schema_ref(t, builds_dir):
    t = xo.memtable({"a": [0, 1], "b": [0, 1]})
    expr = t.filter(t.a == 1).drop("b")
    build_path = build_expr(expr, builds_dir=builds_dir)
    yaml_dict = yaml12.parse_yaml(build_path.joinpath(DumpFiles.expr).read_text())
    assert yaml_dict["expression"][RefEnum.schema_ref]


def test_multi_engine_deferred_reads(builds_dir, parquet_dir):
    con0 = xo.connect()
    con1 = xo.connect()
    con2 = xo.duckdb.connect()
    con3 = xo.connect()

    awards_players = deferred_read_parquet(
        parquet_dir / "awards_players.parquet", con=con0
    ).into_backend(con1)
    batting = deferred_read_parquet(
        parquet_dir / "batting.parquet", con=con2
    ).into_backend(con1)
    expr = (
        awards_players.join(batting, predicates=["playerID", "yearID", "lgID"])
        .into_backend(con3)
        .filter(xo._.G == 1)
    )
    roundtrip_expr = do_roundtrip_expr(expr, builds_dir=builds_dir)
    assert expr.execute().equals(roundtrip_expr.execute())


def test_multi_engine_with_caching(builds_dir, parquet_dir):
    con0 = xo.connect()
    con1 = xo.connect()
    con2 = xo.duckdb.connect()
    con3 = xo.connect()

    awards_players = deferred_read_parquet(
        parquet_dir / "awards_players.parquet", con=con0
    ).into_backend(con1)
    batting = deferred_read_parquet(
        parquet_dir / "batting.parquet", con=con2
    ).into_backend(con1)
    expr = (
        awards_players.join(batting, predicates=["playerID", "yearID", "lgID"])
        .into_backend(con3)
        .filter(xo._.G == 1)
    )
    roundtrip_expr = do_roundtrip_expr(expr, builds_dir=builds_dir)
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
    builds_dir, tmp_path, environment_factory, cli_factory, monkeypatch, parquet_dir
):
    con0 = xo.connect()
    con1 = xo.connect()

    cache = ParquetCache.from_kwargs(source=con1, relative_path=tmp_path)

    expr = (
        deferred_read_parquet(parquet_dir / "awards_players.parquet", con=con0)
        .into_backend(con1)
        .filter(xo._.playerID == "bondto01")
        .cache(cache=cache)
    )

    expected_cache_dir = tmp_path
    if environment_factory is not None:
        cache_dir = environment_factory(tmp_path)
        monkeypatch.setenv("XORQ_CACHE_DIR", str(cache_dir))
        expected_cache_dir = cache_dir.joinpath(tmp_path)

    if cli_factory is not None:
        cli_cache_dir = cli_factory(tmp_path)
        cache_dir = cli_cache_dir
        expected_cache_dir = cli_cache_dir.joinpath(tmp_path)
    else:
        cache_dir = None

    roundtrip_expr = load_expr(
        build_expr(expr, builds_dir=builds_dir, cache_dir=cache_dir)
    )
    assert expr.execute().equals(roundtrip_expr.execute())
    assert expected_cache_dir.exists()


def test_pinned_cache_yaml_roundtrip(
    builds_dir: pathlib.Path, tmp_path: pathlib.Path, parquet_dir: pathlib.Path
) -> None:
    con = xo.connect()
    cache = ParquetCache.from_kwargs(source=con, relative_path=tmp_path)
    expr = (
        deferred_read_parquet(parquet_dir / "awards_players.parquet", con=con)
        .filter(xo._.playerID == "bondto01")
        .cache(cache=cache)
    )
    # materialize the cache so the pinned expr can read it directly
    expr.execute()
    pinned = expr.ls.pin()

    roundtrip_expr = do_roundtrip_expr(pinned, builds_dir=builds_dir)

    assert walk_nodes((CacheTag,), roundtrip_expr)
    assert not walk_nodes((CachedNode,), roundtrip_expr)
    assert pinned.execute().equals(roundtrip_expr.execute())


@pytest.mark.parametrize(
    "table_from_df",
    (
        pytest.param(lambda _, df: xo.memtable(df, name="users"), id="memtable"),
        pytest.param(
            lambda con, df: con.register(df, table_name="users"), id="database_table"
        ),
    ),
)
def test_roundtrip_database_table(builds_dir, users_df, table_from_df):
    original = xo.connect()

    t = table_from_df(original, users_df)
    expr = t.filter(t.age > 30).select(t.user_id, t.name)
    roundtrip_expr = do_roundtrip_expr(expr, builds_dir=builds_dir)
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
def test_roundtrip_database_table_cached(builds_dir, tmp_path, users_df, table_from_df):
    original = xo.connect()
    ddb = xo.duckdb.connect()

    cache = ParquetCache.from_kwargs(source=ddb, relative_path=tmp_path)

    t = table_from_df(original, users_df)
    expr = t.filter(t.age > 30).select(t.user_id, t.name, t.age * 2).cache(cache=cache)
    roundtrip_expr = do_roundtrip_expr(expr, builds_dir=builds_dir)
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
def test_roundtrip_database_table_behind_cache(
    builds_dir, tmp_path, users_df, table_from_df
):
    original = xo.connect()
    ddb = xo.duckdb.connect()

    cache = ParquetCache.from_kwargs(source=ddb, relative_path=tmp_path)

    t = table_from_df(original, users_df)
    expr = (
        t.filter(t.age > 30)
        .cache(cache=cache)
        .select(xo._.user_id, xo._.name, xo._.age * 2)
    )
    roundtrip_expr = do_roundtrip_expr(expr, builds_dir=builds_dir)
    assert_frame_equal(xo.execute(expr), roundtrip_expr.execute())


def test_build_pandas_backend(builds_dir, users_df):
    xo_con = xo.connect()
    pandas_con = xo.pandas.connect()
    t = xo_con.register(users_df, table_name="users")

    expected = (
        t.filter(t.age > 30)
        .select(t.user_id, t.name, t.age * 2)
        .into_backend(pandas_con, name="pandas_users")
    )
    actual = do_roundtrip_expr(expected, builds_dir=builds_dir)
    assert_frame_equal(xo.execute(expected), actual.execute())


@pytest.mark.slow(level=1)
@pytest.mark.snapshot_check
def test_build_file_stability_https(builds_dir, snapshot):
    def with_profile_idx(con, idx):
        profile = con._profile
        con._profile = profile.clone(idx=idx)
        return con

    con0 = with_profile_idx(xo.connect(), 0)
    con1 = with_profile_idx(xo.connect(), 1)
    con2 = with_profile_idx(xo.duckdb.connect(), 2)
    con3 = with_profile_idx(xo.connect(), 3)

    awards_players_path = "https://storage.googleapis.com/letsql-pins/awards_players/20240711T171119Z-886c4/awards_players.parquet"
    batting_path = "https://storage.googleapis.com/letsql-pins/batting/20240711T171118Z-431ef/batting.parquet"

    awards_players = xo.deferred_read_parquet(
        awards_players_path,
        con0,
        "awards_players",
    ).into_backend(con1, "awards_players_into")
    batting = xo.deferred_read_parquet(
        batting_path,
        con2,
        "batting",
    ).into_backend(con1, "batting_into")
    expr = (
        awards_players.join(batting, predicates=["playerID", "yearID", "lgID"])
        .into_backend(con3, "joined_into")
        .filter(xo._.G == 1)
    )
    build_path = build_expr(expr, builds_dir=builds_dir, debug=True)

    actual = json.dumps(
        {
            p.name: hashlib.md5(p.read_bytes()).hexdigest()
            for p in build_path.iterdir()
            if p.name != DumpFiles.build_metadata
        },
        indent=2,
        sort_keys=True,
    )

    snapshot.assert_match(actual, "expected.json")

    # test that it also runs
    roundtrip_expr = load_expr(build_path)
    assert expr.execute().equals(roundtrip_expr.execute())


@pytest.mark.snapshot_check
def test_build_file_stability_local(
    builds_dir,
    parquet_dir,
    tmpdir,
    monkeypatch,
    snapshot,
):
    monkeypatch.chdir(tmpdir)

    def with_profile_idx(con, idx):
        profile = con._profile
        con._profile = profile.clone(idx=idx)
        return con

    batting_path = get_local_path(parquet_dir, "batting")
    awards_players_path = get_local_path(parquet_dir, "awards_players")

    con0 = with_profile_idx(xo.connect(), 0)
    con1 = with_profile_idx(xo.connect(), 1)
    con2 = with_profile_idx(xo.duckdb.connect(), 2)
    con3 = with_profile_idx(xo.connect(), 3)

    awards_players = xo.deferred_read_parquet(
        awards_players_path,
        con0,
        "awards_players",
        # we must hash based on content: inode stat is constantly updating
        normalize_method=normalize_read_path_md5sum,
    ).into_backend(con1, "awards_players_into")
    batting = xo.deferred_read_parquet(
        batting_path,
        con2,
        "batting",
        # we must hash based on content: inode stat is constantly updating
        normalize_method=normalize_read_path_md5sum,
    ).into_backend(con1, "batting_into")
    expr = (
        awards_players.join(batting, predicates=["playerID", "yearID", "lgID"])
        .into_backend(con3, "joined_into")
        .filter(xo._.G == 1)
    )
    build_path = build_expr(expr, builds_dir=builds_dir, debug=True)
    actual = json.dumps(
        {
            p.name: hashlib.md5(p.read_bytes()).hexdigest()
            for p in build_path.iterdir()
            if p.name != DumpFiles.build_metadata
        },
        indent=2,
        sort_keys=True,
    )

    snapshot.assert_match(actual, "expected.json")

    # test that it also runs
    roundtrip_expr = load_expr(build_path)
    assert expr.execute().equals(roundtrip_expr.execute())


@pytest.mark.snapshot_check
def test_build_file_stability_and_relocatability(
    builds_dir, parquet_dir, tmpdir, monkeypatch, snapshot
):
    # path impacts node hash therefore path ***MUST*** be the same
    monkeypatch.chdir(tmpdir)

    path = get_local_path(parquet_dir, "awards_players")
    awards_players = xo.deferred_read_parquet(
        path,
        normalize_method=normalize_read_path_md5sum,
    )
    batting = xo.memtable(
        xo.connect().read_parquet(parquet_dir.joinpath("batting.parquet")).execute()
    )
    on = sorted(set(batting.columns).intersection(awards_players.columns))
    expr = awards_players.select(on).join(batting.select(on), predicates=on)

    build_dir = build_expr(
        expr, builds_dir=builds_dir, read_normalize_method=normalize_read_path_md5sum
    )
    actual = json.dumps(
        {
            p.name: hashlib.md5(p.read_bytes()).hexdigest()
            for p in build_dir.iterdir()
            if p.name != DumpFiles.build_metadata and p.is_file()
        }
        | {
            "build_dir_name": build_dir.name,
        },
        indent=2,
        sort_keys=True,
    )
    snapshot.assert_match(actual, "expected.json")

    # test that it also runs
    roundtrip_expr = load_expr(build_dir)
    (actual, expected) = (
        expr.execute().pipe(lambda t: t.sort_values(list(t.columns), ignore_index=True))
        for expr in (roundtrip_expr, expr)
    )
    assert actual.equals(expected)

    before = tuple(node.name for node in walk_nodes(rel.PhysicalTable, expr))
    assert all(map(get_uid_prefix, before))
    after = tuple(node.name for node in walk_nodes(rel.PhysicalTable, roundtrip_expr))
    assert not any(map(get_uid_prefix, after))


def test_build_pandas_backend_behind_into_backend(builds_dir, users_df):
    xo_con = xo.connect()
    pandas_con = xo.pandas.connect()
    t = xo_con.register(users_df, table_name="users")

    expected = (
        t.filter(t.age > 30)
        .into_backend(pandas_con, name="pandas_users")
        .select(xo._.user_id, xo._.name, xo._.age * 2)
    )
    actual = do_roundtrip_expr(expected, builds_dir=builds_dir)
    assert_frame_equal(xo.execute(expected), actual.execute())


def test_struct_field(builds_dir, tmpdir):
    path = pathlib.Path(tmpdir).joinpath("t.parquet")
    xo.memtable({"a": [{"b": 1, "c": "string"}]}).to_parquet(path)
    t = xo.deferred_read_parquet(
        path,
        xo.connect(),
        table_name="t",
    )
    expr = t.select(t.a.b.name("a-b"))
    roundtrip_expr = do_roundtrip_expr(expr, builds_dir=builds_dir)
    assert_frame_equal(expr.execute(), roundtrip_expr.execute())


def test_no_sql_or_deferred_when_debug_false(builds_dir):
    t = xo.memtable({"a": [1, 2, 3]})
    expr = t.filter(t.a > 1)
    build_path = build_expr(expr, builds_dir=builds_dir, debug=False)
    assert not os.path.exists(build_path / DumpFiles.sql)
    assert not os.path.exists(build_path / DumpFiles.deferred_reads)


def test_into_backend_with_array_filter(builds_dir):
    duckdb_con = xo.duckdb.connect()

    t = duckdb_con.create_table("array_types", array_types_df)
    expr = t.mutate(filtered=t.x.filter(xo._ > 1)).cache(
        SourceCache.from_kwargs(source=xo.connect())
    )
    roundtrip_expr = do_roundtrip_expr(expr, builds_dir=builds_dir, debug=False)
    assert_frame_equal(expr.execute(), roundtrip_expr.execute())
    assert {"duckdb", "xorq_datafusion"}.intersection(
        source.name for source in find_all_sources(roundtrip_expr)
    )


def test_roundtrip_parquet_snapshot_cache(builds_dir, tmp_path, users_df):
    original = xo.connect()
    ddb = xo.duckdb.connect()

    cache = ParquetSnapshotCache.from_kwargs(source=ddb, relative_path=tmp_path)

    t = original.register(users_df, table_name="users")
    expr = t.filter(t.age > 30).select(t.user_id, t.name, t.age * 2).cache(cache=cache)
    roundtrip_expr = do_roundtrip_expr(expr, builds_dir=builds_dir)
    assert_frame_equal(xo.execute(expr), roundtrip_expr.execute())


def test_roundtrip_parquet_ttl_snapshot_cache(builds_dir, tmp_path, users_df):
    original = xo.connect()
    ddb = xo.duckdb.connect()

    cache = ParquetTTLSnapshotCache.from_kwargs(
        source=ddb, relative_path=tmp_path, ttl=datetime.timedelta(hours=2)
    )

    t = original.register(users_df, table_name="users")
    expr = t.filter(t.age > 30).select(t.user_id, t.name, t.age * 2).cache(cache=cache)
    roundtrip_expr = do_roundtrip_expr(expr, builds_dir=builds_dir)
    assert_frame_equal(xo.execute(expr), roundtrip_expr.execute())


def test_roundtrip_source_snapshot_cache(builds_dir, users_df):
    original = xo.connect()

    cache = SourceSnapshotCache.from_kwargs(source=xo.connect())

    t = original.register(users_df, table_name="users")
    expr = t.filter(t.age > 30).select(t.user_id, t.name, t.age * 2).cache(cache=cache)
    roundtrip_expr = do_roundtrip_expr(expr, builds_dir=builds_dir)
    assert_frame_equal(xo.execute(expr), roundtrip_expr.execute())


def test_generated_name_sanitization_parquet(
    builds_dir,
    parquet_dir,
    tmpdir,
    monkeypatch,
    snapshot,
):
    monkeypatch.chdir(tmpdir)

    batting_path = get_local_path(parquet_dir, "batting")
    expr = xo.deferred_read_parquet(batting_path)
    build_path = build_expr(
        expr, builds_dir=builds_dir, read_normalize_method=normalize_read_path_md5sum
    )
    loaded = load_expr(build_path)

    assert (expr_name := expr.op().name) != (build_name := loaded.op().name)
    assert get_uid_prefix(expr_name)
    assert not get_uid_prefix(build_name)
    snapshot.assert_match(build_name, "parquet-build-name.txt")


def test_generated_name_sanitization_memtable(
    builds_dir,
    parquet_dir,
    snapshot,
):
    df = xo.deferred_read_parquet(parquet_dir.joinpath("batting.parquet")).execute()
    expr = xo.memtable(df.sort_values(list(df.columns)))
    build_path = build_expr(expr, builds_dir=builds_dir)
    loaded = load_expr(build_path)

    assert (expr_name := expr.op().name) != (build_name := loaded.op().name)
    assert get_uid_prefix(expr_name)
    assert not get_uid_prefix(build_name)
    snapshot.assert_match(build_name, "memory-build-name.txt")


def test_memtable_cache_key_stable_across_roundtrip(builds_dir, tmp_path):
    cache = ParquetSnapshotCache.from_kwargs(relative_path=tmp_path / "cache")
    expr = xo.memtable({"x": [1, 2, 3]}).cache(cache=cache)

    # ExprDumper sanitizes names before building; replicate that here
    sanitized = _sanitize_generated_names(expr, normalize_method=None)
    build_path = build_expr(expr, builds_dir=builds_dir)
    loaded = load_expr(build_path)

    def cache_key(e):
        (cn,) = walk_nodes((CachedNode,), e)
        return cn.cache.calc_key(cn.parent)

    assert cache_key(sanitized) == cache_key(loaded)


def test_memtable_creates_same_key(builds_dir, tmp_path):
    # The cache file written by the sanitized original expr and the cache file
    # written by the loaded expr must have the same filename — confirming the
    # key is stable across the build/load roundtrip.
    cache_path = tmp_path / "cache"
    cache = ParquetSnapshotCache.from_kwargs(relative_path=cache_path)
    expr = xo.memtable({"x": [1, 2, 3]}).cache(cache=cache)

    # Sanitize names the same way ExprDumper does before building
    sanitized = _sanitize_generated_names(expr, normalize_method=None)
    build_path = build_expr(expr, builds_dir=builds_dir)

    # Execute the sanitized expr — writes the cache file
    sanitized.execute()
    original_files = set(cache_path.glob("*.parquet"))
    assert original_files, "sanitized exec did not create a cache file"

    # Load and execute — must hit the same cache file, not create a new one
    loaded = load_expr(build_path)
    loaded.execute()
    loaded_files = set(cache_path.glob("*.parquet"))

    assert original_files == loaded_files, (
        f"key mismatch: sanitized created {[f.name for f in original_files]}, "
        f"loaded added {[f.name for f in loaded_files - original_files]}"
    )


def test_pandas_memtable_comparison(builds_dir):
    df = pd.DataFrame({"a": [1]})
    expr = xo.memtable(df, name="name")
    expr2 = xo.memtable(df, name="name")
    joined = expr.join(expr2, predicates="a")
    xo.build_expr(joined, builds_dir=builds_dir)


def test_pyarrow_memtable_comparison(builds_dir):
    table = pa.table({"a": [1]})
    expr = xo.memtable(table, name="name")
    expr2 = xo.memtable(table, name="name")
    joined = expr.join(expr2, predicates="a")
    xo.build_expr(joined, builds_dir=builds_dir)


def test_polars_memtable_comparison(builds_dir):
    pl = pytest.importorskip("polars")

    df = pl.DataFrame({"a": [1]})
    expr = xo.memtable(df, name="name")
    expr2 = xo.memtable(df, name="name")
    joined = expr.join(expr2, predicates="a")
    xo.build_expr(joined, builds_dir=builds_dir)


def make_lahman_parquet_dir(tmp_dir: pathlib.Path, n_rows: int = 1_000) -> pathlib.Path:
    parquet_dir = tmp_dir / "lahman_parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    player_ids = [f"player{i:04d}" for i in range(n_rows)]
    years = [1990 + (i % 30) for i in range(n_rows)]
    teams = [f"T{i % 20:02d}" for i in range(n_rows)]

    pd.DataFrame(
        {
            "playerID": player_ids,
            "nameFirst": [f"First{i}" for i in range(n_rows)],
            "nameLast": [f"Last{i}" for i in range(n_rows)],
        }
    ).to_parquet(parquet_dir / "people.parquet", index=False)

    pd.DataFrame(
        {
            "playerID": player_ids,
            "yearID": years,
            "teamID": teams,
            "salary": [100_000 + i * 500 for i in range(n_rows)],
        }
    ).to_parquet(parquet_dir / "salaries.parquet", index=False)

    pd.DataFrame(
        {
            "playerID": player_ids,
            "yearID": years,
            "teamID": teams,
            "POS": [
                ["P", "C", "1B", "2B", "SS", "3B", "LF", "CF", "RF"][i % 9]
                for i in range(n_rows)
            ],
        }
    ).to_parquet(parquet_dir / "fielding.parquet", index=False)

    return parquet_dir


def make_multi_join_expr(parquet_dir: pathlib.Path):
    pg = xo.postgres.connect_examples()
    batting = pg.table("batting")
    pg_backend = batting._find_backend()

    local = xo.connect()

    people = deferred_read_parquet(
        parquet_dir / "people.parquet", local, table_name="people"
    )
    salaries = deferred_read_parquet(
        parquet_dir / "salaries.parquet", local, table_name="salaries"
    )
    fielding = deferred_read_parquet(
        parquet_dir / "fielding.parquet", local, table_name="fielding"
    )

    people_pg = people[["playerID", "nameFirst", "nameLast"]].into_backend(pg_backend)
    salaries_pg = salaries[["playerID", "yearID", "teamID", "salary"]].into_backend(
        pg_backend
    )
    fielding_pg = fielding[["playerID", "yearID", "teamID", "POS"]].into_backend(
        pg_backend
    )

    with_names = (
        batting.filter(batting.AB > 0)
        .join(people_pg, predicates="playerID", how="left")
        .drop("playerID_right")
    )
    with_salary = with_names.join(
        salaries_pg,
        predicates=["playerID", "yearID", "teamID"],
        how="left",
    ).drop("playerID_right", "yearID_right", "teamID_right")
    return with_salary.join(
        fielding_pg,
        predicates=["playerID", "yearID", "teamID"],
        how="left",
    ).drop("playerID_right", "yearID_right", "teamID_right")


def test_multi_join_expr_yaml_line_count(tmp_path, builds_dir):
    parquet_dir = make_lahman_parquet_dir(tmp_path)
    expr = make_multi_join_expr(parquet_dir)
    build_path = build_expr(expr, builds_dir=builds_dir)
    expr_yaml_path = build_path / DumpFiles.expr
    line_count = len(expr_yaml_path.read_text().splitlines())
    assert line_count < 1300, f"expr.yaml has {line_count} lines (expected < 1300)"


def test_build_expr_kind_source(tmp_path):
    expr = xo.memtable({"a": [1, 2, 3]})
    build_dir = build_expr(expr, builds_dir=tmp_path)
    entry = json.loads((build_dir / DumpFiles.expr_metadata).read_text())
    assert entry["kind"] == ExprKind.Source
    assert "schema_out" in entry
    assert "schema_in" not in entry


def test_build_expr_kind_bound(tmp_path):
    expr = xo.memtable({"a": [1, 2, 3]}).filter(xo._.a > 1)
    build_dir = build_expr(expr, builds_dir=tmp_path)
    entry = json.loads((build_dir / DumpFiles.expr_metadata).read_text())
    assert entry["kind"] == ExprKind.Expr
    assert "schema_out" in entry
    assert "schema_in" not in entry


def test_build_expr_kind_partial(tmp_path):
    t = xo.table(schema={"a": "int64"})
    expr = t.filter(t.a > 0)
    build_dir = build_expr(expr, builds_dir=tmp_path)
    entry = json.loads((build_dir / DumpFiles.expr_metadata).read_text())
    assert entry["kind"] == ExprKind.UnboundExpr
    assert "schema_out" in entry
    assert "schema_in" in entry
    assert entry["schema_in"] == {"a": "int64"}


def test_extract_sql_queries_binds_non_none_defaults():
    """Params with non-None defaults are bound into the generated SQL."""
    threshold = xo.param("threshold", "float64", default=1.0)
    t = xo.table(schema={"x": "float64"})
    expr = t.filter(t.x > threshold)
    result = _extract_sql_queries(expr, ExprKind.UnboundExpr)
    assert len(result) == 1
    assert "1.0" in result[0][2]


def test_read_kwargs_contains_hash_path_and_read_path(builds_dir):
    t = xo.memtable({"a": [1, 2], "b": [3, 4]})
    build_path = build_expr(t, builds_dir=builds_dir)
    loaded_yaml = yaml12.parse_yaml(build_path.joinpath(DumpFiles.expr).read_text())
    loaded = load_expr(build_path, raise_on_unbound=False)

    reads = tuple(walk_nodes((Read,), loaded))
    assert not reads, "deferred reads should be converted to memtables after load"

    # inspect the YAML directly to verify both keys are serialized
    def find_read_kwargs(d):
        match d:
            case {"op": "Read", "read_kwargs": rk}:
                return [rk]
            case dict():
                return [rk for v in d.values() for rk in find_read_kwargs(v)]
            case list():
                return [rk for v in d for rk in find_read_kwargs(v)]
            case _:
                return []

    all_read_kwargs = find_read_kwargs(loaded_yaml)
    assert all_read_kwargs

    for rk_list in all_read_kwargs:
        kw = dict(rk_list)
        assert "hash_path" in kw, f"missing hash_path in {kw}"
        assert "read_path" in kw, f"missing read_path in {kw}"
        hash_path = pathlib.Path(kw["hash_path"])
        read_path = pathlib.Path(kw["read_path"])
        assert not read_path.is_absolute(), f"read_path should be relative: {read_path}"
        assert hash_path.name == read_path.name


def test_roundtrip_database_table_preserves_node_type(builds_dir, users_df):
    """Roundtripping a con.register() expression must produce DatabaseTable, not Read or InMemoryTable."""
    con = xo.connect()
    t = con.register(users_df, table_name="users")
    expr = t.filter(t.age > 30).select(t.user_id, t.name)

    roundtrip_expr = do_roundtrip_expr(expr, builds_dir=builds_dir)

    reads = tuple(walk_nodes((Read,), roundtrip_expr))
    assert not reads, "roundtripped database_table should not contain Read nodes"

    inmem = tuple(walk_nodes((rel.InMemoryTable,), roundtrip_expr))
    assert not inmem, (
        "roundtripped database_table should not contain InMemoryTable nodes"
    )

    dts = tuple(
        n
        for n in walk_nodes((rel.DatabaseTable,), roundtrip_expr)
        if type(n) is rel.DatabaseTable
    )
    assert dts, "roundtripped database_table should contain a DatabaseTable node"


def _make_three_table_join(tables, order):
    """Build a three-way join chain: order[0].join(order[1]).select(order[1]).join(order[2]).select(order[2])."""
    first, second, third = (tables[i] for i in order)
    return (
        first.join(second, second.columns)
        .select(second)
        .join(third, third.columns)
        .select(third)
    )


@pytest.fixture
def three_tables_mixed():
    """Return (database_table, deferred_read, memtable) sharing the same data."""
    tf = tempfile.NamedTemporaryFile(suffix=".csv")
    pathlib.Path(tf.name).write_text("a,b\n1,2\n3,4")
    con = xo.connect()
    t0 = con.read_csv(tf.name, table_name="t0")
    t1 = xo.deferred_read_csv(tf.name, con=con, table_name="t1")
    t2 = xo.memtable(t1.execute(), name="t2")
    yield (t0, t1, t2), tf


@pytest.mark.parametrize(
    "order",
    tuple(itertools.permutations(range(3))),
    ids=lambda o: "-".join(f"t{i}" for i in o),
)
def test_join_order_permutations_build_roundtrip(builds_dir, three_tables_mixed, order):
    """build_expr / load_expr roundtrip works for every join ordering of
    (database_table, deferred_read, memtable)."""
    tables, _ = three_tables_mixed
    expr = _make_three_table_join(tables, order)
    expected = expr.execute()

    build_path = build_expr(expr, builds_dir=builds_dir)
    roundtrip_expr = load_expr(build_path)
    actual = roundtrip_expr.execute()

    assert_frame_equal(
        actual.sort_values(list(actual.columns), ignore_index=True),
        expected.sort_values(list(expected.columns), ignore_index=True),
    )


@pytest.mark.parametrize(
    "order",
    tuple(itertools.permutations(range(3))),
    ids=lambda o: "-".join(f"t{i}" for i in o),
)
def test_join_order_permutations_catalog_roundtrip(tmp_path, three_tables_mixed, order):
    """Catalog add / entry.expr roundtrip works for every join ordering of
    (database_table, deferred_read, memtable)."""
    repo = Catalog.init_repo_path(tmp_path / f"repo-{''.join(map(str, order))}")
    catalog = Catalog(backend=GitBackend(repo=repo))

    tables, _ = three_tables_mixed
    expr = _make_three_table_join(tables, order)
    expected = expr.execute()

    entry = catalog.add(expr)
    roundtrip_expr = entry.expr
    actual = roundtrip_expr.execute()

    assert_frame_equal(
        actual.sort_values(list(actual.columns), ignore_index=True),
        expected.sort_values(list(expected.columns), ignore_index=True),
    )


def _build_fitted_pipeline_entry(catalog):
    """Build a FittedPipeline predict entry exercising the tokenize side-channel.
    Returns the ``preds`` entry."""
    from xorq.catalog.bind import bind  # noqa: PLC0415
    from xorq.vendor.ibis.expr import operations as ops  # noqa: PLC0415

    sk_pipeline = pytest.importorskip("sklearn.pipeline")
    sk_preprocessing = pytest.importorskip("sklearn.preprocessing")
    sk_linear_model = pytest.importorskip("sklearn.linear_model")

    training = catalog.add(
        xo.memtable({"f": [1.0, 2.0, 3.0, 4.0], "t": [0.0, 0.0, 1.0, 1.0]}),
        aliases=("training",),
    )
    scoring = catalog.add(
        xo.memtable({"f": [1.5, 2.5], "t": [0.0, 1.0]}),
        aliases=("scoring",),
    )
    unbound = ops.UnboundTable(name="p", schema=training.expr.schema()).to_expr()
    identity = catalog.add(unbound.select("f", "t"), aliases=("identity",))
    sk = sk_pipeline.make_pipeline(
        sk_preprocessing.StandardScaler(),
        sk_linear_model.LinearRegression(),
    )
    pipeline = xo.Pipeline.from_instance(sk)
    fitted = pipeline.fit(bind(training, identity), features=("f",), target="t")
    predict_expr = fitted.predict(bind(scoring, identity))
    _ = scoring  # referenced in predict_expr graph; silence unused-warnings
    return catalog.add(predict_expr, aliases=("preds",))


def test_tokenize_survives_side_channel_read(tmp_path):
    """Canonical repro: the FittedPipeline UDF closure hosts a pre-d2m Expr
    whose Read has a catalog-relative `hash_path`. Before Plan B,
    ``normalize_read`` would raise NotImplementedError on that path. With
    ``read_path`` preferred, tokenize goes through the stable content-addressed
    key."""
    repo = Catalog.init_repo_path(tmp_path / "repo")
    catalog = Catalog(backend=GitBackend(repo=repo))
    preds = _build_fitted_pipeline_entry(catalog)

    token = tokenize(preds.lazy_expr)
    assert isinstance(token, str) and token


def test_tokenize_stable_across_reload(tmp_path):
    """Loading the same catalog entry twice produces distinct extract tempdirs
    (`xorq-catalog-*`), so `hash_path` differs between reloads. The stable,
    catalog-relative `read_path` must make the tokens match anyway."""
    repo = Catalog.init_repo_path(tmp_path / "repo")
    catalog = Catalog(backend=GitBackend(repo=repo))
    preds = _build_fitted_pipeline_entry(catalog)

    load1 = preds.lazy_expr
    load2 = preds.lazy_expr
    assert load1 is not load2
    assert tokenize(load1) == tokenize(load2)


def test_tokenize_non_catalog_read_unchanged(parquet_dir):
    """User-created `deferred_read_parquet` has no `read_path` kwarg and must
    keep taking the existing `hash_path` branch (path-existence check +
    content-md5sum). Non-catalog Reads are unaffected by Plan B."""
    parquet_path = parquet_dir / "awards_players.parquet"
    backend = xo.duckdb.connect()
    t = deferred_read_parquet(parquet_path, backend, table_name="awards_players")

    reads = tuple(walk_nodes((Read,), t))
    assert reads, "deferred_read_parquet must produce a Read"
    assert "read_path" not in dict(reads[0].read_kwargs)

    assert tokenize(t) == tokenize(t)


def test_tokenize_missing_path_still_raises(parquet_dir):
    """A Read without `read_path` whose `hash_path` is a bogus relative path
    must still raise FileNotFoundError. The read_path-preferring branch only
    kicks in when the catalog contract is in effect."""
    parquet_path = parquet_dir / "awards_players.parquet"
    backend = xo.duckdb.connect()
    t = deferred_read_parquet(parquet_path, backend, table_name="awards_players")
    op = t.op()
    bogus = tuple(
        (k, "memtables/does-not-exist.parquet" if k == "hash_path" else v)
        for k, v in op.read_kwargs
    )
    bad = op.__recreate__(
        dict(zip(op.__argnames__, op.__args__)) | {"read_kwargs": bogus}
    )

    with pytest.raises(FileNotFoundError, match="memtables/does-not-exist"):
        tokenize(bad.to_expr())


@pytest.fixture
def sample_parquet(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"x": [1, 2, 3]}), path)
    return path


def test_relocatable_read_parquet(
    builds_dir: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """A relocatable Read should survive deletion of the original file."""
    table = pa.table({"x": [1, 2, 3], "y": [4, 5, 6]})
    parquet_path = tmp_path / "input.parquet"
    pq.write_table(table, parquet_path)

    t = deferred_read_parquet(parquet_path, relocatable=True)
    expr = t.filter(t.x > 1)
    build_path = build_expr(expr, builds_dir=builds_dir)

    reads_dir = build_path / "reads"
    assert reads_dir.exists(), "reads/ directory should be created"
    parquet_files = list(reads_dir.glob("*.parquet"))
    assert len(parquet_files) == 1

    parquet_path.unlink()

    loaded = load_expr(build_path)
    result = loaded.execute()
    assert len(result) == 2
    assert list(result.x) == [2, 3]


def test_relocatable_read_via_relocate_reads_flag(
    builds_dir: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """--relocate-reads should bundle even non-relocatable Read nodes."""
    table = pa.table({"a": [10, 20], "b": [30, 40]})
    parquet_path = tmp_path / "data.parquet"
    pq.write_table(table, parquet_path)

    t = deferred_read_parquet(parquet_path)
    build_path = build_expr(t, builds_dir=builds_dir, relocate_reads=True)

    reads_dir = build_path / "reads"
    assert reads_dir.exists()
    assert list(reads_dir.glob("*.parquet"))

    parquet_path.unlink()

    loaded = load_expr(build_path)
    result = loaded.execute()
    assert len(result) == 2


def test_relocatable_read_csv(builds_dir: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """A CSV Read with relocatable=True should copy the CSV into the archive."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y\n1,4\n2,5\n3,6\n")

    t = deferred_read_csv(csv_path, relocatable=True)
    expr = t.filter(t.x > 1)
    build_path = build_expr(expr, builds_dir=builds_dir)

    reads_dir = build_path / "reads"
    assert reads_dir.exists()
    assert list(reads_dir.glob("*.csv"))

    csv_path.unlink()

    loaded = load_expr(build_path)
    result = loaded.execute()
    assert len(result) == 2
    assert sorted(result.x.tolist()) == [2, 3]


def test_relocatable_read_multiple_joined(
    builds_dir: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Two relocatable reads joined in one expression should both be bundled."""
    pq.write_table(pa.table({"key": [1, 2], "val_a": [10, 20]}), tmp_path / "a.parquet")
    pq.write_table(pa.table({"key": [1, 2], "val_b": [30, 40]}), tmp_path / "b.parquet")

    a = deferred_read_parquet(tmp_path / "a.parquet", relocatable=True)
    b = deferred_read_parquet(tmp_path / "b.parquet", relocatable=True)
    expr = a.join(b, "key")
    build_path = build_expr(expr, builds_dir=builds_dir)

    reads_dir = build_path / "reads"
    assert len(list(reads_dir.glob("*.parquet"))) == 2

    (tmp_path / "a.parquet").unlink()
    (tmp_path / "b.parquet").unlink()

    loaded = load_expr(build_path)
    result = loaded.execute()
    assert len(result) == 2
    assert sorted(result.val_a.tolist()) == [10, 20]
    assert sorted(result.val_b.tolist()) == [30, 40]


def test_relocatable_changes_build_hash(
    builds_dir: pathlib.Path, sample_parquet: pathlib.Path
) -> None:
    """relocatable=True must produce a different build hash than the default."""
    plain = deferred_read_parquet(sample_parquet)
    reloc = deferred_read_parquet(sample_parquet, relocatable=True)

    plain_path = build_expr(plain, builds_dir=builds_dir)
    reloc_path = build_expr(reloc, builds_dir=builds_dir)
    assert plain_path.name != reloc_path.name


def test_relocate_reads_flag_changes_build_hash(
    builds_dir: pathlib.Path, sample_parquet: pathlib.Path
) -> None:
    """--relocate-reads must produce a different build hash than the default."""
    t = deferred_read_parquet(sample_parquet)
    default_path = build_expr(t, builds_dir=builds_dir)
    reloc_path = build_expr(t, builds_dir=builds_dir, relocate_reads=True)
    assert default_path.name != reloc_path.name


def test_relocatable_api_and_flag_produce_same_hash(
    builds_dir: pathlib.Path, sample_parquet: pathlib.Path
) -> None:
    """relocatable=True at construction and --relocate-reads at build time produce the same build hash."""
    api_t = deferred_read_parquet(sample_parquet, relocatable=True)
    api_path = build_expr(api_t, builds_dir=builds_dir / "api")

    plain_t = deferred_read_parquet(sample_parquet)
    flag_path = build_expr(plain_t, builds_dir=builds_dir / "flag", relocate_reads=True)

    assert api_path.name == flag_path.name


def test_relocatable_read_no_local_path_warning(
    builds_dir: pathlib.Path, sample_parquet: pathlib.Path
) -> None:
    """Relocatable reads should not emit the local-path warning during build."""
    t = deferred_read_parquet(sample_parquet, relocatable=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_expr(t, builds_dir=builds_dir)

    local_path_warnings = [
        w for w in caught if "local filesystem path" in str(w.message)
    ]
    assert local_path_warnings == [], [str(w.message) for w in local_path_warnings]


def test_relocatable_survives_round_trip(
    builds_dir: pathlib.Path, sample_parquet: pathlib.Path
) -> None:
    """A relocatable Read should stay relocatable after build → load."""
    t = deferred_read_parquet(sample_parquet, relocatable=True)
    build_path = build_expr(t, builds_dir=builds_dir)
    loaded = load_expr(build_path)

    reads = list(walk_nodes(Read, loaded))
    assert len(reads) == 1
    kw = dict(reads[0].read_kwargs)
    assert kw.get("relocatable") is True
    assert "read_path" in kw


def test_relocatable_rebuild_from_loaded_expr(
    builds_dir: pathlib.Path, sample_parquet: pathlib.Path
) -> None:
    """A loaded relocatable expr can be re-built and the result still executes.

    This is why relocatable reads stay as Read nodes after load rather than
    being eagerly resolved to DatabaseTable: the relocatable marker must
    survive so a subsequent build_expr can re-bundle the file.
    """
    t = deferred_read_parquet(sample_parquet, relocatable=True)
    first_build = build_expr(t, builds_dir=builds_dir / "first")
    loaded = load_expr(first_build)

    second_build = build_expr(loaded, builds_dir=builds_dir / "second")

    sample_parquet.unlink()
    first_build_reads = list((first_build / "reads").glob("*"))
    for f in first_build_reads:
        f.unlink()

    result = load_expr(second_build).execute()
    assert len(result) == 3
    assert sorted(result.x.tolist()) == [1, 2, 3]


def test_mark_reads_relocatable_skips_remote(sample_parquet: pathlib.Path) -> None:
    """_mark_reads_relocatable should not inject relocatable for remote paths."""
    local_t = deferred_read_parquet(sample_parquet)
    local_read = list(walk_nodes(Read, local_t))[0]

    remote_read = Read(
        method_name=local_read.method_name,
        name="remote_table",
        schema=local_read.schema,
        source=local_read.source,
        read_kwargs=(("hash_path", "s3://bucket/data.parquet"),),
        normalize_method=local_read.normalize_method,
    )
    expr = local_t.join(remote_read.to_expr(), "x")

    marked = _mark_reads_relocatable(expr)
    reads = list(walk_nodes(Read, marked))
    for read in reads:
        kw = dict(read.read_kwargs)
        if kw.get("hash_path") == "s3://bucket/data.parquet":
            assert not kw.get("relocatable", False)
        else:
            assert kw.get("relocatable") is True


# ---------------------------------------------------------------------------
# IDENTITY_KEYS — relocatable affects tokenization and snapshot keys
# ---------------------------------------------------------------------------


def test_identity_keys_includes_relocatable() -> None:
    """READ_IDENTITY_KEYS must contain 'relocatable'."""
    assert "relocatable" in READ_IDENTITY_KEYS


def test_tokenize_differs_by_relocatable(sample_parquet: pathlib.Path) -> None:
    """Two Reads differing only in relocatable should produce different tokens."""
    plain = deferred_read_parquet(sample_parquet)
    reloc = deferred_read_parquet(sample_parquet, relocatable=True)

    assert tokenize(plain) != tokenize(reloc)


def test_snapshot_normalize_read_differs_by_relocatable(
    sample_parquet: pathlib.Path,
) -> None:
    """snapshot_normalize_read should yield different results when relocatable differs."""
    plain = deferred_read_parquet(sample_parquet)
    reloc = deferred_read_parquet(sample_parquet, relocatable=True)

    plain_read = list(walk_nodes(Read, plain))[0]
    reloc_read = list(walk_nodes(Read, reloc))[0]

    assert snapshot_normalize_read(plain_read) != snapshot_normalize_read(reloc_read)


# ---------------------------------------------------------------------------
# _is_relocatable_candidate edge cases
# ---------------------------------------------------------------------------


def test_is_relocatable_candidate_not_a_read() -> None:
    """Non-Read nodes are never candidates."""
    node = rel.DatabaseTable(
        name="t",
        schema=ibis.schema({"x": "int64"}),
        source=ibis.duckdb.connect(),
        namespace=rel.Namespace(database=None, catalog=None),
    )
    assert not _is_relocatable_candidate(node)


def test_is_relocatable_candidate_already_relocatable(
    sample_parquet: pathlib.Path,
) -> None:
    """A Read that is already relocatable is not a candidate (no double-marking)."""
    t = deferred_read_parquet(sample_parquet, relocatable=True)
    read = list(walk_nodes(Read, t))[0]
    assert not _is_relocatable_candidate(read)


def test_is_relocatable_candidate_no_hash_path(sample_parquet: pathlib.Path) -> None:
    """A Read with no hash_path is not a candidate."""
    t = deferred_read_parquet(sample_parquet)
    read = list(walk_nodes(Read, t))[0]
    no_hash = Read(
        method_name=read.method_name,
        name=read.name,
        schema=read.schema,
        source=read.source,
        read_kwargs=tuple((k, v) for k, v in read.read_kwargs if k != "hash_path"),
        normalize_method=read.normalize_method,
    )
    assert not _is_relocatable_candidate(no_hash)


def test_is_relocatable_candidate_local_path(sample_parquet: pathlib.Path) -> None:
    """A normal local-file Read IS a candidate."""
    t = deferred_read_parquet(sample_parquet)
    read = list(walk_nodes(Read, t))[0]
    assert _is_relocatable_candidate(read)


# ---------------------------------------------------------------------------
# ArtifactStore.copy_file
# ---------------------------------------------------------------------------


def test_artifact_store_copy_file(tmp_path: pathlib.Path) -> None:
    """copy_file should copy the file and create parent directories."""
    source = tmp_path / "source.txt"
    source.write_text("hello")

    store = ArtifactStore(root_path=tmp_path / "store")
    dest = store.copy_file(source, "sub", "copied.txt")

    assert dest.exists()
    assert dest.read_text() == "hello"
    assert dest == store.get_path("sub", "copied.txt")


def test_artifact_store_copy_file_overwrites(tmp_path: pathlib.Path) -> None:
    """copy_file should overwrite an existing destination."""
    source = tmp_path / "source.txt"
    source.write_text("new content")

    store = ArtifactStore(root_path=tmp_path / "store")
    dest = store.get_path("out.txt")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("old content")

    store.copy_file(source, "out.txt")
    assert dest.read_text() == "new content"


def test_artifact_store_copy_file_missing_source(tmp_path: pathlib.Path) -> None:
    """copy_file should raise when the source file does not exist."""
    store = ArtifactStore(root_path=tmp_path / "store")
    with pytest.raises(FileNotFoundError):
        store.copy_file(tmp_path / "nonexistent.txt", "out.txt")


# ---------------------------------------------------------------------------
# warn_on_local_path changes
# ---------------------------------------------------------------------------


def test_warn_on_local_path_skips_when_relocatable(tmp_path: pathlib.Path) -> None:
    """warn_on_local_path should not warn when relocatable=True."""
    local_file = tmp_path / "file.parquet"
    local_file.write_bytes(b"data")
    items = (
        ("hash_path", str(local_file)),
        ("relocatable", True),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_on_local_path(items)
    assert caught == []


def test_warn_on_local_path_warns_when_read_path_but_not_relocatable(
    tmp_path: pathlib.Path,
) -> None:
    """read_path alone does not suppress the warning — only relocatable does."""
    local_file = tmp_path / "file.parquet"
    local_file.write_bytes(b"data")
    items = (
        ("hash_path", str(local_file)),
        ("read_path", "database_tables/abc123.parquet"),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_on_local_path(items)
    assert len(caught) == 1


def test_warn_on_local_path_warns_for_non_relocatable_local(
    tmp_path: pathlib.Path,
) -> None:
    """Non-relocatable local reads should still warn with the updated message."""
    local_file = tmp_path / "file.parquet"
    local_file.write_bytes(b"data")
    items = (("hash_path", str(local_file)),)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_on_local_path(items)
    assert len(caught) == 1
    assert "relocatable=True" in str(caught[0].message)
    assert "--relocate-reads" in str(caught[0].message)


# ---------------------------------------------------------------------------
# ExprLoader — relocatable reads stay as Read nodes, not DatabaseTable
# ---------------------------------------------------------------------------


def test_loaded_relocatable_is_read_not_database_table(
    builds_dir: pathlib.Path, sample_parquet: pathlib.Path
) -> None:
    """After load, a relocatable Read should remain a Read node, not DatabaseTable."""
    t = deferred_read_parquet(sample_parquet, relocatable=True)
    build_path = build_expr(t, builds_dir=builds_dir)
    loaded = load_expr(build_path)

    reads = list(walk_nodes(Read, loaded))
    assert len(reads) == 1, "relocatable Read should survive as Read, not be converted"

    dts = [
        n
        for n in walk_nodes((rel.DatabaseTable,), loaded)
        if type(n) is rel.DatabaseTable and not isinstance(n, Read)
    ]
    assert dts == [], (
        "relocatable Read should not be converted to a plain DatabaseTable"
    )


def test_loaded_non_relocatable_becomes_database_table(
    builds_dir: pathlib.Path, sample_parquet: pathlib.Path
) -> None:
    """After load, a non-relocatable Read should be resolved to a DatabaseTable."""
    t = deferred_read_parquet(sample_parquet)
    build_path = build_expr(t, builds_dir=builds_dir)
    loaded = load_expr(build_path)

    reads = [r for r in walk_nodes(Read, loaded) if "read_path" in dict(r.read_kwargs)]
    assert reads == [], "non-relocatable Read should be resolved to DatabaseTable"


# ---------------------------------------------------------------------------
# _mark_reads_relocatable is idempotent
# ---------------------------------------------------------------------------


def test_mark_reads_relocatable_is_idempotent(sample_parquet: pathlib.Path) -> None:
    """Calling _mark_reads_relocatable twice should not double-wrap."""
    t = deferred_read_parquet(sample_parquet)
    once = _mark_reads_relocatable(t)
    twice = _mark_reads_relocatable(once)

    once_reads = list(walk_nodes(Read, once))
    twice_reads = list(walk_nodes(Read, twice))
    assert len(once_reads) == len(twice_reads) == 1

    once_kw = dict(once_reads[0].read_kwargs)
    twice_kw = dict(twice_reads[0].read_kwargs)
    assert once_kw == twice_kw

    assert tokenize(once) == tokenize(twice)


def test_cache_dir_reaches_remote_expr_nested_cache(tmp_path: pathlib.Path) -> None:
    cona = xo.connect()
    conb = xo.connect()
    t = cona.register(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), "t")
    cache = ParquetSnapshotCache.from_kwargs(
        source=cona, relative_path="c", base_path=tmp_path / "buildcache"
    )
    cached = t.mutate(s=t.a + t.b).cache(cache=cache)
    expr = cached.into_backend(conb, "moved")

    assert any(
        walk_nodes((CachedNode,), rt.remote_expr)
        for rt in walk_nodes((RemoteTable,), expr)
    )

    build_path = build_expr(expr, builds_dir=tmp_path / "builds")
    target = tmp_path / "cache_target"
    loaded = load_expr(build_path, cache_dir=str(target))

    cached_nodes = walk_nodes((CachedNode,), loaded)
    assert cached_nodes
    for cn in cached_nodes:
        assert cn.cache.storage.base_path == target


# ---------------------------------------------------------------------------
# Execution pipeline descends into opaque sub-expressions
# ---------------------------------------------------------------------------


def test_execution_handles_cache_in_remote_expr(tmp_path: pathlib.Path) -> None:
    """Caches nested inside RemoteTable.remote_expr must be executed."""
    cona = xo.connect()
    conb = xo.connect()
    t = cona.register(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}), "t")
    cache = ParquetSnapshotCache.from_kwargs(
        source=cona, relative_path="c", base_path=tmp_path
    )
    cached = t.mutate(s=t.a + t.b).cache(cache=cache)
    expr = cached.into_backend(conb, "moved")

    # Verify that there really is a CachedNode nested inside a RemoteTable
    remote_tables = walk_nodes((RemoteTable,), expr)
    assert remote_tables, "expected at least one RemoteTable"
    nested_caches = [
        cn for rt in remote_tables for cn in walk_nodes((CachedNode,), rt.remote_expr)
    ]
    assert nested_caches, "expected a CachedNode nested in remote_expr"

    # Should execute without error — the cache inside remote_expr gets found
    result = expr.execute()
    assert len(result) == 3


def test_hashing_handles_remote_table_in_opaque_subexpr(tmp_path: pathlib.Path) -> None:
    """SnapshotStrategy.calc_key must handle RemoteTables nested in opaque
    sub-exprs: the tokenizer recurses into remote_expr / CachedNode.parent on its
    own, so the key is computed without raising and is stable."""
    cona = xo.connect()
    conb = xo.connect()
    t = cona.register(pd.DataFrame({"x": [10, 20]}), "t")
    inner_cache = ParquetSnapshotCache.from_kwargs(
        source=cona, relative_path="inner", base_path=tmp_path / "inner"
    )
    cached = t.cache(cache=inner_cache)
    expr = cached.into_backend(conb, "moved")

    # Hashing the expression should not raise, and the key should be stable
    strategy = SnapshotStrategy(key_prefix="test-")
    key1 = strategy.calc_key(expr)
    key2 = strategy.calc_key(expr)
    assert key1 == key2
    assert key1.startswith("test-snapshot-")


def test_execution_handles_remote_table_in_computed_kwargs_expr(
    tmp_path: pathlib.Path,
) -> None:
    """RemoteTables nested inside ExprScalarUDF.computed_kwargs_expr must be
    materialized lazily when the UDF is compiled, not during the outer
    register_and_transform_remote_tables pass (which intentionally skips
    opaque sub-expressions)."""
    cona = xo.connect()
    conb = xo.connect()

    df = pd.DataFrame({"x": [1, 2, 3]})
    data = cona.register(df, "data").select("x")
    schema = data.schema()

    @udf.agg.pandas_df(schema=schema, return_type=dt.float64, name="my_sum")
    def my_sum(frame):
        return frame["x"].astype(float).sum()

    # Build computed_kwargs_expr with a RemoteTable: aggregate on data moved
    # to a different backend, so the expression tree contains a RemoteTable.
    remote_data = data.into_backend(conb, "remote_data")
    computed_kwargs_expr = my_sum.on_expr(remote_data).name("my_sum").as_table()

    # Verify the RemoteTable is actually nested inside the computed_kwargs_expr
    assert walk_nodes((RemoteTable,), computed_kwargs_expr), (
        "expected a RemoteTable in computed_kwargs_expr"
    )

    predict_udf = udf.make_pandas_expr_udf(
        computed_kwargs_expr=computed_kwargs_expr,
        fn=lambda value, frame, **kw: (frame["x"] + float(value)).astype(float),
        schema=ibis.schema({"x": dt.float64}),
        name="add_remote_sum",
        return_type=dt.float64,
        post_process_fn=identity,
    )

    expr = data.mutate(out=predict_udf.on_expr(data)).as_table()

    # The outer expression contains an ExprScalarUDF whose computed_kwargs_expr
    # holds a RemoteTable — this RemoteTable is opaque to op.replace and must
    # be materialized lazily during _compile_pyarrow_expr_udf.
    assert walk_nodes((ExprScalarUDF,), expr)

    result = expr.execute()
    assert len(result) == 3
    # my_sum(1+2+3) = 6.0; each row: x + 6.0
    expected_out = [7.0, 8.0, 9.0]
    assert list(result["out"]) == expected_out
