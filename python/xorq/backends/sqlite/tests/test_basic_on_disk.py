import xorq.api as xo
from xorq.backends.sqlite import Backend as SqliteBackend
from xorq.caching import ParquetCache, SourceCache
from xorq.ibis_yaml.compiler import build_expr, load_expr
from xorq.tests.util import assert_frame_equal


def test_read_parquet(persistent_sqlite_con, astronauts_parquet_path):
    t = persistent_sqlite_con.read_parquet(astronauts_parquet_path)
    assert not t.execute().empty


def test_can_be_cached(persistent_sqlite_con, astronauts_parquet_path):
    astronauts = persistent_sqlite_con.read_parquet(astronauts_parquet_path)
    expr = (
        astronauts.cache(SourceCache.from_kwargs(source=xo.duckdb.connect()))
        .filter(xo._.number == 104)
        .select(xo._.id, xo._.number, xo._.nationwide_number, xo._.name)
    )

    actual = expr.execute()
    expected = (
        astronauts.filter(xo._.number == 104)
        .select(xo._.id, xo._.number, xo._.nationwide_number, xo._.name)
        .execute()
    )

    assert_frame_equal(expected, actual)


def test_build_over_attached_database_loads(tmp_path):
    other = xo.sqlite.connect(str(tmp_path / "other.sqlite"))
    other.create_table("t", xo.memtable({"a": [1, 2, 3]}))

    con = SqliteBackend().connect(str(tmp_path / "main.sqlite"))
    con.attach("other", tmp_path / "other.sqlite")
    t = con.table("t", database="other")
    build_path = build_expr(t.filter(t.a > 1), builds_dir=tmp_path / "builds")

    assert load_expr(build_path).execute()["a"].tolist() == [2, 3]


def test_reattaching_a_name_replaces_its_recorded_path(tmp_path):
    for name, rows in (("a", [1]), ("b", [2])):
        path = str(tmp_path / f"{name}.sqlite")
        xo.sqlite.connect(path).create_table("t", xo.memtable({"a": rows}))

    con = SqliteBackend().connect(str(tmp_path / "main.sqlite"))
    con.attach("other", tmp_path / "a.sqlite")
    con.raw_sql("DETACH DATABASE other")
    con.attach("other", tmp_path / "b.sqlite")
    t = con.table("t", database="other")
    build_path = build_expr(t, builds_dir=tmp_path / "builds")

    assert load_expr(build_path).execute()["a"].tolist() == [2]


def test_cache_over_table_without_id_column(tmp_path):
    con = xo.sqlite.connect(str(tmp_path / "lite.db"))
    con.create_table("t", xo.memtable({"a": [1, 2], "b": ["x", "y"]}))
    t = con.table("t")
    cache = ParquetCache.from_kwargs(source=xo.connect(), relative_path=tmp_path)
    expr = t.filter(t.a > 1).cache(cache=cache)

    assert expr.execute().to_dict("records") == [{"a": 2, "b": "y"}]


def test_cache_over_table_without_id_column_sees_replaced_row(tmp_path):
    con = xo.sqlite.connect(str(tmp_path / "lite.db"))
    con.create_table("t", xo.memtable({"a": [1, 2], "b": ["x", "y"]}))
    t = con.table("t")
    cache = ParquetCache.from_kwargs(source=xo.connect(), relative_path=tmp_path)
    expr = t.filter(t.a > 1).cache(cache=cache)
    expr.execute()

    # same row count after the swap: only the stamp can tell the tables apart
    con.raw_sql("DELETE FROM t WHERE a = 1")
    con.raw_sql("INSERT INTO t VALUES (3, 'z')")

    assert expr.execute().to_dict("records") == [
        {"a": 2, "b": "y"},
        {"a": 3, "b": "z"},
    ]


def test_cache_over_without_rowid_table_with_uppercase_id(tmp_path):
    con = xo.sqlite.connect(str(tmp_path / "lite.db"))
    con.raw_sql("CREATE TABLE t (ID INTEGER PRIMARY KEY, a INTEGER) WITHOUT ROWID")
    con.raw_sql("INSERT INTO t VALUES (1, 1), (2, 2)")
    t = con.table("t")
    cache = ParquetCache.from_kwargs(source=xo.connect(), relative_path=tmp_path)
    expr = t.filter(t.a > 1).cache(cache=cache)

    assert expr.execute().to_dict("records") == [{"ID": 2, "a": 2}]
