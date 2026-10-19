import xorq.api as xo
from xorq.backends.sqlite import Backend as SqliteBackend
from xorq.caching import SourceCache
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
