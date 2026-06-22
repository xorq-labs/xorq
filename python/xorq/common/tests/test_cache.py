import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import xorq.api as xo
from xorq.caching import ParquetCache
from xorq.caching.strategy import ModificationTimeStrategy, SnapshotStrategy
from xorq.catalog.expr_utils import build_expr_context_zip, load_expr_from_zip
from xorq.common.utils.graph_utils import replace_nodes, walk_nodes
from xorq.expr.relations import RemoteTable


def test_put_get_drop(tmp_path, parquet_dir):
    astronauts_path = parquet_dir.joinpath("astronauts.parquet")

    con = xo.datafusion.connect()
    t = con.read_parquet(astronauts_path, table_name="astronauts")

    cache = ParquetCache.from_kwargs(relative_path=tmp_path, source=con)
    put_node = cache.put(t, t.op())
    assert put_node is not None

    get_node = cache.get(t)
    assert get_node is not None

    cache.drop(t)
    with pytest.raises(KeyError):
        cache.get(t)


def test_default_connection(tmp_path, parquet_dir):
    batting_path = parquet_dir.joinpath("astronauts.parquet")

    con = xo.connect()
    t = con.read_parquet(batting_path, table_name="astronauts")

    # if we do cross source caching, then we get a random name and cache.calc_key result isn't stable
    cache = ParquetCache.from_kwargs(relative_path=tmp_path)
    cache.put(t, t.op())

    get_node = cache.get(t)
    assert get_node is not None
    assert get_node.source.name == con.name
    assert get_node.to_expr().execute is not None


def test_snapshot_strategy_key_is_path_identity(tmp_path):
    # Regression test: SnapshotStrategy must key on path identity only, not on
    # mtime/size/inode. Without this, snapshot keys flip whenever the underlying
    # file is rewritten, defeating the purpose of the strategy.
    #
    # SnapshotStrategy.cached_normalize_read is @functools.cache'd on op identity,
    # which would mask the bug in a single process. Clear it between runs to
    # simulate a fresh process — that is where the bug actually bit users.
    path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"a": [1, 2, 3]}), path)

    con = xo.connect()
    snapshot = SnapshotStrategy()
    mtime = ModificationTimeStrategy()

    expr = xo.deferred_read_parquet(path, con=con, table_name="t")
    snapshot_before = snapshot.calc_key(expr)
    mtime_before = mtime.calc_key(expr)

    pq.write_table(pa.table({"a": list(range(50))}), path)
    expr = xo.deferred_read_parquet(path, con=con, table_name="t")

    assert snapshot.calc_key(expr) == snapshot_before
    # Sanity: ModificationTimeStrategy *should* notice the change. If it doesn't,
    # the test setup isn't actually exercising stat sensitivity and the snapshot
    # invariant above is vacuous.
    assert mtime.calc_key(expr) != mtime_before


def test_snapshot_strategy_calc_key_with_hashing_tag_over_remote_table() -> None:
    t = xo.memtable({"a": [1, 2, 3]})
    con = t._find_backend()
    rt = RemoteTable.from_expr(con, t).to_expr()
    tagged = rt.hashing_tag("my-source", entry_name="test-source", kind="source")

    strategy = SnapshotStrategy()
    key = strategy.calc_key(tagged)
    assert key.startswith(f"{strategy.key_prefix}snapshot-")


def test_snapshot_strategy_key_ignores_remote_table_name() -> None:
    """SnapshotStrategy.calc_key tokenizes the expr directly: it does not rewrite
    RemoteTables, relying on the snapshot key being independent of
    RemoteTable.name (auto-generated, non-deterministic across processes). The
    tokenizer recurses into remote_expr / CachedNode.parent on its own, so
    RemoteTables buried in opaque sub-exprs are covered too.

    Guards the removal of the old _replace_remote_table pass: if a future
    normalize rule makes the snapshot hash name-sensitive, this fails loudly.
    """

    def rename_all_remote_tables(expr):
        # replace_nodes descends into opaque sub-exprs, so this rewrites every
        # RemoteTable name, including those buried under CachedNode.parent and a
        # parent RemoteTable's remote_expr.
        def rename(node, kwargs):
            if isinstance(node, RemoteTable):
                return RemoteTable(
                    name=f"renamed-{id(node)}",
                    schema=node.schema,
                    source=node.source,
                    remote_expr=node.remote_expr,
                    namespace=node.namespace,
                )
            return node.__recreate__(kwargs) if kwargs else node

        return replace_nodes(rename, expr).to_expr()

    strategy = SnapshotStrategy()
    con1, con2, con3 = xo.connect(), xo.connect(), xo.connect()
    base = con1.register(xo.memtable({"a": [1, 2, 3]}), "t")

    # nested into_backend: inner RemoteTable lives in the outer's remote_expr;
    # under_cache: the RemoteTable lives under the opaque CachedNode.parent.
    nested = base.into_backend(con2).filter(lambda x: x.a > 0).into_backend(con3)
    under_cache = base.into_backend(con2).cache()

    for expr in (nested, under_cache):
        # Buried RemoteTables are invisible to a non-descending traversal, so the
        # rename must reach through opaque sub-exprs to exercise them.
        assert len(walk_nodes((RemoteTable,), expr)) > len(expr.op().find(RemoteTable))
        assert strategy.calc_key(rename_all_remote_tables(expr)) == strategy.calc_key(
            expr
        )


@pytest.mark.parametrize(
    "backend_factory",
    (
        pytest.param(lambda: xo.datafusion.connect(), id="datafusion"),
        pytest.param(lambda: xo.duckdb.connect(), id="duckdb"),
    ),
)
def test_loaded_dt_has_stable_token_across_zip_reloads(tmp_path, backend_factory):
    """Two loads of the same build zip produce equal ``.ls.tokenized`` for
    DataFusion- and DuckDB-backed ``DatabaseTable`` nodes.

    Regression for ADR-0007: ``load_expr_from_zip`` extracts each load into a
    fresh ``tempfile.mkdtemp(prefix="xorq-catalog-")``. DataFusion's execution
    plan repr and DuckDB's table DDL embed that tempdir path; without
    canonicalization the DT token diverges per reload, defeating
    content-addressed catalog entries.
    """
    con = backend_factory()
    t = con.create_table("users", pd.DataFrame({"x": [1, 2, 3]}))
    expr = t.select("x")
    with build_expr_context_zip(expr) as zip_path:
        a = load_expr_from_zip(zip_path)
        b = load_expr_from_zip(zip_path)
        assert a.ls.tokenized == b.ls.tokenized
