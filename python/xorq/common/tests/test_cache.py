from __future__ import annotations

import pathlib

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import xorq.api as xo
from xorq.caching import ParquetCache, ParquetSnapshotCache
from xorq.caching.strategy import ModificationTimeStrategy, SnapshotStrategy
from xorq.catalog.expr_utils import build_expr_context_zip, load_expr_from_zip
from xorq.common.utils.graph_utils import replace_nodes, walk_nodes
from xorq.common.utils.provenance_utils import get_expr_hash
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


@pytest.mark.uv_export
@pytest.mark.parametrize(
    "backend_factory",
    (
        pytest.param(lambda: xo.datafusion.connect(), id="datafusion"),
        pytest.param(lambda: xo.duckdb.connect(), id="duckdb"),
    ),
)
def test_loaded_dt_has_stable_token_across_zip_reloads(
    tmp_path: pathlib.Path, backend_factory: object
) -> None:
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


# --- a HashingTag is part of a cache's identity, on every path --------------


@pytest.fixture
def tagged_cache(tmp_path):
    """``make(tag, hashing=True)`` -> a cached expr over one table, tagged with a
    hashing tag (or a plain one), untagged when ``tag`` is None;
    ``written()`` -> keys on disk."""
    con = xo.connect()
    t = con.create_table("t", pd.DataFrame({"a": [1, 2, 3]}))
    # relative_path pinned: another test may leave an absolute default behind,
    # which would send the files outside tmp_path
    cache = ParquetSnapshotCache.from_kwargs(
        source=con, base_path=tmp_path, relative_path="parquet"
    )

    def make(tag, hashing=True):
        tagged = t if tag is None else (t.hashing_tag if hashing else t.tag)(tag)
        return tagged.filter(tagged.a > 1).cache(cache)

    make.written = lambda: {p.stem for p in cache.storage.path.glob("*.parquet")}
    return make


@pytest.mark.parametrize(
    "wrap",
    [
        pytest.param(lambda c: c, id="bare"),
        # a plain tag above the CachedNode: stripped without re-walking into the parent
        pytest.param(lambda c: c.tag("outer"), id="outer-tag"),
        # a hashing tag above the CachedNode: stripped by the boundary pass before
        # cache runs, so the CachedNode is the root the cache pass stamps
        pytest.param(lambda c: c.hashing_tag("outer"), id="outer-hashing-tag"),
        pytest.param(
            lambda c: c.tag("plain").hashing_tag("outer"), id="outer-both-tags"
        ),
        # the cache inside a RemoteTable payload: keyed by that payload's own transform
        pytest.param(lambda c: c.into_backend(xo.connect(), "rt"), id="payload"),
    ],
)
def test_hashing_tag_cache_key_agrees_on_every_path(tagged_cache, wrap):
    """Execution must write the artifact ``ls.get_key()`` named (so
    ``cache_exists`` flips) for a hashing-tagged parent, and stamp it with the
    build hash of the cached expression as written. For this fixture's bare
    ``ParquetSnapshotCache`` root, ``Cache.calc_key`` and
    ``ExprMetadata.projected_cache_key`` reduce to ``get_key`` and are not
    re-asserted (that does not hold for a wrapped root)."""
    cached = tagged_cache("v1")
    key = cached.ls.get_key()
    assert cached.ls.cache_exists() is False
    wrap(cached).execute()
    assert tagged_cache.written() == {key}
    assert cached.ls.cache_exists() is True
    footer = pq.read_schema(cached.ls.get_cache_path()).metadata
    assert footer[b"xorq:expr_hash"] == get_expr_hash(cached).encode()


def test_hashing_tags_cache_distinctly(tagged_cache):
    """Two expressions differing only by hashing-tag metadata write two entries."""
    v1, v2 = tagged_cache("v1"), tagged_cache("v2")
    v1.execute()
    v2.execute()
    assert v1.ls.get_key() != v2.ls.get_key()
    assert len(tagged_cache.written()) == 2


def test_hashing_tag_changes_the_cache_entry(tagged_cache):
    """A hashing tag is part of the cache identity: the tagged expression keys,
    and materializes, apart from the same expression without it."""
    plain, tagged = tagged_cache(None), tagged_cache("v1")
    assert plain.ls.get_key() != tagged.ls.get_key()
    plain.execute()
    tagged.execute()
    assert tagged_cache.written() == {plain.ls.get_key(), tagged.ls.get_key()}


def test_plain_tag_is_transparent_and_hashing_tag_is_not(tagged_cache):
    """At execution a plain tag shares the untagged entry; a hashing tag gets its own."""
    untagged = tagged_cache(None)
    plain = tagged_cache("v1", hashing=False)
    hashing = tagged_cache("v1")
    for expr in (untagged, plain, hashing):
        expr.execute()
    assert plain.ls.get_key() == untagged.ls.get_key()
    assert tagged_cache.written() == {untagged.ls.get_key(), hashing.ls.get_key()}
