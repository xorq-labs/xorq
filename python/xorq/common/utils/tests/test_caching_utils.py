import asyncio
from pathlib import Path

import pytest

import xorq.api as xo
from xorq.caching.strategy import SnapshotStrategy, snapshot_normalize_read
from xorq.common.utils.caching_utils import get_xorq_cache_dir
from xorq.common.utils.dasher import with_caches
from xorq.common.utils.dasher._opaque import (
    _expr_normalize_memo,
    _parent_token_memo,
)
from xorq.common.utils.dasher._relations import _dt_normalize_memo, _normalize_read_xorq
from xorq.common.utils.tests._test_helpers import FakeRead


def test_default_caching_dir():
    actual_dir = get_xorq_cache_dir()
    assert actual_dir is not None
    assert isinstance(actual_dir, Path)

    assert actual_dir.match("**/.cache/xorq/")


# --- SnapshotStrategy.normalize_backend ------------------------------------


@pytest.mark.parametrize("name", ["pandas", "duckdb", "datafusion", "xorq_datafusion"])
def test_snapshot_normalize_backend_in_memory(name):
    """In-memory backends get structure-only (name, None) normalization."""

    class InMemoryBackend:
        pass

    InMemoryBackend.name = name
    assert SnapshotStrategy.normalize_backend(InMemoryBackend()) == (name, None)


def test_snapshot_normalize_backend_remote_falls_through():
    """Remote backends are NOT whitelisted — they fall through to
    data-sensitive HASHER.normalize so different connections produce
    distinct tokens.  We verify the fall-through happened (i.e. the
    fast-path early-return for in-memory backends did NOT fire) rather
    than coupling to xorq_dasher's specific error wording.
    """

    class RemoteBackend:
        name = "postgres"

    # If the in-memory whitelist matched, normalize_backend would return a
    # ``(name, None)`` tuple here.  Falling through to HASHER.normalize on
    # an unregistered fake class raises *some* exception — we accept any.
    with pytest.raises(Exception):  # noqa: B017, PT011
        SnapshotStrategy.normalize_backend(RemoteBackend())


# --- _normalize_read_xorq / snapshot_normalize_read ------------------------


def test_normalize_read_xorq_multi_element_paths(tmp_path):
    """_normalize_read_xorq should handle multi-element path lists the
    same way snapshot_normalize_read does, without raising.
    """
    f1 = tmp_path / "a.parquet"
    f2 = tmp_path / "b.parquet"
    f1.write_bytes(b"data-a")
    f2.write_bytes(b"data-b")

    def fake_normalize_method(p):
        return (("file", str(p)),)

    read = FakeRead([str(f1), str(f2)], normalize_method=fake_normalize_method)

    snap_result = snapshot_normalize_read(read)
    assert snap_result[0] == "snapshot_normalize_read"

    result = _normalize_read_xorq(read)
    assert result[0] == "xorq.Read"
    assert len(result[2]) == 2


def test_normalize_read_xorq_single_path(tmp_path):
    """Single-element path lists already work — regression guard."""
    f = tmp_path / "test.parquet"
    f.write_bytes(b"data")
    path_str = str(f)

    def fake_normalize_method(p):
        return (("file", str(p)),)

    read = FakeRead([path_str], normalize_method=fake_normalize_method)

    snap_result = snapshot_normalize_read(read)
    assert ("path", path_str) in snap_result[2]

    result = _normalize_read_xorq(read)
    assert result[0] == "xorq.Read"


# --- normalization_context installs per-call dasher memos -------------------


def test_normalization_context_installs_per_call_memos():
    """normalization_context must install the three per-call dasher memos
    (_parent_token_memo, _expr_normalize_memo, _dt_normalize_memo) so
    callers that bypass @with_caches still get memoized tokenization."""
    assert _parent_token_memo.get() is None
    assert _expr_normalize_memo.get() is None
    assert _dt_normalize_memo.get() is None

    t = xo.memtable({"a": [1, 2, 3]})
    strategy = SnapshotStrategy()

    with strategy.normalization_context(t) as local:
        assert _parent_token_memo.get() is not None
        assert _expr_normalize_memo.get() is not None
        assert _dt_normalize_memo.get() is not None

        local.tokenize(t)

        assert len(_parent_token_memo.get()) > 0 or len(_expr_normalize_memo.get()) > 0

    assert _parent_token_memo.get() is None
    assert _expr_normalize_memo.get() is None
    assert _dt_normalize_memo.get() is None


# --- with_caches generator / coroutine support ------------------------------


def test_with_caches_sync_generator():
    """with_caches must keep memos alive across yield in sync generators."""

    @with_caches
    def _gen():
        yield 42

    assert _parent_token_memo.get() is None
    items = []
    for item in _gen():
        assert _parent_token_memo.get() is not None
        items.append(item)
    assert items == [42]
    assert _parent_token_memo.get() is None


def test_with_caches_async_generator():
    """with_caches must keep memos alive across yield in async generators."""

    @with_caches
    async def _async_gen():
        yield 42

    async def _run():
        assert _parent_token_memo.get() is None
        items = []
        async for item in _async_gen():
            assert _parent_token_memo.get() is not None
            items.append(item)
        assert items == [42]
        assert _parent_token_memo.get() is None

    asyncio.run(_run())


def test_with_caches_coroutine():
    """with_caches must install memos for plain async def coroutines."""

    @with_caches
    async def _coro():
        assert _parent_token_memo.get() is not None
        return 42

    async def _run():
        assert _parent_token_memo.get() is None
        result = await _coro()
        assert result == 42
        assert _parent_token_memo.get() is None

    asyncio.run(_run())
