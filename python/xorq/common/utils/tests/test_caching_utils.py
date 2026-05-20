from pathlib import Path

import pytest

from xorq.caching.strategy import SnapshotStrategy, snapshot_normalize_read
from xorq.common.utils.caching_utils import get_xorq_cache_dir
from xorq.common.utils.dasher._relations import _normalize_read_xorq


def test_default_caching_dir():
    actual_dir = get_xorq_cache_dir()
    assert actual_dir is not None
    assert isinstance(actual_dir, Path)

    assert actual_dir.match("**/.cache/xorq/")


# --- SnapshotStrategy.normalize_backend ------------------------------------


def test_snapshot_normalize_backend_in_memory():
    """In-memory backends get structure-only (name, None) normalization."""

    for name in ("pandas", "duckdb", "datafusion", "xorq_datafusion"):

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

    class MultiRead:
        schema = "fake-schema"

        def __init__(self, paths):
            self.read_kwargs = (("hash_path", paths),)
            self.normalize_method = fake_normalize_method

    read = MultiRead([str(f1), str(f2)])

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

    class SingleRead:
        schema = "fake-schema"

        def __init__(self, paths):
            self.read_kwargs = (("hash_path", paths),)
            self.normalize_method = fake_normalize_method

    read = SingleRead([path_str])

    snap_result = snapshot_normalize_read(read)
    assert ("path", path_str) in snap_result[2]

    result = _normalize_read_xorq(read)
    assert result[0] == "xorq.Read"
