from __future__ import annotations

import importlib.metadata

from xorq.caching.storage import REMOTE_PUT_BACKENDS, resolve_parquet_cache_path


def test_resolve_parquet_cache_path_uses_xorq_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "xorq.common.utils.caching_utils.get_xorq_cache_dir", lambda: tmp_path
    )
    result = resolve_parquet_cache_path("my_cache", "abc123")
    assert result == tmp_path / "my_cache" / "abc123.parquet"


def test_resolve_parquet_cache_path_explicit_base_path(tmp_path, monkeypatch):
    base = tmp_path / "explicit"
    monkeypatch.setattr(
        "xorq.common.utils.caching_utils.get_xorq_cache_dir",
        lambda: tmp_path / "other",
    )
    result = resolve_parquet_cache_path("my_cache", "abc123", base_path=base)
    assert result == base / "my_cache" / "abc123.parquet"


# ``REMOTE_PUT_BACKENDS`` decides whether ``SourceStorage.put`` stays
# server-side or pulls the whole result through client memory. It is one of
# several per-backend registries kept by hand, and it fell a backend behind
# ``defer_utils._ADBC_BACKENDS`` without anything noticing -- redshift was
# added to that one and not to this one in the same change.
#
# The invariant is NOT that the registries agree: they classify different
# capabilities, and sqlite belongs in one and not the other. It is that every
# registered backend has been *considered* for this one. A new backend that
# nobody classified fails here by name instead of silently defaulting to the
# client-memory branch.
_NOT_REMOTE_PUT = frozenset(
    (
        # local/in-process engines: there is no server to push a CTAS to
        "xorq_datafusion",
        "datafusion",
        "duckdb",
        "pandas",
        "sqlite",
        # remote, but no ``read_record_batches`` to take the out-of-core path
        "trino",
        # remote, with a ``read_record_batches`` (own or inherited), but not yet
        # evaluated for the server-side path (see the FIXME on
        # ``REMOTE_PUT_BACKENDS``)
        "databricks",
        "bigquery",
        "pyiceberg",
        "gizmosql",
    )
)


def test_every_registered_backend_is_classified_for_remote_put() -> None:
    entry_points = importlib.metadata.entry_points(group="xorq.backends")
    registered = frozenset(ep.name for ep in entry_points)

    unclassified = registered - REMOTE_PUT_BACKENDS - _NOT_REMOTE_PUT
    assert not unclassified, (
        f"backend(s) {sorted(unclassified)} are registered but absent from both "
        f"REMOTE_PUT_BACKENDS and the explicit exclusion list. A backend that is "
        f"neither caches by materialising its whole result in client memory, "
        f"which is a silent performance cliff rather than an error."
    )
    # and the exclusion list must not drift into naming backends that no longer exist
    assert not (_NOT_REMOTE_PUT - registered)


def test_redshift_is_registered_for_the_server_side_put_path() -> None:
    """It has both a ``read_record_batches`` and a server-side CTAS, so the
    client-memory branch would pull a warehouse-sized result down only to send
    it straight back.

    Registry membership only: ``SourceStorage.put`` is not exercised. On
    Redshift it cannot complete yet either way -- ``put`` ends in
    ``self.get(key)``, which binds through ``con.table()``, and Redshift's
    table introspection is not in this backend yet."""
    assert "redshift" in REMOTE_PUT_BACKENDS
