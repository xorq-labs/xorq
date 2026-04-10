from xorq.caching.storage import resolve_parquet_cache_path


def test_resolve_parquet_cache_path_uses_xorq_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("xorq.caching.storage.get_xorq_cache_dir", lambda: tmp_path)
    result = resolve_parquet_cache_path("my_cache", "abc123")
    assert result == tmp_path / "my_cache" / "abc123.parquet"


def test_resolve_parquet_cache_path_explicit_base_path(tmp_path, monkeypatch):
    base = tmp_path / "explicit"
    monkeypatch.setattr(
        "xorq.caching.storage.get_xorq_cache_dir", lambda: tmp_path / "other"
    )
    result = resolve_parquet_cache_path("my_cache", "abc123", base_path=base)
    assert result == base / "my_cache" / "abc123.parquet"
