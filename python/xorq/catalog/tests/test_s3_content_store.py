from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest


boto3 = pytest.importorskip("boto3")
from botocore.exceptions import ClientError  # noqa: E402

from xorq.catalog.content_store import (  # noqa: E402
    ContentIntegrityError,
    S3ContentStore,
    S3ContentStoreConfig,
    compute_sha256,
)
from xorq.catalog.s3_utils import make_boto3_client  # noqa: E402


def _mock_s3_client() -> MagicMock:
    """Return a mock boto3 S3 client with enough behavior for S3ContentStore."""
    client = MagicMock()
    _store: dict[str, bytes] = {}

    def upload_file(local_path: str, bucket: str, key: str, **kwargs: object) -> None:
        _store[key] = Path(local_path).read_bytes()

    def download_file(bucket: str, key: str, local_path: str) -> None:
        if key not in _store:
            raise ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}}, "GetObject"
            )
        Path(local_path).write_bytes(_store[key])

    def head_object(Bucket: str, Key: str) -> dict:  # noqa: N803
        if Key not in _store:
            raise ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
            )
        return {"ContentLength": len(_store[Key])}

    def delete_object(Bucket: str, Key: str) -> None:  # noqa: N803
        _store.pop(Key, None)

    class _MockPaginator:
        def __init__(self, store_ref: dict[str, bytes]) -> None:
            self._store_ref = store_ref

        def paginate(self, Bucket: str, Prefix: str = "") -> object:  # noqa: N803
            contents = [
                {"Key": k} for k in sorted(self._store_ref) if k.startswith(Prefix)
            ]
            yield {"Contents": contents} if contents else {}

    def get_paginator(operation_name: str) -> _MockPaginator:
        if operation_name == "list_objects_v2":
            return _MockPaginator(_store)
        raise ValueError(f"unsupported paginator: {operation_name}")

    client.upload_file = MagicMock(side_effect=upload_file)
    client.download_file = MagicMock(side_effect=download_file)
    client.head_object = MagicMock(side_effect=head_object)
    client.delete_object = MagicMock(side_effect=delete_object)
    client.get_paginator = MagicMock(side_effect=get_paginator)
    client._store = _store
    return client


def _make_s3_store(
    monkeypatch: pytest.MonkeyPatch, client: MagicMock, prefix: str = ""
) -> S3ContentStore:
    monkeypatch.setattr(
        "xorq.catalog.content_store.make_boto3_client", lambda **kw: client
    )
    return S3ContentStore(bucket="test-bucket", prefix=prefix)


def test_s3_content_store_put_get_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _mock_s3_client()
    store = _make_s3_store(monkeypatch, client)

    src = tmp_path / "data.zip"
    src.write_bytes(b"hello s3 store")
    store.put("cat/aa/bb/test.zip", src)

    dest = tmp_path / "out.zip"
    store.get("cat/aa/bb/test.zip", dest)
    assert dest.read_bytes() == b"hello s3 store"


def test_s3_content_store_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _mock_s3_client()
    store = _make_s3_store(monkeypatch, client)

    assert not store.exists("cat/aa/bb/missing.zip")

    src = tmp_path / "data.zip"
    src.write_bytes(b"content")
    store.put("cat/aa/bb/exists.zip", src)
    assert store.exists("cat/aa/bb/exists.zip")


def test_s3_content_store_delete_returns_bool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _mock_s3_client()
    store = _make_s3_store(monkeypatch, client)

    src = tmp_path / "data.zip"
    src.write_bytes(b"content")
    store.put("cat/aa/bb/del.zip", src)

    assert store.delete("cat/aa/bb/del.zip") is True
    assert store.delete("cat/aa/bb/del.zip") is False
    assert not store.exists("cat/aa/bb/del.zip")


def test_s3_content_store_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _mock_s3_client()
    store = _make_s3_store(monkeypatch, client, prefix="my-prefix/")

    src = tmp_path / "data.zip"
    src.write_bytes(b"prefixed")
    store.put("cat/aa/bb/test.zip", src)

    assert "my-prefix/cat/aa/bb/test.zip" in client._store
    assert "cat/aa/bb/test.zip" not in client._store


def test_s3_content_store_prefix_trailing_slash_stripped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _mock_s3_client()
    store = _make_s3_store(monkeypatch, client, prefix="my-prefix///")

    src = tmp_path / "data.zip"
    src.write_bytes(b"prefixed")
    store.put("cat/aa/bb/test.zip", src)

    assert "my-prefix/cat/aa/bb/test.zip" in client._store
    assert store.prefix == "my-prefix"


def test_s3_content_store_put_verifies_sha256_before_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _mock_s3_client()
    store = _make_s3_store(monkeypatch, client)

    src = tmp_path / "data.zip"
    src.write_bytes(b"checksummed")
    sha = compute_sha256(src)
    store.put("cat/aa/bb/chk.zip", src, sha256=sha)
    assert store.exists("cat/aa/bb/chk.zip")

    bad_sha = "0" * 64
    with pytest.raises(ContentIntegrityError, match="SHA256 mismatch before upload"):
        store.put("cat/aa/bb/bad.zip", src, sha256=bad_sha)
    assert not store.exists("cat/aa/bb/bad.zip")


def test_s3_content_store_put_checks_size_after_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _mock_s3_client()
    store = _make_s3_store(monkeypatch, client)

    src = tmp_path / "data.zip"
    src.write_bytes(b"content")
    store.put("cat/aa/bb/ok.zip", src)
    assert store.exists("cat/aa/bb/ok.zip")

    # head_object returns the correct size via the mock, so the normal
    # path succeeds.  Sabotage head_object to return a wrong size:

    real_head = client.head_object.side_effect

    def wrong_size_head(Bucket: str, Key: str) -> dict:  # noqa: N803
        resp = real_head(Bucket=Bucket, Key=Key)
        resp["ContentLength"] += 1
        return resp

    client.head_object.side_effect = wrong_size_head

    with pytest.raises(ContentIntegrityError, match="Size mismatch after S3 upload"):
        store.put("cat/aa/bb/bad.zip", src)
    # the corrupted object should be cleaned up
    client.head_object.side_effect = real_head
    assert not store.exists("cat/aa/bb/bad.zip")


def test_s3_content_store_config_make_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XORQ_CONTENT_STORE_S3_AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("XORQ_CONTENT_STORE_S3_AWS_SECRET_ACCESS_KEY", raising=False)

    config = S3ContentStoreConfig(
        catalog_id="test-cat",
        bucket="my-bucket",
        prefix="pfx",
        region="us-west-2",
    )
    store = config.make_store()
    assert isinstance(store, S3ContentStore)
    assert store.bucket == "my-bucket"
    assert store.prefix == "pfx"
    assert store.region == "us-west-2"


def test_s3_content_store_config_resolve_secrets_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XORQ_CONTENT_STORE_S3_AWS_ACCESS_KEY_ID", "env-key")
    monkeypatch.setenv("XORQ_CONTENT_STORE_S3_AWS_SECRET_ACCESS_KEY", "env-secret")

    config = S3ContentStoreConfig(catalog_id="cat", bucket="b")
    secrets = config._resolve_secrets()
    assert secrets["aws_access_key_id"] == "env-key"
    assert secrets["aws_secret_access_key"] == "env-secret"


def test_s3_content_store_config_resolve_secrets_field_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XORQ_CONTENT_STORE_S3_AWS_ACCESS_KEY_ID", "env-key")
    monkeypatch.setenv("XORQ_CONTENT_STORE_S3_AWS_SECRET_ACCESS_KEY", "env-secret")

    config = S3ContentStoreConfig(
        catalog_id="cat",
        bucket="b",
        aws_access_key_id="field-key",
        aws_secret_access_key="field-secret",
    )
    secrets = config._resolve_secrets()
    assert secrets["aws_access_key_id"] == "field-key"
    assert secrets["aws_secret_access_key"] == "field-secret"


def test_make_boto3_client_drops_auto_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_client_fn = MagicMock()
    monkeypatch.setattr(boto3, "client", mock_client_fn)

    make_boto3_client(region="auto")
    call_kwargs = mock_client_fn.call_args[1]
    assert "region_name" not in call_kwargs

    make_boto3_client(region="us-east-1")
    call_kwargs = mock_client_fn.call_args[1]
    assert call_kwargs["region_name"] == "us-east-1"


def test_s3_content_store_config_from_env_missing_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("XORQ_CONTENT_STORE_S3_CATALOG_ID", raising=False)
    monkeypatch.delenv("XORQ_CONTENT_STORE_S3_BUCKET", raising=False)

    with pytest.raises(ValueError, match="requires 'bucket'"):
        S3ContentStoreConfig.from_env()


def test_s3_content_store_config_from_env_gcs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XORQ_CONTENT_STORE_S3_BUCKET", "my-gcs-bucket")
    monkeypatch.setenv("XORQ_CONTENT_STORE_S3_AWS_ACCESS_KEY_ID", "hmac-key")
    monkeypatch.setenv("XORQ_CONTENT_STORE_S3_AWS_SECRET_ACCESS_KEY", "hmac-secret")
    monkeypatch.delenv("XORQ_CONTENT_STORE_S3_CATALOG_ID", raising=False)

    config = S3ContentStoreConfig.from_env_gcs()
    assert config.bucket == "my-gcs-bucket"
    assert config.host == "storage.googleapis.com"
    assert config.protocol == "https"
    assert config.region is None  # "auto" is normalized to None
    assert config.aws_access_key_id == "hmac-key"
    assert config.aws_secret_access_key == "hmac-secret"


def test_s3_content_store_config_from_env_gcs_kwargs_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XORQ_CONTENT_STORE_S3_BUCKET", "default-bucket")
    monkeypatch.delenv("XORQ_CONTENT_STORE_S3_CATALOG_ID", raising=False)
    monkeypatch.delenv("XORQ_CONTENT_STORE_S3_AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("XORQ_CONTENT_STORE_S3_AWS_SECRET_ACCESS_KEY", raising=False)

    config = S3ContentStoreConfig.from_env_gcs(
        bucket="override-bucket", region="us-central1"
    )
    assert config.bucket == "override-bucket"
    assert config.region == "us-central1"
    assert config.host == "storage.googleapis.com"
    assert config.protocol == "https"


def test_s3_content_store_get_missing_key_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _mock_s3_client()
    store = _make_s3_store(monkeypatch, client)

    dest = tmp_path / "out.zip"
    with pytest.raises(ClientError):
        store.get("cat/aa/bb/missing.zip", dest)


def test_s3_content_store_list_keys_no_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _mock_s3_client()
    store = _make_s3_store(monkeypatch, client)

    src = tmp_path / "data.zip"
    src.write_bytes(b"content")
    store.put("cat/aa/bb/file1.zip", src)
    store.put("cat/aa/cc/file2.zip", src)

    keys = sorted(store.list_keys())
    assert keys == ["cat/aa/bb/file1.zip", "cat/aa/cc/file2.zip"]


def test_s3_content_store_list_keys_with_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _mock_s3_client()
    store = _make_s3_store(monkeypatch, client, prefix="my-prefix")

    src = tmp_path / "data.zip"
    src.write_bytes(b"content")
    store.put("cat/aa/bb/file1.zip", src)
    store.put("other/aa/bb/file2.zip", src)

    cat_keys = sorted(store.list_keys(prefix="cat"))
    assert cat_keys == ["cat/aa/bb/file1.zip"]

    all_keys = sorted(store.list_keys())
    assert len(all_keys) == 2


def test_s3_content_store_list_keys_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _mock_s3_client()
    store = _make_s3_store(monkeypatch, client)

    assert list(store.list_keys()) == []
    assert list(store.list_keys(prefix="cat")) == []
