import pytest

from xorq.catalog.annex import (
    DirectoryRemoteConfig,
    RsyncRemoteConfig,
    S3RemoteConfig,
    remote_config_from_dict,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def s3_secrets():
    return {
        "aws_access_key_id": "AKID",
        "aws_secret_access_key": "SECRET",
    }


@pytest.fixture()
def s3_minimal(s3_secrets):
    return S3RemoteConfig(name="mys3", bucket="my-bucket", **s3_secrets)


@pytest.fixture()
def s3_full(s3_secrets):
    return S3RemoteConfig(
        name="mys3",
        bucket="my-bucket",
        host="s3.example.com",
        port="443",
        protocol="https",
        requeststyle="path",
        signature="v4",
        region="us-east-1",
        encryption="none",
        fileprefix="catalog/",
        storageclass="STANDARD",
        chunk="1MiB",
        **s3_secrets,
    )


@pytest.fixture()
def s3_minio(s3_secrets):
    return S3RemoteConfig.make_minio_remote(
        name="minio",
        bucket="test-bucket",
        host="172.19.0.2",
        **s3_secrets,
    )


# ---------------------------------------------------------------------------
# DirectoryRemoteConfig round-trip
# ---------------------------------------------------------------------------


def test_directory_to_dict_includes_type():
    rc = DirectoryRemoteConfig(name="mydir", directory="/tmp/store")
    d = rc.to_dict()
    assert d["type"] == "directory"


def test_directory_to_dict_includes_all_fields():
    rc = DirectoryRemoteConfig(
        name="mydir", directory="/tmp/store", encryption="shared"
    )
    d = rc.to_dict()
    assert d == {
        "type": "directory",
        "name": "mydir",
        "directory": "/tmp/store",
        "encryption": "shared",
    }


def test_directory_round_trip():
    original = DirectoryRemoteConfig(
        name="mydir", directory="/tmp/store", encryption="none"
    )
    restored = DirectoryRemoteConfig.from_dict(original.to_dict())
    assert restored == original


def test_directory_round_trip_via_dispatcher():
    original = DirectoryRemoteConfig(name="d", directory="/data")
    restored = remote_config_from_dict(original.to_dict())
    assert restored == original


def test_directory_from_dict_with_overrides():
    d = {"type": "directory", "name": "old", "directory": "/old"}
    rc = DirectoryRemoteConfig.from_dict(d, directory="/new")
    assert rc.directory == "/new"
    assert rc.name == "old"


# ---------------------------------------------------------------------------
# RsyncRemoteConfig round-trip
# ---------------------------------------------------------------------------


def test_rsync_to_dict_includes_type():
    rc = RsyncRemoteConfig(name="myrsync", rsyncurl="user@host:/data")
    d = rc.to_dict()
    assert d["type"] == "rsync"


def test_rsync_to_dict_includes_all_fields():
    rc = RsyncRemoteConfig(
        name="myrsync", rsyncurl="user@host:/data", encryption="shared"
    )
    d = rc.to_dict()
    assert d == {
        "type": "rsync",
        "name": "myrsync",
        "rsyncurl": "user@host:/data",
        "encryption": "shared",
    }


def test_rsync_to_dict_omits_none():
    rc = RsyncRemoteConfig(name="r", rsyncurl="host:/path")
    d = rc.to_dict()
    assert "autoenable" not in d
    assert "shellescape" not in d


def test_rsync_round_trip():
    original = RsyncRemoteConfig(
        name="myrsync", rsyncurl="user@host:/data", encryption="none"
    )
    restored = RsyncRemoteConfig.from_dict(original.to_dict())
    assert restored == original


def test_rsync_round_trip_via_dispatcher():
    original = RsyncRemoteConfig(name="r", rsyncurl="host:/data")
    restored = remote_config_from_dict(original.to_dict())
    assert restored == original


def test_rsync_from_dict_with_overrides():
    d = {"type": "rsync", "name": "old", "rsyncurl": "host:/old"}
    rc = RsyncRemoteConfig.from_dict(d, rsyncurl="host:/new")
    assert rc.rsyncurl == "host:/new"
    assert rc.name == "old"


# ---------------------------------------------------------------------------
# S3RemoteConfig round-trip
# ---------------------------------------------------------------------------


def test_s3_to_dict_excludes_secrets(s3_minimal):
    d = s3_minimal.to_dict()
    assert "aws_access_key_id" not in d
    assert "aws_secret_access_key" not in d


def test_s3_to_dict_excludes_none_optional_fields(s3_minimal):
    d = s3_minimal.to_dict()
    assert "host" not in d
    assert "region" not in d


def test_s3_to_dict_includes_set_optional_fields(s3_full):
    d = s3_full.to_dict()
    assert d["host"] == "s3.example.com"
    assert d["region"] == "us-east-1"
    assert d["fileprefix"] == "catalog/"


def test_s3_to_dict_type(s3_minimal):
    assert s3_minimal.to_dict()["type"] == "S3"


def test_s3_round_trip_minimal(s3_minimal, s3_secrets):
    restored = S3RemoteConfig.from_dict(s3_minimal.to_dict(), **s3_secrets)
    assert restored == s3_minimal


def test_s3_round_trip_full(s3_full, s3_secrets):
    restored = S3RemoteConfig.from_dict(s3_full.to_dict(), **s3_secrets)
    assert restored == s3_full


def test_s3_round_trip_minio(s3_minio, s3_secrets):
    restored = S3RemoteConfig.from_dict(s3_minio.to_dict(), **s3_secrets)
    assert restored == s3_minio


def test_s3_round_trip_via_dispatcher(s3_full, s3_secrets):
    restored = remote_config_from_dict(s3_full.to_dict(), **s3_secrets)
    assert restored == s3_full


def test_s3_from_dict_requires_secrets_without_embedcreds(s3_minimal):
    d = s3_minimal.to_dict()
    with pytest.raises(TypeError):
        S3RemoteConfig.from_dict(d)


# ---------------------------------------------------------------------------
# embedcreds=yes
# ---------------------------------------------------------------------------


def test_s3_to_dict_includes_secrets_when_embedcreds(s3_secrets):
    rc = S3RemoteConfig(name="pub", bucket="pub-bucket", embedcreds="yes", **s3_secrets)
    d = rc.to_dict()
    assert d["aws_access_key_id"] == "AKID"
    assert d["aws_secret_access_key"] == "SECRET"
    assert d["embedcreds"] == "yes"


def test_s3_to_dict_excludes_secrets_when_embedcreds_not_yes(s3_secrets):
    rc = S3RemoteConfig(
        name="priv", bucket="priv-bucket", embedcreds="no", **s3_secrets
    )
    d = rc.to_dict()
    assert "aws_access_key_id" not in d
    assert "aws_secret_access_key" not in d


def test_s3_round_trip_with_embedcreds_no_kwargs(s3_secrets):
    """With embedcreds=yes, round-trip works without supplying secrets as kwargs."""
    original = S3RemoteConfig(
        name="pub", bucket="pub-bucket", embedcreds="yes", **s3_secrets
    )
    d = original.to_dict()
    restored = S3RemoteConfig.from_dict(d)
    assert restored == original


def test_s3_round_trip_via_dispatcher_with_embedcreds(s3_secrets):
    original = S3RemoteConfig(
        name="pub", bucket="pub-bucket", embedcreds="yes", **s3_secrets
    )
    d = original.to_dict()
    restored = remote_config_from_dict(d)
    assert restored == original


def test_s3_has_embedded_creds_property(s3_secrets):
    yes = S3RemoteConfig(name="a", bucket="b", embedcreds="yes", **s3_secrets)
    no = S3RemoteConfig(name="a", bucket="b", embedcreds="no", **s3_secrets)
    none = S3RemoteConfig(name="a", bucket="b", **s3_secrets)
    assert yes.has_embedded_creds is True
    assert no.has_embedded_creds is False
    assert none.has_embedded_creds is False


def test_s3_from_dict_with_different_secrets(s3_minimal):
    d = s3_minimal.to_dict()
    restored = S3RemoteConfig.from_dict(
        d,
        aws_access_key_id="OTHER_AKID",
        aws_secret_access_key="OTHER_SECRET",
    )
    assert restored.aws_access_key_id == "OTHER_AKID"
    assert restored.bucket == s3_minimal.bucket


# ---------------------------------------------------------------------------
# from_env
# ---------------------------------------------------------------------------


def test_directory_from_env_all(monkeypatch):
    monkeypatch.setenv("XORQ_CATALOG_DIRECTORY_NAME", "env-remote")
    monkeypatch.setenv("XORQ_CATALOG_DIRECTORY_DIRECTORY", "/tmp/env-store")
    rc = DirectoryRemoteConfig.from_env()
    assert rc.name == "env-remote"
    assert rc.directory == "/tmp/env-store"
    assert rc.encryption == "none"


def test_directory_from_env_kwargs_override(monkeypatch):
    monkeypatch.setenv("XORQ_CATALOG_DIRECTORY_NAME", "env-name")
    monkeypatch.setenv("XORQ_CATALOG_DIRECTORY_DIRECTORY", "/tmp/env")
    rc = DirectoryRemoteConfig.from_env(name="kwarg-name")
    assert rc.name == "kwarg-name"
    assert rc.directory == "/tmp/env"


def test_directory_from_env_overrides_template_default(monkeypatch):
    monkeypatch.setenv("XORQ_CATALOG_DIRECTORY_NAME", "r")
    monkeypatch.setenv("XORQ_CATALOG_DIRECTORY_DIRECTORY", "/tmp")
    monkeypatch.setenv("XORQ_CATALOG_DIRECTORY_ENCRYPTION", "shared")
    rc = DirectoryRemoteConfig.from_env()
    assert rc.encryption == "shared"


def test_rsync_from_env_all(monkeypatch):
    monkeypatch.setenv("XORQ_CATALOG_RSYNC_NAME", "env-rsync")
    monkeypatch.setenv("XORQ_CATALOG_RSYNC_RSYNCURL", "user@host:/data")
    rc = RsyncRemoteConfig.from_env()
    assert rc.name == "env-rsync"
    assert rc.rsyncurl == "user@host:/data"
    assert rc.encryption == "none"


def test_rsync_from_env_kwargs_override(monkeypatch):
    monkeypatch.setenv("XORQ_CATALOG_RSYNC_NAME", "env-name")
    monkeypatch.setenv("XORQ_CATALOG_RSYNC_RSYNCURL", "host:/path")
    rc = RsyncRemoteConfig.from_env(name="kwarg-name")
    assert rc.name == "kwarg-name"
    assert rc.rsyncurl == "host:/path"


def test_s3_from_env_secrets(monkeypatch):
    monkeypatch.setenv("XORQ_CATALOG_S3_AWS_ACCESS_KEY_ID", "AKID")
    monkeypatch.setenv("XORQ_CATALOG_S3_AWS_SECRET_ACCESS_KEY", "SECRET")
    rc = S3RemoteConfig.from_env(name="s3", bucket="b")
    assert rc.aws_access_key_id == "AKID"
    assert rc.aws_secret_access_key == "SECRET"


def test_s3_from_env_optional_fields(monkeypatch):
    monkeypatch.setenv("XORQ_CATALOG_S3_AWS_ACCESS_KEY_ID", "AKID")
    monkeypatch.setenv("XORQ_CATALOG_S3_AWS_SECRET_ACCESS_KEY", "SECRET")
    monkeypatch.setenv("XORQ_CATALOG_S3_REGION", "us-west-2")
    monkeypatch.setenv("XORQ_CATALOG_S3_HOST", "s3.example.com")
    rc = S3RemoteConfig.from_env(name="s3", bucket="b")
    assert rc.region == "us-west-2"
    assert rc.host == "s3.example.com"
    assert rc.port is None


def test_s3_from_env_kwargs_override(monkeypatch):
    monkeypatch.setenv("XORQ_CATALOG_S3_AWS_ACCESS_KEY_ID", "AKID")
    monkeypatch.setenv("XORQ_CATALOG_S3_AWS_SECRET_ACCESS_KEY", "SECRET")
    monkeypatch.setenv("XORQ_CATALOG_S3_REGION", "us-west-2")
    rc = S3RemoteConfig.from_env(name="s3", bucket="b", region="eu-west-1")
    assert rc.region == "eu-west-1"


def test_s3_from_env_all(monkeypatch):
    monkeypatch.setenv("XORQ_CATALOG_S3_NAME", "full-env")
    monkeypatch.setenv("XORQ_CATALOG_S3_BUCKET", "env-bucket")
    monkeypatch.setenv("XORQ_CATALOG_S3_AWS_ACCESS_KEY_ID", "AKID")
    monkeypatch.setenv("XORQ_CATALOG_S3_AWS_SECRET_ACCESS_KEY", "SECRET")
    rc = S3RemoteConfig.from_env()
    assert rc.name == "full-env"
    assert rc.bucket == "env-bucket"


# ---------------------------------------------------------------------------
# remote_config_from_dict dispatcher
# ---------------------------------------------------------------------------


def test_remote_config_from_dict_unknown_type_raises():
    with pytest.raises(ValueError, match="unknown remote type"):
        remote_config_from_dict({"type": "bogus"})


def test_remote_config_from_dict_missing_type_raises():
    with pytest.raises(ValueError, match="unknown remote type"):
        remote_config_from_dict({"name": "foo"})


def test_remote_config_from_dict_dispatches_directory():
    d = {"type": "directory", "name": "d", "directory": "/tmp"}
    rc = remote_config_from_dict(d)
    assert isinstance(rc, DirectoryRemoteConfig)


def test_remote_config_from_dict_dispatches_rsync():
    d = {"type": "rsync", "name": "r", "rsyncurl": "host:/data"}
    rc = remote_config_from_dict(d)
    assert isinstance(rc, RsyncRemoteConfig)


def test_remote_config_from_dict_dispatches_s3():
    d = {"type": "S3", "name": "s", "bucket": "b", "encryption": "none"}
    rc = remote_config_from_dict(d, aws_access_key_id="k", aws_secret_access_key="s")
    assert isinstance(rc, S3RemoteConfig)
