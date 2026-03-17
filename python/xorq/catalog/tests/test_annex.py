import pytest

from xorq.catalog.annex import (
    DirectoryRemoteConfig,
    S3RemoteConfig,
    remote_config_from_dict,
)


class TestDirectoryRemoteConfigRoundTrip:
    def test_to_dict_includes_type(self):
        rc = DirectoryRemoteConfig(name="mydir", directory="/tmp/store")
        d = rc.to_dict()
        assert d["type"] == "directory"

    def test_to_dict_includes_all_fields(self):
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

    def test_round_trip(self):
        original = DirectoryRemoteConfig(
            name="mydir", directory="/tmp/store", encryption="none"
        )
        restored = DirectoryRemoteConfig.from_dict(original.to_dict())
        assert restored == original

    def test_round_trip_via_dispatcher(self):
        original = DirectoryRemoteConfig(name="d", directory="/data")
        restored = remote_config_from_dict(original.to_dict())
        assert restored == original

    def test_from_dict_with_overrides(self):
        d = {"type": "directory", "name": "old", "directory": "/old"}
        rc = DirectoryRemoteConfig.from_dict(d, directory="/new")
        assert rc.directory == "/new"
        assert rc.name == "old"


class TestS3RemoteConfigRoundTrip:
    @pytest.fixture()
    def secrets(self):
        return {
            "aws_access_key_id": "AKID",
            "aws_secret_access_key": "SECRET",
        }

    @pytest.fixture()
    def minimal(self, secrets):
        return S3RemoteConfig(name="mys3", bucket="my-bucket", **secrets)

    @pytest.fixture()
    def full(self, secrets):
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
            **secrets,
        )

    @pytest.fixture()
    def minio(self, secrets):
        return S3RemoteConfig.make_minio_remote(
            name="minio",
            bucket="test-bucket",
            host="172.19.0.2",
            **secrets,
        )

    def test_to_dict_excludes_secrets(self, minimal):
        d = minimal.to_dict()
        assert "aws_access_key_id" not in d
        assert "aws_secret_access_key" not in d

    def test_to_dict_excludes_none_optional_fields(self, minimal):
        d = minimal.to_dict()
        assert "host" not in d
        assert "region" not in d

    def test_to_dict_includes_set_optional_fields(self, full):
        d = full.to_dict()
        assert d["host"] == "s3.example.com"
        assert d["region"] == "us-east-1"
        assert d["fileprefix"] == "catalog/"

    def test_to_dict_type(self, minimal):
        assert minimal.to_dict()["type"] == "S3"

    def test_round_trip_minimal(self, minimal, secrets):
        restored = S3RemoteConfig.from_dict(minimal.to_dict(), **secrets)
        assert restored == minimal

    def test_round_trip_full(self, full, secrets):
        restored = S3RemoteConfig.from_dict(full.to_dict(), **secrets)
        assert restored == full

    def test_round_trip_minio(self, minio, secrets):
        restored = S3RemoteConfig.from_dict(minio.to_dict(), **secrets)
        assert restored == minio

    def test_round_trip_via_dispatcher(self, full, secrets):
        restored = remote_config_from_dict(full.to_dict(), **secrets)
        assert restored == full

    def test_from_dict_requires_secrets(self, minimal):
        d = minimal.to_dict()
        with pytest.raises(TypeError):
            S3RemoteConfig.from_dict(d)

    def test_from_dict_with_different_secrets(self, minimal):
        d = minimal.to_dict()
        restored = S3RemoteConfig.from_dict(
            d,
            aws_access_key_id="OTHER_AKID",
            aws_secret_access_key="OTHER_SECRET",
        )
        assert restored.aws_access_key_id == "OTHER_AKID"
        assert restored.bucket == minimal.bucket


class TestRemoteConfigFromDict:
    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="unknown remote type"):
            remote_config_from_dict({"type": "bogus"})

    def test_missing_type_raises(self):
        with pytest.raises(ValueError, match="unknown remote type"):
            remote_config_from_dict({"name": "foo"})

    def test_dispatches_directory(self):
        d = {"type": "directory", "name": "d", "directory": "/tmp"}
        rc = remote_config_from_dict(d)
        assert isinstance(rc, DirectoryRemoteConfig)

    def test_dispatches_s3(self):
        d = {"type": "S3", "name": "s", "bucket": "b", "encryption": "none"}
        rc = remote_config_from_dict(
            d, aws_access_key_id="k", aws_secret_access_key="s"
        )
        assert isinstance(rc, S3RemoteConfig)
