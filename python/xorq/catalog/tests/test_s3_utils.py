"""Unit tests for xorq.catalog.s3_utils — pure-function tests with mocked boto3."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import attr
import pytest

from xorq.catalog.s3_utils import (
    assert_readonly,
    check_bucket,
    make_boto3_client,
    make_endpoint_url,
    probe_write,
    serialize_fields,
)


botocore = pytest.importorskip("botocore")
from botocore.exceptions import ClientError  # noqa: E402


# ---------------------------------------------------------------------------
# make_endpoint_url
# ---------------------------------------------------------------------------


def test_make_endpoint_url_none_host_returns_none() -> None:
    assert make_endpoint_url(None) is None


def test_make_endpoint_url_defaults_to_https() -> None:
    assert make_endpoint_url("s3.example.com") == "https://s3.example.com"


def test_make_endpoint_url_explicit_http() -> None:
    assert make_endpoint_url("localhost", protocol="http") == "http://localhost"


def test_make_endpoint_url_non_default_port() -> None:
    url = make_endpoint_url("localhost", port=9000, protocol="http")
    assert url == "http://localhost:9000"


def test_make_endpoint_url_default_https_port_omitted() -> None:
    url = make_endpoint_url("s3.example.com", port=443, protocol="https")
    assert url == "https://s3.example.com"


def test_make_endpoint_url_default_http_port_omitted() -> None:
    url = make_endpoint_url("localhost", port=80, protocol="http")
    assert url == "http://localhost"


def test_make_endpoint_url_non_default_https_port_included() -> None:
    url = make_endpoint_url("s3.example.com", port=8443)
    assert url == "https://s3.example.com:8443"


def test_make_endpoint_url_port_as_string() -> None:
    url = make_endpoint_url("localhost", port="9000", protocol="http")
    assert url == "http://localhost:9000"


def test_make_endpoint_url_port_none_no_suffix() -> None:
    url = make_endpoint_url("s3.example.com", port=None)
    assert url == "https://s3.example.com"


# ---------------------------------------------------------------------------
# make_boto3_client
# ---------------------------------------------------------------------------


def test_make_boto3_client_minimal_call() -> None:
    with patch("boto3.client") as mock_client:
        make_boto3_client()
        mock_client.assert_called_once_with("s3")


def test_make_boto3_client_with_credentials() -> None:
    with patch("boto3.client") as mock_client:
        make_boto3_client(
            aws_access_key_id="AKID",
            aws_secret_access_key="SECRET",
        )
        kwargs = mock_client.call_args[1]
        assert kwargs["aws_access_key_id"] == "AKID"
        assert kwargs["aws_secret_access_key"] == "SECRET"


def test_make_boto3_client_region_passed() -> None:
    with patch("boto3.client") as mock_client:
        make_boto3_client(region="us-west-2")
        kwargs = mock_client.call_args[1]
        assert kwargs["region_name"] == "us-west-2"


def test_make_boto3_client_region_auto_excluded() -> None:
    with patch("boto3.client") as mock_client:
        make_boto3_client(region="auto")
        kwargs = mock_client.call_args[1]
        assert "region_name" not in kwargs


def test_make_boto3_client_region_none_excluded() -> None:
    with patch("boto3.client") as mock_client:
        make_boto3_client(region=None)
        kwargs = mock_client.call_args[1]
        assert "region_name" not in kwargs


def test_make_boto3_client_endpoint_url_sets_config() -> None:
    with patch("boto3.client") as mock_client:
        make_boto3_client(endpoint_url="http://localhost:9000")
        kwargs = mock_client.call_args[1]
        assert kwargs["endpoint_url"] == "http://localhost:9000"
        assert "config" in kwargs


def test_make_boto3_client_no_endpoint_url_no_config() -> None:
    with patch("boto3.client") as mock_client:
        make_boto3_client()
        kwargs = mock_client.call_args[1]
        assert "config" not in kwargs


def test_make_boto3_client_empty_credentials_excluded() -> None:
    with patch("boto3.client") as mock_client:
        make_boto3_client(
            aws_access_key_id="",
            aws_secret_access_key="",
        )
        kwargs = mock_client.call_args[1]
        assert "aws_access_key_id" not in kwargs
        assert "aws_secret_access_key" not in kwargs


# ---------------------------------------------------------------------------
# check_bucket / probe_write / assert_readonly
# ---------------------------------------------------------------------------


def test_check_bucket_basic() -> None:
    client = MagicMock()
    client.list_objects_v2.return_value = {"KeyCount": 5}
    result = check_bucket(client, "my-bucket")
    assert result["bucket"] == "my-bucket"
    assert result["key_count"] == 5
    assert result["endpoint_url"] == "(AWS default)"
    assert "writable" not in result
    client.head_bucket.assert_called_once_with(Bucket="my-bucket")


def test_check_bucket_custom_endpoint_url() -> None:
    client = MagicMock()
    client.list_objects_v2.return_value = {"KeyCount": 0}
    result = check_bucket(client, "b", endpoint_url="http://localhost:9000")
    assert result["endpoint_url"] == "http://localhost:9000"


def test_check_bucket_check_write_true() -> None:
    client = MagicMock()
    client.list_objects_v2.return_value = {"KeyCount": 0}
    client.create_multipart_upload.return_value = {"UploadId": "uid"}
    result = check_bucket(client, "b", check_write=True)
    assert "writable" in result
    assert result["writable"] is True


def test_probe_write_writable_bucket() -> None:
    client = MagicMock()
    client.create_multipart_upload.return_value = {"UploadId": "uid"}
    assert probe_write(client, "b") is True
    client.abort_multipart_upload.assert_called_once()


def test_probe_write_access_denied() -> None:
    client = MagicMock()
    client.create_multipart_upload.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "denied"}},
        "CreateMultipartUpload",
    )
    assert probe_write(client, "b") is False


def test_probe_write_403_code() -> None:
    client = MagicMock()
    client.create_multipart_upload.side_effect = ClientError(
        {"Error": {"Code": "403", "Message": "forbidden"}},
        "CreateMultipartUpload",
    )
    assert probe_write(client, "b") is False


def test_probe_write_other_error_reraises() -> None:
    client = MagicMock()
    client.create_multipart_upload.side_effect = ClientError(
        {"Error": {"Code": "InternalError", "Message": "oops"}},
        "CreateMultipartUpload",
    )
    with pytest.raises(ClientError, match="InternalError"):
        probe_write(client, "b")


def test_probe_write_with_prefix() -> None:
    client = MagicMock()
    client.create_multipart_upload.return_value = {"UploadId": "uid"}
    probe_write(client, "b", prefix="my/prefix")
    key = client.create_multipart_upload.call_args[1]["Key"]
    assert key.startswith("my/prefix/")


def test_assert_readonly_raises_when_writable() -> None:
    client = MagicMock()
    client.list_objects_v2.return_value = {"KeyCount": 0}
    client.create_multipart_upload.return_value = {"UploadId": "uid"}
    with pytest.raises(ValueError, match="write access"):
        assert_readonly(client, "b")


def test_assert_readonly_passes_when_readonly() -> None:
    client = MagicMock()
    client.list_objects_v2.return_value = {"KeyCount": 0}
    client.create_multipart_upload.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "denied"}},
        "CreateMultipartUpload",
    )
    assert_readonly(client, "b")


# ---------------------------------------------------------------------------
# serialize_fields
# ---------------------------------------------------------------------------


def test_serialize_fields_filters_none_and_secrets() -> None:
    @attr.s(auto_attribs=True)
    class _Cfg:
        bucket: str = "b"
        aws_access_key_id: str | None = "AKID"
        region: str | None = None

    obj = _Cfg()
    result = serialize_fields(obj)
    assert result == {"bucket": "b"}
    assert "aws_access_key_id" not in result
    assert "region" not in result


def test_serialize_fields_include_secrets() -> None:
    @attr.s(auto_attribs=True)
    class _Cfg:
        bucket: str = "b"
        aws_access_key_id: str = "AKID"

    result = serialize_fields(_Cfg(), include_secrets=True)
    assert result["aws_access_key_id"] == "AKID"


def test_serialize_fields_path_converted_to_str() -> None:
    @attr.s(auto_attribs=True)
    class _Cfg:
        directory: Path = Path("/tmp/store")

    result = serialize_fields(_Cfg())
    assert result["directory"] == "/tmp/store"
    assert isinstance(result["directory"], str)
