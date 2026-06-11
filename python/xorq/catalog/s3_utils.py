from __future__ import annotations

import uuid
from pathlib import PurePath
from typing import Any

import attr


S3_SECRET_FIELDS = frozenset(
    {"aws_access_key_id", "aws_secret_access_key", "aws_session_token"}
)

_DEFAULT_PORTS = {"http": "80", "https": "443"}


def serialize_fields(obj: Any, *, include_secrets: bool = False) -> dict[str, Any]:
    """Return non-None attrs fields as a dict, stringifying paths and stripping secrets."""
    return {
        k: str(v) if isinstance(v, PurePath) else v
        for k, v in attr.asdict(obj, recurse=False).items()
        if v is not None and (include_secrets or k not in S3_SECRET_FIELDS)
    }


def make_endpoint_url(
    host: str | None,
    port: str | int | None = None,
    protocol: str | None = None,
) -> str | None:
    """Build a ``protocol://host[:port]`` URL, omitting default ports."""
    if host is None:
        return None
    protocol = protocol or "https"
    default_port = _DEFAULT_PORTS.get(protocol)
    port_str = str(port) if port is not None else None
    port_suffix = f":{port_str}" if port_str and port_str != default_port else ""
    return f"{protocol}://{host}{port_suffix}"


def make_boto3_client(
    *,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_session_token: str | None = None,
    region: str | None = None,
    endpoint_url: str | None = None,
) -> Any:
    """Create a boto3 S3 client, with checksum workarounds for S3-compatible services."""
    import boto3  # noqa: PLC0415
    from botocore.config import Config as BotoConfig  # noqa: PLC0415

    kwargs: dict[str, str | BotoConfig] = {}
    if aws_access_key_id:
        kwargs["aws_access_key_id"] = aws_access_key_id
    if aws_secret_access_key:
        kwargs["aws_secret_access_key"] = aws_secret_access_key
    if aws_session_token:
        kwargs["aws_session_token"] = aws_session_token
    if region and region != "auto":
        kwargs["region_name"] = region
    if endpoint_url:
        kwargs["endpoint_url"] = endpoint_url
        # S3-compatible services (GCS, R2, MinIO) don't understand the
        # x-amz-checksum-* headers that newer boto3 injects by default;
        # the extra headers break the v4 signature.
        kwargs["config"] = BotoConfig(request_checksum_calculation="when_required")
    return boto3.client("s3", **kwargs)


def check_bucket(
    client: Any,
    bucket: str,
    *,
    endpoint_url: str | None = None,
    check_write: bool = False,
    probe_prefix: str = "",
) -> dict[str, Any]:
    """Verify credentials can reach *bucket*; optionally probe write access."""
    client.head_bucket(Bucket=bucket)
    listing = client.list_objects_v2(Bucket=bucket, MaxKeys=1)
    result: dict[str, Any] = {
        "bucket": bucket,
        "endpoint_url": endpoint_url or "(AWS default)",
        "key_count": listing.get("KeyCount", 0),
    }
    if check_write:
        result["writable"] = probe_write(client, bucket, prefix=probe_prefix)
    return result


def probe_write(client: Any, bucket: str, *, prefix: str = "") -> bool:
    """Test write access by initiating and aborting a multipart upload."""
    from botocore.exceptions import ClientError  # noqa: PLC0415

    probe_suffix = f".xorq-write-probe-{uuid.uuid4().hex[:12]}"
    probe_key = f"{prefix}/{probe_suffix}" if prefix else probe_suffix
    try:
        resp = client.create_multipart_upload(Bucket=bucket, Key=probe_key)
        client.abort_multipart_upload(
            Bucket=bucket, Key=probe_key, UploadId=resp["UploadId"]
        )
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("AccessDenied", "403"):
            return False
        raise


def assert_readonly(
    client: Any,
    bucket: str,
    *,
    endpoint_url: str | None = None,
    probe_prefix: str = "",
) -> None:
    """Raise ``ValueError`` if the credentials can write to *bucket*."""
    result = check_bucket(
        client,
        bucket,
        endpoint_url=endpoint_url,
        check_write=True,
        probe_prefix=probe_prefix,
    )
    if result["writable"]:
        raise ValueError(
            f"Credentials for bucket {bucket!r} have write access; "
            f"expected read-only credentials."
        )


class S3ClientMixin:
    """Mixin for attrs classes that need ``check_bucket`` / ``assert_readonly``.

    Subclasses must define ``bucket``, ``_make_boto3_client()``,
    ``_boto3_endpoint_url``, and ``_probe_prefix``.
    """

    @property
    def _boto3_endpoint_url(self) -> str | None:
        return make_endpoint_url(self.host, self.port, self.protocol)

    @property
    def _probe_prefix(self) -> str:
        return ""

    def check_bucket(self, check_write: bool = False) -> dict[str, Any]:
        return check_bucket(
            self._make_boto3_client(),
            self.bucket,
            endpoint_url=self._boto3_endpoint_url,
            check_write=check_write,
            probe_prefix=self._probe_prefix,
        )

    def assert_readonly(self) -> None:
        assert_readonly(
            self._make_boto3_client(),
            self.bucket,
            endpoint_url=self._boto3_endpoint_url,
            probe_prefix=self._probe_prefix,
        )
