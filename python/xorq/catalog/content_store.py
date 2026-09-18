from __future__ import annotations

import abc
import base64
import hashlib
import ipaddress
import json
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from functools import cached_property
from http.client import HTTPException
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import SplitResult, urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)

import attr
import toolz
import yaml12
from attr import field, frozen
from attr.validators import in_, instance_of, matches_re, optional

from xorq.catalog.enums import ContentStoreType
from xorq.catalog.exceptions import (
    ContentIntegrityError,
    ContentStoreCapabilityError,
    ContentStoreError,
)
from xorq.catalog.s3_utils import (
    S3_SECRET_FIELDS,
    S3ClientMixin,
    make_boto3_client,
    make_endpoint_url,
    serialize_fields,
)
from xorq.common.utils.env_utils import EnvConfigable, env_templates_dir
from xorq.common.utils.file_utils import file_digest


POINTER_VERSION = "xorq-pointer v1"

# validated to prevent path traversal from untrusted cloned repos
_SAFE_CATALOG_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_POINTER_RE = re.compile(
    rb"xorq-pointer v1\nsha256 ([0-9a-f]{64})\nsize (0|[1-9][0-9]*)\n\Z"
)

_DEFAULT_CACHE_DIR = Path("~/.cache/xorq/content")
_DEFAULT_CACHE_MAX_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB
_MAX_PRESIGNED_BLOB_BYTES = 5_000_000_000
_CONTROL_RESPONSE_MAX_BYTES = 256 * 1024
_CONTROL_TIMEOUT_SECONDS = 300
_PRESIGNED_BATCH_SIZE = 10
_PRESIGNED_EXPIRY_MARGIN_SECONDS = 5
_TRANSFER_TIMEOUT_SECONDS = 300
_TRANSFER_CHUNK_BYTES = 1024 * 1024
_CATALOG_TOKEN_ENV = "XORQ_CATALOG_TOKEN"
_CATALOG_TOKEN_SERVICE_ENV = "XORQ_CATALOG_TOKEN_SERVICE_URL"

_PresignedRequest = tuple[str, dict[str, str], datetime]


def _normalize_region(value: str | None) -> str | None:
    if value == "auto":
        return None
    return value


def _strip_trailing_slashes(value: str) -> str:
    return value.rstrip("/")


def _non_empty_str(instance: Any, attribute: attr.Attribute, value: Any) -> None:
    if not value:
        raise ValueError(f"'{attribute.name}' must not be empty")


def _coerce_port(value: int | str | None) -> int | None:
    if value is None:
        return None
    port = int(value)
    if not (1 <= port <= 65535):
        raise ValueError(f"port must be 1-65535, got {port}")
    return port


def _canonical_uuid(value: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (AttributeError, ValueError):
        raise ValueError(f"invalid catalog UUID: {value!r}") from None


def _is_loopback_host(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _split_http_url(value: str, *, label: str, allow_username: bool) -> SplitResult:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            f"{label} must be a non-empty URL without surrounding whitespace"
        )
    if len(value) > 2048:
        raise ValueError(f"{label} is too long")
    try:
        parsed = urlsplit(value)
        # Accessing port forces urllib to validate it.
        parsed.port
    except ValueError as exc:
        raise ValueError(f"{label} is invalid") from exc
    if (
        not parsed.hostname
        or parsed.password is not None
        or (parsed.username is not None and not allow_username)
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            f"{label} must not contain credentials, a query, or a fragment"
        )
    if parsed.scheme == "https":
        return parsed
    if parsed.scheme == "http" and _is_loopback_host(parsed.hostname):
        return parsed
    raise ValueError(
        f"{label} must use HTTPS (loopback HTTP is allowed for development)"
    )


def _validate_service_url(value: str) -> str:
    _split_http_url(value, label="service_url", allow_username=False)
    return value


def _service_url_identity(value: str, *, label: str) -> tuple[str, str, int, str]:
    parsed = _split_http_url(value, label=label, allow_username=False)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return (
        parsed.scheme,
        parsed.hostname.lower(),
        port,
        parsed.path.rstrip("/") or "/",
    )


def _validate_remote_binding(service_url: str, remote_url: str) -> None:
    service = _split_http_url(service_url, label="service_url", allow_username=False)
    remote = _split_http_url(remote_url, label="Git remote URL", allow_username=True)
    service_port = service.port or (443 if service.scheme == "https" else 80)
    remote_port = remote.port or (443 if remote.scheme == "https" else 80)
    if (
        service.scheme != remote.scheme
        or service.hostname.lower() != remote.hostname.lower()
        or service_port != remote_port
    ):
        raise ValueError("presigned service_url does not match the catalog Git remote")

    service_path = service.path.rstrip("/")
    prefix = f"{service_path}/" if service_path else "/"
    relative = remote.path.removeprefix(prefix)
    if relative == remote.path or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}/"
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}\.git/?",
        relative,
    ):
        raise ValueError(
            "catalog Git remote must be an owner/slug.git URL below service_url"
        )


def compute_sha256(path: str | Path) -> str:
    return file_digest(path, hashlib.sha256)


def compute_content_key(catalog_id: str, sha256: str) -> str:
    if not _SAFE_CATALOG_ID_RE.match(catalog_id):
        raise ValueError(f"Unsafe catalog_id: {catalog_id!r}")
    if not _SHA256_RE.match(sha256):
        raise ValueError(f"Invalid sha256: {sha256!r}")
    return f"{catalog_id}/{sha256[:2]}/{sha256[2:4]}/{sha256}.zip"


@frozen
class ContentSpec:
    """Immutable identity and expected size for one content-addressed object."""

    key: str = field(validator=(instance_of(str), _non_empty_str))
    sha256: str = field(validator=(instance_of(str), matches_re(_SHA256_RE)))
    size: int = field(validator=instance_of(int))

    def __attrs_post_init__(self) -> None:
        if isinstance(self.size, bool) or self.size < 0:
            raise ValueError("content size must be a non-negative integer")


@contextmanager
def atomic_write(dest: Path) -> Iterator[Path]:
    """Yield a tmp path in the same directory; on success replace *dest*, on error clean up."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".tmp")
    tmp_path = Path(tmp)
    try:
        os.close(fd)
        if dest.exists():
            os.chmod(tmp, dest.stat().st_mode)
        yield tmp_path
        tmp_path.replace(dest)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def write_pointer(path: str | Path, sha256: str, size: int) -> None:
    with atomic_write(Path(path)) as tmp:
        tmp.write_text(f"{POINTER_VERSION}\nsha256 {sha256}\nsize {size}\n")


def parse_pointer(path: str | Path, *, canonical: bool = False) -> tuple[str, int]:
    if canonical:
        match = _CANONICAL_POINTER_RE.fullmatch(Path(path).read_bytes())
        if match is None:
            raise ValueError(f"Invalid pointer file: {path}")
        sha256 = match.group(1).decode("ascii")
        size = int(match.group(2))
        if size > _MAX_PRESIGNED_BLOB_BYTES:
            raise ValueError(f"Invalid pointer file: {path}")
        return sha256, size

    lines = Path(path).read_text().strip().splitlines()
    if len(lines) != 3 or lines[0] != POINTER_VERSION:
        raise ValueError(f"Invalid pointer file: {path}")
    sha_parts = lines[1].split(" ", 1)
    size_parts = lines[2].split(" ", 1)
    if (
        len(sha_parts) != 2
        or sha_parts[0] != "sha256"
        or len(size_parts) != 2
        or size_parts[0] != "size"
    ):
        raise ValueError(f"Invalid pointer file: {path}")
    sha256 = sha_parts[1]
    if not _SHA256_RE.match(sha256):
        raise ValueError(f"Invalid pointer file: {path}")
    try:
        size = int(size_parts[1])
    except ValueError:
        raise ValueError(f"Invalid pointer file: {path}") from None
    if size < 0:
        raise ValueError(f"Invalid pointer file: {path}")
    return sha256, size


class ContentStore(abc.ABC):
    """ABC for external content storage backends."""

    client_managed_lifecycle = True

    @abc.abstractmethod
    def put(
        self, key: str, local_path: str | Path, *, sha256: str | None = None
    ) -> None: ...

    @abc.abstractmethod
    def get(self, key: str, local_path: str | Path) -> None: ...

    @abc.abstractmethod
    def exists(self, key: str) -> bool: ...

    @abc.abstractmethod
    def delete(self, key: str) -> bool: ...

    @abc.abstractmethod
    def list_keys(self, prefix: str = "") -> Iterator[str]: ...

    def ensure_present(
        self, key: str, local_path: str | Path, *, sha256: str | None = None
    ) -> bool:
        """Upload one object if needed and report whether bytes were transferred."""
        local_path = Path(local_path)
        digest = sha256 or compute_sha256(local_path)
        spec = ContentSpec(key=key, sha256=digest, size=local_path.stat().st_size)
        return key in self.ensure_present_many(((spec, local_path),))

    def ensure_present_many(
        self, objects: Iterable[tuple[ContentSpec, str | Path]]
    ) -> set[str]:
        """Default multi-object upload implementation for client-managed stores."""
        uploaded: set[str] = set()
        for spec, local_path in objects:
            if not self.exists(spec.key):
                self.put(spec.key, local_path, sha256=spec.sha256)
                uploaded.add(spec.key)
        return uploaded

    def get_many(self, objects: Iterable[tuple[ContentSpec, str | Path]]) -> None:
        """Default multi-object download implementation."""
        for spec, local_path in objects:
            # Keep legacy ContentStore implementations source-compatible. Stores
            # that need pointer metadata (the hosted adapter) override this method.
            self.get(spec.key, local_path)


@frozen
class DirectoryContentStore(ContentStore):
    """Content store backed by a local directory."""

    directory: Path = field(validator=instance_of(Path), converter=Path)

    def _key_path(self, key: str) -> Path:
        return self.directory / key

    def put(
        self, key: str, local_path: str | Path, *, sha256: str | None = None
    ) -> None:
        with atomic_write(self._key_path(key)) as tmp:
            shutil.copy2(local_path, tmp)
            if sha256 is not None:
                actual = compute_sha256(tmp)
                if actual != sha256:
                    raise ContentIntegrityError(
                        f"SHA256 mismatch after copy: expected {sha256}, got {actual}"
                    )

    def get(self, key: str, local_path: str | Path) -> None:
        src = self._key_path(key)
        if not src.exists():
            raise FileNotFoundError(f"Content not found in store: {key}")
        local_path = Path(local_path)
        with atomic_write(local_path) as tmp:
            shutil.copy2(src, tmp)

    def exists(self, key: str) -> bool:
        return self._key_path(key).exists()

    def delete(self, key: str) -> bool:
        p = self._key_path(key)
        if p.exists():
            p.unlink()
            return True
        return False

    def list_keys(self, prefix: str = "") -> Iterator[str]:
        search_dir = self.directory / prefix if prefix else self.directory
        if not search_dir.is_dir():
            return
        for p in search_dir.rglob("*"):
            if p.is_file() and not p.name.endswith(".tmp"):
                yield p.relative_to(self.directory).as_posix()


@frozen
class S3ContentStore(ContentStore):
    """Content store backed by an S3-compatible bucket.

    attrs @frozen allows cached_property (see GitPointerBackend docstring).
    """

    bucket: str = field(validator=instance_of(str))
    prefix: str = field(
        validator=instance_of(str), converter=_strip_trailing_slashes, default=""
    )
    region: str | None = field(validator=optional(instance_of(str)), default=None)
    aws_access_key_id: str | None = field(
        validator=optional(instance_of(str)), default=None, repr=False
    )
    aws_secret_access_key: str | None = field(
        validator=optional(instance_of(str)), default=None, repr=False
    )
    aws_session_token: str | None = field(
        validator=optional(instance_of(str)), default=None, repr=False
    )
    host: str | None = field(validator=optional(instance_of(str)), default=None)
    port: int | None = field(converter=_coerce_port, default=None)
    protocol: str | None = field(validator=optional(instance_of(str)), default=None)

    def _s3_key(self, key: str) -> str:
        if self.prefix:
            return f"{self.prefix}/{key}"
        return key

    @cached_property
    def _client(self) -> Any:
        return make_boto3_client(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
            region=self.region,
            endpoint_url=make_endpoint_url(self.host, self.port, self.protocol),
        )

    def put(
        self, key: str, local_path: str | Path, *, sha256: str | None = None
    ) -> None:
        local_path = Path(local_path)
        if sha256 is not None:
            actual = compute_sha256(local_path)
            if actual != sha256:
                raise ContentIntegrityError(
                    f"SHA256 mismatch before upload: expected {sha256}, got {actual}"
                )
        expected_size = local_path.stat().st_size
        s3_key = self._s3_key(key)
        self._client.upload_file(str(local_path), self.bucket, s3_key)
        resp = self._client.head_object(Bucket=self.bucket, Key=s3_key)
        actual_size = resp["ContentLength"]
        if actual_size != expected_size:
            try:
                self._client.delete_object(Bucket=self.bucket, Key=s3_key)
            except Exception:
                import structlog  # noqa: PLC0415

                structlog.get_logger(__name__).warning(
                    "Failed to delete corrupt S3 object %s/%s during cleanup",
                    self.bucket,
                    s3_key,
                    exc_info=True,
                )
            raise ContentIntegrityError(
                f"Size mismatch after S3 upload for {key}: "
                f"expected {expected_size}, got {actual_size}"
            )

    def get(self, key: str, local_path: str | Path) -> None:
        client = self._client
        s3_key = self._s3_key(key)
        local_path = Path(local_path)
        resp = client.head_object(Bucket=self.bucket, Key=s3_key)
        expected_size = resp["ContentLength"]
        with atomic_write(local_path) as tmp:
            client.download_file(self.bucket, s3_key, str(tmp))
            actual_size = tmp.stat().st_size
            if actual_size != expected_size:
                raise ContentIntegrityError(
                    f"Size mismatch after S3 download for {key}: "
                    f"expected {expected_size}, got {actual_size}"
                )

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError  # noqa: PLC0415

        client = self._client
        try:
            client.head_object(Bucket=self.bucket, Key=self._s3_key(key))
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return False
            raise

    def delete(self, key: str) -> bool:
        existed = self.exists(key)
        self._client.delete_object(Bucket=self.bucket, Key=self._s3_key(key))
        return existed

    def list_keys(self, prefix: str = "") -> Iterator[str]:
        s3_prefix = self._s3_key(prefix)
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=s3_prefix):
            for obj in page.get("Contents", ()):
                full_key = obj["Key"]
                if self.prefix:
                    yield full_key.removeprefix(self.prefix).lstrip("/")
                else:
                    yield full_key


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _CatalogServiceHTTPError(ContentStoreError):
    def __init__(
        self,
        message: str,
        *,
        status: int,
        error_code: str | None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.error_code = error_code


@frozen
class PresignedContentStore(ContentStore):
    """Hosted blob client using service-issued presigned URLs."""

    client_managed_lifecycle = False

    catalog_id: str = field(converter=_canonical_uuid)
    service_url: str = field(converter=_validate_service_url)
    remote_url: str = field(validator=(instance_of(str), _non_empty_str))

    def __attrs_post_init__(self) -> None:
        _validate_remote_binding(self.service_url, self.remote_url)

    @cached_property
    def _opener(self):
        return build_opener(_RejectRedirects())

    @cached_property
    def _direct_opener(self):
        return build_opener(ProxyHandler({}), _RejectRedirects())

    def _endpoint(self, suffix: str) -> str:
        return (
            f"{self.service_url.rstrip('/')}/v1/catalogs/"
            f"{self.catalog_id}/blobs/{suffix}"
        )

    def _token(self, *, required: bool) -> str | None:
        token = os.environ.get(_CATALOG_TOKEN_ENV)
        if token is None:
            if required:
                raise ContentStoreError(
                    f"{_CATALOG_TOKEN_ENV} is required for authenticated hosted "
                    "catalog requests"
                )
            return None
        token_service_url = os.environ.get(_CATALOG_TOKEN_SERVICE_ENV)
        if token_service_url is None:
            if required:
                raise ContentStoreError(
                    f"{_CATALOG_TOKEN_SERVICE_ENV} is required to scope "
                    f"{_CATALOG_TOKEN_ENV} to one hosted service"
                )
            return None
        try:
            token_service = _service_url_identity(
                token_service_url,
                label=_CATALOG_TOKEN_SERVICE_ENV,
            )
        except ValueError as exc:
            if required:
                raise ContentStoreError(str(exc)) from exc
            return None
        if token_service != _service_url_identity(
            self.service_url,
            label="service_url",
        ):
            if required:
                raise ContentStoreError(
                    f"{_CATALOG_TOKEN_ENV} is scoped to a different hosted service"
                )
            return None
        if (
            not token
            or len(token) > 16 * 1024
            or token != token.strip()
            or any(
                ord(character) < 0x21 or ord(character) > 0x7E for character in token
            )
        ):
            raise ContentStoreError(
                f"{_CATALOG_TOKEN_ENV} must contain one non-empty HTTP token"
            )
        return token

    @staticmethod
    def _http_failure(error: HTTPError, operation: str) -> ContentStoreError:
        detail = ""
        error_code = None
        try:
            body = error.read(_CONTROL_RESPONSE_MAX_BYTES + 1)
            if len(body) <= _CONTROL_RESPONSE_MAX_BYTES:
                payload = json.loads(body)
                error_code = payload.get("error") if isinstance(payload, dict) else None
                if isinstance(error_code, str) and re.fullmatch(
                    r"[a-z][a-z0-9_]{0,63}", error_code
                ):
                    detail = f" ({error_code})"
        except (OSError, HTTPException, UnicodeDecodeError, json.JSONDecodeError):
            pass
        return _CatalogServiceHTTPError(
            f"{operation} failed with HTTP status {error.code}{detail}",
            status=error.code,
            error_code=error_code if isinstance(error_code, str) else None,
        )

    def _open(self, request: Request, *, timeout: int, operation: str):
        hostname = urlsplit(request.full_url).hostname
        opener = (
            self._direct_opener
            if hostname is not None and _is_loopback_host(hostname)
            else self._opener
        )
        try:
            return opener.open(request, timeout=timeout)
        except HTTPError as exc:
            raise self._http_failure(exc, operation) from None
        except (OSError, URLError, HTTPException) as exc:
            raise ContentStoreError(
                f"{operation} failed due to a network error"
            ) from exc

    def _control_post(
        self,
        suffix: str,
        payload: dict[str, Any] | None,
        *,
        token: str | None,
    ) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        if payload is None:
            body = b""
        else:
            body = json.dumps(payload, separators=(",", ":")).encode()
            headers["Content-Type"] = "application/json"
        request = Request(
            self._endpoint(suffix),
            data=body,
            headers=headers,
            method="POST",
        )
        with self._open(
            request,
            timeout=_CONTROL_TIMEOUT_SECONDS,
            operation="catalog service request",
        ) as response:
            if response.status != 200:
                raise ContentStoreError(
                    f"catalog service request returned HTTP status {response.status}"
                )
            try:
                raw = response.read(_CONTROL_RESPONSE_MAX_BYTES + 1)
            except (OSError, HTTPException) as exc:
                raise ContentStoreError(
                    "catalog service response failed during transfer"
                ) from exc
        if len(raw) > _CONTROL_RESPONSE_MAX_BYTES:
            raise ContentStoreError("catalog service response is too large")
        try:
            decoded = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContentStoreError("catalog service returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise ContentStoreError(
                "catalog service returned an invalid response shape"
            )
        return decoded

    def _validate_spec(self, spec: ContentSpec) -> None:
        if spec.size > _MAX_PRESIGNED_BLOB_BYTES:
            raise ContentStoreError(
                f"hosted blobs cannot exceed {_MAX_PRESIGNED_BLOB_BYTES} bytes"
            )
        expected_key = compute_content_key(self.catalog_id, spec.sha256)
        if spec.key != expected_key:
            raise ContentStoreError(
                "content key does not match the hosted catalog and SHA-256"
            )

    @staticmethod
    def _validate_local(spec: ContentSpec, local_path: Path) -> None:
        try:
            actual_size = local_path.stat().st_size
        except OSError as exc:
            raise ContentStoreError("content file is not readable") from exc
        if actual_size != spec.size:
            raise ContentIntegrityError(
                f"Size mismatch before upload: expected {spec.size}, got {actual_size}"
            )
        actual_sha256 = compute_sha256(local_path)
        if actual_sha256 != spec.sha256:
            raise ContentIntegrityError(
                "SHA256 mismatch before upload: "
                f"expected {spec.sha256}, got {actual_sha256}"
            )

    @staticmethod
    def _response_identity(item: Any) -> tuple[str, int]:
        if not isinstance(item, dict):
            raise ContentStoreError("catalog service returned an invalid response item")
        sha256 = item.get("sha256")
        size = item.get("size")
        if (
            not isinstance(sha256, str)
            or isinstance(size, bool)
            or not isinstance(size, int)
        ):
            raise ContentStoreError(
                "catalog service returned an invalid object identity"
            )
        return sha256, size

    @staticmethod
    def _upload_id(item: dict[str, Any]) -> str:
        try:
            return str(uuid.UUID(item.get("upload_id")))
        except (AttributeError, ValueError):
            raise ContentStoreError(
                "catalog service returned an invalid upload ID"
            ) from None

    @classmethod
    def _response_items(
        cls,
        payload: dict[str, Any],
        field_name: str,
        specs: dict[tuple[str, int], ContentSpec],
    ) -> dict[tuple[str, int], dict[str, Any]]:
        raw_items = payload.get(field_name)
        if not isinstance(raw_items, list):
            raise ContentStoreError(
                "catalog service returned an invalid response shape"
            )
        items: dict[tuple[str, int], dict[str, Any]] = {}
        for raw_item in raw_items:
            identity = cls._response_identity(raw_item)
            if identity not in specs or identity in items:
                raise ContentStoreError("catalog service returned an unexpected object")
            items[identity] = raw_item
        if items.keys() != specs.keys():
            raise ContentStoreError("catalog service omitted a requested object")
        return items

    @staticmethod
    def _spec_batches(
        specs: dict[tuple[str, int], ContentSpec],
    ) -> Iterator[dict[tuple[str, int], ContentSpec]]:
        items = tuple(specs.items())
        for start in range(0, len(items), _PRESIGNED_BATCH_SIZE):
            yield dict(items[start : start + _PRESIGNED_BATCH_SIZE])

    def _control_batch(
        self,
        suffix: str,
        specs: dict[tuple[str, int], ContentSpec],
        *,
        token_required: bool,
    ) -> dict[str, Any]:
        request_payload = {
            "objects": [
                {"sha256": spec.sha256, "size": spec.size} for spec in specs.values()
            ]
        }
        token = self._token(required=token_required)
        try:
            payload = self._control_post(
                suffix,
                request_payload,
                token=token,
            )
        except _CatalogServiceHTTPError as exc:
            rejected_optional_credentials = (exc.status, exc.error_code) in {
                (401, "unauthorized"),
                (403, "forbidden"),
            }
            if token_required or not rejected_optional_credentials:
                raise
            if token is None:
                raise
            return self._control_post(
                suffix,
                request_payload,
                token=None,
            )
        return payload

    def _batch_results(
        self,
        suffix: str,
        field_name: str,
        specs: dict[tuple[str, int], ContentSpec],
        *,
        token_required: bool,
    ) -> dict[tuple[str, int], dict[str, Any]]:
        try:
            payload = self._control_batch(
                suffix,
                specs,
                token_required=token_required,
            )
        except _CatalogServiceHTTPError as exc:
            if (
                exc.status == 400
                and exc.error_code == "invalid_request"
                and len(specs) > 1
            ):
                items = tuple(specs.items())
                midpoint = len(items) // 2
                left = self._batch_results(
                    suffix,
                    field_name,
                    dict(items[:midpoint]),
                    token_required=token_required,
                )
                right = self._batch_results(
                    suffix,
                    field_name,
                    dict(items[midpoint:]),
                    token_required=token_required,
                )
                return left | right
            raise
        return self._response_items(payload, field_name, specs)

    @staticmethod
    def _presigned_request(
        raw: Any,
    ) -> _PresignedRequest:
        if not isinstance(raw, dict):
            raise ContentStoreError("catalog service omitted a presigned request")
        url = raw.get("url")
        headers = raw.get("headers")
        expires_at = raw.get("expires_at")
        if (
            not isinstance(url, str)
            or len(url) > 16 * 1024
            or not isinstance(headers, dict)
            or not isinstance(expires_at, str)
            or not expires_at
        ):
            raise ContentStoreError(
                "catalog service returned an invalid presigned request"
            )
        try:
            parsed = urlsplit(url)
            parsed.port
        except ValueError as exc:
            raise ContentStoreError(
                "catalog service returned an invalid presigned URL"
            ) from exc
        if (
            not (
                parsed.scheme == "https"
                or (
                    parsed.scheme == "http"
                    and parsed.hostname is not None
                    and _is_loopback_host(parsed.hostname)
                )
            )
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise ContentStoreError(
                "catalog service returned an invalid presigned URL; HTTPS is "
                "required except for loopback testing"
            )
        for key, value in headers.items():
            if (
                not isinstance(key, str)
                or re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", key) is None
                or not isinstance(value, str)
                or len(value) > 16 * 1024
                or any(character in value for character in "\r\n\0")
            ):
                raise ContentStoreError(
                    "catalog service returned invalid signed headers"
                )
        try:
            normalized_expiry = (
                f"{expires_at[:-1]}+00:00" if expires_at.endswith("Z") else expires_at
            )
            expiry = datetime.fromisoformat(normalized_expiry)
        except ValueError as exc:
            raise ContentStoreError(
                "catalog service returned an invalid presigned expiry"
            ) from exc
        if expiry.tzinfo is None or expiry.utcoffset() is None:
            raise ContentStoreError(
                "catalog service returned a timezone-naive presigned expiry"
            )
        return url, dict(headers), expiry.astimezone(timezone.utc)

    @staticmethod
    def _request_expires_soon(request: _PresignedRequest) -> bool:
        _url, _headers, expiry = request
        return expiry <= datetime.now(timezone.utc) + timedelta(
            seconds=_PRESIGNED_EXPIRY_MARGIN_SECONDS
        )

    def _put_presigned(
        self, spec: ContentSpec, local_path: Path, request_data: _PresignedRequest
    ) -> None:
        url, headers, expiry = request_data
        if expiry <= datetime.now(timezone.utc):
            raise ContentStoreError("presigned upload request has expired")
        folded_headers = {key.lower(): value for key, value in headers.items()}
        if folded_headers.get("content-length") != str(spec.size):
            raise ContentStoreError("presigned upload has an invalid content-length")
        checksum = base64.b64encode(bytes.fromhex(spec.sha256)).decode()
        if folded_headers.get("x-amz-checksum-sha256") != checksum:
            raise ContentStoreError("presigned upload has an invalid SHA-256 header")
        try:
            with local_path.open("rb") as source:
                request = Request(url, data=source, headers=headers, method="PUT")
                with self._open(
                    request,
                    timeout=_TRANSFER_TIMEOUT_SECONDS,
                    operation="presigned upload",
                ) as response:
                    if not 200 <= response.status < 300:
                        raise ContentStoreError(
                            f"presigned upload returned HTTP status {response.status}"
                        )
        except (OSError, HTTPException) as exc:
            raise ContentStoreError("content file could not be uploaded") from exc

    def put(
        self, key: str, local_path: str | Path, *, sha256: str | None = None
    ) -> None:
        self.ensure_present(key, local_path, sha256=sha256)

    def ensure_present_many(
        self, objects: Iterable[tuple[ContentSpec, str | Path]]
    ) -> set[str]:
        by_identity: dict[tuple[str, int], tuple[ContentSpec, Path]] = {}
        sizes_by_sha256: dict[str, int] = {}
        for spec, raw_path in objects:
            self._validate_spec(spec)
            local_path = Path(raw_path)
            self._validate_local(spec, local_path)
            previous_size = sizes_by_sha256.setdefault(spec.sha256, spec.size)
            if previous_size != spec.size:
                raise ContentStoreError(
                    "the same content digest has conflicting upload sizes"
                )
            identity = (spec.sha256, spec.size)
            by_identity.setdefault(identity, (spec, local_path))
        if not by_identity:
            return set()

        specs = {identity: value[0] for identity, value in by_identity.items()}
        uploaded: set[str] = set()
        upload_ids: set[str] = set()
        for batch in self._spec_batches(specs):
            results = self._batch_results(
                "uploads:batch",
                "uploads",
                batch,
                token_required=True,
            )
            for identity, spec in batch.items():
                local_path = by_identity[identity][1]
                result = results[identity]
                status = result.get("status")
                request_data = None
                if status == "upload_required":
                    request_data = self._presigned_request(result.get("request"))
                    if self._request_expires_soon(request_data):
                        result = self._batch_results(
                            "uploads:batch",
                            "uploads",
                            {identity: spec},
                            token_required=True,
                        )[identity]
                        status = result.get("status")
                        if status == "upload_required":
                            request_data = self._presigned_request(
                                result.get("request")
                            )
                upload_id = self._upload_id(result)
                if upload_id in upload_ids:
                    raise ContentStoreError(
                        "catalog service returned a duplicate upload ID"
                    )
                upload_ids.add(upload_id)
                if status == "already_verified":
                    continue
                if status != "upload_required":
                    raise ContentStoreError(
                        "catalog service returned an unknown upload status"
                    )
                assert request_data is not None
                self._put_presigned(spec, local_path, request_data)
                completed = self._control_post(
                    f"uploads/{upload_id}/complete",
                    None,
                    token=self._token(required=True),
                )
                if self._response_identity(completed) != identity:
                    raise ContentStoreError(
                        "catalog service completed a different object"
                    )
                if self._upload_id(completed) != upload_id:
                    raise ContentStoreError(
                        "catalog service completed a different upload"
                    )
                if completed.get("status") != "verified":
                    raise ContentStoreError("catalog service did not verify the upload")
                uploaded.add(spec.key)
        return uploaded

    def get(
        self,
        key: str,
        local_path: str | Path,
        *,
        sha256: str | None = None,
        size: int | None = None,
    ) -> None:
        if sha256 is None or size is None:
            raise ContentStoreError(
                "hosted downloads require the pointer SHA-256 and expected size"
            )
        self.get_many(((ContentSpec(key=key, sha256=sha256, size=size), local_path),))

    def _download_presigned(
        self, spec: ContentSpec, local_path: Path, request_data: _PresignedRequest
    ) -> None:
        url, headers, expiry = request_data
        if expiry <= datetime.now(timezone.utc):
            raise ContentStoreError("presigned download request has expired")
        request = Request(url, headers=headers, method="GET")
        digest = hashlib.sha256()
        total = 0
        with atomic_write(local_path) as tmp:
            try:
                with (
                    self._open(
                        request,
                        timeout=_TRANSFER_TIMEOUT_SECONDS,
                        operation="presigned download",
                    ) as response,
                    tmp.open("wb") as destination,
                ):
                    if response.status != 200:
                        raise ContentStoreError(
                            f"presigned download returned HTTP status {response.status}"
                        )
                    while chunk := response.read(_TRANSFER_CHUNK_BYTES):
                        total += len(chunk)
                        if total > spec.size:
                            raise ContentIntegrityError(
                                f"Size mismatch after download: expected {spec.size}, got more"
                            )
                        digest.update(chunk)
                        destination.write(chunk)
            except (OSError, HTTPException) as exc:
                raise ContentStoreError(
                    "presigned download failed during transfer"
                ) from exc
            if total != spec.size:
                raise ContentIntegrityError(
                    f"Size mismatch after download: expected {spec.size}, got {total}"
                )
            actual = digest.hexdigest()
            if actual != spec.sha256:
                raise ContentIntegrityError(
                    f"SHA256 mismatch after download: expected {spec.sha256}, got {actual}"
                )

    def get_many(self, objects: Iterable[tuple[ContentSpec, str | Path]]) -> None:
        by_identity: dict[tuple[str, int], tuple[ContentSpec, list[Path]]] = {}
        sizes_by_sha256: dict[str, int] = {}
        for spec, raw_path in objects:
            self._validate_spec(spec)
            previous_size = sizes_by_sha256.setdefault(spec.sha256, spec.size)
            if previous_size != spec.size:
                raise ContentStoreError(
                    "the same content digest has conflicting download sizes"
                )
            identity = (spec.sha256, spec.size)
            if identity in by_identity:
                by_identity[identity][1].append(Path(raw_path))
            else:
                by_identity[identity] = (spec, [Path(raw_path)])
        if not by_identity:
            return

        specs = {identity: value[0] for identity, value in by_identity.items()}
        for batch in self._spec_batches(specs):
            results = self._batch_results(
                "downloads:batch",
                "downloads",
                batch,
                token_required=False,
            )
            for identity, spec in batch.items():
                request_data = self._presigned_request(results[identity].get("request"))
                for local_path in by_identity[identity][1]:
                    if self._request_expires_soon(request_data):
                        refreshed = self._batch_results(
                            "downloads:batch",
                            "downloads",
                            {identity: spec},
                            token_required=False,
                        )
                        request_data = self._presigned_request(
                            refreshed[identity].get("request")
                        )
                    self._download_presigned(spec, local_path, request_data)

    @staticmethod
    def _unsupported(operation: str) -> ContentStoreCapabilityError:
        return ContentStoreCapabilityError(
            f"{operation} is unavailable to hosted clients; blob lifecycle is "
            "owned by the hosted catalog service"
        )

    def exists(self, key: str) -> bool:
        raise self._unsupported("blob existence checks")

    def delete(self, key: str) -> bool:
        raise self._unsupported("blob deletion")

    def list_keys(self, prefix: str = "") -> Iterator[str]:
        raise self._unsupported("blob enumeration")


@frozen
class ContentCache:
    """LRU disk cache for content store objects.

    *max_bytes* semantics: positive → bounded LRU, 0 → disabled (no
    persistent caching; ``fetch_from`` still downloads to a temporary
    location), negative → unlimited (never evict).
    """

    cache_dir: Path = field(validator=instance_of(Path))
    max_bytes: int = field(validator=instance_of(int))

    def __attrs_post_init__(self) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OSError(
                f"Content cache directory is not writable: {self.cache_dir}"
            ) from exc
        if not os.access(self.cache_dir, os.W_OK):
            raise OSError(f"Content cache directory is not writable: {self.cache_dir}")

    @property
    def disabled(self) -> bool:
        return self.max_bytes == 0

    def _path(self, key: str) -> Path:
        return self.cache_dir / key

    def contains(self, key: str) -> bool:
        if self.disabled:
            return False
        return self._path(key).exists()

    def get_path(self, key: str) -> Path | None:
        if self.disabled:
            return None
        p = self._path(key)
        if not p.exists():
            return None
        os.utime(p)
        return p

    def put(self, key: str, local_path: str | Path) -> None:
        if self.disabled:
            return
        dest = self._path(key)
        with atomic_write(dest) as tmp:
            shutil.copy2(local_path, tmp)
        os.utime(dest)
        self._maybe_evict(protect=key)

    def fetch_from(
        self,
        store: ContentStore,
        key: str,
    ) -> Path:
        if self.disabled:
            fd, tmp = tempfile.mkstemp(suffix=".xorq")
            tmp_path = Path(tmp)
            try:
                os.close(fd)
                store.get(key, tmp_path)
                return tmp_path
            except BaseException:
                tmp_path.unlink(missing_ok=True)
                raise
        dest = self._path(key)
        with atomic_write(dest) as tmp:
            store.get(key, tmp)
        os.utime(dest)
        self._maybe_evict(protect=key)
        return dest

    def _maybe_evict(self, protect: str | None = None) -> None:
        if self.max_bytes < 0:
            return
        protect_path = self._path(protect) if protect is not None else None
        entries: list[tuple[float, int, Path]] = []
        total = 0
        for p in self.cache_dir.rglob("*"):
            if p.is_file() and not p.name.endswith(".tmp"):
                st = p.stat()
                entries.append((st.st_atime, st.st_size, p))
                total += st.st_size
        if total <= self.max_bytes:
            return
        entries.sort()
        for _atime, size, path in entries:
            if total <= self.max_bytes:
                break
            if path == protect_path:
                continue
            path.unlink(missing_ok=True)
            total -= size

    EnvConfig = EnvConfigable.subclass_from_env_file(
        env_templates_dir.joinpath(".env.catalog.content_cache.template"),
        prefix="XORQ_CONTENT_CACHE_",
    )

    @classmethod
    def default(cls) -> ContentCache:
        env_config = cls.EnvConfig.from_env()
        cache_dir = Path(env_config.dir or _DEFAULT_CACHE_DIR).expanduser()
        max_bytes = int(env_config.max_bytes or _DEFAULT_CACHE_MAX_BYTES)
        return cls(cache_dir=cache_dir, max_bytes=max_bytes)


class ContentStoreConfig(abc.ABC):
    """Typed, serializable configuration for constructing a ``ContentStore``."""

    @abc.abstractmethod
    def make_store(self, *, repo: Any = None) -> ContentStore: ...

    @abc.abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    _required_env_field: str = ""
    _required_env_fields: tuple[str, ...] = ()
    _env_field_hint: str = ""

    @classmethod
    def from_env(cls, **kwargs: Any) -> ContentStoreConfig:
        env_config = cls.EnvConfig.from_env()
        env = {
            k: v
            for k, v in attr.asdict(env_config, recurse=False).items()
            if k != "env_file" and v
        }
        merged = {**env, **kwargs}
        required = cls._required_env_fields
        if not required and cls._required_env_field:
            required = (cls._required_env_field,)
        missing = tuple(name for name in required if name not in merged)
        if missing:
            names = ", ".join(repr(name) for name in missing)
            raise ValueError(
                f"{cls.__name__}.from_env() requires {names} "
                f"via {cls._env_field_hint} or as a kwarg"
            )
        return cls(**merged)

    def write_yaml(self, path: str | Path) -> None:
        with atomic_write(Path(path)) as tmp:
            tmp.write_text(yaml12.format_yaml(self.to_dict()))

    @classmethod
    def resolve_fields(
        cls, fields: dict[str, Any], resolve_dir: Path
    ) -> dict[str, Any]:
        return fields

    @classmethod
    def from_dict(
        cls,
        dct: dict[str, Any],
        *,
        ignore_unknown: bool = False,
        resolve_dir: Path | None = None,
    ) -> ContentStoreConfig:
        dct = dict(dct)
        raw_type = dct.pop("type", None)
        if raw_type is None:
            raise ValueError("content store config missing required 'type' field")
        try:
            type_ = ContentStoreType(raw_type)
        except ValueError:
            raise ValueError(f"unknown content store type: {raw_type!r}") from None
        config_cls = _CONTENT_STORE_CONFIG_CLASSES[type_]
        valid_keys = {a.name for a in attr.fields(config_cls)}
        if not ignore_unknown:
            unknown = set(dct) - valid_keys
            if unknown:
                raise ValueError(
                    f"unknown fields for {type_!r} content store config: {unknown}"
                )
        filtered = {k: v for k, v in dct.items() if k in valid_keys}
        if resolve_dir is not None:
            filtered = config_cls.resolve_fields(filtered, resolve_dir)
        return config_cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ContentStoreConfig:
        path = Path(path)
        data = yaml12.read_yaml(path)
        return cls.from_dict(data, resolve_dir=path.parent)


@frozen
class DirectoryContentStoreConfig(ContentStoreConfig):
    _required_env_field = "directory"
    _env_field_hint = "XORQ_CONTENT_STORE_DIRECTORY_DIRECTORY"

    directory: Path = field(
        validator=instance_of(Path), converter=lambda v: Path(v).resolve()
    )
    catalog_id: str = field(
        validator=(instance_of(str), matches_re(_SAFE_CATALOG_ID_RE)),
        factory=lambda: str(uuid.uuid4()),
    )

    EnvConfig = EnvConfigable.subclass_from_env_file(
        env_templates_dir.joinpath(".env.catalog.content_store.directory.template"),
        prefix="XORQ_CONTENT_STORE_DIRECTORY_",
    )

    def make_store(self, *, repo: Any = None) -> DirectoryContentStore:
        return DirectoryContentStore(directory=self.directory)

    def to_dict(self) -> dict[str, Any]:
        return {"type": ContentStoreType.DIRECTORY} | serialize_fields(self)

    def write_yaml(self, path: str | Path) -> None:
        path = Path(path)
        dct = self.to_dict()
        base = path.parent.resolve()
        rel = Path(os.path.relpath(self.directory, base))  # xorq-style: disable=os-path
        dct["directory"] = str(rel)
        with atomic_write(path) as tmp:
            tmp.write_text(yaml12.format_yaml(dct))

    @classmethod
    def resolve_fields(
        cls, fields: dict[str, Any], resolve_dir: Path
    ) -> dict[str, Any]:
        if "directory" in fields and not Path(fields["directory"]).is_absolute():
            fields = dict(fields)
            fields["directory"] = str((resolve_dir / fields["directory"]).resolve())
        return fields


@frozen
class S3ContentStoreConfig(S3ClientMixin, ContentStoreConfig):
    _required_env_field = "bucket"
    _env_field_hint = "XORQ_CONTENT_STORE_S3_BUCKET"

    bucket: str = field(validator=(instance_of(str), _non_empty_str))
    catalog_id: str = field(
        validator=(instance_of(str), matches_re(_SAFE_CATALOG_ID_RE)),
        factory=lambda: str(uuid.uuid4()),
    )
    prefix: str = field(
        validator=instance_of(str), converter=_strip_trailing_slashes, default=""
    )
    region: str | None = field(
        validator=optional(instance_of(str)), converter=_normalize_region, default=None
    )
    aws_access_key_id: str | None = field(
        validator=optional(instance_of(str)), default=None, repr=False
    )
    aws_secret_access_key: str | None = field(
        validator=optional(instance_of(str)), default=None, repr=False
    )
    aws_session_token: str | None = field(
        validator=optional(instance_of(str)), default=None, repr=False
    )
    host: str | None = field(validator=optional(instance_of(str)), default=None)
    port: int | None = field(converter=_coerce_port, default=None)
    protocol: str | None = field(
        validator=optional(in_(("http", "https"))), default=None
    )

    EnvConfig = EnvConfigable.subclass_from_env_file(
        env_templates_dir.joinpath(".env.catalog.content_store.s3.template"),
        prefix="XORQ_CONTENT_STORE_S3_",
    )

    def _resolve_secrets(self) -> dict[str, str]:
        secrets: dict[str, str] = {}
        env_config = self.EnvConfig.from_env()
        for name in S3_SECRET_FIELDS:
            val = getattr(self, name, None)
            if val is None:
                val = getattr(env_config, name, None)
            # explicit empty string means "no secret", blocking env fallback
            if val:
                secrets[name] = val
        return secrets

    def _make_boto3_client(self) -> Any:
        return make_boto3_client(
            **self._resolve_secrets(),
            region=self.region,
            endpoint_url=make_endpoint_url(self.host, self.port, self.protocol),
        )

    @property
    def _probe_prefix(self) -> str:
        return self.prefix

    def make_store(self, *, repo: Any = None) -> S3ContentStore:
        kwargs = (
            toolz.dissoc(
                serialize_fields(self, include_secrets=False),
                "catalog_id",
            )
            | self._resolve_secrets()
        )
        return S3ContentStore(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {"type": ContentStoreType.S3} | serialize_fields(self)

    @classmethod
    def from_env_gcs(cls, **kwargs: Any) -> S3ContentStoreConfig:
        """Like ``from_env``, but with GCS defaults for host/protocol/region."""
        return cls.from_env(**{**_S3_GCS_DEFAULTS, **kwargs})


def _single_remote_url(repo: Any) -> str:
    if repo is None:
        raise ValueError("a Git repository is required for a presigned content store")
    remotes = tuple(repo.remotes)
    if len(remotes) != 1:
        raise ValueError("a presigned catalog requires exactly one Git remote")
    urls = tuple(remotes[0].urls)
    if len(urls) != 1:
        raise ValueError("the presigned catalog Git remote must have exactly one URL")
    return urls[0]


@frozen
class PresignedContentStoreConfig(ContentStoreConfig):
    """Strict committed configuration for the hosted catalog service."""

    _required_env_fields = ("catalog_id", "service_url")
    _env_field_hint = (
        "XORQ_CONTENT_STORE_PRESIGNED_CATALOG_ID and "
        "XORQ_CONTENT_STORE_PRESIGNED_SERVICE_URL"
    )

    catalog_id: str = field(converter=_canonical_uuid)
    service_url: str = field(converter=_validate_service_url)

    EnvConfig = EnvConfigable.subclass_from_env_file(
        env_templates_dir.joinpath(".env.catalog.content_store.presigned.template"),
        prefix="XORQ_CONTENT_STORE_PRESIGNED_",
    )

    def validate_remote_url(self, remote_url: str) -> None:
        """Validate the hosted Git origin before creating or mutating a repo."""
        _validate_remote_binding(self.service_url, remote_url)

    def bound_remote_url(self, repo: Any) -> str:
        """Return the current sole Git URL after validating its service binding."""
        remote_url = _single_remote_url(repo)
        self.validate_remote_url(remote_url)
        return remote_url

    def make_store(self, *, repo: Any = None) -> PresignedContentStore:
        remote_url = self.bound_remote_url(repo)
        return PresignedContentStore(
            catalog_id=self.catalog_id,
            service_url=self.service_url,
            remote_url=remote_url,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"type": ContentStoreType.PRESIGNED} | serialize_fields(self)


_S3_GCS_DEFAULTS: dict[str, str] = {
    "host": "storage.googleapis.com",
    "protocol": "https",
    "region": "auto",
}


_CONTENT_STORE_CONFIG_CLASSES: dict[ContentStoreType, type[ContentStoreConfig]] = {
    ContentStoreType.DIRECTORY: DirectoryContentStoreConfig,
    ContentStoreType.PRESIGNED: PresignedContentStoreConfig,
    ContentStoreType.S3: S3ContentStoreConfig,
}
