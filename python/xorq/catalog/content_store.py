from __future__ import annotations

import abc
import hashlib
import logging
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from typing import Any

import attr
import toolz
import yaml12
from attr import field, frozen
from attr.validators import in_, instance_of, matches_re, optional

from xorq.catalog.constants import ContentStoreType
from xorq.catalog.s3_utils import (
    S3_SECRET_FIELDS,
    S3ClientMixin,
    make_boto3_client,
    make_endpoint_url,
    serialize_fields,
)
from xorq.common.exceptions import XorqError
from xorq.common.utils.env_utils import EnvConfigable, env_templates_dir


logger = logging.getLogger(__name__)

POINTER_VERSION = "xorq-pointer v1"

# validated to prevent path traversal from untrusted cloned repos
_SAFE_CATALOG_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_DEFAULT_CACHE_DIR = Path("~/.cache/xorq/content")
_DEFAULT_CACHE_MAX_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB


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


def compute_sha256(path: str | Path) -> str:
    from xorq.common.utils.defer_utils import file_digest  # noqa: PLC0415

    return file_digest(path, hashlib.sha256)


def content_key(catalog_id: str, sha256: str) -> str:
    if not _SAFE_CATALOG_ID_RE.match(catalog_id):
        raise ValueError(f"Unsafe catalog_id: {catalog_id!r}")
    if not _SHA256_RE.match(sha256):
        raise ValueError(f"Invalid sha256: {sha256!r}")
    return f"{catalog_id}/{sha256[:2]}/{sha256[2:4]}/{sha256}.zip"


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


def parse_pointer(path: str | Path) -> tuple[str, int]:
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


class ContentIntegrityError(XorqError):
    """Raised when content does not match the expected checksum."""


class ContentStore(abc.ABC):
    """ABC for external content storage backends."""

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
                yield str(p.relative_to(self.directory))


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
                logger.warning(
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

    def fetch_from(self, store: ContentStore, key: str) -> Path:
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
    def make_store(self) -> ContentStore: ...

    @abc.abstractmethod
    def to_dict(self) -> dict[str, Any]: ...

    _required_env_field: str = ""
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
        req = cls._required_env_field
        if req and req not in merged:
            raise ValueError(
                f"{cls.__name__}.from_env() requires '{req}' "
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

    def make_store(self) -> DirectoryContentStore:
        return DirectoryContentStore(directory=self.directory)

    def to_dict(self) -> dict[str, Any]:
        return {"type": ContentStoreType.DIRECTORY} | serialize_fields(self)

    def write_yaml(self, path: str | Path) -> None:
        path = Path(path)
        dct = self.to_dict()
        base = path.parent.resolve()
        rel = Path(os.path.relpath(self.directory, base))
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

    def make_store(self) -> S3ContentStore:
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


_S3_GCS_DEFAULTS: dict[str, str] = {
    "host": "storage.googleapis.com",
    "protocol": "https",
    "region": "auto",
}


_CONTENT_STORE_CONFIG_CLASSES: dict[ContentStoreType, type[ContentStoreConfig]] = {
    ContentStoreType.DIRECTORY: DirectoryContentStoreConfig,
    ContentStoreType.S3: S3ContentStoreConfig,
}
