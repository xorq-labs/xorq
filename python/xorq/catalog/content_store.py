import abc
import functools
import hashlib
import os
import re
import shutil
from pathlib import Path

import yaml12
from attr import field, frozen
from attr.validators import instance_of, optional

from xorq.common.utils.env_utils import EnvConfigable, env_templates_dir


POINTER_VERSION = "xorq-pointer v1"

# catalog_id and sha256 are interpolated into filesystem keys (cache and store
# paths) and may originate from an untrusted cloned repo, so they are validated
# to prevent path traversal (e.g. a catalog_id of "../..").
_SAFE_CATALOG_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_DEFAULT_CACHE_DIR = Path("~/.cache/xorq/content")
_DEFAULT_CACHE_MAX_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB


def compute_sha256(path):
    from xorq.common.utils.defer_utils import _file_digest  # noqa: PLC0415

    return _file_digest(path, hashlib.sha256)


def content_key(catalog_id, sha256):
    if not _SAFE_CATALOG_ID_RE.match(catalog_id):
        raise ValueError(f"Unsafe catalog_id: {catalog_id!r}")
    if not _SHA256_RE.match(sha256):
        raise ValueError(f"Invalid sha256: {sha256!r}")
    return f"{catalog_id}/{sha256[:2]}/{sha256[2:4]}/{sha256}.zip"


def write_pointer(path, sha256, size):
    Path(path).write_text(f"{POINTER_VERSION}\nsha256 {sha256}\nsize {size}\n")


def parse_pointer(path):
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
    try:
        size = int(size_parts[1])
    except ValueError:
        raise ValueError(f"Invalid pointer file: {path}") from None
    return sha256, size


class ContentStore(abc.ABC):
    """ABC for external content storage backends."""

    @abc.abstractmethod
    def put(self, key, local_path): ...

    @abc.abstractmethod
    def get(self, key, local_path): ...

    @abc.abstractmethod
    def exists(self, key): ...

    @abc.abstractmethod
    def delete(self, key): ...


@frozen
class DirectoryContentStore(ContentStore):
    """Content store backed by a local directory."""

    directory = field(validator=instance_of((str, Path)))

    @property
    def root(self):
        return Path(self.directory)

    def _key_path(self, key):
        return self.root / key

    def put(self, key, local_path):
        dest = self._key_path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)

    def get(self, key, local_path):
        src = self._key_path(key)
        if not src.exists():
            raise FileNotFoundError(f"Content not found in store: {key}")
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local_path)

    def exists(self, key):
        return self._key_path(key).exists()

    def delete(self, key):
        p = self._key_path(key)
        if p.exists():
            p.unlink()


@frozen
class S3ContentStore(ContentStore):
    """Content store backed by an S3-compatible bucket."""

    bucket = field(validator=instance_of(str))
    prefix = field(validator=instance_of(str), default="")
    region = field(validator=optional(instance_of(str)), default=None)
    aws_access_key_id = field(
        validator=optional(instance_of(str)), default=None, repr=False
    )
    aws_secret_access_key = field(
        validator=optional(instance_of(str)), default=None, repr=False
    )
    host = field(validator=optional(instance_of(str)), default=None)
    port = field(validator=optional(instance_of(str)), default=None)
    protocol = field(validator=optional(instance_of(str)), default=None)

    def _s3_key(self, key):
        if self.prefix:
            return f"{self.prefix.rstrip('/')}/{key}"
        return key

    @functools.cached_property
    def _client(self):
        import boto3  # noqa: PLC0415

        kwargs = {}
        if self.aws_access_key_id:
            kwargs["aws_access_key_id"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            kwargs["aws_secret_access_key"] = self.aws_secret_access_key
        if self.region:
            kwargs["region_name"] = self.region
        if self.host:
            protocol = self.protocol or "https"
            port_suffix = f":{self.port}" if self.port else ""
            kwargs["endpoint_url"] = f"{protocol}://{self.host}{port_suffix}"
        return boto3.client("s3", **kwargs)

    def put(self, key, local_path):
        client = self._client
        client.upload_file(str(local_path), self.bucket, self._s3_key(key))

    def get(self, key, local_path):
        client = self._client
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        client.download_file(self.bucket, self._s3_key(key), str(local_path))

    def exists(self, key):
        from botocore.exceptions import ClientError  # noqa: PLC0415

        client = self._client
        try:
            client.head_object(Bucket=self.bucket, Key=self._s3_key(key))
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                return False
            raise

    def delete(self, key):
        client = self._client
        client.delete_object(Bucket=self.bucket, Key=self._s3_key(key))


@frozen
class ContentCache:
    """LRU disk cache for content store objects."""

    cache_dir = field(validator=instance_of(Path))
    max_bytes = field(validator=instance_of(int))

    def _path(self, key):
        return self.cache_dir / key

    def contains(self, key):
        return self._path(key).exists()

    def get_path(self, key):
        p = self._path(key)
        if not p.exists():
            return None
        os.utime(p)
        return p

    def put(self, key, local_path):
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)
        os.utime(dest)  # copy2 preserves source atime; mark as freshly used for LRU
        self._maybe_evict(protect=key)

    def fetch_from(self, store, key):
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        store.get(key, dest)
        os.utime(dest)  # store.get may preserve source atime; mark as freshly used
        self._maybe_evict(protect=key)
        return dest

    def _maybe_evict(self, protect=None):
        if self.max_bytes <= 0:
            return
        protect_path = self._path(protect) if protect is not None else None
        entries = []
        total = 0
        for p in self.cache_dir.rglob("*"):
            if p.is_file():
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
    def default(cls):
        env_config = cls.EnvConfig.from_env()
        cache_dir = Path(env_config.dir or _DEFAULT_CACHE_DIR).expanduser()
        max_bytes = int(env_config.max_bytes or _DEFAULT_CACHE_MAX_BYTES)
        return cls(cache_dir=cache_dir, max_bytes=max_bytes)


_CONTENT_STORE_CLASSES = {
    "directory": DirectoryContentStore,
    "s3": S3ContentStore,
}


_RESERVED_CONFIG_KEYS = ("type", "catalog_id")


def _no_reserved_config_keys(instance, attribute, value):
    bad = [k for k in _RESERVED_CONFIG_KEYS if k in value]
    if bad:
        raise ValueError(f"content_store config must not contain reserved keys: {bad}")


@frozen
class ContentStoreConfig:
    """Serializable configuration for constructing a ``ContentStore``."""

    type = field(validator=instance_of(str))
    catalog_id = field(validator=instance_of(str))
    config = field(
        validator=[instance_of(dict), _no_reserved_config_keys], factory=dict
    )

    _S3EnvConfig = EnvConfigable.subclass_from_env_file(
        env_templates_dir.joinpath(".env.catalog.content_store.s3.template"),
        prefix="XORQ_CONTENT_STORE_",
    )

    def to_dict(self):
        return {"type": self.type, "catalog_id": self.catalog_id, **self.config}

    def write_yaml(self, path):
        Path(path).write_text(yaml12.format_yaml(self.to_dict()))

    def make_store(self):
        store_cls = _CONTENT_STORE_CLASSES[self.type]
        kwargs = dict(self.config)
        if self.type == "s3":
            env_config = self._S3EnvConfig.from_env()
            env = {
                a.name: getattr(env_config, a.name)
                for a in env_config.__attrs_attrs__
                if a.name != "env_file" and getattr(env_config, a.name)
            }
            kwargs = {**env, **kwargs}
        return store_cls(**kwargs)

    @classmethod
    def from_dict(cls, dct):
        dct = dict(dct)
        type_ = dct.pop("type")
        catalog_id = dct.pop("catalog_id")
        return cls(type=type_, catalog_id=catalog_id, config=dct)

    @classmethod
    def from_yaml(cls, path):
        data = yaml12.read_yaml(path)
        return cls.from_dict(data)
