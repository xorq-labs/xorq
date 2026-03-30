import abc
import base64
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import attr
import toolz
from attr import (
    field,
    frozen,
)
from attr.validators import (
    instance_of,
    optional,
)
from git import Repo

from xorq.common.utils.env_utils import EnvConfigable, env_templates_dir


abspath = toolz.compose(Path.absolute, Path)

GIT_ANNEX_COMMAND = "git-annex"

POLL_TIMEOUT_SECONDS = 30.0


class AnnexError(RuntimeError):
    """Raised when a git-annex command fails."""


def require_git_annex():
    """Raise a clear error if the git-annex binary is not on $PATH."""
    if shutil.which(GIT_ANNEX_COMMAND) is None:
        raise AnnexError(
            f"'{GIT_ANNEX_COMMAND}' not found on $PATH. "
            "Install git-annex, or pass annex=False for a plain-git catalog."
        )


def _do_inside(repo_path, *args, env=None):
    cmd = [GIT_ANNEX_COMMAND, *args]
    run_env = None
    if env:
        run_env = {**os.environ, **env}
    result = subprocess.run(
        cmd,
        cwd=repo_path,
        capture_output=True,
        text=True,
        env=run_env,
    )
    return result.returncode, result.stdout, result.stderr


def _check_output_do_inside(repo_path, *args, check_stderr=True, env=None):
    returncode, stdout, stderr = _do_inside(repo_path, *args, env=env)
    if returncode != 0:
        raise AnnexError(f"git-annex {args} failed: {stderr}")
    if check_stderr and stderr:
        raise AnnexError(f"git-annex {args} stderr: {stderr}")
    return stdout


@frozen
class Annex:
    repo_path = field(validator=instance_of(Path), converter=abspath)
    env = field(validator=optional(instance_of(dict)), default=None)
    poll_interval_seconds = field(validator=instance_of(float), default=0.001)

    def __attrs_post_init__(self):
        require_git_annex()
        if not self.annex_path.exists():
            raise ValueError(f"git-annex not initialized at {self.repo_path}")

    @property
    def annex_path(self):
        return self.repo_path.joinpath(".git", "annex")

    @property
    def annex_objects_path(self):
        return self.annex_path.joinpath("objects")

    def _do(self, *args):
        returncode, stdout, stderr = _do_inside(self.repo_path, *args, env=self.env)
        if returncode != 0:
            raise AnnexError(f"git-annex {args} failed (rc={returncode}): {stderr}")
        return stdout, stderr

    def _check_output_do(self, *args, **kwargs):
        return _check_output_do_inside(self.repo_path, *args, env=self.env, **kwargs)

    @property
    def remote_name(self):
        """Return the name of the single configured special remote, or None."""
        remote_log = self.remote_log
        if not remote_log:
            return None
        gen = filter(None, (dct.get("name") for dct in remote_log.values()))
        name, rest = next(gen, None), tuple(gen)
        if rest:
            return None
        else:
            return name

    def add(self, relpath, copy_to_remote=True):
        path = self.repo_path.joinpath(relpath)
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        self._do("add", str(relpath))
        if not path.is_symlink():
            raise AnnexError(f"git-annex add completed but {relpath} is not a symlink")
        if copy_to_remote and (name := self.remote_name) is not None:
            self.copy(to=name, path=str(relpath))

    def init(self):
        self._do("init")

    def get(self, *paths):
        self._do("get", *(str(p) for p in paths) or (".",))

    def copy(self, to=None, from_=None, path="."):
        if (to is None) == (from_ is None):
            raise ValueError("specify exactly one of to/from_")
        direction = f"--to={to}" if to else f"--from={from_}"
        self._do("copy", direction, str(path))

    def drop(self, path="."):
        self._do("drop", "--force", str(path))

    def sync(self, **kwargs):
        self._do("sync")

    def push(self):
        self._do("push")

    def pull(self):
        self._do("pull")

    def info(self, name=None):
        args = ("info", "--json") if name is None else ("info", name, "--json")
        out = self._check_output_do(*args, check_stderr=False)
        return json.loads(out)

    def _merge_annex_branch(self):
        """Flush the git-annex journal to the git-annex branch."""
        _do_inside(self.repo_path, "merge", env=self.env)

    @property
    def remote_log(self):
        """Parse the git-annex branch's remote.log into {uuid: {key: value}}."""
        self._merge_annex_branch()
        branch = Repo(self.repo_path).commit("git-annex")
        try:
            blob = branch.tree / "remote.log"
        except KeyError:
            # no git-annex in the repo
            return {}
        result = {}
        for line in blob.data_stream[3].read().decode().strip().splitlines():
            parts = line.split()
            uuid = parts[0]
            config = dict(part.split("=", 1) for part in parts[1:] if "=" in part)
            result[uuid] = config
        return result

    # Maps AWS env-var names (as stored in Annex.env) back to attrs field names
    # so resolve_remote_config can recover credentials from self.env.
    _ENV_TO_FIELD = {
        "AWS_ACCESS_KEY_ID": "aws_access_key_id",
        "AWS_SECRET_ACCESS_KEY": "aws_secret_access_key",
    }

    def resolve_remote_config(self, **kwargs):
        """Recover the RemoteConfig from the git-annex branch, env vars, and kwargs.

        Precedence (highest to lowest):

        1. *kwargs* — explicit overrides from the caller (e.g. credentials
           passed to ``from_repo_path``).
        2. ``remote.log`` on the ``git-annex`` branch — non-secret config
           (and secrets when ``embedcreds=yes``).
        3. ``self.env`` — credentials the Annex was constructed with.
        4. ``XORQ_CATALOG_S3_*`` / ``XORQ_CATALOG_DIRECTORY_*`` environment
           variables — fallback for fields missing from remote.log.

        Returns ``None`` when no remote is configured.
        """
        if not (remote_log := self.remote_log):
            return None
        config, *rest = remote_log.values()
        if rest:
            raise ValueError("can only handle one remote")
        remote_type = config.get("type")
        cls = _REMOTE_CONFIG_CLASSES.get(remote_type)
        if cls is None:
            raise ValueError(f"unknown remote type: {remote_type!r}")
        # when embedcreds=yes, remote.log has the credentials — git-annex
        # stores S3 creds as base64 in a single "s3creds" field
        if config.get("embedcreds") == "yes":
            s3creds = config.get("s3creds")
            if s3creds and "aws_access_key_id" not in config:
                lines = base64.b64decode(s3creds).decode().strip().splitlines()
                config = {
                    **config,
                    "aws_access_key_id": lines[0],
                    "aws_secret_access_key": lines[1],
                }
            return cls.from_dict(config, **kwargs)
        # recover creds from self.env (set at construction time)
        instance_fallback = {
            field_name: self.env[env_key]
            for env_key, field_name in self._ENV_TO_FIELD.items()
            if self.env and self.env.get(env_key) and field_name not in config
        }
        # fill in fields missing from remote.log (e.g. secrets) from env vars
        env_config = cls.EnvConfig.from_env()
        env_fallback = {
            a.name: getattr(env_config, a.name)
            for a in env_config.__attrs_attrs__
            if a.name != "env_file"
            and getattr(env_config, a.name)
            and a.name not in config
        }
        return cls.from_dict(config, **{**env_fallback, **instance_fallback, **kwargs})

    @property
    def remote_config(self):
        """Convenience property — calls ``resolve_remote_config()`` with no overrides."""
        return self.resolve_remote_config()

    def findkeys(self):
        out = self._check_output_do("findkeys", check_stderr=False)
        return out.split()

    def dropkey(self, key):
        self._do("dropkey", key, "--force")

    def uninit(self):
        self._do("uninit")

    @staticmethod
    def init_repo_path(repo_path, remote_config=None):
        require_git_annex()
        _check_output_do_inside(repo_path, "init", check_stderr=False)
        if remote_config:
            remote_config.initremote(repo_path)


def teardown_local(git_annex):
    annex = git_annex.annex
    keys = annex.findkeys()
    for key in keys:
        annex.dropkey(key)
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    while annex.annex_objects_path.exists() and tuple(
        annex.annex_objects_path.iterdir()
    ):
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"annex objects not cleaned up within {POLL_TIMEOUT_SECONDS}s"
            )
        time.sleep(annex.poll_interval_seconds)
    annex.uninit()
    shutil.rmtree(git_annex.repo_path)


def teardown_remote(git_annex, remote_config=None):
    """Remove all content from a remote and purge it from the git-annex branch."""
    if remote_config is None:
        remote_config = git_annex.annex.remote_config
    if remote_config is None:
        raise ValueError("no remote found and no remote_config provided")
    repo_path = git_annex.repo_path
    env = getattr(remote_config, "env", None)
    name = remote_config.name
    _check_output_do_inside(
        repo_path, "drop", "--force", f"--from={name}", ".", check_stderr=False, env=env
    )
    _check_output_do_inside(repo_path, "dead", name, check_stderr=False, env=env)
    _check_output_do_inside(
        repo_path, "forget", "--drop-dead", "--force", check_stderr=False, env=env
    )


def teardown(git_annex, remote_config=None):
    teardown_remote(git_annex, remote_config=remote_config)
    teardown_local(git_annex)


class AnnexConfig:
    """Base for all annex configuration objects.

    Passing any ``AnnexConfig`` instance as ``annex`` to a Catalog classmethod
    selects the git-annex backend.  ``None`` selects plain git.
    """


@frozen
class LocalAnnexConfig(AnnexConfig):
    """Annex with no special remote — content lives only in the local repo."""


LOCAL_ANNEX = LocalAnnexConfig()


class RemoteConfig(AnnexConfig, abc.ABC):
    @abc.abstractmethod
    def initremote(self, repo_path): ...

    def enableremote(self, repo_path): ...  # noqa: B027

    @abc.abstractmethod
    def validate_config(self, repo_path): ...

    @abc.abstractmethod
    def to_dict(self): ...

    @classmethod
    def from_dict(cls, d, **kwargs):
        valid_keys = {a.name for a in attr.fields(cls)}
        d = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**{**d, **kwargs})

    @classmethod
    def from_env(cls, **kwargs):
        env_config = cls.EnvConfig.from_env()
        env = {
            a.name: getattr(env_config, a.name)
            for a in env_config.__attrs_attrs__
            if a.name != "env_file" and getattr(env_config, a.name)
        }
        return cls(**{**env, **kwargs})


@frozen
class DirectoryRemoteConfig(RemoteConfig):
    name = field(validator=instance_of(str))
    directory = field(validator=instance_of(str))
    encryption = field(validator=instance_of(str), default="none")
    autoenable = field(validator=optional(instance_of(str)), default=None)

    EnvConfig = EnvConfigable.subclass_from_env_file(
        env_templates_dir.joinpath(".env.catalog.directory.template"),
        prefix="XORQ_CATALOG_DIRECTORY_",
    )

    def initremote(self, repo_path):
        Path(self.directory).mkdir(exist_ok=True, parents=True)
        params = [
            self.name,
            "type=directory",
            f"directory={self.directory}",
            f"encryption={self.encryption}",
        ]
        if self.autoenable is not None:
            params.append(f"autoenable={self.autoenable}")
        _check_output_do_inside(
            repo_path,
            "initremote",
            *params,
            check_stderr=False,
        )

    def validate_config(self, repo_path):
        out = _check_output_do_inside(
            repo_path, "info", self.name, "--json", check_stderr=False
        )
        info = json.loads(out)
        if info["type"] != "directory":
            raise ValueError(f"expected remote type 'directory', got {info['type']!r}")

    def to_dict(self):
        d = {"type": "directory", **attr.asdict(self)}
        return {k: v for k, v in d.items() if v is not None}


_REQUIRED_S3_FIELDS = frozenset(
    {"name", "bucket", "aws_access_key_id", "aws_secret_access_key", "encryption"}
)


@frozen
class S3RemoteConfig(RemoteConfig):
    name = field(validator=instance_of(str))
    bucket = field(validator=instance_of(str))
    aws_access_key_id = field(validator=instance_of(str))
    aws_secret_access_key = field(validator=instance_of(str), repr=False)
    encryption = field(validator=instance_of(str), default="none")
    datacenter = field(validator=optional(instance_of(str)), default=None)
    region = field(validator=optional(instance_of(str)), default=None)
    host = field(validator=optional(instance_of(str)), default=None)
    port = field(validator=optional(instance_of(str)), default=None)
    protocol = field(validator=optional(instance_of(str)), default=None)
    requeststyle = field(validator=optional(instance_of(str)), default=None)
    signature = field(validator=optional(instance_of(str)), default=None)
    fileprefix = field(validator=optional(instance_of(str)), default=None)
    storageclass = field(validator=optional(instance_of(str)), default=None)
    chunk = field(validator=optional(instance_of(str)), default=None)
    public = field(validator=optional(instance_of(str)), default=None)
    publicurl = field(validator=optional(instance_of(str)), default=None)
    versioning = field(validator=optional(instance_of(str)), default=None)
    partsize = field(validator=optional(instance_of(str)), default=None)
    embedcreds = field(validator=optional(instance_of(str)), default=None)
    autoenable = field(validator=optional(instance_of(str)), default=None)

    EnvConfig = EnvConfigable.subclass_from_env_file(
        env_templates_dir.joinpath(".env.catalog.s3.template"),
        prefix="XORQ_CATALOG_S3_",
    )

    _SECRET_FIELDS = ("aws_access_key_id", "aws_secret_access_key")

    @property
    def _optional_params(self):
        """Derive optional params from attrs fields (all fields not in the required set)."""
        return tuple(
            a.name
            for a in attr.fields(type(self))
            if a.name not in _REQUIRED_S3_FIELDS and not a.name.startswith("_")
        )

    @property
    def env(self):
        return {
            "AWS_ACCESS_KEY_ID": self.aws_access_key_id,
            "AWS_SECRET_ACCESS_KEY": self.aws_secret_access_key,
            # clear session/temporary credentials that may be in the
            # environment so git-annex doesn't try to use STS tokens
            "AWS_SESSION_TOKEN": "",
            "AWS_SECURITY_TOKEN": "",
            "AWS_CREDENTIAL_EXPIRATION": "",
        }

    _DEFAULT_PORTS = {"http": "80", "https": "443"}

    @property
    def endpoint_url(self):
        if self.host is None:
            return None
        protocol = self.protocol or "https"
        default_port = self._DEFAULT_PORTS.get(protocol)
        port_suffix = f":{self.port}" if self.port and self.port != default_port else ""
        return f"{protocol}://{self.host}{port_suffix}"

    @property
    def boto3_endpoint_url(self):
        """Endpoint URL suitable for boto3 (always HTTPS for public hosts)."""
        if self.host is None:
            return None
        return f"https://{self.host}"

    def _make_boto3_client(self):
        import boto3  # noqa: PLC0415

        # boto3 SigV4 needs a real region in the credential scope;
        # "auto" works for git-annex but not for boto3 PUT signing.
        region = "us-east-1" if self.region in (None, "auto") else self.region
        return boto3.client(
            "s3",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            endpoint_url=self.boto3_endpoint_url,
            region_name=region,
        )

    def check_bucket(self, check_write=False):
        """Verify that credentials can reach the bucket.

        Returns a dict with connection details on success, raises on failure.
        When *check_write* is True, probes write access by uploading and
        deleting a zero-byte sentinel object under the configured fileprefix.
        Requires boto3.
        """
        client = self._make_boto3_client()
        # head_bucket verifies both auth and bucket existence
        client.head_bucket(Bucket=self.bucket)
        # list a single key to confirm read access
        listing = client.list_objects_v2(Bucket=self.bucket, MaxKeys=1)
        result = {
            "bucket": self.bucket,
            "endpoint_url": self.boto3_endpoint_url or "(AWS default)",
            "key_count": listing.get("KeyCount", 0),
        }
        if check_write:
            result["writable"] = self._probe_write(client)
        return result

    def _probe_write(self, client=None):
        """Upload and delete a zero-byte sentinel to test write access.

        Returns True if the write succeeds, False on AccessDenied.
        """
        from botocore.exceptions import ClientError  # noqa: PLC0415

        if client is None:
            client = self._make_boto3_client()
        probe_key = f"{self.fileprefix or ''}.xorq-write-probe-{uuid.uuid4().hex[:12]}"
        try:
            resp = client.create_multipart_upload(
                Bucket=self.bucket,
                Key=probe_key,
            )
            client.abort_multipart_upload(
                Bucket=self.bucket,
                Key=probe_key,
                UploadId=resp["UploadId"],
            )
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("AccessDenied", "403"):
                return False
            raise

    @property
    def initremote_params(self):
        params = [
            self.name,
            "type=S3",
            f"bucket={self.bucket}",
            f"encryption={self.encryption}",
        ] + [
            f"{key}={value}"
            for (key, value) in (
                (key, getattr(self, key)) for key in self._optional_params
            )
            if value is not None
        ]
        return params

    def initremote(self, repo_path):
        _check_output_do_inside(
            repo_path,
            "initremote",
            *self.initremote_params,
            check_stderr=False,
            env=self.env,
        )

    def enableremote(self, repo_path):
        _check_output_do_inside(
            repo_path,
            "enableremote",
            *self.initremote_params,
            check_stderr=False,
            env=self.env,
        )

    def validate_config(self, repo_path):
        out = _check_output_do_inside(
            repo_path, "info", self.name, "--json", check_stderr=False
        )
        info = json.loads(out)
        if info["type"] != "S3":
            raise ValueError(f"expected remote type 'S3', got {info['type']!r}")
        if info["bucket"] != self.bucket:
            raise ValueError(f"expected bucket {self.bucket!r}, got {info['bucket']!r}")

    @property
    def has_embedded_creds(self):
        return self.embedcreds == "yes"

    def to_dict(self):
        d = {"type": "S3"} | {
            a.name: getattr(self, a.name)
            for a in attr.fields(type(self))
            if not a.name.startswith("_")
            and (self.has_embedded_creds or a.name not in self._SECRET_FIELDS)
            and getattr(self, a.name) is not None
        }
        return d

    @classmethod
    def make_s3_remote(
        cls, name, bucket, aws_access_key_id, aws_secret_access_key, **kwargs
    ):
        return cls(
            name=name,
            bucket=bucket,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            **kwargs,
        )

    @classmethod
    def make_minio_remote(
        cls,
        name,
        bucket,
        host,
        aws_access_key_id,
        aws_secret_access_key,
        port="9000",
        **kwargs,
    ):
        return cls(
            name=name,
            bucket=bucket,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            host=host,
            port=port,
            protocol="http",
            requeststyle="path",
            signature="v2",
            **kwargs,
        )

    _GCS_DEFAULTS = {
        "host": "storage.googleapis.com",
        "protocol": "https",
        "requeststyle": "path",
        "signature": "v4",
        "region": "auto",
    }

    @classmethod
    def make_gcs_remote(
        cls,
        name,
        bucket,
        aws_access_key_id,
        aws_secret_access_key,
        **kwargs,
    ):
        """Create an S3-compatible remote pointing at Google Cloud Storage.

        GCS exposes an S3-compatible API (interoperability mode).  The
        *aws_access_key_id* / *aws_secret_access_key* values should be
        HMAC keys generated from the GCS console (Settings > Interoperability).
        """
        return cls(
            name=name,
            bucket=bucket,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            **{**cls._GCS_DEFAULTS, **kwargs},
        )

    @classmethod
    def make_gcs_remote_from_env(cls, **kwargs):
        """Like ``from_env``, but with GCS defaults for host/protocol/requeststyle."""
        return cls.from_env(**{**cls._GCS_DEFAULTS, **kwargs})


_REMOTE_CONFIG_CLASSES = {
    "directory": DirectoryRemoteConfig,
    "S3": S3RemoteConfig,
}


def remote_config_from_dict(d, **kwargs):
    """Reconstruct a remote config from a dict (as stored in catalog.yaml).

    Secrets (e.g. aws_secret_access_key) are not stored in the dict and
    must be passed as kwargs.
    """
    remote_type = d.get("type")
    cls = _REMOTE_CONFIG_CLASSES.get(remote_type)
    if cls is None:
        raise ValueError(f"unknown remote type: {remote_type!r}")
    return cls.from_dict(d, **kwargs)
