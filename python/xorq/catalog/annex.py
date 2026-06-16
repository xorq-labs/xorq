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

from xorq.catalog.constants import ANNEX_BRANCH
from xorq.catalog.s3_utils import (
    S3_SECRET_FIELDS,
    S3ClientMixin,
    make_boto3_client,
    make_endpoint_url,
    serialize_fields,
)
from xorq.common.utils.env_utils import EnvConfigable, env_templates_dir, parse_env_file


abspath = toolz.compose(Path.absolute, Path)

GIT_ANNEX_COMMAND = "git-annex"

POLL_TIMEOUT_SECONDS = 30.0

_ADR_0009_PATH = "docs/adr/0009-bucket-fileprefix-name-uuid-namespace.md"


class AnnexError(RuntimeError):
    """Raised when a git-annex command fails."""


def require_git_annex():
    """Raise a clear error if the git-annex binary is not on $PATH."""
    if shutil.which(GIT_ANNEX_COMMAND) is None:
        raise AnnexError(
            f"'{GIT_ANNEX_COMMAND}' not found on $PATH. "
            "Install git-annex, or pass annex=False for a plain-git catalog."
        )


def _do_inside(
    repo_path: str | Path, *args: str, env: dict | None = None
) -> tuple[int, str, str]:
    cmd = [GIT_ANNEX_COMMAND, *args]
    run_env = None
    if env:
        run_env = {**os.environ, **dict(env)}
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
    env = field(
        validator=optional(instance_of(tuple)),
        default=None,
        converter=lambda v: tuple(v.items()) if isinstance(v, dict) else v,
    )
    poll_interval_seconds = field(validator=instance_of(float), default=0.001)
    # cached remote.log dict, populated on first read and invalidated after writes.
    # Mutated via object.__setattr__ to bypass frozen.
    _remote_log_cache = field(init=False, default=None, eq=False, repr=False)

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
        """Parse the git-annex branch's remote.log into {uuid: {key: value}}.

        Cached for the lifetime of the instance; ``initremote``/``enableremote``
        invalidate the cache after writing.
        """
        cached = self._remote_log_cache
        if cached is not None:
            return cached
        from git import Repo  # noqa: PLC0415

        self._merge_annex_branch()
        branch = Repo(self.repo_path).commit(ANNEX_BRANCH)
        try:
            blob = branch.tree / "remote.log"
        except KeyError:
            result = {}
        else:
            result = {}
            for line in blob.data_stream[3].read().decode().strip().splitlines():
                parts = line.split()
                uuid_ = parts[0]
                config = dict(part.split("=", 1) for part in parts[1:] if "=" in part)
                result[uuid_] = config
        object.__setattr__(self, "_remote_log_cache", result)
        return result

    def _invalidate_remote_log(self):
        object.__setattr__(self, "_remote_log_cache", None)

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
        env_dict = dict(self.env) if self.env else {}
        instance_fallback = {
            field_name: env_dict[env_key]
            for env_key, field_name in self._ENV_TO_FIELD.items()
            if env_dict.get(env_key) and field_name not in config
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
        merged = config | env_fallback | instance_fallback | kwargs
        # check that all required fields (no default) are present
        required = {a.name for a in attr.fields(cls) if a.default is attr.NOTHING}
        if not required.issubset(merged):
            return None
        return cls.from_dict(merged)

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

    @classmethod
    def from_repo_path(cls, repo_path, **kwargs):
        """Construct an ``Annex`` from a repo path (``str`` or ``Path``)."""
        return cls(repo_path=Path(repo_path), **kwargs)

    def initremote(self, remote_config):
        """Initialize a special remote with a pre-generated UUID baked into fileprefix.

        Generates the remote UUID up front and passes it to ``git annex
        initremote`` via ``uuid=<value>`` so any name/uuid namespacing the
        config wants to apply (see ``RemoteConfig.augment_fileprefix``) is
        enforced from the very first write — no orphaned ``annex-uuid``
        sentinel at the parent prefix.
        """
        remote_uuid = str(uuid.uuid4())
        augmented = remote_config.augment_fileprefix(remote_uuid)
        augmented.initremote(self.repo_path, remote_uuid=remote_uuid)
        self._invalidate_remote_log()
        return augmented

    def enableremote(self, remote_config):
        """Enable an existing special remote.

        Looks up the remote's UUID from ``remote.log`` so any namespacing
        is stable across clones.  Refuses to operate on a remote whose
        existing ``remote.log`` fileprefix is not properly namespaced
        (catalog initialized before the namespacing scheme — bucket
        objects must be migrated first; see ADR-0011).  Falls back to
        ``initremote`` only when ``remote.log`` is empty (genuinely fresh
        repo); a non-empty ``remote.log`` without a matching name is an
        error so callers cannot accidentally create a second remote.
        """
        remote_log = self.remote_log
        remote_uuid = next(
            (
                ru
                for ru, cfg in remote_log.items()
                if cfg.get("name") == remote_config.name
            ),
            None,
        )
        if remote_uuid is None:
            if remote_log:
                names = sorted(
                    filter(None, {cfg.get("name") for cfg in remote_log.values()})
                )
                raise AnnexError(
                    f"remote {remote_config.name!r} is not registered in this "
                    f"catalog's remote.log (existing remotes: {names}). "
                    "Use initremote to create a new remote."
                )
            return self.initremote(remote_config)
        existing_fileprefix = remote_log[remote_uuid].get("fileprefix", "")
        remote_config.verify_fileprefix(remote_uuid, existing_fileprefix)
        augmented = remote_config.augment_fileprefix(remote_uuid)
        augmented.enableremote(self.repo_path)
        self._invalidate_remote_log()
        return augmented

    @staticmethod
    def init_repo_path(repo_path, remote_config=None):
        require_git_annex()
        _check_output_do_inside(repo_path, "init", check_stderr=False)
        if remote_config:
            Annex.from_repo_path(repo_path).initremote(remote_config)


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
    def initremote(self, repo_path, *, remote_uuid): ...

    def enableremote(self, repo_path): ...  # noqa: B027

    def augment_fileprefix(self, remote_uuid):
        """Return self with ``{name}/{remote_uuid}/`` namespacing applied.

        Default is a no-op — only remotes that store content under a
        configurable bucket prefix (S3) override this to namespace the
        prefix.  Directory and rsync remotes already isolate via the
        directory/URL itself, so namespacing is unnecessary.
        """
        return self

    def verify_fileprefix(self, remote_uuid, existing_fileprefix):
        """Raise ``AnnexError`` if *existing_fileprefix* is not properly namespaced.

        Default is a no-op for the same reason ``augment_fileprefix`` is.
        """

    @abc.abstractmethod
    def validate_config(self, repo_path): ...

    @abc.abstractmethod
    def to_dict(self): ...

    @classmethod
    def from_dict(cls, d, **kwargs):
        valid_keys = {a.name for a in attr.fields(cls)}
        d = {k: v for k, v in d.items() if k in valid_keys}
        kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
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

    @property
    def _remote_params(self):
        params = [
            self.name,
            "type=directory",
            f"directory={self.directory}",
            f"encryption={self.encryption}",
        ]
        if self.autoenable is not None:
            params.append(f"autoenable={self.autoenable}")
        return params

    def initremote(self, repo_path, *, remote_uuid):
        Path(self.directory).mkdir(exist_ok=True, parents=True)
        _check_output_do_inside(
            repo_path,
            "initremote",
            *self._remote_params,
            f"uuid={remote_uuid}",
            check_stderr=False,
        )

    def enableremote(self, repo_path):
        _check_output_do_inside(
            repo_path,
            "enableremote",
            *self._remote_params,
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


@frozen
class RsyncRemoteConfig(RemoteConfig):
    name = field(validator=instance_of(str))
    rsyncurl = field(validator=instance_of(str))
    encryption = field(validator=instance_of(str), default="none")
    autoenable = field(validator=optional(instance_of(str)), default=None)
    shellescape = field(validator=optional(instance_of(str)), default=None)

    EnvConfig = EnvConfigable.subclass_from_env_file(
        env_templates_dir.joinpath(".env.catalog.rsync.template"),
        prefix="XORQ_CATALOG_RSYNC_",
    )

    @property
    def _remote_params(self):
        params = [
            self.name,
            "type=rsync",
            f"rsyncurl={self.rsyncurl}",
            f"encryption={self.encryption}",
        ]
        for key in ("autoenable", "shellescape"):
            value = getattr(self, key)
            if value is not None:
                params.append(f"{key}={value}")
        return params

    def initremote(self, repo_path, *, remote_uuid):
        _check_output_do_inside(
            repo_path,
            "initremote",
            *self._remote_params,
            f"uuid={remote_uuid}",
            check_stderr=False,
        )

    def enableremote(self, repo_path):
        _check_output_do_inside(
            repo_path,
            "enableremote",
            *self._remote_params,
            check_stderr=False,
        )

    def validate_config(self, repo_path):
        out = _check_output_do_inside(
            repo_path, "info", self.name, "--json", check_stderr=False
        )
        info = json.loads(out)
        if info["type"] != "rsync":
            raise ValueError(f"expected remote type 'rsync', got {info['type']!r}")

    def to_dict(self):
        d = {"type": "rsync", **attr.asdict(self)}
        return {k: v for k, v in d.items() if v is not None}


_REQUIRED_S3_FIELDS = frozenset(
    {"name", "bucket", "aws_access_key_id", "aws_secret_access_key", "encryption"}
)


@frozen
class S3RemoteConfig(S3ClientMixin, RemoteConfig):
    name = field(validator=instance_of(str))
    bucket = field(validator=instance_of(str))
    aws_access_key_id = field(validator=instance_of(str))
    aws_secret_access_key = field(validator=instance_of(str), repr=False)
    aws_session_token = field(validator=optional(instance_of(str)), default=None)
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

    _SECRET_FIELDS = S3_SECRET_FIELDS

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
        env = [
            ("AWS_ACCESS_KEY_ID", self.aws_access_key_id),
            ("AWS_SECRET_ACCESS_KEY", self.aws_secret_access_key),
            ("AWS_SESSION_TOKEN", self.aws_session_token or ""),
            ("AWS_SECURITY_TOKEN", ""),
            ("AWS_CREDENTIAL_EXPIRATION", ""),
        ]
        return tuple(env)

    @property
    def endpoint_url(self):
        return make_endpoint_url(self.host, self.port, self.protocol)

    @property
    def _boto3_endpoint_url(self) -> str | None:
        """Endpoint URL suitable for boto3 (always HTTPS for public hosts)."""
        if self.host is None:
            return None
        return f"https://{self.host}"

    @property
    def _probe_prefix(self) -> str:
        return self.fileprefix or ""

    def _make_boto3_client(self) -> object:
        return make_boto3_client(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
            region=self.region,
            endpoint_url=self._boto3_endpoint_url,
        )

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

    def initremote(self, repo_path, *, remote_uuid):
        _check_output_do_inside(
            repo_path,
            "initremote",
            *self.initremote_params,
            f"uuid={remote_uuid}",
            check_stderr=False,
            env=self.env,
        )

    def _fileprefix_suffix(self, remote_uuid):
        return f"{self.name}/{remote_uuid}/"

    def augment_fileprefix(self, remote_uuid):
        """Return self with ``{name}/{remote_uuid}/`` appended to ``fileprefix``.

        The remote name namespaces the catalog within the bucket; the
        special remote's own UUID specializes within that namespace so
        independent catalogs sharing a name write to different subdirs.
        Because the remote UUID is stable across all clones of a catalog,
        clones agree on the path and content-addressed dedup works.
        Idempotent for the *same* ``remote_uuid`` — if the suffix is
        already present the input is returned unchanged. Calling with a
        different ``remote_uuid`` re-appends, so callers must thread the
        same UUID throughout a catalog's lifetime.
        """
        suffix = self._fileprefix_suffix(remote_uuid)
        current = self.fileprefix or ""
        if current.endswith(suffix):
            return self
        base = current if not current or current.endswith("/") else f"{current}/"
        return attr.evolve(self, fileprefix=f"{base}{suffix}")

    def verify_fileprefix(self, remote_uuid, existing_fileprefix):
        """Raise if *existing_fileprefix* in remote.log disagrees with what
        augmenting *self* with *remote_uuid* would produce.

        Two failure modes are distinguished:

        * **Suffix missing**: catalog initialized before the
          name+remote-uuid namespacing scheme; bucket objects must be
          migrated.
        * **Base prefix mismatch**: existing remote was initialized with
          a different ``XORQ_CATALOG_S3_FILEPREFIX`` than is currently
          configured. Continuing would silently rewrite ``remote.log``
          and orphan bucket objects under the previous base.
        """
        suffix = self._fileprefix_suffix(remote_uuid)
        existing = existing_fileprefix or ""
        if not existing.endswith(suffix):
            raise AnnexError(
                f"S3 remote {self.name!r} has fileprefix {existing!r} in "
                f"remote.log, which is not namespaced by {suffix!r}. This "
                "catalog was initialized before the name+remote-uuid "
                "namespacing scheme; bucket objects must be copied from "
                f"{existing!r} to a path ending in {suffix!r} (and "
                f"remote.log updated to match) before reopening. "
                f"See {_ADR_0009_PATH}."
            )
        expected = self.augment_fileprefix(remote_uuid).fileprefix
        if existing != expected:
            raise AnnexError(
                f"S3 remote {self.name!r} has fileprefix {existing!r} in "
                f"remote.log but the configured base prefix would produce "
                f"{expected!r}. The existing remote uses a different base "
                "prefix; either align your configuration to match the "
                f"existing layout, or migrate bucket objects from "
                f"{existing!r} to {expected!r} (and update remote.log) "
                f"before reopening. See {_ADR_0009_PATH}."
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
        return {"type": "S3"} | serialize_fields(
            self, include_secrets=self.has_embedded_creds
        )

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
    "rsync": RsyncRemoteConfig,
    "S3": S3RemoteConfig,
}


_PREFIX_TO_REMOTE_CONFIG = {
    "XORQ_CATALOG_S3_": S3RemoteConfig,
    "XORQ_CATALOG_DIRECTORY_": DirectoryRemoteConfig,
    "XORQ_CATALOG_RSYNC_": RsyncRemoteConfig,
}


def remote_config_from_prefix(prefix, *, gcs=False):
    """Return a RemoteConfig from environment variables matching *prefix*.

    When *gcs* is True and the prefix resolves to S3RemoteConfig,
    GCS-compatible defaults (host, protocol, etc.) are applied.
    """
    cls = _PREFIX_TO_REMOTE_CONFIG.get(prefix)
    if cls is None:
        raise ValueError(
            f"Unknown prefix {prefix!r}. "
            f"Expected one of: {', '.join(_PREFIX_TO_REMOTE_CONFIG)}"
        )
    if gcs and cls is S3RemoteConfig:
        return cls.make_gcs_remote_from_env()
    return cls.from_env()


def remote_config_from_env_file(env_file, *, gcs=False):
    """Load an env file and return the matching RemoteConfig.

    Detects the remote type from the env var prefix
    (``XORQ_CATALOG_S3_``, ``XORQ_CATALOG_DIRECTORY_``, ``XORQ_CATALOG_RSYNC_``).
    When *gcs* is True and the file contains S3 keys,
    GCS-compatible defaults are applied.
    """
    env_vars = parse_env_file(env_file)
    os.environ.update(env_vars)
    keys = env_vars.keys()
    for prefix in _PREFIX_TO_REMOTE_CONFIG:
        if any(k.startswith(prefix) for k in keys):
            return remote_config_from_prefix(prefix, gcs=gcs)
    raise ValueError(
        f"Could not determine remote type from {env_file}. "
        f"Expected keys with prefix: {', '.join(_PREFIX_TO_REMOTE_CONFIG)}"
    )


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
