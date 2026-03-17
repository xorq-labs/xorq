import json
import os
import shutil
import subprocess
import time
from contextlib import contextmanager
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


abspath = toolz.compose(Path.absolute, Path)

GIT_ANNEX_COMMAND = "git-annex"


def _do_inside(repo_path, *args, env=None):
    cmd = " ".join((GIT_ANNEX_COMMAND, *args))
    run_env = None
    if env:
        run_env = {**os.environ, **env}
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=repo_path,
        capture_output=True,
        text=True,
        env=run_env,
    )
    return result.returncode, result.stdout, result.stderr


def _check_output_do_inside(repo_path, *args, check_stderr=True, env=None):
    returncode, stdout, stderr = _do_inside(repo_path, *args, env=env)
    assert returncode == 0, f"git-annex {args} failed: {stderr}"
    if check_stderr:
        assert not stderr, f"git-annex {args} stderr: {stderr}"
    return stdout


@frozen
class Annex:
    repo_path = field(validator=instance_of(Path), converter=abspath)
    env = field(validator=optional(instance_of(dict)), default=None)
    poll_interval_seconds = field(validator=instance_of(float), default=0.001)

    def __attrs_post_init__(self):
        assert self.repo_path == abspath(self.repo_path)
        assert self.annex_path.exists(), (
            f"git-annex not initialized at {self.repo_path}"
        )

    @property
    def annex_path(self):
        return self.repo_path.joinpath(".git", "annex")

    @property
    def annex_objects_path(self):
        return self.annex_path.joinpath("objects")

    def _do(self, *args):
        return _do_inside(self.repo_path, *args, env=self.env)

    def _check_output_do(self, *args, **kwargs):
        return _check_output_do_inside(self.repo_path, *args, env=self.env, **kwargs)

    def add(self, relpath):
        assert (path := self.repo_path.joinpath(relpath)).exists()
        self._do("add", str(relpath))
        while not path.is_symlink():
            time.sleep(self.poll_interval_seconds)

    def init(self):
        self._do("init")

    def get(self, path="."):
        self._do("get", str(path))

    def copy(self, to=None, frm=None, path="."):
        assert (to is None) != (frm is None), "specify exactly one of to/frm"
        direction = f"--to={to}" if to else f"--from={frm}"
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

    def findkeys(self):
        out = self._check_output_do("findkeys", check_stderr=False)
        return out.split()

    def dropkey(self, key):
        self._do("dropkey", key, "--force")

    def uninit(self):
        self._do("uninit")

    def teardown(self):
        keys = self.findkeys()
        for key in keys:
            self.dropkey(key)
        while self.annex_objects_path.exists() and tuple(
            self.annex_objects_path.iterdir()
        ):
            time.sleep(self.poll_interval_seconds)
        self.uninit()

    @staticmethod
    def init_repo_path(repo_path, external_remote_config=None):
        _do_inside(repo_path, "init")
        if external_remote_config:
            external_remote_config.initremote(repo_path)


@frozen
class ExternalRemoteConfig:
    name = field(validator=instance_of(str))
    externaltype = field(validator=instance_of(str))
    params_tuple = field(validator=instance_of(tuple))
    encryption = field(validator=instance_of(str), default="none")

    @property
    def params(self):
        return dict(self.params_tuple)

    @property
    def initremote_args(self):
        return " ".join(
            (
                self.name,
                "type=external",
                f"externaltype={self.externaltype}",
                f"encryption={self.encryption}",
                " ".join(map("=".join, self.params_tuple)),
            )
        )

    def initremote(self, repo_path):
        _do_inside(repo_path, "initremote", self.initremote_args)
        _do_inside(repo_path, "sync")

    def validate_config(self, repo_path):
        out = _check_output_do_inside(repo_path, "info", self.name, "--json")
        info = json.loads(out)
        expected = {
            name: getattr(self, name) for name in ("externaltype", "encryption")
        } | self.params
        actual = {name: info[name] for name in expected}
        diffs = {
            name: (a, e)
            for (name, a, e) in (
                (name, a, expected[name]) for name, a in actual.items()
            )
            if a != e
        }
        if diffs:
            raise ValueError(diffs)

    @classmethod
    def make_directory_remote(cls, name, directory, **kwargs):
        return cls(
            name=name,
            externaltype="directory",
            params_tuple=(("directory", directory),),
            **kwargs,
        )

    @classmethod
    def make_requests_remote(cls, name, url, **kwargs):
        return cls(
            name=name,
            externaltype="requests",
            params_tuple=(("url", url),),
            **kwargs,
        )


@frozen
class DirectoryRemoteConfig:
    name = field(validator=instance_of(str))
    directory = field(validator=instance_of(str))
    encryption = field(validator=instance_of(str), default="none")

    def initremote(self, repo_path):
        _check_output_do_inside(
            repo_path,
            "initremote",
            self.name,
            "type=directory",
            f"directory={self.directory}",
            f"encryption={self.encryption}",
            check_stderr=False,
        )

    def validate_config(self, repo_path):
        out = _check_output_do_inside(
            repo_path, "info", self.name, "--json", check_stderr=False
        )
        info = json.loads(out)
        assert info["type"] == "directory"

    def to_dict(self):
        return {"type": "directory", **attr.asdict(self)}

    @classmethod
    def from_dict(cls, d, **kwargs):
        d = {k: v for k, v in d.items() if k != "type"}
        return cls(**{**d, **kwargs})


@frozen
class S3RemoteConfig:
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

    _OPTIONAL_PARAMS = (
        "datacenter",
        "region",
        "host",
        "port",
        "protocol",
        "requeststyle",
        "signature",
        "fileprefix",
        "storageclass",
        "chunk",
        "public",
        "publicurl",
        "versioning",
        "partsize",
        "embedcreds",
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

    @property
    def initremote_params(self):
        params = [
            self.name,
            "type=S3",
            f"bucket={self.bucket}",
            f"encryption={self.encryption}",
        ]
        for key in self._OPTIONAL_PARAMS:
            value = getattr(self, key)
            if value is not None:
                params.append(f"{key}={value}")
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
        assert info["type"] == "S3"
        assert info["bucket"] == self.bucket

    _SECRET_FIELDS = ("aws_access_key_id", "aws_secret_access_key")

    def to_dict(self):
        d = {"type": "S3"}
        for a in attr.fields(type(self)):
            if a.name.startswith("_") or a.name in self._SECRET_FIELDS:
                continue
            value = getattr(self, a.name)
            if value is not None:
                d[a.name] = value
        return d

    @classmethod
    def from_dict(cls, d, **kwargs):
        d = {k: v for k, v in d.items() if k != "type" and not k.startswith("_")}
        return cls(**{**d, **kwargs})

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


@frozen
class GitAnnex:
    repo = field(validator=instance_of(Repo))
    annex = field(validator=instance_of(Annex))

    def __attrs_post_init__(self):
        assert Path(self.repo.working_dir).absolute() == self.annex.repo_path

    @property
    def repo_path(self):
        return Path(self.repo.working_dir)

    @property
    def index(self):
        return self.repo.index

    def get_relpath(self, path):
        return Path(path).relative_to(self.repo_path)

    def stage(self, path):
        self.repo.index.add([str(path)])

    def stage_annex(self, path):
        relpath = self.get_relpath(path)
        self.annex.add(relpath)
        self.repo.index.add([str(path)])

    def stage_unlink(self, path):
        self.repo.index.remove([str(path)])
        Path(path).unlink()

    @contextmanager
    def commit_context(self, message):
        yield self.repo.index
        self.repo.index.commit(message)

    def teardown(self, remove=False):
        self.annex.teardown()
        if remove:
            shutil.rmtree(self.repo_path)

    @classmethod
    def from_repo_path(cls, repo_path, external_remote_config=None, exist_ok=False):
        repo_path = Path(repo_path).absolute()
        match (repo_path.exists(), exist_ok):
            case (True, True):
                pass
            case (True, False):
                raise ValueError(f"{repo_path} exists and exist_ok is {exist_ok}")
            case (False, _):
                cls.init_repo_path(
                    repo_path, external_remote_config=external_remote_config
                )
        env = getattr(external_remote_config, "env", None)
        return cls(
            repo=Repo(repo_path),
            annex=Annex(repo_path=repo_path, env=env),
        )

    @staticmethod
    def init_repo_path(repo_path, external_remote_config=None):
        repo_path = Path(repo_path)
        assert not repo_path.exists(), f"repo already exists at {repo_path}"
        repo_path.mkdir(parents=True)
        repo = Repo.init(repo_path)
        repo.index.commit("initial commit")
        Annex.init_repo_path(repo_path, external_remote_config=external_remote_config)
        return repo
