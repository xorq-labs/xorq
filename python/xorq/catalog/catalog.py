#!/usr/bin/env python
import json
import shutil
import subprocess
import tempfile
import zipfile
from contextlib import (
    contextmanager,
    nullcontext,
)
from functools import cached_property, partial
from pathlib import Path
from subprocess import Popen
from urllib.parse import urlparse

import toolz
import yaml
from attr import (
    field,
    frozen,
)
from attr.validators import (
    deep_iterable,
    instance_of,
    optional,
)
from git import (
    Blob,
    Remote,
    Repo,
)

from xorq.catalog.annex import Annex, GitAnnex, remote_config_from_dict
from xorq.catalog.constants import (
    CATALOG_REMOTE_KEY,
    CATALOG_YAML_NAME,
    METADATA_APPEND,
    PREFERRED_SUFFIX,
    CatalogInfix,
)
from xorq.catalog.expr_utils import (
    build_expr_context,
    build_expr_context_zip,
    load_expr_from_zip,
)
from xorq.catalog.git_utils import (
    add_as_submodule,
    commit_context,
)
from xorq.catalog.zip_utils import BuildZip, make_zip_context, with_pure_suffix
from xorq.ibis_yaml.enums import DumpFiles, ExprKind


abspath = toolz.compose(Path.absolute, Path)
popen_shell = partial(Popen, shell=True)


@frozen
class Catalog:
    """A git-annex-backed registry for versioned build artifacts.

    A catalog is a git repository containing serialized xorq expressions
    as content-addressed zip archives.  Archives are tracked by git-annex
    so that cloning downloads only metadata; artifact content is fetched
    on demand.

    Construct via the classmethods ``from_name``, ``from_repo_path``,
    ``from_default``, ``clone_from``, or the dispatch helper ``from_kwargs``.
    """

    git_annex = field(validator=instance_of(GitAnnex))
    check_consistency: bool = field(default=True, repr=False)

    by_name_base_path = Path("~/.local/share/xorq/annex-catalogs").expanduser()
    submodule_rel_path = Path(".xorq/annex-catalogs")

    @property
    def repo(self):
        return self.git_annex.repo

    def __attrs_post_init__(self):
        self._ensure_catalog_yaml()
        if self.check_consistency:
            self.assert_consistency()

    def _ensure_catalog_yaml(self):
        assert not self.repo.bare
        if not any(
            self.catalog_yaml.yaml_relpath.name == blob.name
            for blob in self.repo.head.commit.tree.list_traverse()
        ):
            with self.commit_context(f"add {CATALOG_YAML_NAME}"):
                self.git_annex.stage(self.catalog_yaml.yaml_path)

    @property
    def repo_path(self):
        return Path(self.repo.working_dir)

    @property
    def catalog_yaml(self):
        return CatalogYAML(self.repo_path)

    def set_remote_config(self, remote_config):
        """Persist a remote config dict to catalog.yaml and commit."""
        self.catalog_yaml.set_remote(remote_config.to_dict())
        with self.commit_context(f"set remote: {remote_config.name}"):
            self.git_annex.stage(self.catalog_yaml.yaml_path)

    def get_remote_config(self, **kwargs):
        """Load the remote config from catalog.yaml, if any.

        Secrets (e.g. aws_secret_access_key) must be passed as kwargs.
        """
        remote_dict = self.catalog_yaml.remote_config
        if remote_dict is None:
            return None
        return remote_config_from_dict(remote_dict, **kwargs)

    def _add_zip(self, path, sync=True, aliases=()):
        # should we enable not syncing?
        with self.maybe_synchronizing(sync):
            catalog_addition = CatalogAddition(BuildZip(path), self, aliases=aliases)
            catalog_entry = catalog_addition.add()
            self.assert_consistency()
            return catalog_entry

    def _add_build_dir(self, build_dir, sync=True, aliases=()):
        with make_zip_context(build_dir) as zip_path:
            return self._add_zip(zip_path, sync=sync, aliases=aliases)

    def _add_expr(self, expr, sync=True, aliases=()):
        with build_expr_context(expr) as path:
            return self._add_build_dir(path, sync=sync, aliases=aliases)

    def add(self, obj, sync=True, aliases=()):
        """Add a build to the catalog.

        *obj* may be a ``Path`` to a zip archive, a ``Path`` to a build
        directory, or an xorq ``Expr``.  Returns the created ``CatalogEntry``.
        """
        from xorq.api import Expr  # noqa: PLC0415

        match obj:
            case Path() if obj.is_dir():
                f = self._add_build_dir
            case Path() if obj.is_file():
                f = self._add_zip
            case Expr():
                f = self._add_expr
            case _:
                raise ValueError(f"don't know how to handle type={type(obj)}")
        return f(obj, sync=sync, aliases=aliases)

    def remove(self, name, sync=True):
        """Remove an entry (and its aliases) from the catalog by name."""
        with self.maybe_synchronizing(sync):
            catalog_removal = CatalogRemoval.from_name_catalog(name, self)
            catalog_entry = catalog_removal.remove()
            self.assert_consistency()
            return catalog_entry

    def list(self):
        """Return the list of entry names in the catalog."""
        return self.catalog_yaml.contents[CatalogInfix.ENTRY]

    @property
    def _git_remotes(self):
        """Remotes that are real git remotes (have a fetch refspec), excluding annex special remotes."""

        def _has_fetch_refspec(remote):
            try:
                remote.config_reader.get("fetch")
                return True
            except (KeyError, Exception):
                return False

        return tuple(r for r in self.repo.remotes if _has_fetch_refspec(r))

    def fetch(self):
        """Fetch from all git remotes (excludes annex-only special remotes)."""
        return tuple(map(Remote.fetch, self._git_remotes))

    def push(self):
        """Push to all git remotes after verifying consistency."""
        self.assert_consistency()
        return tuple(map(Remote.push, self._git_remotes))

    def pull(self):
        """Pull from all git remotes."""
        return tuple(map(Remote.pull, self._git_remotes))

    @contextmanager
    def synchronizing(self):
        self.pull()
        yield
        self.push()

    @contextmanager
    def maybe_synchronizing(self, sync):
        ctx = self.synchronizing if sync else nullcontext
        with ctx():
            yield

    def sync(self):
        """Pull then push — shorthand for a full round-trip synchronization."""
        with self.synchronizing():
            pass

    def contains(self, name):
        """Return True if an entry with *name* exists in the catalog."""
        catalog_entry = CatalogEntry(name, self, require_exists=False)
        return catalog_entry.exists()

    def get_catalog_entry(self, name):
        """Look up a ``CatalogEntry`` by name.  Raises if not found."""
        assert name in self.list(), f"Entry '{name}' not found in catalog"
        catalog_entry = CatalogEntry(name, self)
        return catalog_entry

    def get_zip(self, name, dir_path=None):
        """Export an entry's archive to *dir_path* (default: cwd).  Returns the output path."""
        catalog_entry = self.get_catalog_entry(name)
        return catalog_entry.get(dir_path)

    @property
    def catalog_entries(self):
        return tuple(CatalogEntry(name, self) for name in self.list())

    @contextmanager
    def commit_context(self, message):
        with self.git_annex.commit_context(message) as index:
            yield index

    def add_alias(self, name, alias, sync=True):
        """Create an alias pointing at entry *name*.  Overwrites if the alias already exists."""
        with self.maybe_synchronizing(sync):
            catalog_entry = CatalogEntry(name, self)
            catalog_alias = CatalogAlias(alias=alias, catalog_entry=catalog_entry)
            catalog_alias.add()
            return catalog_alias

    def list_aliases(self):
        """Return the list of alias names in the catalog."""
        return self.catalog_yaml.contents[CatalogInfix.ALIAS]

    @property
    def catalog_aliases(self):
        return tuple(
            CatalogAlias(
                alias=alias,
                catalog_entry=CatalogEntry(
                    self.repo_path.joinpath(
                        CatalogInfix.ALIAS, alias + PREFERRED_SUFFIX
                    )
                    .readlink()
                    .with_suffix("")
                    .name,
                    self,
                ),
            )
            for alias in self.list_aliases()
        )

    def assert_consistency(self):
        """Verify that catalog.yaml, entries, metadata, and aliases are all in agreement."""
        # catalog_yaml is in repo
        catalog_yaml_relpath_string = str(self.catalog_yaml.yaml_relpath)
        path_strings = tuple(
            blob.path
            for blob in self.repo.head.commit.tree.list_traverse()
            if isinstance(blob, Blob)
        )
        assert catalog_yaml_relpath_string in path_strings

        # yaml aliases match filesystem aliases
        assert sorted(self.catalog_yaml.contents[CatalogInfix.ALIAS]) == sorted(
            ca.alias for ca in self.catalog_aliases
        )

        # everything else in repo is either catalog_path or metadata_path from an entry the catalog_yaml knows about, or an alias symlink
        actual = sorted(el for el in path_strings if el != catalog_yaml_relpath_string)
        expected = sorted(
            (
                *(
                    str(path.relative_to(self.repo_path))
                    for catalog_entry in self.catalog_entries
                    for path in (
                        catalog_entry.metadata_path,
                        catalog_entry.catalog_path,
                    )
                ),
                *(
                    str(catalog_alias.alias_path.relative_to(self.repo_path))
                    for catalog_alias in self.catalog_aliases
                ),
            )
        )
        assert actual == expected

    def add_as_submodule(self, root_repo):
        message = f"add submodule: {self.repo_path.name}"
        with commit_context(root_repo, message):
            add_as_submodule(root_repo, self.repo)

    @classmethod
    def clone_from(cls, url, repo_path=None, git_config=None, **remote_kwargs):
        """Clone a catalog repo, init git-annex, and fetch content.

        If catalog.yaml contains a ``remote`` key, the remote is enabled
        automatically.  For S3 remotes the caller must supply credentials::

            Catalog.clone_from(
                url,
                aws_access_key_id="...",
                aws_secret_access_key="...",
            )

        Use *git_config* to set repo-local git config before annex init
        (e.g. ``{"annex.security.allowed-ip-addresses": "all"}``).
        """
        if repo_path is None:
            name = Path(urlparse(url).path).stem
            repo_path = cls.name_to_repo_path(name)
        repo = Repo.clone_from(url, repo_path)

        # apply local git config before annex init (e.g. allowed-ip-addresses)
        if git_config:
            for key, value in git_config.items():
                result = subprocess.run(
                    ["git", "config", key, value],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    raise ValueError(
                        f"git config {key}={value} failed: {result.stderr}"
                    )
        Annex.init_repo_path(repo_path)

        # read remote config from catalog.yaml (written by set_remote_config)
        catalog_yaml = CatalogYAML(repo_path)
        remote_dict = catalog_yaml.remote_config
        remote_config = None
        if remote_dict is not None:
            remote_config = remote_config_from_dict(remote_dict, **remote_kwargs)
            remote_config.enableremote(repo_path)

        env = getattr(remote_config, "env", None)
        annex = Annex(repo_path=Path(repo.working_dir), env=env)
        annex.get()
        git_annex = GitAnnex(repo=repo, annex=annex)
        return cls(git_annex=git_annex)

    @classmethod
    def from_repo_path(
        cls, repo_path, init=None, check_consistency=True, remote_config=None
    ):
        init = not Path(repo_path).exists() if init is None else init
        if init:
            repo = cls.init_repo_path(repo_path, remote_config=remote_config)
        else:
            repo = Repo(repo_path)
        if not init:
            # temporary Annex without env to read remote.log and resolve
            # remote_config before we know what env should be
            annex = Annex(repo_path=Path(repo.working_dir))
            disk_config = annex.remote_config
            if remote_config is None:
                remote_config = disk_config
            elif disk_config is None:
                raise ValueError(
                    f"remote_config was passed but no remote is registered in git-annex remote.log at {repo_path}; "
                    f"pass init=True to create a new catalog"
                )
            else:
                passed = remote_config.to_dict()
                stored = disk_config.to_dict()
                # only compare fields explicitly set in the passed config;
                # git-annex may store extra defaults (storageclass, datacenter, etc.)
                diffs = {
                    k: (passed[k], stored.get(k))
                    for k in passed
                    if passed[k] != stored.get(k)
                }
                if diffs:
                    raise ValueError(
                        f"remote_config disagrees with git-annex remote.log: {diffs}"
                    )
        env = getattr(remote_config, "env", None)
        annex = Annex(repo_path=Path(repo.working_dir), env=env)
        git_annex = GitAnnex(repo=repo, annex=annex)
        catalog = cls(git_annex=git_annex, check_consistency=check_consistency)
        if init and remote_config is not None:
            catalog.set_remote_config(remote_config)
        return catalog

    @classmethod
    def from_name(cls, name, init=None, check_consistency=True, remote_config=None):
        repo_path = cls.name_to_repo_path(name)
        return cls.from_repo_path(
            repo_path,
            init=init,
            check_consistency=check_consistency,
            remote_config=remote_config,
        )

    @classmethod
    def from_default(cls, init=None, check_consistency=True, remote_config=None):
        return cls.from_name(
            name="default",
            init=init,
            check_consistency=check_consistency,
            remote_config=remote_config,
        )

    @classmethod
    def clone_from_as_submodule(cls, root_repo, url):
        name = Path(urlparse(url).path).stem
        repo_path = Path(root_repo.working_dir).joinpath(cls.submodule_rel_path, name)
        self = cls.clone_from(url, repo_path)
        self.add_as_submodule(root_repo)
        return self

    @classmethod
    def from_name_as_submodule(cls, root_repo, name, init=None):
        repo_path = Path(root_repo.working_dir).joinpath(cls.submodule_rel_path, name)
        self = cls.from_repo_path(repo_path, init=init)
        self.add_as_submodule(root_repo)
        return self

    @classmethod
    def from_kwargs(cls, name=None, path=None, url=None, root_repo=None, init=None):
        if isinstance(root_repo, (str, Path)):
            root_repo = Repo(root_repo)
        if root_repo:
            match (name, url, path):
                case (None, str(), None):
                    return cls.clone_from_as_submodule(root_repo=root_repo, url=url)
                case (str(), None, None):
                    return cls.from_name_as_submodule(root_repo=root_repo, name=name)
                case _:
                    raise ValueError(
                        f"With `root_repo`, provide exactly one of `name` or `url` "
                        f"(not `path`). Got: name={name!r}, url={url!r}, path={path!r}"
                    )
        elif url:
            if name:
                raise ValueError(
                    f"`url` and `name` are mutually exclusive. "
                    f"Got: name={name!r}, url={url!r}"
                )
            else:
                return cls.clone_from(url=url, repo_path=path)
        else:
            match (name, path):
                case (None, None):
                    return cls.from_default(init=init)
                case (str(), None):
                    return cls.from_name(name=name, init=init)
                case (None, str() | Path()):
                    catalog = Catalog.from_repo_path(Path(path), init=init)
                case _:
                    raise ValueError("`name` and `path` are mutually exclusive.")
            return catalog

    @classmethod
    def name_to_repo_path(cls, name):
        repo_path = cls.by_name_base_path.joinpath(name)
        return repo_path

    @staticmethod
    def init_repo_path(repo_path, bare=False, remote_config=None):
        assert not (repo_path := Path(repo_path)).exists(), (
            f"Catalog repo already exists at {repo_path}"
        )
        repo = Repo.init(repo_path, mkdir=True, bare=bare)
        repo.index.commit("initial commit")
        Annex.init_repo_path(repo_path, remote_config=remote_config)
        return repo


@frozen
class CatalogAddition:
    """Encapsulates the operation of adding a build archive to a catalog.

    Normally created internally by ``Catalog.add``; use ``from_expr`` to
    build directly from an xorq expression.
    """

    build_zip = field(validator=instance_of(BuildZip))
    catalog = field(validator=instance_of(Catalog))
    aliases = field(validator=deep_iterable(instance_of(str)), default=())
    _maybe_tmpfile = field(
        validator=optional(instance_of(tempfile._TemporaryFileWrapper)),
        default=None,
    )

    @property
    def name(self):
        return self.build_zip.name

    @property
    def metadata(self):
        return {"md5sum": self.build_zip.md5sum}

    @property
    def catalog_entry(self):
        return CatalogEntry(self.name, self.catalog, require_exists=False)

    def ensure_dirs(self):
        for p in (self.catalog_entry.metadata_path, self.catalog_entry.catalog_path):
            p.parent.mkdir(exist_ok=True, parents=True)

    @property
    def catalog_aliases(self):
        return tuple(CatalogAlias(alias, self.catalog_entry) for alias in self.aliases)

    @property
    def message(self):
        alias_message = f" (aliases {', '.join(self.aliases)})" if self.aliases else ""
        message = f"add: {self.name}{alias_message}"
        return message

    def _add(self):
        assert not self.catalog.contains(self.name), (
            f"Entry '{self.name}' already exists in catalog"
        )
        self.ensure_dirs()
        catalog_entry = self.catalog_entry
        catalog_entry.metadata_path.write_text(yaml.safe_dump(self.metadata))
        shutil.copy(self.build_zip.path, catalog_entry.catalog_path)
        ga = self.catalog.git_annex
        #
        self.catalog.catalog_yaml.add(self.name)
        ga.stage_annex(catalog_entry.catalog_path)
        ga.stage(catalog_entry.metadata_path)
        ga.stage(self.catalog.catalog_yaml.yaml_path)
        for catalog_alias in self.catalog_aliases:
            catalog_alias._add()
        return CatalogEntry(self.name, self.catalog, require_exists=True)

    def add(self):
        with self.catalog.commit_context(self.message):
            return self._add()

    @classmethod
    def from_expr(cls, expr, catalog):
        ntfh = tempfile.NamedTemporaryFile(suffix=PREFERRED_SUFFIX)
        with build_expr_context_zip(expr) as zip_path:
            shutil.copy(zip_path, ntfh.name)
        return cls(BuildZip(ntfh.name), catalog, maybe_tmpfile=ntfh)


@frozen
class CatalogEntry:
    """A single versioned entry in a catalog, identified by its content-derived name.

    Provides access to the entry's archive, metadata, deserialized expression,
    kind, backends, and any aliases pointing to it.
    """

    name = field(validator=instance_of(str))
    catalog = field(validator=instance_of(Catalog))
    require_exists = field(validator=instance_of(bool), default=True)

    def __attrs_post_init__(self):
        self.assert_consistency()
        if self.require_exists:
            assert self.exists(), f"Catalog entry '{self.name}' does not exist"

    @property
    def repo_path(self):
        return self.catalog.repo_path

    @property
    def metadata_path(self):
        metadata_path = self.repo_path.joinpath(
            CatalogInfix.METADATA, self.name + PREFERRED_SUFFIX + METADATA_APPEND
        )
        return metadata_path

    @property
    def catalog_path(self):
        catalog_path = self.repo_path.joinpath(
            CatalogInfix.ENTRY, self.name
        ).with_suffix(PREFERRED_SUFFIX)
        return catalog_path

    @property
    def expr(self):
        return load_expr_from_zip(self.catalog_path)

    @cached_property
    def kind(self) -> ExprKind:
        data = self._read_zip_member(DumpFiles.expr_metadata, json.loads)
        if not isinstance(data, dict):
            raise ValueError(
                f"Expected {DumpFiles.expr_metadata!r} to contain a JSON object in {self.catalog_path}"
            )
        return ExprKind(data["kind"])

    @cached_property
    def backends(self) -> tuple[str, ...]:
        data = self._read_zip_member(DumpFiles.profiles, yaml.safe_load)
        if not isinstance(data, dict):
            raise ValueError(
                f"Expected {DumpFiles.profiles!r} to contain a YAML mapping in {self.catalog_path}"
            )
        if non_dicts := tuple(v for v in data.values() if not isinstance(v, dict)):
            raise ValueError(
                f"Expected all profile entries to be mappings in {self.catalog_path}, got: {non_dicts!r}"
            )
        return tuple(value["con_name"] for value in data.values())

    @property
    def aliases(self):
        return tuple(
            catalog_alias
            for catalog_alias in self.catalog.catalog_aliases
            if catalog_alias.catalog_entry.name == self.name
        )

    @property
    def _exists_components(self):
        # catalog_path may be an annex symlink pointing to content not yet
        # available locally; lexists / is_symlink handles that case.
        catalog_path = self.catalog_path
        return {
            "metadata_path": self.metadata_path.exists(),
            "catalog_path": catalog_path.exists() or catalog_path.is_symlink(),
            "catalog_yaml_contents": self.catalog.catalog_yaml.contains(self.name),
        }

    def get(self, dir_path=None):
        target = (
            Path(dir_path or Path.cwd())
            .joinpath(self.name)
            .with_suffix(PREFERRED_SUFFIX)
        )
        shutil.copy(self.catalog_path, target)
        return target

    def assert_consistency(self):
        values = self._exists_components.values()
        assert all(values) or not any(values)

    def exists(self):
        return all(self._exists_components.values())

    def _read_zip_member(self, filename, read_f):
        with zipfile.ZipFile(self.catalog_path, "r") as zf:
            member_path = f"{self.name}/{filename}"
            if member_path not in zf.namelist():
                raise ValueError(f"{filename} not found in archive")
            return read_f(zf.read(member_path))


@frozen
class CatalogAlias:
    """A human-readable symlink pointing at a ``CatalogEntry``.

    Aliases live in the ``aliases/`` directory and can be reassigned over
    time.  Use ``list_revisions`` to view the git history of an alias.
    """

    alias = field(validator=instance_of(str))
    catalog_entry = field(validator=instance_of(CatalogEntry))

    @property
    def alias_path(self):
        return self.catalog_entry.repo_path.joinpath(
            CatalogInfix.ALIAS, self.alias
        ).with_suffix(PREFERRED_SUFFIX)

    @property
    def target(self):
        return Path("..").joinpath(
            CatalogInfix.ENTRY, self.catalog_entry.name + PREFERRED_SUFFIX
        )

    def ensure_dirs(self):
        self.alias_path.parent.mkdir(exist_ok=True, parents=True)

    def _add(self):
        alias_path = self.alias_path
        if alias_path.exists():
            if not alias_path.is_symlink():
                raise ValueError(f"non symlink already exists at {alias_path}")
            alias_path.unlink()
        else:
            self.ensure_dirs()
        alias_path.symlink_to(self.target)
        catalog = self.catalog_entry.catalog
        catalog_yaml = catalog.catalog_yaml
        ga = catalog.git_annex
        #
        catalog_yaml.add_alias(self.alias)
        ga.stage(alias_path)
        ga.stage(catalog_yaml.yaml_path)
        return self

    def add(self):
        message = f"add alias: {self.alias} -> {self.catalog_entry.name}"
        with self.catalog_entry.catalog.commit_context(message):
            self._add()

    def list_revisions(self):
        repo = self.catalog_entry.catalog.repo
        catalog = self.catalog_entry.catalog
        alias_relpath = str(self.alias_path.relative_to(self.catalog_entry.repo_path))
        blobs_commits = tuple(
            (
                # commit.tree.join raises KeyError for non-existent paths
                toolz.excepts(KeyError, commit.tree.join)(
                    Path(CatalogInfix.ALIAS, self.alias + PREFERRED_SUFFIX)
                ),
                commit,
            )
            for commit in repo.iter_commits(paths=alias_relpath)
        )
        result = tuple(
            (
                CatalogEntry(
                    with_pure_suffix(Path(blob.data_stream.read().decode()), "").name,
                    catalog,
                    require_exists=False,
                ),
                commit,
            )
            for blob, commit in blobs_commits
            if blob
        )
        return result

    def _remove(self):
        alias_path = self.alias_path
        assert alias_path.is_symlink(), f"no alias symlink at {alias_path}"
        catalog = self.catalog_entry.catalog
        catalog_yaml = catalog.catalog_yaml
        ga = catalog.git_annex
        #
        catalog_yaml.remove_alias(self.alias)
        ga.stage(catalog_yaml.yaml_path)
        ga.stage_unlink(alias_path)
        return self

    def remove(self):
        message = f"rm alias: {self.alias}"
        with self.catalog_entry.catalog.commit_context(message):
            self._remove()
        return self

    @classmethod
    def from_name(cls, name, catalog):
        alias_path = catalog.repo_path.joinpath(
            CatalogInfix.ALIAS,
            name,
        ).with_suffix(PREFERRED_SUFFIX)
        if alias_path.exists():
            catalog_alias = CatalogAlias(
                name,
                CatalogEntry(alias_path.readlink().with_suffix("").name, catalog),
            )
            return catalog_alias
        else:
            raise ValueError(f"no such alias {name}")


@frozen
class CatalogRemoval:
    """Encapsulates the operation of removing an entry (and its aliases) from a catalog."""

    catalog_entry = field(validator=instance_of(CatalogEntry))

    @property
    def message(self):
        name = with_pure_suffix(self.catalog_entry.catalog_path, "").name
        catalog_aliases = self.catalog_entry.aliases
        aliases_message = (
            f" (aliases {', '.join(catalog_alias.alias for catalog_alias in catalog_aliases)})"
            if catalog_aliases
            else ""
        )
        message = f"rm: {name}{aliases_message}"
        return message

    def _remove(self):
        catalog_entry = self.catalog_entry
        catalog = catalog_entry.catalog
        assert catalog_entry.exists(), (
            f"Cannot remove entry '{catalog_entry.name}': not found in catalog"
        )
        ga = catalog.git_annex
        #
        for catalog_alias in self.catalog_entry.aliases:
            catalog_alias._remove()
        catalog.catalog_yaml.remove(catalog_entry.name)
        ga.stage(catalog.catalog_yaml.yaml_path)
        for path in (catalog_entry.metadata_path, catalog_entry.catalog_path):
            ga.stage_unlink(path)
        return catalog_entry

    def remove(self):
        with self.catalog_entry.catalog.commit_context(self.message):
            return self._remove()

    @classmethod
    def from_name_catalog(cls, name, catalog):
        return cls(CatalogEntry(name=name, catalog=catalog))


@frozen
class CatalogYAML:
    repo_path = field(validator=instance_of(Path), converter=abspath)

    def __attrs_post_init__(self):
        if not self.yaml_path.exists():
            self.yaml_path.write_text(
                yaml.safe_dump(
                    {str(CatalogInfix.ENTRY): [], str(CatalogInfix.ALIAS): []}
                )
            )

    @property
    def yaml_path(self):
        return self.repo_path.joinpath(CATALOG_YAML_NAME)

    @property
    def yaml_relpath(self):
        return self.yaml_path.relative_to(self.repo_path)

    @property
    def contents(self):
        raw = yaml.safe_load(self.yaml_path.read_text())
        if isinstance(raw, list):
            # legacy format: plain list of entry names, no aliases section
            return {str(CatalogInfix.ENTRY): raw, str(CatalogInfix.ALIAS): []}
        return raw

    def set_contents(self, contents):
        self.yaml_path.write_text(yaml.safe_dump(contents))
        return self.yaml_path

    def contains(self, entry):
        return entry in self.contents[CatalogInfix.ENTRY]

    def add(self, entry):
        assert not self.contains(entry)
        contents = self.contents
        contents[CatalogInfix.ENTRY] = contents[CatalogInfix.ENTRY] + [entry]
        return self.set_contents(contents)

    def remove(self, entry):
        contents = self.contents
        contents[CatalogInfix.ENTRY] = [
            el for el in contents[CatalogInfix.ENTRY] if el != entry
        ]
        return self.set_contents(contents)

    def contains_alias(self, alias):
        return alias in self.contents[CatalogInfix.ALIAS]

    def add_alias(self, alias):
        if not self.contains_alias(alias):
            contents = self.contents
            contents[CatalogInfix.ALIAS] = contents[CatalogInfix.ALIAS] + [alias]
            self.set_contents(contents)
        return self.yaml_path

    def remove_alias(self, alias):
        contents = self.contents
        contents[CatalogInfix.ALIAS] = [
            el for el in contents[CatalogInfix.ALIAS] if el != alias
        ]
        return self.set_contents(contents)

    @property
    def remote_config(self):
        return self.contents.get(CATALOG_REMOTE_KEY)

    def set_remote(self, remote_dict):
        contents = self.contents
        contents[CATALOG_REMOTE_KEY] = remote_dict
        return self.set_contents(contents)
