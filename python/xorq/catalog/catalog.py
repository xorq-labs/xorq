#!/usr/bin/env python
import json
import shutil
import subprocess
import tempfile
from contextlib import (
    contextmanager,
    nullcontext,
)
from functools import cached_property, partial
from pathlib import Path
from subprocess import Popen
from urllib.parse import urlparse

import attr
import toolz
import yaml12
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

from xorq.catalog.annex import (
    LOCAL_ANNEX,
    Annex,
    AnnexConfig,
    AnnexError,
    RemoteConfig,
)
from xorq.catalog.backend import CatalogBackend, GitAnnexBackend, GitBackend
from xorq.catalog.constants import (
    ANNEX_BRANCH,
    CATALOG_YAML_NAME,
    MAIN_BRANCH,
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


def _has_annex_branch(repo):
    """Check whether a Repo has a git-annex branch (local or remote-tracking)."""
    return any(ref.name.endswith(ANNEX_BRANCH) for ref in repo.refs)


def _try_resolve_annex_remote(repo_path, **remote_kwargs):
    """Try to resolve and enable an annex remote, with graceful degradation.

    Reads remote.log from the git-annex branch and resolves credentials
    from embedded creds, env vars, and *remote_kwargs*.  Returns the
    resolved ``RemoteConfig`` on success, or ``None`` if credentials are
    unavailable or the remote cannot be enabled.
    """
    try:
        tmp_annex = Annex(repo_path=Path(repo_path))
        rc = tmp_annex.resolve_remote_config(**remote_kwargs)
        if rc is not None:
            rc.enableremote(repo_path)
            return rc
    except (ValueError, AnnexError):
        if remote_kwargs:
            raise
    return None


@frozen
class Catalog:
    """A git-backed registry for versioned build artifacts.

    A catalog is a git repository containing serialized xorq expressions
    as content-addressed zip archives.  When backed by git-annex, cloning
    downloads only metadata and artifact content is fetched on demand.
    A plain-git backend stores archives as regular blobs.

    Construct via the classmethods ``from_name``, ``from_repo_path``,
    ``from_default``, ``clone_from``, or the dispatch helper ``from_kwargs``.
    """

    backend = field(validator=instance_of(CatalogBackend))

    by_name_base_path = Path("~/.local/share/xorq/catalogs").expanduser()
    submodule_rel_path = Path(".xorq/catalogs")

    @property
    def repo(self):
        return self.backend.repo

    def __attrs_post_init__(self):
        self._ensure_catalog_yaml()

    def _ensure_catalog_yaml(self):
        assert not self.repo.bare
        if not any(
            self.catalog_yaml.yaml_relpath.name == blob.name
            for blob in self.repo.head.commit.tree.list_traverse()
        ):
            with self.commit_context(f"add {CATALOG_YAML_NAME}"):
                self.backend.stage(self.catalog_yaml.yaml_path)

    @property
    def repo_path(self):
        return Path(self.repo.working_dir)

    @property
    def catalog_yaml(self):
        return CatalogYAML(self.repo_path)

    @property
    def remote_config(self):
        """The resolved remote config, or None.

        On GitAnnexBackend catalogs, resolves from remote.log using the
        credentials the Annex was constructed with (embedded, env vars,
        or explicit kwargs passed at construction time).
        """
        annex = getattr(self.backend, "annex", None)
        if annex is None:
            return None
        return annex.resolve_remote_config()

    def set_remote_config(self, remote_config):
        """Update the git-annex special remote configuration.

        Calls ``enableremote`` to write the config to remote.log on the
        git-annex branch.  Use ``catalog.remote_config`` to read it back.
        """
        remote_config.enableremote(self.repo_path)

    def embed_readonly(self, readonly_config):
        """Embed read-only credentials into the git-annex branch.

        Verifies that *readonly_config* cannot write to the bucket, then
        sets ``embedcreds=yes`` and writes the config to remote.log.

        Raises ``ValueError`` if the credentials have write access.
        """
        readonly_config.assert_readonly()
        embed_config = attr.evolve(readonly_config, embedcreds="yes")
        self.set_remote_config(embed_config)

    def _add_zip(self, path, sync=True, aliases=(), exist_ok=False):
        # should we enable not syncing?
        with self.maybe_synchronizing(sync):
            catalog_addition = CatalogAddition(BuildZip(path), self, aliases=aliases)
            catalog_entry = catalog_addition.add(exist_ok=exist_ok)
            self.assert_consistency()
            return catalog_entry

    def _add_build_dir(self, build_dir, sync=True, aliases=(), exist_ok=False):
        with make_zip_context(build_dir) as zip_path:
            return self._add_zip(
                zip_path, sync=sync, aliases=aliases, exist_ok=exist_ok
            )

    def _add_expr(self, expr, sync=True, aliases=(), exist_ok=False):
        with build_expr_context(expr) as path:
            return self._add_build_dir(
                path, sync=sync, aliases=aliases, exist_ok=exist_ok
            )

    def add(self, obj, sync=True, aliases=(), exist_ok=False):
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
        return f(obj, sync=sync, aliases=aliases, exist_ok=exist_ok)

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

    def fetch_entries(self, *entries):
        """Fetch annex content for the given entries in a single operation.

        Each element can be a ``CatalogEntry`` or a string (entry name).
        No-op for plain-git backends.
        """
        paths = []
        for entry in entries:
            if isinstance(entry, str):
                entry = self.get_catalog_entry(entry)
            if not entry.is_content_local:
                paths.append(entry.catalog_path)
        if paths:
            self.backend.fetch_content(*paths)

    def push(self):
        """Push to all git remotes after verifying consistency."""
        self.assert_consistency()
        results = tuple(map(Remote.push, self._git_remotes))
        for remote in self._git_remotes:
            remote.push(ANNEX_BRANCH)
        return results

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

    def get_catalog_entry(self, name, maybe_alias: bool = False):
        """Look up a ``CatalogEntry`` by name.  Raises if not found."""
        if maybe_alias:
            if name in self.list_aliases():
                return CatalogAlias.from_name(name, self).catalog_entry

        if name not in self.list():
            raise ValueError(f"Entry '{name}' not found in catalog")
        catalog_entry = CatalogEntry(name, self)
        return catalog_entry

    def load(self, name_or_alias, con=None):
        """Return a tagged RemoteTable expression for a catalog entry (by hash or alias)."""
        from xorq.catalog.bind import _make_source_expr  # noqa: PLC0415

        entry = self.get_catalog_entry(name_or_alias, maybe_alias=True)
        alias = name_or_alias if name_or_alias in self.list_aliases() else None
        return _make_source_expr(entry, con=con, alias=alias)

    def bind(self, source_entry, *transforms, con=None):
        """Bind a source entry through one or more transform entries."""
        from xorq.catalog.bind import bind  # noqa: PLC0415

        return bind(source_entry, *transforms, con=con)

    def add_builder(self, spec, script_path, sync=True, aliases=(), exist_ok=False):
        """Add a Builder to the catalog as a builder entry.

        *script_path* is the path to the calling script; the enclosing project
        is discovered by walking up to ``pyproject.toml`` and an sdist is
        bundled into the catalog entry for isolated execution via
        ``SdistRunner``.
        """
        import tempfile  # noqa: PLC0415

        from xorq.common.utils.zip_utils import copy_path  # noqa: PLC0415
        from xorq.ibis_yaml.packager import BUILD_SDIST_NAME, Sdister  # noqa: PLC0415

        sdister = Sdister.from_script_path(script_path)

        with tempfile.TemporaryDirectory() as tmp:
            build_dir = spec.to_build_dir(Path(tmp))
            copy_path(sdister.sdist_path, build_dir / BUILD_SDIST_NAME)

            return self._add_build_dir(
                build_dir, sync=sync, aliases=aliases, exist_ok=exist_ok
            )

    def get_builder(self, name_or_alias):
        """Retrieve a Builder from a catalog entry (by hash or alias)."""
        import json as json_mod  # noqa: PLC0415

        from xorq.expr.builders import (  # noqa: PLC0415
            BUILDER_META_FILENAME,
            get_registry,
        )

        entry = self.get_catalog_entry(name_or_alias, maybe_alias=True)
        meta = entry._read_zip_member(BUILDER_META_FILENAME, json_mod.loads)
        builder_type = meta["type"]
        registry = get_registry()
        if builder_type not in registry:
            raise ValueError(
                f"Builder type {builder_type!r} not found in registry — "
                f"is the required package installed?"
            )
        builder_cls = registry[builder_type]
        return entry._with_extracted_build_dir(builder_cls.from_build_dir)

    def get_builder_from_expr(self, expr):
        """Recover a Builder from tags on a cataloged expression."""
        from xorq.common.utils.graph_utils import walk_nodes  # noqa: PLC0415
        from xorq.expr.builders import get_registry  # noqa: PLC0415
        from xorq.expr.relations import HashingTag, Tag  # noqa: PLC0415

        registry = get_registry()
        tag_nodes = walk_nodes((Tag, HashingTag), expr)
        for tag_node in tag_nodes:
            tag_name = tag_node.metadata.get("tag")
            if tag_name == "bsl":
                from xorq.expr.builders.semantic_model import (  # noqa: PLC0415
                    SemanticModelBuilder,
                )

                return SemanticModelBuilder.from_tagged(tag_node)
            if tag_name in registry:
                return registry[tag_name].from_tagged(tag_node)
        raise ValueError("No builder tags found in expression")

    def get_zip(self, name, dir_path=None):
        """Export an entry's archive to *dir_path* (default: cwd).  Returns the output path."""
        catalog_entry = self.get_catalog_entry(name)
        return catalog_entry.get(dir_path)

    @property
    def catalog_entries(self):
        return tuple(CatalogEntry(name, self) for name in self.list())

    @contextmanager
    def commit_context(self, message):
        with self.backend.commit_context(message) as index:
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
    def clone_from(
        cls,
        url,
        repo_path=None,
        check_consistency=True,
        annex=None,
        git_config=None,
        **remote_kwargs,
    ):
        """Clone a catalog repo and optionally init git-annex.

        *annex* controls the backend:

        - ``None`` (default) — auto-detect.  If the cloned repo has a
          ``git-annex`` branch, git-annex is initialised and the remote
          is enabled when credentials are available (embedded, env vars,
          or *remote_kwargs*).  Otherwise falls back to plain git.
        - ``False`` — force plain git, even if the repo has a
          ``git-annex`` branch.
        - Any ``AnnexConfig`` instance — git-annex is initialised and
          the remote is enabled if remote.log has a special remote configured.

        Content is **not** fetched eagerly; it is retrieved on demand
        when ``entry.expr`` is accessed (via ``fetch_content``).
        For S3 remotes without embedded credentials, the caller
        can supply credentials via *remote_kwargs* or environment
        variables (``XORQ_CATALOG_S3_*``).

        Use *git_config* to set repo-local git config before annex init
        (e.g. ``{"annex.security.allowed-ip-addresses": "all"}``).
        """
        if repo_path is None:
            name = Path(urlparse(url).path).stem
            repo_path = cls.name_to_repo_path(name)
        repo = Repo.clone_from(url, repo_path)

        # annex=False → force plain git
        if annex is False:
            backend = GitBackend(repo=repo)
            catalog = cls(backend=backend)
            if check_consistency:
                catalog.assert_consistency()
            return catalog

        # annex=None → auto-detect from git-annex branch
        if annex is None:
            if _has_annex_branch(repo):
                annex = LOCAL_ANNEX
            else:
                backend = GitBackend(repo=repo)
                catalog = cls(backend=backend)
                if check_consistency:
                    catalog.assert_consistency()
                return catalog

        if not isinstance(annex, AnnexConfig):
            raise TypeError(
                f"annex must be None, False, or an AnnexConfig; got {type(annex)}"
            )

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

        # resolve remote config with graceful degradation
        remote_config = annex if isinstance(annex, RemoteConfig) else None
        if remote_config is not None:
            remote_config.enableremote(repo_path)
        else:
            remote_config = _try_resolve_annex_remote(repo_path, **remote_kwargs)

        env = getattr(remote_config, "env", None)
        annex_obj = Annex(repo_path=Path(repo.working_dir), env=env)
        backend = GitAnnexBackend(repo=repo, annex=annex_obj)
        catalog = cls(backend=backend)
        if check_consistency:
            catalog.assert_consistency()
        return catalog

    @classmethod
    def from_repo_path(
        cls,
        repo_path,
        init=None,
        check_consistency=True,
        annex=None,
        **remote_kwargs,
    ):
        remote_config = annex if isinstance(annex, RemoteConfig) else None
        init = not Path(repo_path).exists() if init is None else init
        if init:
            repo = cls.init_repo_path(repo_path, annex=annex)
        else:
            repo = Repo(repo_path)

        # annex=False → force plain git
        if annex is False:
            backend = GitBackend(repo=repo)
            catalog = cls(backend=backend)
            if check_consistency:
                catalog.assert_consistency()
            return catalog

        # annex=None → auto-detect from git-annex branch (existing repos only)
        if annex is None:
            if not init and _has_annex_branch(repo):
                annex = LOCAL_ANNEX
            else:
                backend = GitBackend(repo=repo)
                catalog = cls(backend=backend)
                if check_consistency:
                    catalog.assert_consistency()
                return catalog

        if not isinstance(annex, AnnexConfig):
            raise TypeError(
                f"annex must be None, False, or an AnnexConfig; got {type(annex)}"
            )

        if not init:
            # ensure annex is initialized locally (e.g. after
            # git submodule update --init, which clones but doesn't annex init)
            Annex.init_repo_path(repo_path)
            if remote_config is None:
                # try to resolve remote config from remote.log + env vars;
                # gracefully degrade to None when credentials/fields are
                # unavailable (e.g. directory remote without env vars set)
                remote_config = _try_resolve_annex_remote(repo_path, **remote_kwargs)
            else:
                # ensure the special remote is enabled locally (e.g. after
                # git submodule add, which clones but doesn't enableremote)
                remote_config.enableremote(repo_path)

        env = getattr(remote_config, "env", None)
        annex_obj = Annex(repo_path=Path(repo.working_dir), env=env)
        backend = GitAnnexBackend(repo=repo, annex=annex_obj)
        catalog = cls(backend=backend)
        if check_consistency:
            catalog.assert_consistency()
        return catalog

    @classmethod
    def from_name(
        cls, name, init=None, check_consistency=True, annex=None, **remote_kwargs
    ):
        repo_path = cls.name_to_repo_path(name)
        return cls.from_repo_path(
            repo_path,
            init=init,
            check_consistency=check_consistency,
            annex=annex,
            **remote_kwargs,
        )

    @classmethod
    def from_default(
        cls, init=None, check_consistency=True, annex=None, **remote_kwargs
    ):
        return cls.from_name(
            name="default",
            init=init,
            check_consistency=check_consistency,
            annex=annex,
            **remote_kwargs,
        )

    @classmethod
    def clone_from_as_submodule(
        cls, root_repo, url, check_consistency=True, annex=None
    ):
        name = Path(urlparse(url).path).stem
        repo_path = Path(root_repo.working_dir).joinpath(cls.submodule_rel_path, name)
        self = cls.clone_from(
            url, repo_path, check_consistency=check_consistency, annex=annex
        )
        self.add_as_submodule(root_repo)
        return self

    @classmethod
    def from_name_as_submodule(
        cls, root_repo, name, init=None, check_consistency=True, annex=None
    ):
        repo_path = Path(root_repo.working_dir).joinpath(cls.submodule_rel_path, name)
        self = cls.from_repo_path(
            repo_path, init=init, check_consistency=check_consistency, annex=annex
        )
        self.add_as_submodule(root_repo)
        return self

    @classmethod
    def from_kwargs(
        cls,
        name=None,
        path=None,
        url=None,
        root_repo=None,
        init=None,
        check_consistency=True,
        annex=None,
    ):
        if isinstance(root_repo, (str, Path)):
            root_repo = Repo(root_repo)
        if root_repo:
            match (name, url, path):
                case (None, str(), None):
                    return cls.clone_from_as_submodule(
                        root_repo=root_repo,
                        url=url,
                        check_consistency=check_consistency,
                        annex=annex,
                    )
                case (str(), None, None):
                    return cls.from_name_as_submodule(
                        root_repo=root_repo,
                        name=name,
                        check_consistency=check_consistency,
                        annex=annex,
                    )
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
                return cls.clone_from(
                    url=url,
                    repo_path=path,
                    check_consistency=check_consistency,
                    annex=annex,
                )
        else:
            match (name, path):
                case (None, None):
                    return cls.from_default(
                        init=init,
                        check_consistency=check_consistency,
                        annex=annex,
                    )
                case (str(), None):
                    return cls.from_name(
                        name=name,
                        init=init,
                        check_consistency=check_consistency,
                        annex=annex,
                    )
                case (None, str() | Path()):
                    catalog = Catalog.from_repo_path(
                        Path(path),
                        init=init,
                        check_consistency=check_consistency,
                        annex=annex,
                    )
                case _:
                    raise ValueError("`name` and `path` are mutually exclusive.")
            return catalog

    @classmethod
    def name_to_repo_path(cls, name):
        repo_path = cls.by_name_base_path.joinpath(name)
        return repo_path

    @staticmethod
    def init_repo_path(repo_path, bare=False, annex=None):
        assert not (repo_path := Path(repo_path)).exists(), (
            f"Catalog repo already exists at {repo_path}"
        )
        repo = Repo.init(repo_path, mkdir=True, bare=bare, initial_branch=MAIN_BRANCH)
        repo.index.commit("initial commit")
        if isinstance(annex, AnnexConfig):
            remote_config = annex if isinstance(annex, RemoteConfig) else None
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

    @cached_property
    def metadata(self):
        prefix = self.build_zip.internal_prefix
        expr_data = self.build_zip.read_member(
            f"{prefix}/{DumpFiles.expr_metadata}", json.loads
        )
        profiles_data = self.build_zip.read_member(
            f"{prefix}/{DumpFiles.profiles}", yaml12.parse_yaml
        )
        backends = [
            v["con_name"] for v in profiles_data.values() if isinstance(v, dict)
        ]
        return {
            k: v
            for k, v in {
                "md5sum": self.build_zip.md5sum,
                "backends": backends,
                "expr_metadata": expr_data,
            }.items()
            if v is not None
        }

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

    def _add(self, exist_ok=False):
        if self.catalog.contains(self.name):
            if not exist_ok:
                raise ValueError(f"Entry '{self.name}' already exists in catalog")
            for catalog_alias in self.catalog_aliases:
                catalog_alias._add()
            return None
        self.ensure_dirs()
        catalog_entry = self.catalog_entry
        catalog_entry.metadata_path.write_text(yaml12.format_yaml(self.metadata))
        shutil.copy(self.build_zip.path, catalog_entry.catalog_path)
        backend = self.catalog.backend
        #
        self.catalog.catalog_yaml.add(self.name)
        backend.stage_content(catalog_entry.catalog_path)
        backend.stage(catalog_entry.metadata_path)
        backend.stage(self.catalog.catalog_yaml.yaml_path)
        for catalog_alias in self.catalog_aliases:
            catalog_alias._add()
        return CatalogEntry(self.name, self.catalog, require_exists=True)

    def add(self, exist_ok=False):
        with self.catalog.commit_context(self.message):
            return self._add(exist_ok=exist_ok)

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

    Prefer sidecar-backed properties (``metadata``, ``kind``, ``columns``,
    ``backends``, ``composed_from``, ``root_tag``) over ``expr`` /
    ``lazy_expr``.  The sidecar is always available; ``expr`` auto-fetches
    annex content from the remote if not local.  See ADR-0003 for extension
    guidelines.
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
        if not self.is_content_local:
            self.fetch()
        return load_expr_from_zip(self.catalog_path)

    @property
    def lazy_expr(self):
        if not self.is_content_local:
            self.fetch()
        return load_expr_from_zip(self.catalog_path, lazy=True)

    @property
    def parquet_cache_paths(self) -> tuple[str]:
        return self.metadata.parquet_cache_paths

    @cached_property
    def sidecar_metadata(self) -> dict:
        """Always-available metadata from the git-tracked sidecar file."""
        return yaml12.parse_yaml(self.metadata_path.read_text()) or {}

    @cached_property
    def metadata(self):
        from xorq.vendor.ibis.expr.types.core import ExprMetadata  # noqa: PLC0415

        try:
            return ExprMetadata.from_dict(self.sidecar_metadata["expr_metadata"])
        except (KeyError, TypeError):
            return None

    @property
    def kind(self) -> ExprKind | None:
        return self.metadata.kind if self.metadata is not None else None

    @property
    def columns(self) -> tuple[str]:
        return tuple(self.metadata.schema_out)

    @property
    def root_tag(self) -> str:
        return self.metadata.root_tag or ""

    @cached_property
    def composed_from(self) -> tuple:
        """Catalog entry references this entry was composed from."""
        return self.metadata.composed_from

    @cached_property
    def backends(self) -> tuple[str, ...]:
        return tuple(self.sidecar_metadata["backends"])

    @property
    def aliases(self):
        return tuple(
            catalog_alias
            for catalog_alias in self.catalog.catalog_aliases
            if catalog_alias.catalog_entry.name == self.name
        )

    @property
    def is_content_local(self):
        """True when the entry's archive can be read right now."""
        return self.catalog.backend.is_content_local(self.catalog_path)

    @property
    def is_available(self):
        """True when the entry is registered *and* its content is local."""
        return self.exists() and self.is_content_local

    def fetch(self):
        """Fetch this entry's annex content from the remote.

        No-op for plain-git backends.
        """
        self.catalog.backend.fetch_content(self.catalog_path)

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
        if not self.is_content_local:
            self.fetch()
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
        """True when the entry is registered in the catalog (content may not be local)."""
        return all(self._exists_components.values())

    def _read_zip_member(self, filename, read_f):
        with zipfile.ZipFile(self.catalog_path, "r") as zf:
            member_path = f"{self.name}/{filename}"
            if member_path not in zf.namelist():
                raise ValueError(f"{filename} not found in archive")
            return read_f(zf.read(member_path))

    def _with_extracted_build_dir(self, fn):
        """Extract archive to a temp directory and call fn(build_dir)."""
        with zipfile.ZipFile(self.catalog_path, "r") as zf:
            with tempfile.TemporaryDirectory() as tmp:
                zf.extractall(tmp)
                build_dir = Path(tmp) / self.name
                return fn(build_dir)

    @cached_property
    def builder_meta(self):
        """Read builder_meta.json if present, else None."""
        from xorq.expr.builders import BUILDER_META_FILENAME  # noqa: PLC0415

        try:
            return self._read_zip_member(BUILDER_META_FILENAME, json.loads)
        except ValueError:
            return None

    @property
    def is_builder(self) -> bool:
        return self.builder_meta is not None


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
            if (
                alias_path.resolve()
                == self.alias_path.parent.joinpath(self.target).resolve()
            ):
                return None
            alias_path.unlink()
        else:
            self.ensure_dirs()
        alias_path.symlink_to(self.target)
        catalog = self.catalog_entry.catalog
        catalog_yaml = catalog.catalog_yaml
        backend = catalog.backend
        #
        catalog_yaml.add_alias(self.alias)
        backend.stage(alias_path)
        backend.stage(catalog_yaml.yaml_path)
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
        backend = catalog.backend
        #
        catalog_yaml.remove_alias(self.alias)
        backend.stage(catalog_yaml.yaml_path)
        backend.stage_unlink(alias_path)
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
        backend = catalog.backend
        #
        for catalog_alias in self.catalog_entry.aliases:
            catalog_alias._remove()
        catalog.catalog_yaml.remove(catalog_entry.name)
        backend.stage(catalog.catalog_yaml.yaml_path)
        for path in (catalog_entry.metadata_path, catalog_entry.catalog_path):
            backend.stage_unlink(path)
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
                yaml12.format_yaml(
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
        raw = yaml12.read_yaml(self.yaml_path)
        if isinstance(raw, list):
            # legacy format: plain list of entry names, no aliases section
            return {str(CatalogInfix.ENTRY): raw, str(CatalogInfix.ALIAS): []}
        return raw

    def set_contents(self, contents):
        self.yaml_path.write_text(yaml12.format_yaml(contents))
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
