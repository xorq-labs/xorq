#!/usr/bin/env python
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from configparser import NoOptionError, NoSectionError
from contextlib import (
    contextmanager,
    nullcontext,
)
from functools import cached_property, partial
from pathlib import Path
from subprocess import Popen
from typing import TYPE_CHECKING, Literal
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
    PushInfo,
    Remote,
    Repo,
)
from git.exc import GitCommandError

from xorq.catalog import constants as catalog_constants
from xorq.catalog.annex import (
    LOCAL_ANNEX,
    Annex,
    AnnexConfig,
    AnnexError,
    RemoteConfig,
)
from xorq.catalog.backend import (
    CatalogBackend,
    GitAnnexBackend,
    GitBackend,
    GitPointerBackend,
)
from xorq.catalog.constants import (
    ANNEX_BRANCH,
    CATALOG_YAML_NAME,
    CONTENT_STORE_YAML,
    MAIN_BRANCH,
    METADATA_APPEND,
    PREFERRED_SUFFIX,
)
from xorq.catalog.content_store import ContentStoreConfig, atomic_write
from xorq.catalog.enums import CatalogInfix
from xorq.catalog.exceptions import CatalogConfigurationError, CatalogPushError
from xorq.catalog.expr_utils import (
    build_expr_context,
    build_expr_context_zip,
    load_expr_from_zip,
)
from xorq.catalog.git_utils import (
    add_as_submodule,
    commit_context,
)
from xorq.catalog.s3_utils import S3_SECRET_FIELDS
from xorq.catalog.zip_utils import BuildZip, make_zip_context, with_pure_suffix
from xorq.common.utils.logging_utils import get_logger
from xorq.ibis_yaml.enums import DumpFiles, ExprKind


if TYPE_CHECKING:
    from xorq.api import Expr


logger = get_logger(__name__)


def _check_backend_exclusive(
    content_store_config: ContentStoreConfig | None,
    annex: AnnexConfig | None | Literal[False],
) -> None:
    if content_store_config is not None and annex is not None:
        raise ValueError(
            "content_store_config and annex are mutually exclusive; "
            "use one backend or the other"
        )


abspath = toolz.compose(Path.absolute, Path)


_PUSH_FAILURE_FLAGS = (
    PushInfo.REJECTED
    | PushInfo.REMOTE_REJECTED
    | PushInfo.REMOTE_FAILURE
    | PushInfo.ERROR
)


def _format_push_failures(
    push_infos: list[PushInfo], remote_name: str
) -> tuple[str, ...]:
    return tuple(
        f"{remote_name}/{info.local_ref or '?'}: {(info.summary or '').strip()}"
        for info in push_infos
        if info.flags & _PUSH_FAILURE_FLAGS
    )


def _ensure_wheel_artifacts(build_dir, project_path=None):
    """Ensure a build directory contains a wheel and requirements.txt.

    If either is missing, builds them from the given project_path
    (or the nearest pyproject.toml found by walking up from cwd).
    """
    build_dir = Path(build_dir)
    has_wheel = bool(list(build_dir.glob("*.whl")))
    reqs_path = build_dir / DumpFiles.requirements
    if has_wheel and reqs_path.exists():
        return
    from xorq.ibis_yaml.packager import (  # noqa: PLC0415
        PYPROJECT_NAME,
        WheelPackager,
        find_file_upwards,
    )

    if project_path is None:
        try:
            project_path = find_file_upwards(Path.cwd(), PYPROJECT_NAME).parent
        except ValueError as e:
            raise ValueError(
                f"cannot locate a {PYPROJECT_NAME} to build wheel artifacts from: "
                f"current working directory {Path.cwd()!s} has no {PYPROJECT_NAME} "
                f"in it or any parent. Pass project_path= to catalog.add() when "
                f"calling from outside the project (e.g. a Jupyter kernel)."
            ) from e
    packager = WheelPackager(project_path)
    bundle = packager.build()
    if not has_wheel:
        shutil.copy2(bundle.wheel_path, build_dir / bundle.wheel_path.name)
    if not reqs_path.exists():
        shutil.copy2(bundle.requirements_path, reqs_path)


popen_shell = partial(Popen, shell=True)


def _has_annex_branch(repo):
    """Check whether a Repo has a git-annex branch (local or remote-tracking)."""
    return any(ref.name.endswith(ANNEX_BRANCH) for ref in repo.refs)


def _has_local_annex_branch(repo):
    """Check whether a Repo has a local git-annex branch (i.e. a checked-out head)."""
    return ANNEX_BRANCH in repo.heads


def _try_resolve_annex_remote(repo_path, **remote_kwargs):
    """Try to resolve and enable an annex remote, with graceful degradation.

    Reads remote.log from the git-annex branch and resolves credentials
    from embedded creds, env vars, and *remote_kwargs*.  Returns the
    resolved ``RemoteConfig`` on success, or ``None`` if credentials are
    unavailable or the remote cannot be enabled.
    """
    try:
        tmp_annex = Annex.from_repo_path(repo_path)
        rc = tmp_annex.resolve_remote_config(**remote_kwargs)
        if rc is not None:
            tmp_annex.enableremote(rc)
            return rc
    except (ValueError, AnnexError):
        if remote_kwargs:
            raise
    return None


def _parse_catalog_yaml_blob(blob):
    """Parse a ``catalog.yaml`` blob from a merge stage; return defaults
    when the blob is missing (the file did not exist in that stage)."""
    if blob is None:
        return {CatalogInfix.ENTRY: [], CatalogInfix.ALIAS: []}
    raw = yaml12.parse_yaml(blob.data_stream.read().decode())
    if isinstance(raw, list):
        # Legacy on-disk format predates the entries/aliases dict schema.
        return {CatalogInfix.ENTRY: raw, CatalogInfix.ALIAS: []}
    return {
        CatalogInfix.ENTRY: raw.get(CatalogInfix.ENTRY, []),
        CatalogInfix.ALIAS: raw.get(CatalogInfix.ALIAS, []),
    }


def _three_way_list_merge(base, ours, theirs):
    """3-way merge of ordered lists treated as sets with ours-first ordering.

    An item in ``base`` that's been dropped by one side is treated as a
    removal and excluded from the result.  Items added by either side
    survive.  Duplicates collapse.
    """
    base_set = set(base)
    ours_set = set(ours)
    theirs_set = set(theirs)
    result = []
    seen = set()
    for item in (*ours, *theirs):
        if item in seen:
            continue
        if item in base_set and (item not in ours_set or item not in theirs_set):
            # present in base and dropped by at least one side -> remove
            continue
        result.append(item)
        seen.add(item)
    return result


class CatalogPullError(RuntimeError):
    """Raised by ``Catalog.pull`` for failures other than an unresolved
    merge — e.g. a detached HEAD pre-flight check, or a non-conflict
    ``git merge`` failure where re-raising the underlying error gives
    the user the actionable message.

    Distinct from ``CatalogMergeConflict``, which is raised specifically
    when a merge leaves files unresolved and the recovery recipe applies.
    Symmetric with ``CatalogPushError`` from #1899.
    """


class CatalogMergeConflict(Exception):
    """Raised by ``Catalog.pull`` when a merge leaves files unresolved.

    The ``conflicted`` attribute is a tuple of repo-relative paths still
    in the unmerged state — typically alias symlinks under ``aliases/``
    (add/add with diverging targets, or modify/delete).  The
    ``remote_name`` attribute is the name of the remote whose merge
    raised, when known.  The merge is left in-progress in the working
    tree so the user can resolve it.

    Recovery options:

    1. Abort the pull and keep your local state::

        catalog.repo.git.merge("--abort")

    2. Resolve manually, then commit::

        # for each path in exc.conflicted, pick a side or write the
        # resolved content, then:
        catalog.repo.git.add(*exc.conflicted)
        catalog.repo.git.commit("--no-edit")

    3. Take theirs wholesale for the conflicted alias::

        catalog.repo.git.checkout("--theirs", *exc.conflicted)
        catalog.repo.git.add(*exc.conflicted)
        catalog.repo.git.commit("--no-edit")

    The catalog's ``catalog.yaml`` has already been auto-resolved and
    staged before this exception is raised, so manual resolution only
    needs to address the paths in ``conflicted``.
    """

    def __init__(self, conflicted, remote_name=None):
        self.conflicted = tuple(conflicted)
        self.remote_name = remote_name
        prefix = (
            f"pull from remote {remote_name!r} "
            if remote_name is not None
            else "catalog.pull() "
        )
        super().__init__(
            f"{prefix}left {len(self.conflicted)} path(s) in conflict: "
            f"{', '.join(self.conflicted)}"
        )


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
        Annex.from_repo_path(self.repo_path).enableremote(remote_config)

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

    def _add_build_dir(
        self, build_dir, sync=True, aliases=(), exist_ok=False, project_path=None
    ):
        _ensure_wheel_artifacts(build_dir, project_path=project_path)
        with make_zip_context(build_dir) as zip_path:
            return self._add_zip(
                zip_path, sync=sync, aliases=aliases, exist_ok=exist_ok
            )

    def _add_expr(
        self,
        expr: Expr,
        sync: bool = True,
        aliases: tuple[str, ...] = (),
        exist_ok: bool = False,
        project_path: Path | None = None,
        relocate_reads: bool = False,
    ) -> CatalogEntry:
        with build_expr_context(expr, relocate_reads=relocate_reads) as path:
            return self._add_build_dir(
                path,
                sync=sync,
                aliases=aliases,
                exist_ok=exist_ok,
                project_path=project_path,
            )

    def add(
        self,
        obj: Expr | Path,
        sync: bool = True,
        aliases: tuple[str, ...] = (),
        exist_ok: bool = False,
        project_path: Path | None = None,
        relocate_reads: bool = False,
    ) -> CatalogEntry:
        """Add a build to the catalog.

        *obj* may be a ``Path`` to a zip archive, a ``Path`` to a build
        directory, or an xorq ``Expr``.  Returns the created ``CatalogEntry``.

        *project_path* is the directory containing the ``pyproject.toml`` used
        to build the wheel and requirements sidecars.  If omitted, the packager
        walks upward from the current working directory to find one.  Passing
        it explicitly is required when the caller's cwd is not inside the
        project (e.g. Jupyter kernels started from ``/tmp``).  Ignored for zip
        inputs, which are already complete build archives.

        *relocate_reads* controls how the build is produced and so only applies
        to an ``Expr`` input; ``Path`` inputs are already-built artifacts whose
        reads were settled at build time. Defaults to ``False`` (the CLI prefers
        ``True``, and pin/unpin pass it explicitly) until the fuse/bind execute
        path resolves relocated reads' base_path; see #2133.
        """
        from xorq.api import Expr  # noqa: PLC0415

        if relocate_reads and not isinstance(obj, Expr):
            raise ValueError(
                "relocate_reads only applies to an Expr input; "
                f"{type(obj)} is an already-built artifact"
            )

        shared = {"sync": sync, "aliases": aliases, "exist_ok": exist_ok}
        match obj:
            case Path() if obj.is_dir():
                return self._add_build_dir(obj, project_path=project_path, **shared)
            case Path() if obj.is_file():
                return self._add_zip(obj, **shared)
            case Expr():
                return self._add_expr(
                    obj,
                    project_path=project_path,
                    relocate_reads=relocate_reads,
                    **shared,
                )
            case _:
                raise ValueError(f"don't know how to handle type={type(obj)}")

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

    @staticmethod
    def _has_fetch_refspec(remote):
        try:
            remote.config_reader.get("fetch")
            return True
        except (NoOptionError, NoSectionError):
            return False

    @property
    def _git_remotes(self):
        """Remotes that are real git remotes (have a fetch refspec), excluding annex special remotes."""
        return tuple(r for r in self.repo.remotes if self._has_fetch_refspec(r))

    def _validated_git_remotes(self):
        """Return ``self._git_remotes`` after enforcing the single-remote contract.

        ``_git_remotes`` is a property that re-reads the underlying git
        config on each access. Callers want a stable snapshot for the
        duration of a single operation: this helper reads the property
        once, raises ``CatalogConfigurationError`` if there are two or
        more git remotes (ADR-0011), and returns the validated tuple. The
        error message names every remote found so the user can see what
        triggered the failure.
        """
        remotes = self._git_remotes
        if len(remotes) > 1:
            names = ", ".join(r.name for r in remotes)
            raise CatalogConfigurationError(
                f"catalog supports a single git remote (ADR-0011); "
                f"found {len(remotes)}: {names}. "
                f"Use Catalog.set_remote(name, url, force=True) to replace existing remotes, "
                f"or open an issue if you have a multi-remote use case."
            )
        return remotes

    def set_remote(self, name, url, force=False):
        """Configure the catalog's git remote.

        The catalog supports at most one git remote (ADR-0011). When the
        repo has no git remote, ``set_remote`` creates one with the given
        *name* and *url* and returns it.

        When a git remote is already configured, ``set_remote`` raises
        ``CatalogConfigurationError`` unless ``force=True`` is passed. The
        guard exists because silent replacement turns a typo in the
        remote name into the deletion of the existing remote with no
        signal — failing by default forces explicit opt-in. With
        ``force=True``, every existing git remote is deleted and replaced.
        """
        existing_remotes = self._git_remotes
        if existing_remotes and not force:
            existing = ", ".join(f"{r.name} -> {r.url}" for r in existing_remotes)
            raise CatalogConfigurationError(
                f"catalog has a git remote already configured ({existing}); "
                f"pass force=True to replace it."
            )
        for remote in existing_remotes:
            self.repo.delete_remote(remote)
        return self.repo.create_remote(name, url)

    def fetch(self):
        """Fetch from the configured git remote (no-op if no remote is configured)."""
        return tuple(map(Remote.fetch, self._validated_git_remotes()))

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
        """Push to the configured git remote after verifying consistency.

        Pushes ``main``, then ``git-annex`` (if present). Both pushes are
        always attempted — raises a single ``CatalogPushError`` listing
        every rejection or transport failure across both. No-op when no
        git remote is configured.

        Returns ``()``, ``(main_result,)``, or ``(main_result, annex_result)``.
        """
        remotes = self._validated_git_remotes()
        if not remotes:
            return ()
        self.assert_consistency()
        (remote,) = remotes
        main_result = remote.push()
        failure_messages = _format_push_failures(main_result, remote.name)
        if _has_local_annex_branch(self.repo):
            annex_result = remote.push(ANNEX_BRANCH)
            failure_messages += _format_push_failures(annex_result, remote.name)
            results = (main_result, annex_result)
        else:
            logger.debug(
                "skipping annex branch push: no local branch",
                annex_branch=ANNEX_BRANCH,
                repo_path=str(self.repo_path),
            )
            results = (main_result,)
        if failure_messages:
            raise CatalogPushError(f"push failed: {'; '.join(failure_messages)}")
        return results

    def pull(self):
        """Fetch and merge from the catalog's git remote; raise on unmerged paths.

        Replaces ``git pull`` (which inherits the user's ``pull.rebase``
        config and bails on divergent branches by default) with explicit
        ``git fetch`` + ``git merge``.  When the merge leaves
        ``catalog.yaml`` conflicted (typical when both sides appended
        to the entries or aliases lists), a Python 3-way list-merge
        resolves it: items present in the merge base and removed by
        one side are propagated as removals; items added by either
        side survive; duplicates are collapsed.  Anything still
        unmerged after that — typically alias symlinks at the same
        path with diverging targets — surfaces as
        ``CatalogMergeConflict`` with the conflicted paths and the
        remote name; the merge is left in-progress so the user can
        resolve it (see ``CatalogMergeConflict`` for recovery recipes).

        Pre-flights:

        - HEAD must be on a branch (the catalog API never detaches HEAD
          on its own — this only fails if the repo was put in detached
          state outside xorq).  Raises ``CatalogPullError``.
        - ``catalog.yaml`` in both ours (HEAD) and the remote tip must
          exist, parse, and have the expected dict-or-list shape.  The
          resolver assumes well-formed input on both sides; without
          this check, a ``catalog.yaml`` deleted on the remote tip
          would be silently treated as "theirs removed every entry"
          and the 3-way list merge would drop every prior entry, while
          a malformed or scalar-shaped yaml would leak a bare
          ``ValueError`` / ``AttributeError`` from inside the
          resolver.  Raises ``CatalogPullError`` naming the corrupt
          side.
        - A non-conflict ``git merge`` failure (e.g. the remote ref
          doesn't exist, the working tree is dirty, a hook rejected the
          merge commit) re-raises the original ``GitCommandError``
          rather than swallowing it and falling through to a misleading
          ``git commit --no-edit``.

        A catalog has at most one git remote (see ADR on single-remote
        catalogs).  No remote → no-op.
        """
        if self.repo.head.is_detached:
            raise CatalogPullError(
                "catalog.pull() requires a checked-out branch; HEAD is detached"
            )
        remotes = self._validated_git_remotes()
        if not remotes:
            return ()
        (remote,) = remotes
        branch = self.repo.active_branch.name
        fetch_result = remote.fetch()
        self._validate_catalog_yaml_in_commit(self.repo.head.commit, "ours")
        try:
            remote_commit = self.repo.commit(f"{remote.name}/{branch}")
        except Exception:
            # Remote ref didn't resolve (e.g. remote has no branches
            # yet).  Skip the theirs-side pre-flight and let `git
            # merge` raise the actionable ref-missing error below.
            remote_commit = None
        if remote_commit is not None:
            self._validate_catalog_yaml_in_commit(
                remote_commit, f"remote {remote.name!r}"
            )
        try:
            self.repo.git.merge(f"{remote.name}/{branch}", "--no-edit")
        except GitCommandError:
            if not (Path(self.repo.git_dir) / "MERGE_HEAD").exists():
                raise
            unmerged = self._resolve_yaml_conflict_if_any()
            if unmerged:
                raise CatalogMergeConflict(unmerged, remote_name=remote.name) from None
            self.repo.git.commit("--no-edit")
        return (fetch_result,)

    def _resolve_yaml_conflict_if_any(self):
        """Resolve ``catalog.yaml`` in the unmerged index and return remaining unmerged paths."""
        yaml_relpath = str(self.catalog_yaml.yaml_relpath)
        unmerged = self.repo.index.unmerged_blobs()
        if yaml_relpath in unmerged:
            stages = dict(unmerged[yaml_relpath])
            base = _parse_catalog_yaml_blob(stages.get(1))
            ours = _parse_catalog_yaml_blob(stages.get(2))
            theirs = _parse_catalog_yaml_blob(stages.get(3))
            merged = {
                CatalogInfix.ENTRY: _three_way_list_merge(
                    base[CatalogInfix.ENTRY],
                    ours[CatalogInfix.ENTRY],
                    theirs[CatalogInfix.ENTRY],
                ),
                CatalogInfix.ALIAS: _three_way_list_merge(
                    base[CatalogInfix.ALIAS],
                    ours[CatalogInfix.ALIAS],
                    theirs[CatalogInfix.ALIAS],
                ),
            }
            self.catalog_yaml.set_contents(merged)
            self.repo.git.add(yaml_relpath)
        remaining = self.repo.index.unmerged_blobs()
        return tuple(remaining.keys())

    def _validate_catalog_yaml_in_commit(self, commit, side_label):
        """Pre-flight check that ``catalog.yaml`` in *commit*'s tree
        exists, parses, and has the expected dict-or-list shape.

        Raises ``CatalogPullError`` (with *side_label* in the message)
        if the file is missing, fails to parse, or parses to anything
        other than a dict (current schema) or a list (legacy schema).
        See the ``pull`` docstring for the failure modes this guards
        against.
        """
        yaml_relpath = str(self.catalog_yaml.yaml_relpath)
        blob = next(
            (
                b
                for b in commit.tree.list_traverse()
                if isinstance(b, Blob) and b.path == yaml_relpath
            ),
            None,
        )
        if blob is None:
            raise CatalogPullError(
                f"{side_label} commit {commit.hexsha[:8]} is missing {yaml_relpath}"
            )
        try:
            raw = yaml12.parse_yaml(blob.data_stream.read().decode())
        except Exception as e:
            raise CatalogPullError(
                f"{side_label} commit {commit.hexsha[:8]} has malformed "
                f"{yaml_relpath}: {e}"
            ) from e
        if not isinstance(raw, (dict, list)):
            raise CatalogPullError(
                f"{side_label} commit {commit.hexsha[:8]} has unexpected "
                f"{yaml_relpath} shape: {type(raw).__name__} "
                f"(expected dict or list)"
            )
        if isinstance(raw, dict):
            for key in (CatalogInfix.ENTRY, CatalogInfix.ALIAS):
                val = raw.get(key)
                if val is not None and not isinstance(val, list):
                    raise CatalogPullError(
                        f"{side_label} commit {commit.hexsha[:8]} has "
                        f"non-list {key!r} in {yaml_relpath}: "
                        f"{type(val).__name__}"
                    )

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
        # Guard here directly so the check doesn't depend on pull/push internals.
        self._validated_git_remotes()
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
        config_paths = set(self.backend.repo_config_paths())
        if config_paths:
            actual = sorted(el for el in actual if el not in config_paths)
        expected = sorted(
            (
                *(
                    str(path.relative_to(self.repo_path))
                    for catalog_entry in self.catalog_entries
                    for path in (
                        catalog_entry.metadata_path,
                        self.backend.entry_tracked_path(catalog_entry.catalog_path),
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

    @staticmethod
    def _validate_content_store_config(
        repo_path: Path,
        content_store_config: ContentStoreConfig | None,
        annex: AnnexConfig | None | Literal[False],
    ) -> None:
        content_store_path = repo_path / CONTENT_STORE_YAML
        if not content_store_path.exists():
            return
        if isinstance(annex, AnnexConfig):
            raise ValueError(
                f"repo at {repo_path} uses the pointer backend "
                f"({CONTENT_STORE_YAML} present); cannot use git-annex"
            )
        if annex is False:
            raise ValueError(
                f"repo at {repo_path} uses the pointer backend "
                f"({CONTENT_STORE_YAML} present); cannot force plain git"
            )
        if content_store_config is not None:
            config = ContentStoreConfig.from_yaml(content_store_path)
            passed = {
                k: v
                for k, v in attr.asdict(content_store_config, recurse=False).items()
                if v is not None and k not in S3_SECRET_FIELDS
            }
            on_disk = {
                k: v
                for k, v in attr.asdict(config, recurse=False).items()
                if v is not None and k not in S3_SECRET_FIELDS
            }
            if passed != on_disk:
                raise ValueError(
                    f"explicit content_store_config conflicts with committed "
                    f"{CONTENT_STORE_YAML} at {repo_path}: "
                    f"passed {content_store_config}, on disk {config}"
                )

    @classmethod
    def clone_from(
        cls,
        url,
        repo_path=None,
        check_consistency=True,
        annex=None,
        content_store_config: ContentStoreConfig | None = None,
        git_config=None,
        **remote_kwargs,
    ):
        """Clone a catalog repo and detect the backend automatically.

        *content_store_config* and *annex* are mutually exclusive.
        If the cloned repo contains a ``content_store.yaml``, the pointer
        backend is used.  Otherwise *annex* controls the backend:

        - ``None`` (default) — auto-detect.  If the cloned repo has a
          ``git-annex`` branch, git-annex is initialised and the remote
          is enabled when credentials are available (embedded, env vars,
          or *remote_kwargs*).  Otherwise falls back to plain git.
        - ``False`` — force plain git, even if the repo has a
          ``git-annex`` branch.
        - Any ``AnnexConfig`` instance — git-annex is initialised and
          the remote is enabled if remote.log has a special remote
          configured.

        Content is **not** fetched eagerly; it is retrieved on demand
        when ``entry.expr`` is accessed (via ``fetch_content``).
        For S3 remotes without embedded credentials, the caller can
        supply credentials via *remote_kwargs* or environment variables
        (``XORQ_CATALOG_S3_*`` / ``XORQ_CONTENT_STORE_S3_*``).
        """
        _check_backend_exclusive(content_store_config, annex)
        if repo_path is None:
            name = Path(urlparse(url).path).stem
            repo_path = cls.name_to_repo_path(name)
        repo = Repo.clone_from(url, repo_path)

        # pointer backend detection: content_store.yaml in cloned repo
        content_store_path = Path(repo_path) / CONTENT_STORE_YAML
        if content_store_path.exists():
            try:
                cls._validate_content_store_config(
                    Path(repo_path), content_store_config, annex
                )
                backend = GitPointerBackend.from_repo(repo)
                catalog = cls(backend=backend)
                if check_consistency:
                    catalog.assert_consistency()
            except BaseException:
                shutil.rmtree(repo_path, ignore_errors=True)
                raise
            return catalog

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
            Annex.from_repo_path(repo_path).enableremote(remote_config)
        else:
            remote_config = _try_resolve_annex_remote(repo_path, **remote_kwargs)

        env = getattr(remote_config, "env", None)
        annex_obj = Annex.from_repo_path(repo.working_dir, env=env)
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
        content_store_config: ContentStoreConfig | None = None,
        **remote_kwargs,
    ):
        """Open or initialise a catalog; *content_store_config* and *annex* are mutually exclusive."""
        _check_backend_exclusive(content_store_config, annex)
        remote_config = annex if isinstance(annex, RemoteConfig) else None
        init = not Path(repo_path).exists() if init is None else init
        if init:
            repo = cls.init_repo_path(
                repo_path, annex=annex, content_store_config=content_store_config
            )
        else:
            repo = Repo(repo_path)

        # content_store_config detection: explicit param or content_store.yaml on disk
        content_store_path = Path(repo_path) / CONTENT_STORE_YAML
        if content_store_config is not None or content_store_path.exists():
            if content_store_path.exists():
                cls._validate_content_store_config(
                    Path(repo_path),
                    None if init else content_store_config,
                    annex,
                )
            else:
                # content_store_config provided but no committed content_store.yaml:
                # this is only reachable with init=False, where the config would never
                # be persisted (init_repo_path writes it). Building an ephemeral pointer
                # backend would silently produce a broken catalog on the next open.
                raise ValueError(
                    f"content_store_config was provided but {CONTENT_STORE_YAML} is "
                    f"absent at {repo_path}; create a pointer catalog with init=True "
                    "(the config is only persisted at init time)"
                )
            try:
                backend = GitPointerBackend.from_repo(repo)
                catalog = cls(backend=backend)
                if check_consistency:
                    catalog.assert_consistency()
            except BaseException:
                if init:
                    shutil.rmtree(repo_path, ignore_errors=True)
                raise
            return catalog

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
                Annex.from_repo_path(repo_path).enableremote(remote_config)

        env = getattr(remote_config, "env", None)
        annex_obj = Annex.from_repo_path(repo.working_dir, env=env)
        backend = GitAnnexBackend(repo=repo, annex=annex_obj)
        catalog = cls(backend=backend)
        if check_consistency:
            catalog.assert_consistency()
        return catalog

    @classmethod
    def from_name(
        cls,
        name,
        init=None,
        check_consistency=True,
        annex=None,
        content_store_config: ContentStoreConfig | None = None,
        **remote_kwargs,
    ):
        repo_path = cls.name_to_repo_path(name)
        return cls.from_repo_path(
            repo_path,
            init=init,
            check_consistency=check_consistency,
            annex=annex,
            content_store_config=content_store_config,
            **remote_kwargs,
        )

    @classmethod
    def _resolve_default_name(cls) -> str:
        from xorq.vendor.ibis.config import env_config  # noqa: PLC0415

        if name := env_config.XORQ_DEFAULT_CATALOG:
            return name
        try:
            name = catalog_constants.DEFAULT_CATALOG_CONFIG.read_text().strip()
        except FileNotFoundError:
            return catalog_constants.DEFAULT_CATALOG_NAME
        return name or catalog_constants.DEFAULT_CATALOG_NAME

    @classmethod
    def from_default(
        cls,
        init=None,
        check_consistency=True,
        annex=None,
        content_store_config: ContentStoreConfig | None = None,
        **remote_kwargs,
    ):
        name = cls._resolve_default_name()
        return cls.from_name(
            name=name,
            init=init,
            check_consistency=check_consistency,
            annex=annex,
            content_store_config=content_store_config,
            **remote_kwargs,
        )

    @classmethod
    def clone_from_as_submodule(
        cls,
        root_repo,
        url,
        check_consistency=True,
        annex=None,
        content_store_config: ContentStoreConfig | None = None,
    ):
        name = Path(urlparse(url).path).stem
        repo_path = Path(root_repo.working_dir).joinpath(cls.submodule_rel_path, name)
        self = cls.clone_from(
            url,
            repo_path,
            check_consistency=check_consistency,
            annex=annex,
            content_store_config=content_store_config,
        )
        self.add_as_submodule(root_repo)
        return self

    @classmethod
    def from_name_as_submodule(
        cls,
        root_repo,
        name,
        init=None,
        check_consistency=True,
        annex=None,
        content_store_config: ContentStoreConfig | None = None,
    ):
        repo_path = Path(root_repo.working_dir).joinpath(cls.submodule_rel_path, name)
        self = cls.from_repo_path(
            repo_path,
            init=init,
            check_consistency=check_consistency,
            annex=annex,
            content_store_config=content_store_config,
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
        content_store_config: ContentStoreConfig | None = None,
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
                        content_store_config=content_store_config,
                    )
                case (str(), None, None):
                    return cls.from_name_as_submodule(
                        root_repo=root_repo,
                        name=name,
                        check_consistency=check_consistency,
                        annex=annex,
                        content_store_config=content_store_config,
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
                    content_store_config=content_store_config,
                )
        else:
            match (name, path):
                case (None, None):
                    return cls.from_default(
                        init=init,
                        check_consistency=check_consistency,
                        annex=annex,
                        content_store_config=content_store_config,
                    )
                case (str(), None):
                    return cls.from_name(
                        name=name,
                        init=init,
                        check_consistency=check_consistency,
                        annex=annex,
                        content_store_config=content_store_config,
                    )
                case (None, str() | Path()):
                    catalog = Catalog.from_repo_path(
                        Path(path),
                        init=init,
                        check_consistency=check_consistency,
                        annex=annex,
                        content_store_config=content_store_config,
                    )
                case _:
                    raise ValueError("`name` and `path` are mutually exclusive.")
            return catalog

    @classmethod
    def name_to_repo_path(cls, name):
        repo_path = cls.by_name_base_path.joinpath(name)
        return repo_path

    @staticmethod
    def init_repo_path(
        repo_path: str | Path,
        bare: bool = False,
        annex: AnnexConfig | None = None,
        content_store_config: ContentStoreConfig | None = None,
    ) -> Repo:
        _check_backend_exclusive(content_store_config, annex)
        if content_store_config is not None and bare:
            raise ValueError("content_store_config is not supported with bare repos")
        repo_path = Path(repo_path)
        if repo_path.exists():
            raise FileExistsError(f"Catalog repo already exists at {repo_path}")
        repo = Repo.init(repo_path, mkdir=True, bare=bare, initial_branch=MAIN_BRANCH)
        try:
            if content_store_config is not None:
                if not isinstance(content_store_config, ContentStoreConfig):
                    raise TypeError(
                        f"content_store_config must be a ContentStoreConfig; "
                        f"got {type(content_store_config)}"
                    )
                content_store_config.write_yaml(repo_path / CONTENT_STORE_YAML)
                gitignore_path = repo_path / ".gitignore"
                with atomic_write(gitignore_path) as tmp:
                    tmp.write_text("entries/*.zip\n")
                repo.index.add([CONTENT_STORE_YAML, ".gitignore"])
            repo.index.commit("initial commit")
            if isinstance(annex, AnnexConfig):
                remote_config = annex if isinstance(annex, RemoteConfig) else None
                Annex.init_repo_path(repo_path, remote_config=remote_config)
        except BaseException:
            shutil.rmtree(repo_path, ignore_errors=True)
            raise
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
            return CatalogEntry(self.name, self.catalog, require_exists=True)
        self.ensure_dirs()
        catalog_entry = self.catalog_entry
        catalog_entry.metadata_path.write_text(yaml12.format_yaml(self.metadata))
        backend = self.catalog.backend
        self.catalog.catalog_yaml.add(self.name)
        backend.stage_content(self.build_zip.path, catalog_entry.catalog_path)
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
    # When False, skip assert_consistency and existence check (e.g. for removals).
    require_exists = field(validator=instance_of(bool), default=True)

    def __attrs_post_init__(self):
        if self.require_exists:
            self.assert_consistency()
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

    def load_expr(self, lazy=False, read_only_parquet_metadata=False, cache_dir=None):
        if not self.is_content_local:
            self.fetch()
        return load_expr_from_zip(
            self.catalog_path,
            lazy=lazy,
            read_only_parquet_metadata=read_only_parquet_metadata,
            cache_dir=cache_dir,
        )

    @property
    def expr(self):
        return self.load_expr()

    @property
    def lazy_expr(self):
        return self.load_expr(lazy=True)

    @property
    def projected_cache_key(self):
        return self.metadata.projected_cache_key

    @cached_property
    def sidecar_metadata(self) -> dict:
        """Always-available metadata from the git-tracked sidecar file."""
        return yaml12.parse_yaml(self.metadata_path.read_text()) or {}

    @cached_property
    def metadata(self):
        from xorq.vendor.ibis.expr.types.core import ExprMetadata  # noqa: PLC0415

        return ExprMetadata.from_dict(self.sidecar_metadata["expr_metadata"])

    @property
    def kind(self) -> ExprKind:
        return self.metadata.kind

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
        return tuple(self.sidecar_metadata.get("backends", ()))

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
        # available locally; exists / is_symlink handles that case.
        # For the pointer backend, the tracked path is a .pointer file.
        tracked = self.catalog.backend.entry_tracked_path(self.catalog_path)
        return {
            "has_metadata": self.metadata_path.exists(),
            "has_catalog_entry": (
                tracked.exists()
                or self.catalog_path.is_symlink()
                or self.catalog_path.exists()
            ),
            "in_catalog_yaml": self.catalog.catalog_yaml.contains(self.name),
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
        if alias_path.is_symlink() or alias_path.exists():
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
        # No public API exposes the component dict; _exists_components is
        # intentionally internal to CatalogEntry, accessed here within the
        # same module for removal logic.
        components = catalog_entry._exists_components  # noqa: SLF001  # xorq-style: disable=protected-access
        if not any(components.values()):
            raise ValueError(
                f"Cannot remove entry '{catalog_entry.name}': not found in catalog"
            )
        backend = catalog.backend
        for catalog_alias in self.catalog_entry.aliases:
            catalog_alias._remove()
        if components["in_catalog_yaml"]:
            catalog.catalog_yaml.remove(catalog_entry.name)
            backend.stage(catalog.catalog_yaml.yaml_path)
        if components["has_catalog_entry"]:
            backend.stage_unlink(catalog_entry.catalog_path)
        if components["has_metadata"]:
            backend.stage_unlink(catalog_entry.metadata_path)
        return catalog_entry

    def remove(self):
        with self.catalog_entry.catalog.commit_context(self.message):
            return self._remove()

    @classmethod
    def from_name_catalog(cls, name, catalog):
        return cls(CatalogEntry(name=name, catalog=catalog, require_exists=False))


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
