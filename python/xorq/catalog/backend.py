from __future__ import annotations

import abc
import os
import re
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from typing import Any

from attr import (
    Attribute,
    field,
    frozen,
)
from attr.validators import instance_of
from git import IndexFile, Remote, Repo
from git.exc import GitCommandError

from xorq.catalog.annex import Annex, AnnexError
from xorq.catalog.constants import (
    ANNEX_BRANCH,
    CONTENT_STORE_YAML,
    POINTER_SUFFIX,
)
from xorq.catalog.content_store import (
    ContentCache,
    ContentIntegrityError,
    ContentSpec,
    ContentStore,
    ContentStoreCapabilityError,
    ContentStoreConfig,
    PresignedContentStoreConfig,
    atomic_write,
    compute_content_key,
    compute_sha256,
    parse_pointer,
    write_pointer,
)
from xorq.catalog.enums import CatalogInfix
from xorq.catalog.git_utils import commit_context


_HOSTED_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


def _repo_has_annex_artifacts(repo: Repo) -> bool:
    repo_path = Path(repo.working_dir)
    if (repo_path / ".git" / "annex").is_dir():
        return True
    return any(
        ref.name == ANNEX_BRANCH or ref.name.endswith("/" + ANNEX_BRANCH)
        for ref in repo.refs
    )


def _repo_has_pointer_artifacts(repo: Repo) -> bool:
    repo_path = Path(repo.working_dir)
    if (repo_path / CONTENT_STORE_YAML).exists():
        return True
    entries_dir = repo_path / CatalogInfix.ENTRY
    if entries_dir.is_dir():
        return any(entries_dir.glob(f"*{POINTER_SUFFIX}"))
    return False


class CatalogBackend(abc.ABC):
    """ABC for the storage layer that Catalog delegates to."""

    @property
    @abc.abstractmethod
    def repo(self) -> Repo: ...

    @property
    def repo_path(self) -> Path:
        return Path(self.repo.working_dir)

    @abc.abstractmethod
    def stage(self, path: str | Path) -> None: ...

    @abc.abstractmethod
    def stage_content(
        self, source_path: str | Path, catalog_path: str | Path
    ) -> None: ...

    @abc.abstractmethod
    def stage_unlink(self, path: str | Path) -> None: ...

    @contextmanager
    def commit_context(self, message: str) -> Iterator[IndexFile]:
        # Delegates to git_utils.commit_context (skips empty commits) bound to this repo.
        with commit_context(self.repo, message) as index:
            yield index

    @abc.abstractmethod
    def is_content_local(self, path: str | Path) -> bool: ...

    @abc.abstractmethod
    def fetch_content(self, *paths: str | Path) -> None: ...

    def entry_tracked_path(self, catalog_path: str | Path) -> Path:
        """The path tracked in git for a given catalog entry (e.g. .pointer file)."""
        return Path(catalog_path)

    def repo_config_paths(self) -> tuple[str, ...]:
        """Repo-relative paths that assert_consistency should ignore."""
        return ()

    def validate_catalog_component(self, value: str, *, label: str) -> None:
        """Reject path traversal while preserving legacy local naming."""
        if not value or value in {".", ".."} or any(c in value for c in "/\\\0"):
            raise ValueError(f"{label} must be one safe path component")

    def validate_remote_url(self, remote_url: str) -> None:  # noqa: B027
        """Validate a prospective Git remote against backend-specific policy."""

    def validate_remote(self, remote: Remote) -> None:  # noqa: B027
        """Validate the effective URLs used by an existing Git remote."""

    def preflight_content_write(self) -> None:  # noqa: B027
        """Validate backend dependencies before mutating catalog metadata."""


@frozen
class GitBackend(CatalogBackend):
    """Plain-git backend — archives are stored as regular blobs."""

    repo: Repo = field(validator=instance_of(Repo))

    def stage(self, path: str | Path) -> None:
        self.repo.index.add([str(path)])

    def stage_content(self, source_path: str | Path, catalog_path: str | Path) -> None:
        with atomic_write(Path(catalog_path)) as tmp:
            shutil.copy(source_path, tmp)
        self.stage(catalog_path)

    def stage_unlink(self, path: str | Path) -> None:
        self.repo.index.remove([str(path)])
        Path(path).unlink()

    def is_content_local(self, path: str | Path) -> bool:
        return Path(path).exists()

    def fetch_content(self, *paths: str | Path) -> None:
        pass

    @classmethod
    def from_repo(cls, repo: Repo) -> GitBackend:
        return cls(repo=repo)


@frozen
class GitAnnexBackend(CatalogBackend):
    """Git-annex backend — archives are managed by git-annex with optional special remotes."""

    repo: Repo = field(validator=instance_of(Repo))
    annex: Annex = field(validator=instance_of(Annex))

    def __attrs_post_init__(self) -> None:
        if Path(self.repo.working_dir).absolute() != self.annex.repo_path:
            raise ValueError(
                f"repo working_dir {self.repo.working_dir} does not match "
                f"annex repo_path {self.annex.repo_path}"
            )
        if _repo_has_pointer_artifacts(self.repo):
            raise ValueError(
                f"repo at {self.repo.working_dir} has pointer-backend artifacts "
                f"({CONTENT_STORE_YAML} or {POINTER_SUFFIX} files); "
                f"cannot use the git-annex backend"
            )

    def get_relpath(self, path: str | Path) -> Path:
        return Path(path).relative_to(self.repo_path)

    def stage(self, path: str | Path) -> None:
        self.repo.index.add([str(path)])

    def stage_content(self, source_path: str | Path, catalog_path: str | Path) -> None:
        with atomic_write(Path(catalog_path)) as tmp:
            shutil.copy(source_path, tmp)
        relpath = self.get_relpath(catalog_path)
        self.annex.add(relpath)
        self.repo.index.add([str(catalog_path)])

    def stage_unlink(self, path: str | Path) -> None:
        self.repo.index.remove([str(path)])
        Path(path).unlink()

    def is_content_local(self, path: str | Path) -> bool:
        p = Path(path)
        return p.exists() and not (p.is_symlink() and not p.resolve().exists())

    def _has_any_remote(self) -> bool:
        if self.annex.remote_name is not None:
            return True
        return bool(self.repo.remotes)

    def fetch_content(self, *paths: str | Path) -> None:
        if not self._has_any_remote():
            missing = [p for p in paths if not self.is_content_local(p)]
            if missing:
                raise AnnexError(
                    f"Content not local and no remote configured: {missing}"
                )
            return
        relpaths = [self.get_relpath(p) for p in paths]
        self.annex.get(*relpaths)

    @classmethod
    def from_repo(cls, repo: Repo, env: Any = None) -> GitAnnexBackend:
        annex = Annex.from_repo_path(repo.working_dir, env=env)
        return cls(repo=repo, annex=annex)


def _validate_content_cache(instance: Any, attribute: Attribute, value: Any) -> None:
    if not isinstance(value, ContentCache):
        raise TypeError(
            f"'{attribute.name}' must be a ContentCache "
            f"(got {value!r} that is a {type(value)!r})"
        )


@frozen
class GitPointerBackend(CatalogBackend):
    """Pointer-file backend — archives are stored in an external content store.

    attrs @frozen uses a custom __setattr__ that does not prevent
    cached_property descriptors from writing to the instance __dict__,
    so the lazy properties below work correctly on frozen classes.
    """

    repo: Repo = field(validator=instance_of(Repo))
    cache: ContentCache = field(validator=_validate_content_cache)

    def __attrs_post_init__(self) -> None:
        if _repo_has_annex_artifacts(self.repo):
            raise ValueError(
                f"repo at {self.repo.working_dir} has git-annex artifacts; "
                f"cannot use the pointer backend"
            )

    @cached_property
    def _config(self) -> ContentStoreConfig:
        return ContentStoreConfig.from_yaml(
            Path(self.repo.working_dir) / CONTENT_STORE_YAML
        )

    @cached_property
    def _content_store(self) -> ContentStore:
        # Keep third-party ContentStoreConfig implementations compatible with
        # the original no-argument make_store() contract.
        return self._config.make_store()

    @cached_property
    def _presigned_stores(self) -> dict[str, ContentStore]:
        return {}

    @property
    def content_store(self) -> ContentStore:
        if not isinstance(self._config, PresignedContentStoreConfig):
            return self._content_store

        # Re-read and validate the sole Git remote for every hosted operation.
        # Stores remain cached by URL so their transport objects can be reused.
        remote_url = self._config.bound_remote_url(self.repo)
        (remote,) = tuple(self.repo.remotes)
        self.validate_remote(remote)
        store = self._presigned_stores.get(remote_url)
        if store is None:
            store = self._config.make_store(repo=self.repo)
            self._presigned_stores[remote_url] = store
        return store

    @cached_property
    def catalog_id(self) -> str:
        return self._config.catalog_id

    def _pointer_path(self, catalog_path: str | Path) -> Path:
        return Path(catalog_path).with_suffix(POINTER_SUFFIX)

    def validate_catalog_component(self, value: str, *, label: str) -> None:
        super().validate_catalog_component(value, label=label)
        if (
            isinstance(self._config, PresignedContentStoreConfig)
            and _HOSTED_COMPONENT_RE.fullmatch(value) is None
        ):
            raise ValueError(
                f"hosted {label} must be 1-128 ASCII characters: "
                "an alphanumeric first character followed by alphanumerics, '_' or '-'"
            )

    def validate_remote_url(self, remote_url: str) -> None:
        if isinstance(self._config, PresignedContentStoreConfig):
            self._config.validate_remote_url(remote_url)

    @staticmethod
    def _remote_urls(remote: Remote, *, push: bool) -> tuple[str, ...]:
        args = ["get-url"]
        if push:
            args.append("--push")
        args.extend(("--all", remote.name))
        try:
            output = remote.repo.git.remote(*args)
        except GitCommandError as exc:
            raise ValueError(
                "the presigned catalog Git remote must have exactly one URL"
            ) from exc
        return tuple(output.splitlines()) if output else ()

    def validate_remote(self, remote: Remote) -> None:
        if not isinstance(self._config, PresignedContentStoreConfig):
            return
        fetch_urls = self._remote_urls(remote, push=False)
        push_urls = self._remote_urls(remote, push=True)
        if len(fetch_urls) != 1 or len(push_urls) != 1:
            raise ValueError(
                "the presigned catalog Git remote must have exactly one fetch URL "
                "and one effective push URL"
            )
        if fetch_urls != push_urls:
            raise ValueError(
                "the presigned catalog Git remote fetch and push URLs must match"
            )
        self._config.validate_remote_url(fetch_urls[0])

    def preflight_content_write(self) -> None:
        _ = self.content_store

    @property
    def _client_managed_lifecycle(self) -> bool:
        if isinstance(self._config, PresignedContentStoreConfig):
            return False
        return self.content_store.client_managed_lifecycle

    def _parse_pointer(self, path: str | Path) -> tuple[str, int]:
        return parse_pointer(
            path,
            canonical=isinstance(self._config, PresignedContentStoreConfig),
        )

    def stage(self, path: str | Path) -> None:
        self.repo.index.add([str(path)])

    def _remove_from_index(self, path: str | Path) -> None:
        try:
            self.repo.index.remove([str(path)])
        except GitCommandError as exc:
            if "did not match any files" not in str(exc):
                raise

    def stage_content(self, source_path: str | Path, catalog_path: str | Path) -> None:
        # local copy is kept intentionally: it's read from at use time
        archive_path = Path(catalog_path)
        uploaded = False
        store = self.content_store
        with atomic_write(archive_path) as tmp:
            shutil.copy(source_path, tmp)
            sha256 = compute_sha256(tmp)
            size = tmp.stat().st_size
            key = compute_content_key(self.catalog_id, sha256)

        try:
            uploaded = store.ensure_present(
                key,
                archive_path,
                sha256=sha256,
            )
        except BaseException:
            archive_path.unlink(missing_ok=True)
            raise

        pointer_path = self._pointer_path(catalog_path)
        try:
            write_pointer(pointer_path, sha256, size)
            self.repo.index.add([str(pointer_path)])
        except BaseException:
            pointer_path.unlink(missing_ok=True)
            archive_path.unlink(missing_ok=True)
            if (
                uploaded
                and store.client_managed_lifecycle
                and not self._has_references(sha256)
            ):
                store.delete(key)
            raise

    def stage_unlink(self, path: str | Path) -> None:
        pointer_path = self._pointer_path(path)
        if pointer_path.exists():
            store = self.content_store if self._client_managed_lifecycle else None
            try:
                sha256, _ = self._parse_pointer(pointer_path)
            except (ValueError, OSError):
                import structlog  # noqa: PLC0415

                structlog.get_logger(__name__).warning(
                    "corrupt pointer file %s; removing without content store cleanup",
                    pointer_path,
                )
                sha256 = None

            self._remove_from_index(pointer_path)
            pointer_path.unlink()

            if sha256 is not None and store is not None:
                key = compute_content_key(self.catalog_id, sha256)
                if not self._has_references(sha256):
                    store.delete(key)

            archive_path = Path(path)
            if archive_path.exists():
                archive_path.unlink()
        else:
            self._remove_from_index(path)
            Path(path).unlink(missing_ok=True)

    def _iter_pointer_sha256s(self) -> Iterator[str]:
        # flat scan — entries_dir is intentionally flat (no subdirectories)
        entries_dir = self.repo_path / CatalogInfix.ENTRY
        if not entries_dir.is_dir():
            return
        for p in entries_dir.glob(f"*{POINTER_SUFFIX}"):
            try:
                sha256, _ = self._parse_pointer(p)
            except (ValueError, OSError):
                import structlog  # noqa: PLC0415

                structlog.get_logger(__name__).warning(
                    "corrupt pointer file %s; skipping for reference counting",
                    p,
                )
                continue
            yield sha256

    def _has_references(self, sha256: str) -> bool:
        return any(s == sha256 for s in self._iter_pointer_sha256s())

    def is_content_local(self, path: str | Path) -> bool:
        return Path(path).exists()

    def _verify_content(
        self, local: Path, path: str | Path, sha256: str, size: int
    ) -> None:
        actual_size = local.stat().st_size
        if actual_size != size:
            local.unlink(missing_ok=True)
            raise ContentIntegrityError(
                f"Size mismatch for {path}: expected {size}, got {actual_size}"
            )
        actual = compute_sha256(local)
        if actual != sha256:
            local.unlink(missing_ok=True)
            raise ContentIntegrityError(
                f"SHA256 mismatch for {path}: expected {sha256}, got {actual}"
            )

    def fetch_content(self, *paths: str | Path) -> None:
        pending: dict[str, tuple[ContentSpec, list[Path]]] = {}
        for path in paths:
            archive_path = Path(path)
            if archive_path.exists():
                continue
            pointer_path = self._pointer_path(path)
            if not pointer_path.exists():
                raise FileNotFoundError(
                    f"Pointer file missing for {path}: {pointer_path}"
                )
            try:
                sha256, size = self._parse_pointer(pointer_path)
            except (ValueError, OSError) as exc:
                raise ContentIntegrityError(
                    f"corrupt pointer file for {path}: {pointer_path}"
                ) from exc
            key = compute_content_key(self.catalog_id, sha256)
            spec = ContentSpec(key=key, sha256=sha256, size=size)
            previous = pending.get(key)
            if previous is not None:
                if previous[0] != spec:
                    raise ContentIntegrityError(
                        f"Conflicting pointer metadata for content key {key}"
                    )
                if archive_path not in previous[1]:
                    previous[1].append(archive_path)
                continue

            cached = self.cache.get_path(key)
            if cached is not None:
                try:
                    self._verify_content(cached, path, sha256, size)
                    with atomic_write(archive_path) as tmp_path:
                        shutil.copy2(cached, tmp_path)
                except FileNotFoundError:
                    # The cache entry can be evicted between get_path() and copy.
                    pass
                else:
                    continue

            pending[key] = (spec, [archive_path])

        if not pending:
            return

        downloads: dict[str, Path] = {}
        try:
            for key in pending:
                fd, tmp = tempfile.mkstemp(suffix=".xorq")
                try:
                    downloads[key] = Path(tmp)
                finally:
                    os.close(fd)

            self.content_store.get_many(
                (spec, downloads[key]) for key, (spec, _paths) in pending.items()
            )

            for key, (spec, archive_paths) in pending.items():
                downloaded = downloads[key]
                self._verify_content(
                    downloaded,
                    archive_paths[0],
                    spec.sha256,
                    spec.size,
                )

            for key, (_spec, archive_paths) in pending.items():
                downloaded = downloads[key]
                self.cache.put(key, downloaded)
                for archive_path in archive_paths:
                    with atomic_write(archive_path) as tmp_path:
                        shutil.copy2(downloaded, tmp_path)
        finally:
            for downloaded in downloads.values():
                downloaded.unlink(missing_ok=True)

    def gc_content_store(self, dry_run: bool = True) -> list[str]:
        """Find and optionally delete content store keys not referenced by any pointer file."""
        if isinstance(self._config, PresignedContentStoreConfig):
            raise ContentStoreCapabilityError(
                "hosted blob garbage collection is server-owned; "
                "client-side catalog gc is unavailable"
            )
        store = self.content_store
        if not store.client_managed_lifecycle:
            raise ContentStoreCapabilityError(
                "blob garbage collection is unavailable for this content store"
            )
        referenced = {
            compute_content_key(self.catalog_id, sha)
            for sha in self._iter_pointer_sha256s()
        }
        orphans = [
            key
            for key in store.list_keys(prefix=f"{self.catalog_id}/")
            if key not in referenced
        ]

        if not dry_run:
            for key in orphans:
                store.delete(key)

        return orphans

    def entry_tracked_path(self, catalog_path: str | Path) -> Path:
        return self._pointer_path(catalog_path)

    def repo_config_paths(self) -> tuple[str, ...]:
        return (".gitignore", CONTENT_STORE_YAML)

    @classmethod
    def from_repo(
        cls, repo: Repo, cache: ContentCache | None = None
    ) -> GitPointerBackend:
        return cls(
            repo=repo,
            cache=cache or ContentCache.default(),
        )
