from __future__ import annotations

import abc
import shutil
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
from git import IndexFile, Repo
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
    ContentStore,
    ContentStoreConfig,
    compute_content_key,
    compute_sha256,
    parse_pointer,
    write_pointer,
)
from xorq.catalog.enums import CatalogInfix
from xorq.common.utils.file_utils import atomic_write


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
        yield self.repo.index
        self.repo.index.commit(message)

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
    def content_store(self) -> ContentStore:
        return self._config.make_store()

    @cached_property
    def catalog_id(self) -> str:
        return self._config.catalog_id

    def _pointer_path(self, catalog_path: str | Path) -> Path:
        return Path(catalog_path).with_suffix(POINTER_SUFFIX)

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
        with atomic_write(archive_path) as tmp:
            shutil.copy(source_path, tmp)
            sha256 = compute_sha256(tmp)
            size = tmp.stat().st_size
            key = compute_content_key(self.catalog_id, sha256)

        try:
            if not self.content_store.exists(key):
                self.content_store.put(key, archive_path, sha256=sha256)
                uploaded = True
        except BaseException:
            archive_path.unlink(missing_ok=True)
            if uploaded and not self._has_references(sha256):
                self.content_store.delete(key)
            raise

        pointer_path = self._pointer_path(catalog_path)
        try:
            write_pointer(pointer_path, sha256, size)
            self.repo.index.add([str(pointer_path)])
        except BaseException:
            pointer_path.unlink(missing_ok=True)
            archive_path.unlink(missing_ok=True)
            if uploaded and not self._has_references(sha256):
                self.content_store.delete(key)
            raise

    def stage_unlink(self, path: str | Path) -> None:
        pointer_path = self._pointer_path(path)
        if pointer_path.exists():
            try:
                sha256, _ = parse_pointer(pointer_path)
            except (ValueError, OSError):
                import structlog  # noqa: PLC0415

                structlog.get_logger(__name__).warning(
                    "corrupt pointer file %s; removing without content store cleanup",
                    pointer_path,
                )
                sha256 = None

            self._remove_from_index(pointer_path)
            pointer_path.unlink()

            if sha256 is not None:
                key = compute_content_key(self.catalog_id, sha256)
                if not self._has_references(sha256):
                    self.content_store.delete(key)

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
                sha256, _ = parse_pointer(p)
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
        if Path(path).exists():
            return True
        pointer_path = self._pointer_path(path)
        if not pointer_path.exists():
            return False
        try:
            sha256, _ = parse_pointer(pointer_path)
        except (ValueError, OSError):
            return False
        key = compute_content_key(self.catalog_id, sha256)
        return self.cache.contains(key)

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
                sha256, size = parse_pointer(pointer_path)
            except (ValueError, OSError) as exc:
                raise ContentIntegrityError(
                    f"corrupt pointer file for {path}: {pointer_path}"
                ) from exc
            key = compute_content_key(self.catalog_id, sha256)

            cached = self.cache.get_path(key)
            fetched = False
            if cached is None:
                cached = self.cache.fetch_from(self.content_store, key)
                fetched = True

            try:
                try:
                    self._verify_content(cached, path, sha256, size)
                    with atomic_write(archive_path) as tmp_path:
                        shutil.copy2(cached, tmp_path)
                except FileNotFoundError:
                    if fetched:
                        raise
                    cached = self.cache.fetch_from(self.content_store, key)
                    fetched = True
                    self._verify_content(cached, path, sha256, size)
                    with atomic_write(archive_path) as tmp_path:
                        shutil.copy2(cached, tmp_path)
            finally:
                if fetched and self.cache.disabled:
                    cached.unlink(missing_ok=True)

    def gc_content_store(self, dry_run: bool = True) -> list[str]:
        """Find and optionally delete content store keys not referenced by any pointer file."""
        referenced = {
            compute_content_key(self.catalog_id, sha)
            for sha in self._iter_pointer_sha256s()
        }
        orphans = [
            key
            for key in self.content_store.list_keys(prefix=f"{self.catalog_id}/")
            if key not in referenced
        ]

        if not dry_run:
            for key in orphans:
                self.content_store.delete(key)

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
