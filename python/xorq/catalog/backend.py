import abc
import shutil
from contextlib import contextmanager
from pathlib import Path

from attr import (
    field,
    frozen,
)
from attr.validators import instance_of
from git import Repo

from xorq.catalog.annex import Annex, AnnexError
from xorq.catalog.constants import CONTENT_STORE_YAML, POINTER_SUFFIX
from xorq.catalog.content_store import (
    ContentCache,
    ContentIntegrityError,
    ContentStore,
    compute_sha256,
    content_key,
    parse_pointer,
    write_pointer,
)


class CatalogBackend(abc.ABC):
    """ABC for the storage layer that Catalog delegates to."""

    @abc.abstractmethod
    def stage(self, path): ...

    @abc.abstractmethod
    def stage_content(self, source_path, catalog_path): ...

    @abc.abstractmethod
    def stage_unlink(self, path): ...

    @abc.abstractmethod
    def commit_context(self, message): ...

    @abc.abstractmethod
    def is_content_local(self, path) -> bool: ...

    @abc.abstractmethod
    def fetch_content(self, *paths): ...

    def entry_tracked_path(self, catalog_path):
        """The path tracked in git for a given catalog entry (e.g. .pointer file)."""
        return Path(catalog_path)

    def repo_config_paths(self):
        """Repo-relative paths that assert_consistency should ignore."""
        return ()


@frozen
class GitBackend(CatalogBackend):
    """Plain-git backend — archives are stored as regular blobs."""

    repo = field(validator=instance_of(Repo))

    @property
    def repo_path(self):
        return Path(self.repo.working_dir)

    def stage(self, path):
        self.repo.index.add([str(path)])

    def stage_content(self, source_path, catalog_path):
        shutil.copy(source_path, catalog_path)
        self.stage(catalog_path)

    def stage_unlink(self, path):
        self.repo.index.remove([str(path)])
        Path(path).unlink()

    @contextmanager
    def commit_context(self, message):
        yield self.repo.index
        self.repo.index.commit(message)

    def is_content_local(self, path):
        return Path(path).exists()

    def fetch_content(self, *paths):
        pass


@frozen
class GitAnnexBackend(CatalogBackend):
    repo = field(validator=instance_of(Repo))
    annex = field(validator=instance_of(Annex))

    def __attrs_post_init__(self):
        if Path(self.repo.working_dir).absolute() != self.annex.repo_path:
            raise ValueError(
                f"repo working_dir {self.repo.working_dir} does not match "
                f"annex repo_path {self.annex.repo_path}"
            )

    @property
    def repo_path(self):
        return Path(self.repo.working_dir)

    def get_relpath(self, path):
        return Path(path).relative_to(self.repo_path)

    def stage(self, path):
        self.repo.index.add([str(path)])

    def stage_content(self, source_path, catalog_path):
        shutil.copy(source_path, catalog_path)
        relpath = self.get_relpath(catalog_path)
        self.annex.add(relpath)
        self.repo.index.add([str(catalog_path)])

    def stage_unlink(self, path):
        self.repo.index.remove([str(path)])
        Path(path).unlink()

    @contextmanager
    def commit_context(self, message):
        yield self.repo.index
        self.repo.index.commit(message)

    def is_content_local(self, path):
        p = Path(path)
        return p.exists() and not (p.is_symlink() and not p.resolve().exists())

    def _has_any_remote(self):
        """True if the repo has any git remote or annex special remote."""
        if self.annex.remote_name is not None:
            return True
        return bool(self.repo.remotes)

    def fetch_content(self, *paths):
        if not self._has_any_remote():
            missing = [p for p in paths if not self.is_content_local(p)]
            if missing:
                raise AnnexError(
                    f"Content not local and no remote configured: {missing}"
                )
            return
        relpaths = [self.get_relpath(p) for p in paths]
        self.annex.get(*relpaths)


@frozen
class GitPointerBackend(CatalogBackend):
    """Pointer-file backend — archives are stored in an external content store."""

    repo = field(validator=instance_of(Repo))
    content_store = field(validator=instance_of(ContentStore))
    cache = field(validator=instance_of(ContentCache))
    catalog_id = field(validator=instance_of(str))

    @property
    def repo_path(self):
        return Path(self.repo.working_dir)

    def _pointer_path(self, catalog_path):
        return Path(catalog_path).with_suffix(POINTER_SUFFIX)

    def stage(self, path):
        self.repo.index.add([str(path)])

    def stage_content(self, source_path, catalog_path):
        shutil.copy(source_path, catalog_path)
        archive_path = Path(catalog_path)
        sha256 = compute_sha256(archive_path)
        size = archive_path.stat().st_size
        key = content_key(self.catalog_id, sha256)

        self.content_store.put(key, archive_path)
        self.cache.put(key, archive_path)

        pointer_path = self._pointer_path(catalog_path)
        write_pointer(pointer_path, sha256, size)
        self.repo.index.add([str(pointer_path)])

    def stage_unlink(self, path):
        pointer_path = self._pointer_path(path)
        if pointer_path.exists():
            self.repo.index.remove([str(pointer_path)])
            pointer_path.unlink()
            archive_path = Path(path)
            if archive_path.exists():
                archive_path.unlink()
        else:
            self.repo.index.remove([str(path)])
            Path(path).unlink()

    @contextmanager
    def commit_context(self, message):
        yield self.repo.index
        self.repo.index.commit(message)

    def is_content_local(self, path) -> bool:
        if Path(path).exists():
            return True
        pointer_path = self._pointer_path(path)
        if not pointer_path.exists():
            return False
        sha256, _ = parse_pointer(pointer_path)
        key = content_key(self.catalog_id, sha256)
        return self.cache.contains(key)

    def fetch_content(self, *paths):
        for path in paths:
            archive_path = Path(path)
            if archive_path.exists():
                continue
            pointer_path = self._pointer_path(path)
            if not pointer_path.exists():
                raise FileNotFoundError(
                    f"Pointer file missing for {path}: {pointer_path}"
                )
            sha256, _size = parse_pointer(pointer_path)
            key = content_key(self.catalog_id, sha256)

            cached = self.cache.get_path(key)
            if cached is None:
                cached = self.cache.fetch_from(self.content_store, key)

            actual = compute_sha256(cached)
            if actual != sha256:
                # corrupt cache entry — drop so the next fetch re-pulls
                cached.unlink(missing_ok=True)
                raise ContentIntegrityError(
                    f"SHA256 mismatch for {path}: expected {sha256}, got {actual}"
                )

            # atomic copy: partial file at archive_path would fool the exists() guard
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = archive_path.with_name(archive_path.name + ".tmp")
            try:
                shutil.copy2(cached, tmp_path)
                tmp_path.replace(archive_path)
            except BaseException:
                tmp_path.unlink(missing_ok=True)
                raise

    def entry_tracked_path(self, catalog_path):
        return self._pointer_path(catalog_path)

    def repo_config_paths(self):
        return (".gitignore", CONTENT_STORE_YAML)

    @classmethod
    def from_config(cls, repo, config, cache=None):
        return cls(
            repo=repo,
            content_store=config.make_store(),
            cache=cache or ContentCache.default(),
            catalog_id=config.catalog_id,
        )
