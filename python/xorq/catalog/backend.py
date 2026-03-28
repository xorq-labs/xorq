import abc
from contextlib import contextmanager
from pathlib import Path

from attr import (
    field,
    frozen,
)
from attr.validators import instance_of
from git import Repo

from xorq.catalog.annex import Annex


class CatalogBackend(abc.ABC):
    """ABC for the storage layer that Catalog delegates to."""

    @abc.abstractmethod
    def stage(self, path): ...

    @abc.abstractmethod
    def stage_content(self, path): ...

    @abc.abstractmethod
    def stage_unlink(self, path): ...

    @abc.abstractmethod
    def commit_context(self, message): ...

    @abc.abstractmethod
    def is_content_local(self, path) -> bool: ...

    @abc.abstractmethod
    def fetch_content(self, *paths): ...


@frozen
class GitBackend(CatalogBackend):
    """Plain-git backend — archives are stored as regular blobs."""

    repo = field(validator=instance_of(Repo))

    @property
    def repo_path(self):
        return Path(self.repo.working_dir)

    def stage(self, path):
        self.repo.index.add([str(path)])

    def stage_content(self, path):
        self.stage(path)

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

    def stage_content(self, path):
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

    def is_content_local(self, path):
        p = Path(path)
        return p.exists() and not (p.is_symlink() and not p.resolve().exists())

    def fetch_content(self, *paths):
        relpaths = [self.get_relpath(p) for p in paths]
        self.annex.get(*relpaths)
