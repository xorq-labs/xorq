from __future__ import annotations

import abc
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from attr import (
    field,
    frozen,
)
from attr.validators import instance_of
from git import Repo

from xorq.catalog.annex import Annex, AnnexError


class CatalogBackend(abc.ABC):
    """ABC for the storage layer that Catalog delegates to."""

    @abc.abstractmethod
    def stage(self, path: Path | str) -> None: ...

    @abc.abstractmethod
    def stage_content(self, path: Path | str) -> None: ...

    @abc.abstractmethod
    def stage_unlink(self, path: Path | str) -> None: ...

    @abc.abstractmethod
    def commit_context(self, message: str) -> Iterator[Any]: ...

    @abc.abstractmethod
    def is_content_local(self, path: Path | str) -> bool: ...

    @abc.abstractmethod
    def fetch_content(self, *paths: Path | str) -> None: ...


@frozen
class GitBackend(CatalogBackend):
    """Plain-git backend — archives are stored as regular blobs."""

    repo = field(validator=instance_of(Repo))

    @property
    def repo_path(self) -> Path:
        return Path(self.repo.working_dir)

    def stage(self, path: Path | str) -> None:
        self.repo.index.add([str(path)])

    def stage_content(self, path: Path | str) -> None:
        self.stage(path)

    def stage_unlink(self, path: Path | str) -> None:
        self.repo.index.remove([str(path)])
        Path(path).unlink()

    @contextmanager
    def commit_context(self, message: str) -> Iterator[Any]:
        yield self.repo.index
        self.repo.index.commit(message)

    def is_content_local(self, path: Path | str) -> bool:
        return Path(path).exists()

    def fetch_content(self, *paths: Path | str) -> None:
        pass


@frozen
class GitAnnexBackend(CatalogBackend):
    repo = field(validator=instance_of(Repo))
    annex = field(validator=instance_of(Annex))

    def __attrs_post_init__(self) -> None:
        if Path(self.repo.working_dir).absolute() != self.annex.repo_path:
            raise ValueError(
                f"repo working_dir {self.repo.working_dir} does not match "
                f"annex repo_path {self.annex.repo_path}"
            )

    @property
    def repo_path(self) -> Path:
        return Path(self.repo.working_dir)

    def get_relpath(self, path: Path | str) -> Path:
        return Path(path).relative_to(self.repo_path)

    def stage(self, path: Path | str) -> None:
        self.repo.index.add([str(path)])

    def stage_content(self, path: Path | str) -> None:
        relpath = self.get_relpath(path)
        self.annex.add(relpath)
        self.repo.index.add([str(path)])

    def stage_unlink(self, path: Path | str) -> None:
        self.repo.index.remove([str(path)])
        Path(path).unlink()

    @contextmanager
    def commit_context(self, message: str) -> Iterator[Any]:
        yield self.repo.index
        self.repo.index.commit(message)

    def is_content_local(self, path: Path | str) -> bool:
        p = Path(path)
        return p.exists() and not (p.is_symlink() and not p.resolve().exists())

    def _has_any_remote(self) -> bool:
        """True if the repo has any git remote or annex special remote."""
        if self.annex.remote_name is not None:
            return True
        return bool(self.repo.remotes)

    def fetch_content(self, *paths: Path | str) -> None:
        if not self._has_any_remote():
            missing = [p for p in paths if not self.is_content_local(p)]
            if missing:
                raise AnnexError(
                    f"Content not local and no remote configured: {missing}"
                )
            return
        relpaths = [self.get_relpath(p) for p in paths]
        self.annex.get(*relpaths)
