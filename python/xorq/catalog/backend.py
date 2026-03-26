import abc
from contextlib import contextmanager
from pathlib import Path

from attr import (
    field,
    frozen,
)
from attr.validators import instance_of
from git import Repo

from xorq.catalog.annex import Annex, remote_config_from_dict


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

    @staticmethod
    def init_repo_path(repo_path):
        repo_path = Path(repo_path)
        if repo_path.exists():
            raise FileExistsError(f"repo already exists at {repo_path}")
        repo_path.mkdir(parents=True)
        repo = Repo.init(repo_path)
        repo.index.commit("initial commit")
        return repo


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

    def set_remote_config(self, catalog_yaml, remote_config):
        """Persist a remote config dict to catalog.yaml and commit."""
        catalog_yaml.set_remote(remote_config.to_dict())
        with self.commit_context(f"set remote: {remote_config.name}"):
            self.stage(catalog_yaml.yaml_path)

    def get_remote_config(self, catalog_yaml, **kwargs):
        """Load the remote config from catalog.yaml, if any.

        Secrets (e.g. aws_secret_access_key) must be passed as kwargs.
        """
        remote_dict = catalog_yaml.remote_config
        if remote_dict is None:
            return None
        return remote_config_from_dict(remote_dict, **kwargs)

    @staticmethod
    def init_repo_path(repo_path, remote_config=None):
        repo_path = Path(repo_path)
        if repo_path.exists():
            raise FileExistsError(f"repo already exists at {repo_path}")
        repo_path.mkdir(parents=True)
        repo = Repo.init(repo_path)
        repo.index.commit("initial commit")
        Annex.init_repo_path(repo_path, remote_config=remote_config)
        return repo
