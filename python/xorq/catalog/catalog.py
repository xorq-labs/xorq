#!/usr/bin/env python
import hashlib
import shutil
import tempfile
from contextlib import (
    contextmanager,
    nullcontext,
)
from functools import partial
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
    instance_of,
    optional,
)
from git import (
    Blob,
    Remote,
    Repo,
)

from xorq.catalog.constants import (
    CATALOG_YAML_NAME,
    ENTRY_INFIX,
    METADATA_APPEND,
    METADATA_INFIX,
    PREFERRED_SUFFIX,
    VALID_SUFFIXES,
)
from xorq.catalog.expr_utils import (
    build_expr_context,
    build_expr_context_tgz,
    load_expr_from_tgz,
)
from xorq.catalog.git_utils import (
    add_as_submodule,
    commit_context,
)
from xorq.catalog.tar_utils import (
    make_tgz_context,
    test_tgz,
)


abspath = toolz.compose(Path.absolute, Path)
popen_shell = partial(Popen, shell=True)


def with_pure_suffix(path, suffix=""):
    return path.with_name(path.name.removesuffix("".join(path.suffixes))).with_suffix(
        suffix
    )


@frozen
class Catalog:
    repo = field(validator=instance_of(Repo))

    by_name_base_path = Path("~/.local/share/xorq/git-catalogs").expanduser()
    submodule_rel_path = Path(".xorq/git-catalogs")

    def __attrs_post_init__(self):
        self._ensure_catalog_yaml()
        self.assert_consistency()

    def _ensure_catalog_yaml(self):
        assert not self.repo.bare
        if not any(
            self.catalog_yaml.yaml_relpath.name == blob.name
            for blob in self.repo.head.commit.tree.list_traverse()
        ):
            with self.commit_context(f"add {CATALOG_YAML_NAME}") as index:
                index.add(self.catalog_yaml.yaml_path)

    @property
    def repo_path(self):
        return Path(self.repo.working_dir)

    @property
    def catalog_yaml(self):
        return CatalogYAML(self.repo_path)

    def _add_tgz(self, path, sync=True):
        # should we enable not syncing?
        with self.maybe_synchronizing(sync):
            catalog_addition = CatalogAddition(BuildTgz(path), self)
            catalog_entry = catalog_addition.add()
            self.assert_consistency()
            return catalog_entry

    def _add_build_dir(self, build_dir, sync=True):
        with make_tgz_context(build_dir) as tgz_path:
            return self._add_tgz(tgz_path, sync=sync)

    def _add_expr(self, expr, sync=True):
        with build_expr_context(expr) as path:
            return self._add_build_dir(path, sync=sync)

    def add(self, obj, sync=True):
        from xorq.api import Expr

        match obj:
            case Path() if obj.is_dir():
                f = self._add_build_dir
            case Path() if obj.is_file():
                f = self._add_tgz
            case Expr():
                f = self._add_expr
            case _:
                raise ValueError(f"don't know how to handle type={type(obj)}")
        return f(obj, sync=sync)

    def remove(self, name, sync=True):
        with self.maybe_synchronizing(sync):
            catalog_removal = CatalogRemoval.from_name_catalog(name, self)
            catalog_entry = catalog_removal.remove()
            self.assert_consistency()
            return catalog_entry

    def list(self):
        return self.catalog_yaml.contents

    def fetch(self):
        return tuple(map(Remote.fetch, self.repo.remotes))

    def push(self):
        self.assert_consistency()
        return tuple(map(Remote.push, self.repo.remotes))

    def pull(self):
        return tuple(map(Remote.pull, self.repo.remotes))

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
        with self.synchronizing():
            pass

    def contains(self, name):
        catalog_entry = CatalogEntry(name, self, require_exists=False)
        return catalog_entry.exists()

    def get_catalog_entry(self, name):
        assert name in self.list()
        catalog_entry = CatalogEntry(name, self)
        return catalog_entry

    def get_tgz(self, name, dir_path=None):
        catalog_entry = self.get_catalog_entry(name)
        return catalog_entry.get(dir_path)

    @property
    def catalog_entries(self):
        return tuple(CatalogEntry(name, self) for name in self.list())

    @contextmanager
    def commit_context(self, message):
        with commit_context(self.repo, message) as index:
            yield index

    def assert_consistency(self):
        # catalog_yaml is in repo
        catalog_yaml_relpath_string = str(self.catalog_yaml.yaml_relpath)
        path_strings = tuple(
            blob.path
            for blob in self.repo.head.commit.tree.list_traverse()
            if isinstance(blob, Blob)
        )
        assert catalog_yaml_relpath_string in path_strings

        # everything else in repo is either catalog_path or metadata_path from an entry the catalog_yaml knows about
        actual = sorted(el for el in path_strings if el != catalog_yaml_relpath_string)
        expected = sorted(
            str(path.relative_to(self.repo_path))
            for catalog_entry in self.catalog_entries
            for path in (catalog_entry.metadata_path, catalog_entry.catalog_path)
        )
        assert actual == expected

    def add_as_submodule(self, root_repo):
        message = f"add submodule: {self.repo_path.name}"
        with commit_context(root_repo, message):
            add_as_submodule(root_repo, self.repo)

    @classmethod
    def clone_from(cls, url, repo_path=None):
        if repo_path is None:
            name = Path(urlparse(url).path).stem
            repo_path = cls.name_to_repo_path(name)
        repo = Repo.clone_from(url, repo_path)
        return cls(repo=repo)

    @classmethod
    def from_repo_path(cls, repo_path, init=None):
        init = not Path(repo_path).exists() if init is None else init
        repo = cls.init_repo_path(repo_path) if init else Repo(repo_path)
        return cls(repo=repo)

    @classmethod
    def from_name(cls, name, init=None):
        repo_path = cls.name_to_repo_path(name)
        return cls.from_repo_path(repo_path, init=init)

    @classmethod
    def from_default(cls, init=None):
        return cls.from_name(name="default", init=init)

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
    def init_repo_path(repo_path, bare=False):
        assert not (repo_path := Path(repo_path)).exists()
        repo = Repo.init(repo_path, mkdir=True, bare=bare)
        repo.index.commit("initial commit")
        return repo


@frozen
class BuildTgz:
    path = field(validator=instance_of(Path), converter=Path)

    def __attrs_post_init__(self):
        assert "".join(self.path.suffixes) in VALID_SUFFIXES
        assert self.path.exists()
        test_tgz(self.path)

    @property
    def name(self):
        return with_pure_suffix(self.path, "").name

    @property
    def md5sum(self):
        from xorq.common.utils.dask_normalize.dask_normalize_utils import file_digest

        return file_digest(self.path, hashlib.md5)


@frozen
class CatalogAddition:
    build_tgz = field(validator=instance_of(BuildTgz))
    catalog = field(validator=instance_of(Catalog))
    _maybe_tmpfile = field(
        validator=optional(instance_of(tempfile._TemporaryFileWrapper)),
        default=None,
    )

    @property
    def name(self):
        return self.build_tgz.name

    @property
    def metadata(self):
        return {"md5sum": self.build_tgz.md5sum}

    @property
    def catalog_entry(self):
        return CatalogEntry(self.name, self.catalog, require_exists=False)

    def ensure_dirs(self):
        for p in (self.catalog_entry.metadata_path, self.catalog_entry.catalog_path):
            p.parent.mkdir(exist_ok=True, parents=True)

    @property
    def message(self):
        message = f"add: {self.name}"
        return message

    def add(self):
        assert not self.catalog.contains(self.name)
        self.ensure_dirs()
        catalog_entry = self.catalog_entry
        catalog_entry.metadata_path.write_text(yaml.safe_dump(self.metadata))
        shutil.copy(self.build_tgz.path, catalog_entry.catalog_path)
        with self.catalog.commit_context(self.message) as index:
            catalog_entry.catalog_yaml.add(self.name)
            index.add(
                (
                    catalog_entry.catalog_path,
                    catalog_entry.metadata_path,
                    catalog_entry.catalog_yaml.yaml_path,
                )
            )
        return CatalogEntry(self.name, self.catalog, require_exists=True)

    @classmethod
    def from_expr(cls, expr, catalog):
        ntfh = tempfile.NamedTemporaryFile(suffix=PREFERRED_SUFFIX)
        with build_expr_context_tgz(expr) as tgz_path:
            shutil.copy(tgz_path, ntfh.name)
        return cls(BuildTgz(ntfh.name), catalog, maybe_tmpfile=ntfh)


@frozen
class CatalogEntry:
    name = field(validator=instance_of(str))
    catalog = field(validator=instance_of(Catalog))
    require_exists = field(validator=instance_of(bool), default=True)

    def __attrs_post_init__(self):
        self.assert_consistency()
        if self.require_exists:
            assert self.exists()

    @property
    def repo_path(self):
        return self.catalog.repo_path

    @property
    def catalog_yaml(self):
        return CatalogYAML(self.repo_path)

    @property
    def metadata_path(self):
        metadata_path = self.repo_path.joinpath(
            METADATA_INFIX, self.name + PREFERRED_SUFFIX + METADATA_APPEND
        )
        return metadata_path

    @property
    def catalog_path(self):
        catalog_path = self.repo_path.joinpath(ENTRY_INFIX, self.name).with_suffix(
            PREFERRED_SUFFIX
        )
        return catalog_path

    @property
    def expr(self):
        return load_expr_from_tgz(self.catalog_path)

    @property
    def _exists_components(self):
        return {
            "metadata_path": self.metadata_path.exists(),
            "catalog_path": self.catalog_path.exists(),
            "catalog_yaml_contents": self.name in self.catalog_yaml.contents,
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


@frozen
class CatalogRemoval:
    catalog_entry = field(validator=instance_of(CatalogEntry))

    @property
    def message(self):
        message = f"rm: {with_pure_suffix(self.catalog_entry.catalog_path, '').name}"
        return message

    def remove(self):
        catalog_entry = self.catalog_entry
        catalog = catalog_entry.catalog
        assert catalog_entry.exists()
        with catalog.commit_context(self.message) as index:
            catalog.catalog_yaml.remove(catalog_entry.name)
            index.add((catalog.catalog_yaml.yaml_path,))
            paths = (
                catalog_entry.metadata_path,
                catalog_entry.catalog_path,
            )
            index.remove(paths)
            for path in paths:
                path.unlink()
        return catalog_entry

    @classmethod
    def from_name_catalog(cls, name, catalog):
        return cls(CatalogEntry(name=name, catalog=catalog))


@frozen
class CatalogYAML:
    repo_path = field(validator=instance_of(Path), converter=abspath)

    def __attrs_post_init__(self):
        if not self.yaml_path.exists():
            self.yaml_path.write_text(yaml.safe_dump([]))

    @property
    def yaml_path(self):
        return self.repo_path.joinpath(CATALOG_YAML_NAME)

    @property
    def yaml_relpath(self):
        return self.yaml_path.relative_to(self.repo_path)

    @property
    def contents(self):
        return yaml.safe_load(self.yaml_path.read_text())

    def set_contents(self, contents):
        self.yaml_path.write_text(yaml.safe_dump(contents))
        return self.yaml_path

    def contains(self, entry):
        return entry in self.contents

    def add(self, entry):
        assert not self.contains(entry)
        contents = self.contents + [entry]
        return self.set_contents(contents)

    def remove(self, entry):
        contents = [el for el in self.contents if el != entry]
        return self.set_contents(contents)
