#!/usr/bin/env python
import hashlib
import shutil
import tarfile
import tempfile
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

from xorq.catalog.constants import (
    CATALOG_YAML_NAME,
    METADATA_APPEND,
    PREFERRED_SUFFIX,
    VALID_SUFFIXES,
    CatalogInfix,
)
from xorq.catalog.git_utils import (
    add_as_submodule,
    commit_context,
)
from xorq.ibis_yaml.compiler import DumpFiles, ExprKind


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

    def _add_tgz(self, path, sync=True, aliases=()):
        # should we enable not syncing?
        with self.maybe_synchronizing(sync):
            catalog_addition = CatalogAddition(BuildTgz(path), self, aliases=aliases)
            catalog_entry = catalog_addition.add()
            self.assert_consistency()
            return catalog_entry

    def _add_build_dir(self, build_dir, sync=True, aliases=()):
        from xorq.catalog.tar_utils import make_tgz_context  # noqa: PLC0415

        with make_tgz_context(build_dir) as tgz_path:
            return self._add_tgz(tgz_path, sync=sync, aliases=aliases)

    def _add_expr(self, expr, sync=True, aliases=()):
        from xorq.catalog.expr_utils import build_expr_context  # noqa: PLC0415

        with build_expr_context(expr) as path:
            return self._add_build_dir(path, sync=sync, aliases=aliases)

    def add(self, obj, sync=True, aliases=()):
        from xorq.api import Expr  # noqa: PLC0415

        match obj:
            case Path() if obj.is_dir():
                f = self._add_build_dir
            case Path() if obj.is_file():
                f = self._add_tgz
            case Expr():
                f = self._add_expr
            case _:
                raise ValueError(f"don't know how to handle type={type(obj)}")
        return f(obj, sync=sync, aliases=aliases)

    def remove(self, name, sync=True):
        with self.maybe_synchronizing(sync):
            catalog_removal = CatalogRemoval.from_name_catalog(name, self)
            catalog_entry = catalog_removal.remove()
            self.assert_consistency()
            return catalog_entry

    def list(self):
        return self.catalog_yaml.contents[CatalogInfix.ENTRY]

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
        assert name in self.list(), f"Entry '{name}' not found in catalog"
        catalog_entry = CatalogEntry(name, self)
        return catalog_entry

    def get_entry(self, name_or_alias):
        """Resolve an entry name or alias to a CatalogEntry."""
        if name_or_alias in self.list():
            return CatalogEntry(name_or_alias, self)
        if name_or_alias in self.list_aliases():
            return CatalogAlias.from_name(name_or_alias, self).catalog_entry
        raise KeyError(f"No entry or alias named {name_or_alias!r}")

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

    def add_alias(self, name, alias, sync=True):
        with self.maybe_synchronizing(sync):
            catalog_entry = CatalogEntry(name, self)
            catalog_alias = CatalogAlias(alias=alias, catalog_entry=catalog_entry)
            catalog_alias.add()
            return catalog_alias

    def list_aliases(self):
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
                    .resolve()
                    .with_suffix("")
                    .name,
                    self,
                ),
            )
            for alias in self.list_aliases()
        )

    def assert_consistency(self):
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
        assert not (repo_path := Path(repo_path)).exists(), (
            f"Catalog repo already exists at {repo_path}"
        )
        repo = Repo.init(repo_path, mkdir=True, bare=bare)
        repo.index.commit("initial commit")
        return repo


@frozen
class BuildTgz:
    path = field(validator=instance_of(Path), converter=Path)

    def __attrs_post_init__(self):
        from xorq.catalog.tar_utils import test_tgz  # noqa: PLC0415

        assert "".join(self.path.suffixes) in VALID_SUFFIXES, (
            f"Invalid archive suffix '{self.path.suffixes}', expected one of {VALID_SUFFIXES}"
        )
        assert self.path.exists(), f"Build archive not found at {self.path}"
        test_tgz(self.path)

    @property
    def name(self):
        return with_pure_suffix(self.path, "").name

    @property
    def md5sum(self):
        from xorq.common.utils.dask_normalize.dask_normalize_utils import (  # noqa: PLC0415
            file_digest,
        )

        return file_digest(self.path, hashlib.md5)


@frozen
class CatalogAddition:
    build_tgz = field(validator=instance_of(BuildTgz))
    catalog = field(validator=instance_of(Catalog))
    aliases = field(validator=deep_iterable(instance_of(str)), default=())
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
        shutil.copy(self.build_tgz.path, catalog_entry.catalog_path)
        index = self.catalog.repo.index
        #
        self.catalog.catalog_yaml.add(self.name)
        index.add(
            (
                catalog_entry.catalog_path,
                catalog_entry.metadata_path,
                self.catalog.catalog_yaml.yaml_path,
            )
        )
        for catalog_alias in self.catalog_aliases:
            catalog_alias._add()
        return CatalogEntry(self.name, self.catalog, require_exists=True)

    def add(self):
        with self.catalog.commit_context(self.message):
            return self._add()

    @classmethod
    def from_expr(cls, expr, catalog):
        from xorq.catalog.expr_utils import build_expr_context_tgz  # noqa: PLC0415

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
        from xorq.catalog.expr_utils import load_expr_from_tgz  # noqa: PLC0415

        return load_expr_from_tgz(self.catalog_path)

    @cached_property
    def kind(self) -> str:
        default_value = str(ExprKind.Expr)
        data = self._read_tgz_yaml(DumpFiles.expr)
        if not isinstance(data, dict):
            return default_value
        return data.get("kind", default_value)

    @cached_property
    def schema_out(self) -> dict | None:
        """Output schema as {column_name: dtype_str}, read directly from expr.yaml."""
        data = self._read_tgz_yaml(DumpFiles.expr)
        return CatalogEntry._parse_schema_out(data)

    @cached_property
    def schema_in(self) -> dict | None:
        """For unbound expressions, the expected input schema as {column_name: dtype_str}."""
        if self.kind != str(ExprKind.UnboundExpr):
            return None
        data = self._read_tgz_yaml(DumpFiles.expr)
        return CatalogEntry._parse_schema_in(data)

    @staticmethod
    def _resolve_schema_ref(ref_val) -> str | None:
        """Extract the schema key string from a schema_ref value.

        In expr.yaml, schema_ref is stored as {'schema_ref': 'schema_<hash>'}.
        """
        if isinstance(ref_val, dict):
            return ref_val.get("schema_ref")
        if isinstance(ref_val, str):
            return ref_val
        return None

    @staticmethod
    def _dtype_dict_to_str(d) -> str:
        """Convert a serialized DataType dict from expr.yaml to a human-readable string."""
        import xorq.vendor.ibis.expr.datatypes as dt  # noqa: PLC0415

        if not isinstance(d, dict) or d.get("op") != "DataType":
            return str(d)
        typ_cls = getattr(dt, d["type"], None)
        if typ_cls is None:
            return d["type"].lower()
        kwargs = {
            k: CatalogEntry._dtype_dict_to_str(v) if isinstance(v, dict) else v
            for k, v in d.items()
            if k not in ("op", "type")
        }

        # Re-parse nested dtype strings back to dt objects for the constructor
        def _parse_val(k, v):
            try:
                arg_annotation = typ_cls.__dataclass_fields__.get(k)
                if arg_annotation and isinstance(v, str):
                    return dt.dtype(v)
            except Exception:
                pass
            return v

        kwargs = {k: _parse_val(k, v) for k, v in kwargs.items()}
        try:
            return str(typ_cls(**kwargs))
        except Exception:
            return d["type"].lower()

    @staticmethod
    def _schema_dict_to_str_dict(schema_dict) -> dict[str, str]:
        return {
            col: CatalogEntry._dtype_dict_to_str(dtype_val)
            for col, dtype_val in schema_dict.items()
        }

    @staticmethod
    def _parse_schema_out(data) -> dict | None:
        if not isinstance(data, dict):
            return None
        schemas = data.get("definitions", {}).get("schemas", {})
        schema_ref = CatalogEntry._resolve_schema_ref(
            data.get("expression", {}).get("schema_ref")
        )
        if not schema_ref:
            return None
        schema_dict = schemas.get(schema_ref)
        if not isinstance(schema_dict, dict):
            return None
        return CatalogEntry._schema_dict_to_str_dict(schema_dict)

    @staticmethod
    def _parse_schema_in(data) -> dict | None:
        if not isinstance(data, dict):
            return None
        definitions = data.get("definitions", {})
        nodes = definitions.get("nodes", {})
        schemas = definitions.get("schemas", {})
        for node_def in nodes.values():
            if isinstance(node_def, dict) and node_def.get("op") == "UnboundTable":
                schema_ref = CatalogEntry._resolve_schema_ref(
                    node_def.get("schema_ref")
                )
                if schema_ref:
                    schema_dict = schemas.get(schema_ref)
                    if isinstance(schema_dict, dict):
                        return CatalogEntry._schema_dict_to_str_dict(schema_dict)
        return None

    @cached_property
    def backends(self) -> tuple:
        data = self._read_tgz_yaml(DumpFiles.profiles)
        if not isinstance(data, dict):
            return ()
        return tuple(
            pdata.get("con_name", "?")
            for pdata in data.values()
            if isinstance(pdata, dict)
        )

    @property
    def aliases(self):
        return tuple(
            catalog_alias
            for catalog_alias in self.catalog.catalog_aliases
            if catalog_alias.catalog_entry.name == self.name
        )

    @property
    def _exists_components(self):
        return {
            "metadata_path": self.metadata_path.exists(),
            "catalog_path": self.catalog_path.exists(),
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

    def _read_tgz_yaml(self, filename):
        with tarfile.open(self.catalog_path, "r:gz") as tf:
            f = tf.extractfile(f"{self.name}/{filename}")
            if f is None:
                return None
            return yaml.safe_load(f.read())


@frozen
class CatalogAlias:
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
        catalog_yaml = self.catalog_entry.catalog.catalog_yaml
        #
        catalog_yaml.add_alias(self.alias)
        self.catalog_entry.catalog.repo.index.add([alias_path, catalog_yaml.yaml_path])
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
        catalog_yaml = self.catalog_entry.catalog.catalog_yaml
        index = self.catalog_entry.catalog.repo.index
        #
        catalog_yaml.remove_alias(self.alias)
        index.add([catalog_yaml.yaml_path])
        index.remove([alias_path])
        alias_path.unlink()
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
                CatalogEntry(alias_path.resolve().with_suffix("").name, catalog),
            )
            return catalog_alias
        else:
            raise ValueError(f"no such alias {name}")


@frozen
class CatalogRemoval:
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
        index = catalog.repo.index
        #
        for catalog_alias in self.catalog_entry.aliases:
            catalog_alias._remove()
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
