import contextlib
import functools
import json
import operator
import pathlib
import sys
from pathlib import Path
from typing import Any, Dict

import dask
import toolz
import yaml
from attr import (
    evolve,
    field,
    frozen,
)
from attr.validators import (
    instance_of,
    optional,
    or_,
)

import xorq
import xorq.common.utils.logging_utils as lu
import xorq.ibis_yaml.translate  #  noqa: F401
import xorq.vendor.ibis as ibis
import xorq.vendor.ibis.expr.types as ir
from xorq.caching import (
    ParquetCache,
    ParquetSnapshotCache,
    SnapshotStrategy,
)
from xorq.common.utils.caching_utils import get_xorq_cache_dir
from xorq.common.utils.dask_normalize.dask_normalize_utils import (
    normalize_read_path_md5sum,
)
from xorq.common.utils.graph_utils import (
    find_all_sources,
    opaque_ops,
    replace_nodes,
    walk_nodes,
)
from xorq.config import _backend_init
from xorq.expr.api import deferred_read_parquet, read_parquet
from xorq.expr.relations import (
    CachedNode,
    Read,
)
from xorq.ibis_yaml.common import (
    RefEnum,
    Registry,
    RegistryEnum,
    TranslationContext,
    translate_from_yaml,
    translate_to_yaml,
)
from xorq.ibis_yaml.config import config
from xorq.ibis_yaml.sql import generate_sql_plans
from xorq.ibis_yaml.utils import freeze
from xorq.vendor.ibis.backends import Profile
from xorq.vendor.ibis.common.collections import FrozenOrderedDict
from xorq.vendor.ibis.expr.operations import DatabaseTable, InMemoryTable


try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum


class DumpFiles(StrEnum):
    deferred_reads = "deferred_reads.yaml"
    expr = "expr.yaml"
    metadata = "metadata.json"
    profiles = "profiles.yaml"
    sql = "sql.yaml"


class MemtableTypes(StrEnum):
    inmemory = "memtables"
    database_table = "database_tables"


memory_backends = ("pandas", "duckdb", "datafusion", "xorq")
table_like_ops = tuple(o for o in opaque_ops if issubclass(o, DatabaseTable))


class CleanDictYAMLDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

    def represent_enum(self, data):
        return self.represent_scalar("tag:yaml.org,2002:str", data.name)

    def represent_frozenordereddict(self, data):
        return self.represent_dict(dict(data))

    def represent_ibis_schema(self, data):
        schema_dict = {name: str(dtype) for name, dtype in zip(data.names, data.types)}
        return self.represent_mapping("tag:yaml.org,2002:map", schema_dict)

    def represent_posix_path(self, data):
        return self.represent_scalar("tag:yaml.org,2002:str", str(data))

    yaml_representer_pairs = (
        (RefEnum, represent_enum),
        (RegistryEnum, represent_enum),
        (FrozenOrderedDict, represent_frozenordereddict),
        (ibis.Schema, represent_ibis_schema),
        (pathlib.PosixPath, represent_posix_path),
    )

    @classmethod
    def add_representers(cls):
        for to_register, representer in cls.yaml_representer_pairs:
            cls.add_representer(to_register, representer)


CleanDictYAMLDumper.add_representers()


@frozen
class ArtifactStore:
    root_path = field(validator=instance_of(Path), converter=Path)

    def __attrs_post_init__(self):
        self.root_path.mkdir(parents=True, exist_ok=True)

    def get_path(self, *parts) -> pathlib.Path:
        return self.root_path.joinpath(*parts)

    def _read(self, read_f, *parts):
        path = self.get_path(*parts)
        with path.open("r") as f:
            return read_f(f)

    def read_yaml(self, *path_parts) -> Dict[str, Any]:
        return self._read(yaml.safe_load, *path_parts)

    def read_json(self, *path_parts) -> Dict[str, Any]:
        return self._read(json.load, *path_parts)

    def read_text(self, *path_parts) -> str:
        return self._read(operator.methodcaller("read"), *path_parts)

    @contextlib.contextmanager
    def _write(self, *path_parts):
        path = self.get_path(*path_parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yield (path, f)

    def write_yaml(self, data: Dict[str, Any], *path_parts) -> pathlib.Path:
        with self._write(*path_parts) as (path, f):
            yaml.dump(
                data,
                f,
                Dumper=CleanDictYAMLDumper,
                default_flow_style=False,
                sort_keys=False,
            )
        return path

    def write_text(self, content: str, *path_parts) -> pathlib.Path:
        with self._write(*path_parts) as (path, f):
            f.write(content)
        return path

    def write_parquet(self, table, *path_parts) -> pathlib.Path:
        import pyarrow.parquet as pq

        with self._write(*path_parts) as (path, f):
            pq.write_table(table, path)
        return path

    def exists(self, *path_parts) -> bool:
        return self.get_path(*path_parts).exists()

    def save_yaml(self, yaml_dict: Dict[str, Any], filename) -> pathlib.Path:
        return self.write_yaml(yaml_dict, filename)

    def load_yaml(self, filename) -> Dict[str, Any]:
        return self.read_yaml(filename)

    @staticmethod
    def get_expr_hash(expr) -> str:
        with SnapshotStrategy().normalization_context(expr):
            expr_hash = dask.base.tokenize(expr)[: config.hash_length]
            return expr_hash

    @classmethod
    def from_path_and_expr(cls, builds_dir, expr):
        return cls(root_path=builds_dir.joinpath(cls.get_expr_hash(expr)))


class YamlExpressionTranslator:
    @staticmethod
    def to_yaml(expr: ir.Expr, profiles=(), cache_dir=None) -> Dict[str, Any]:
        context = TranslationContext(
            profiles=freeze(dict(profiles)),
            cache_dir=cache_dir,
        )
        with SnapshotStrategy().normalization_context(expr):
            expr_dict = translate_to_yaml(expr, context)
            expr_dict = freeze(
                expr_dict
                | {
                    RefEnum.schema_ref: context.registry.register_schema(expr.schema())
                    if hasattr(expr, "schema")
                    else None,
                }
            )
            return freeze(
                {
                    "definitions": context.definitions,
                    "expression": expr_dict,
                }
            )

    @staticmethod
    def from_yaml(
        yaml_dict: Dict[str, Any],
        profiles=(),
    ) -> ir.Expr:
        context = TranslationContext(
            registry=Registry(**yaml_dict.get("definitions", {})),
            profiles=freeze(dict(profiles)),
        )
        expr_dict = freeze(yaml_dict["expression"])
        return translate_from_yaml(expr_dict, context)


def dehydrate_cons(cons):
    dehydrated = dict(
        sorted(
            (
                profile.hash_name,
                profile.as_dict()
                | {
                    "kwargs_tuple": dict(profile.as_dict()["kwargs_tuple"]),
                },
            )
            for profile in (con._profile for con in cons)
        )
    )
    return dehydrated


def hydrate_cons(hash_to_profile_kwargs):
    def kwargs_to_con(kwargs):
        match dct := dict(kwargs):
            case {"kwargs_tuple": dict()}:
                dct["kwargs_tuple"] = tuple(dct["kwargs_tuple"].items())
            case _:
                dct["kwargs_tuple"] = tuple(map(tuple, dct["kwargs_tuple"]))
        con = Profile(**dct).get_con()
        return con

    profiles = toolz.valmap(
        kwargs_to_con,
        hash_to_profile_kwargs,
    )
    return profiles


def make_read_op(parquet_path, read_kwargs, con=_backend_init()):
    op = deferred_read_parquet(parquet_path, con, **read_kwargs).op()
    args = dict(zip(op.__argnames__, op.__args__))
    op = op.__recreate__(args)
    return op


@frozen
class ExprDumper:
    """
    expr: the expr to be built
    builds_dir: root directory where expr builds are stored
    cache_dir: optional directory for parquet cache files
    debug: when True, output SQL files and debug artifacts (sql.yaml, deferred_reads.yaml)
    """

    expr = field(validator=instance_of(ir.Expr))
    builds_dir = field(validator=instance_of(Path), converter=Path, default="./builds")
    cache_dir = field(validator=optional(instance_of(Path)), factory=get_xorq_cache_dir)
    debug = field(validator=instance_of(bool), default=False)

    def __attrs_post_init__(self):
        attrname = "cache_dir"
        match value := getattr(self, attrname):
            case None:
                object.__setattr__(self, attrname, get_xorq_cache_dir())
            case Path():
                pass
            case _:
                object.__setattr__(self, attrname, Path(value))

    @property
    @functools.cache
    def artifact_store(self):
        return ArtifactStore.from_path_and_expr(self.builds_dir, self.expr)

    @property
    @functools.cache
    def expr_path(self):
        return self.artifact_store.root_path

    @property
    def expr_hash(self):
        return self.expr_path.name

    def _prepare_expr_file(self, expr, profiles):
        path = self.artifact_store.get_path(DumpFiles.expr)
        # we can't translate to yaml until the memtable parquets are written: they will be tokenized
        writer = toolz.compose(
            functools.partial(self.artifact_store.save_yaml, filename=DumpFiles.expr),
            functools.partial(
                YamlExpressionTranslator.to_yaml, expr, profiles, self.cache_dir
            ),
        )
        return (path, writer)

    def _prepare_sql_file(self, sql: str) -> str:
        sql_hash = dask.base.tokenize(sql)[: config.hash_length]
        filename = f"{sql_hash}.sql"
        path = self.artifact_store.get_path(filename)
        writer = functools.partial(self.artifact_store.write_text, sql, filename)
        return (path, writer)

    def _prepare_memtable(self, mt, which):
        assert which in MemtableTypes
        table = mt.to_expr().to_pyarrow()
        filename = f"{dask.base.tokenize(table)}.parquet"
        path_parts = (which, filename)
        path = self.artifact_store.get_path(*path_parts)
        writer = functools.partial(
            self.artifact_store.write_parquet,
            table,
            *path_parts,
        )
        return (path, writer)

    def _prepare_sql_plans(
        self,
        sql_plans: Dict[str, Any],
    ) -> Dict[str, Any]:
        queries = {}
        path_to_writer = {}
        for query_name, query_info in sql_plans["queries"].items():
            path, writer = self._prepare_sql_file(query_info["sql"])
            path_to_writer[path] = writer
            queries[query_name] = toolz.dissoc(query_info, "sql") | {
                "sql_file": path.name
            }
        updated_plans = {"queries": queries}
        path_to_writer[self.artifact_store.get_path(DumpFiles.sql)] = functools.partial(
            self.artifact_store.save_yaml,
            updated_plans,
            filename=DumpFiles.sql,
        )
        return path_to_writer

    def _prepare_deferred_reads(
        self,
        deferred_reads: Dict[str, Any],
    ) -> Dict[str, Any]:
        reads = {}
        path_to_writer = {}
        for read_name, read_info in deferred_reads["reads"].items():
            path, writer = self._prepare_sql_file(read_info["sql"])
            path_to_writer[path] = writer
            reads[read_name] = toolz.dissoc(read_info, "sql") | {"sql_file": path.name}
        updated_reads = {"reads": reads}
        path_to_writer[self.artifact_store.get_path(DumpFiles.deferred_reads)] = (
            functools.partial(
                self.artifact_store.save_yaml,
                updated_reads,
                filename=DumpFiles.deferred_reads,
            )
        )
        return path_to_writer

    @staticmethod
    def _make_metadata() -> str:
        metadata = {
            "current_library_version": xorq.__version__,
            "metadata_version": "0.0.0",  # TODO: make it a real thing
            "git_state": lu.get_git_state(hash_diffs=False)
            if lu._git_is_present()
            else None,
            "sys-version_info": tuple(sys.version_info),
        }
        metadata_json = json.dumps(metadata, indent=2)
        return metadata_json

    def _prepare_metadata_file(self):
        path = self.artifact_store.get_path(DumpFiles.metadata)
        writer = functools.partial(
            self.artifact_store.write_text,
            self._make_metadata(),
            DumpFiles.metadata,
        )
        return path, writer

    def _prepare_profiles_file(self, profiles):
        path = self.artifact_store.get_path(DumpFiles.profiles)
        writer = functools.partial(
            self.artifact_store.save_yaml,
            profiles,
            DumpFiles.profiles,
        )
        return path, writer

    def _prepare_debug_info(self):
        sql_plans, deferred_reads = generate_sql_plans(self.expr)
        path_to_writer0 = self._prepare_sql_plans(sql_plans)
        path_to_writer1 = self._prepare_deferred_reads(deferred_reads)
        path_to_writer = path_to_writer0 | path_to_writer1
        return path_to_writer

    def _memtables_to_deferred_reads(self, expr):
        path_to_writer = {}
        op = expr.op()
        mts = walk_nodes((InMemoryTable,), expr)
        for mt in mts:
            path, writer = self._prepare_memtable(mt, MemtableTypes.inmemory)
            dr_op = make_read_op(
                parquet_path=path,
                read_kwargs={
                    "table_name": mt.name,
                    "schema": mt.schema,
                    str(MemtableTypes.inmemory): True,
                },
            )
            path_to_writer[path] = writer
            op = replace_nodes(replace_from_to(mt, dr_op), expr)
        new_expr = op.to_expr()
        return new_expr, path_to_writer

    def _replace_inmemory_backend_tables(self, expr):
        op = expr.op()
        tables = (
            table
            for table in walk_nodes((DatabaseTable,), expr)
            if not isinstance(table, table_like_ops)
            and table.source.name in memory_backends
        )
        path_to_writer = {}
        for table in tables:
            path, writer = self._prepare_memtable(table, MemtableTypes.database_table)
            dr_op = make_read_op(
                parquet_path=path,
                read_kwargs={
                    "table_name": table.name,
                    # we normalize based on content so we can reproducible hash
                    "normalize_method": normalize_read_path_md5sum,
                    "schema": table.schema,
                },
                con=table.source,
            )
            path_to_writer[path] = writer
            op = replace_nodes(replace_from_to(table, dr_op), expr)
        new_expr = op.to_expr()
        return new_expr, path_to_writer

    def dump_expr(self) -> str:
        # we will mutate the expr below
        expr = self.expr

        # write in-memory data to build dir
        expr, path_to_writer0 = self._memtables_to_deferred_reads(expr)
        expr, path_to_writer1 = self._replace_inmemory_backend_tables(expr)

        profiles = dehydrate_cons(find_all_sources(expr))
        path_to_writer2 = {
            path: writer
            for (path, writer) in (
                self._prepare_metadata_file(),
                self._prepare_profiles_file(profiles),
            )
        }
        path_to_writer = path_to_writer0 | path_to_writer1 | path_to_writer2
        if self.debug:
            # write SQL plan and deferred-read artifacts if debug enabled
            path_to_writer |= self._prepare_debug_info()
        for writer in path_to_writer.values():
            writer()

        # expr must be written last
        _, writer = self._prepare_expr_file(expr, profiles)
        writer()
        return self.expr_path


@frozen
class ExprLoader:
    expr_path = field(validator=instance_of(Path), converter=Path)
    cache_dir = field(
        validator=optional(or_(instance_of(Path), instance_of(str))), default=None
    )

    @property
    def expr_hash(self):
        return self.expr_path.name

    @property
    @functools.cache
    def artifact_store(self):
        return ArtifactStore(self.expr_path)

    def load_expr(self):
        profiles = hydrate_cons(self.artifact_store.load_yaml(DumpFiles.profiles))
        yaml_dict = self.artifact_store.load_yaml(DumpFiles.expr)
        expr = YamlExpressionTranslator.from_yaml(yaml_dict, profiles=profiles)
        expr = self.deferred_reads_to_memtables(expr)
        if self.cache_dir:
            expr = self.replace_base_path(expr, base_path=Path(self.cache_dir))
        return expr

    @staticmethod
    def deferred_reads_to_memtables(loaded):
        def deferred_read_to_memtable(dr):
            assert any(key == MemtableTypes.inmemory for key, _ in dr.read_kwargs)
            path = next(v for k, v in dr.read_kwargs if k == "path")
            df = read_parquet(path).execute()
            mt = ibis.memtable(df, schema=dr.schema, name=dr.name)
            return mt

        drs = tuple(
            dr
            for dr in walk_nodes(Read, loaded)
            if MemtableTypes.inmemory in dict(dr.read_kwargs)
        )
        op = loaded.op()
        for dr in drs:
            mt = deferred_read_to_memtable(dr)
            op = replace_nodes(replace_from_to(dr, mt), op)
        return op.to_expr()

    @staticmethod
    def replace_base_path(expr, base_path):
        def replace(node, kwargs):
            if isinstance(node, CachedNode) and isinstance(
                node.cache, (ParquetCache, ParquetSnapshotCache)
            ):
                evolved = evolve(
                    node.cache,
                    storage=evolve(
                        node.cache.storage,
                        base_path=base_path,
                    ),
                )
                return node.__recreate__(
                    dict(zip(node.argnames, node.args)) | {"cache": evolved}
                )
            elif kwargs:
                return node.__recreate__(kwargs)
            else:
                return node

        return expr.op().replace(replace).to_expr()


@functools.wraps(ExprLoader)
def load_expr(expr_path, **kwargs):
    expr_loader = ExprLoader(expr_path, **kwargs)
    expr = expr_loader.load_expr()
    return expr


# todo: rename to dump_expr
@functools.wraps(ExprDumper)
def build_expr(expr, **kwargs):
    expr_dumper = ExprDumper(expr, **kwargs)
    expr_path = expr_dumper.dump_expr()
    return expr_path


@toolz.curry
def replace_from_to(from_, to_, node, kwargs):
    if node == from_:
        return to_
    elif kwargs:
        return node.__recreate__(kwargs)
    else:
        return node
