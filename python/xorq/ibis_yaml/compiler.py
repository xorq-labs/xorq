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
from xorq.caching import SnapshotStrategy
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
from xorq.expr.relations import Read
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


DEFERRED_READS_YAML_FILENAME = "deferred_reads.yaml"
EXPR_YAML_FILENAME = "expr.yaml"
METADATA_JSON_FILENAME = "metadata.json"
PROFILES_YAML_FILENAME = "profiles.yaml"
SQL_YAML_FILENAME = "sql.yaml"


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

    def ensure_dir(self, *parts) -> pathlib.Path:
        path = self.get_path(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

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

    def exists(self, *path_parts) -> bool:
        return self.get_path(*path_parts).exists()

    def get_expr_hash(self, expr) -> str:
        with SnapshotStrategy().normalization_context(expr):
            expr_hash = dask.base.tokenize(expr)[: config.hash_length]
            return expr_hash

    def get_expr_path(self, expr) -> Path:
        return self.get_path(self.get_expr_hash(expr))

    def save_yaml(self, yaml_dict: Dict[str, Any], expr_hash, filename) -> pathlib.Path:
        return self.write_yaml(yaml_dict, expr_hash, filename)

    def load_yaml(self, expr_hash: str, filename) -> Dict[str, Any]:
        return self.read_yaml(expr_hash, filename)

    def get_build_path(self, expr_hash: str) -> pathlib.Path:
        return self.ensure_dir(expr_hash)


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


@frozen
class BuildManager:
    root_path = field(validator=instance_of(Path), converter=Path)
    cache_dir = field(
        validator=optional(or_(instance_of(Path), instance_of(str))), default=None
    )
    debug = field(validator=instance_of(bool), default=False)

    def __attrs_post_init__(self):
        """
        build_dir: root directory where build artifacts are stored
        cache_dir: optional directory for parquet cache files
        debug: when True, output SQL files and debug artifacts (sql.yaml, deferred_reads.yaml)
        """
        match self.cache_dir:
            case None:
                object.__setattr__(self, "cache_dir", get_xorq_cache_dir())
            case Path():
                pass
            case _:
                object.__setattr__(self, "cache_dir", Path(self.cache_dir))

    @property
    @functools.cache
    def artifact_store(self):
        return ArtifactStore(self.root_path)

    def _write_sql_file(self, sql: str, expr_hash: str, query_name: str) -> str:
        sql_hash = dask.base.tokenize(sql)[: config.hash_length]
        filename = f"{sql_hash}.sql"
        sql_path = self.artifact_store.get_build_path(expr_hash) / filename
        sql_path.write_text(sql)
        return filename

    @staticmethod
    def _write_memtable(build_dir, mt, which):
        import pyarrow.parquet as pq

        assert which in ("database_tables", "memtables")
        table = mt.to_expr().to_pyarrow()
        parquet_path = build_dir.joinpath(which, dask.base.tokenize(table)).with_suffix(
            ".parquet"
        )
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, parquet_path)
        return parquet_path

    @staticmethod
    def _table_to_read_op(parquet_path, read_kwargs, args_values, con=_backend_init()):
        dr = deferred_read_parquet(parquet_path, con, **read_kwargs)
        op = dr.op()
        args = dict(zip(op.__argnames__, op.__args__))
        args["values"] = args_values
        op = op.__recreate__(args)
        return op

    def _process_sql_plans(
        self, sql_plans: Dict[str, Any], expr_hash: str
    ) -> Dict[str, Any]:
        queries = {
            query_name: toolz.dissoc(query_info, "sql")
            | {
                "sql_file": self._write_sql_file(
                    query_info["sql"], expr_hash, query_name
                ),
            }
            for (query_name, query_info) in sql_plans["queries"].items()
        }
        updated_plans = {"queries": queries}
        return updated_plans

    def _process_deferred_reads(
        self, deferred_reads: Dict[str, Any], expr_hash: str
    ) -> Dict[str, Any]:
        reads = {
            read_name: toolz.dissoc(read_info, "sql")
            | {
                "sql_file": self._write_sql_file(
                    read_info["sql"],
                    expr_hash,
                    read_name,
                )
            }
            for read_name, read_info in deferred_reads["reads"].items()
        }
        updated_reads = {"reads": reads}
        return updated_reads

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

    def _write_debug_info(self, expr, expr_hash):
        sql_plans, deferred_reads = generate_sql_plans(expr)
        updated_sql_plans = self._process_sql_plans(sql_plans, expr_hash)
        self.artifact_store.save_yaml(updated_sql_plans, expr_hash, SQL_YAML_FILENAME)
        updated_deferred_reads = self._process_deferred_reads(deferred_reads, expr_hash)
        self.artifact_store.save_yaml(
            updated_deferred_reads,
            expr_hash,
            DEFERRED_READS_YAML_FILENAME,
        )

    def compile_expr(self, expr: ir.Expr) -> str:
        expr_build_dir = self.artifact_store.get_expr_path(expr)
        expr_hash = expr_build_dir.name

        # this is writing to the artifact_store?
        expr = memtables_to_deferred_reads(expr_build_dir, expr)
        expr = replace_inmemory_backend_tables(expr_build_dir, expr)

        profiles = dehydrate_cons(find_all_sources(expr))
        yaml_dict = YamlExpressionTranslator.to_yaml(expr, profiles, self.cache_dir)
        metadata_json = self._make_metadata()
        self.artifact_store.save_yaml(yaml_dict, expr_hash, EXPR_YAML_FILENAME)
        self.artifact_store.save_yaml(profiles, expr_hash, PROFILES_YAML_FILENAME)
        self.artifact_store.write_text(metadata_json, expr_hash, METADATA_JSON_FILENAME)

        # write SQL plan and deferred-read artifacts if debug enabled
        if self.debug:
            self._write_debug_info(expr, expr_hash)

        return expr_hash


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
        return ArtifactStore(self.expr_path.parent)

    def load_expr(self):
        profiles = hydrate_cons(
            self.artifact_store.load_yaml(self.expr_hash, PROFILES_YAML_FILENAME)
        )
        yaml_dict = self.artifact_store.load_yaml(self.expr_hash, EXPR_YAML_FILENAME)
        expr = YamlExpressionTranslator.from_yaml(yaml_dict, profiles=profiles)
        expr = deferred_reads_to_memtables(expr)
        if self.cache_dir:
            expr = replace_base_path(expr, base_path=self.cache_dir)
        return expr


def load_expr(expr_path, cache_dir=None):
    expr_loader = ExprLoader(expr_path, cache_dir=cache_dir)
    expr = expr_loader.load_expr()
    return expr


def build_expr(expr, build_dir="builds", cache_dir=None, **kwargs):
    build_manager = BuildManager(build_dir, cache_dir=cache_dir, **kwargs)
    expr_hash = build_manager.compile_expr(expr)
    path = build_manager.artifact_store.get_path(expr_hash)
    return path


IS_INMEMORY = "is-inmemory"
IS_DATABASE_TABLE = "is-database-table"


@toolz.curry
def replace_from_to(from_, to_, node, kwargs):
    if node == from_:
        return to_
    elif kwargs:
        return node.__recreate__(kwargs)
    else:
        return node


def replace_base_path(expr, base_path):
    from xorq.caching import (
        ParquetCache,
        ParquetSnapshotCache,
    )
    from xorq.expr.relations import CachedNode

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


def deferred_reads_to_memtables(loaded):
    def deferred_read_to_memtable(dr):
        assert dr.values.get(IS_INMEMORY)
        path = next(v for k, v in dr.read_kwargs if k == "path")
        df = read_parquet(path).execute()
        mt = ibis.memtable(df, schema=dr.schema, name=dr.name)
        return mt

    drs = tuple(dr for dr in loaded.op().find(Read) if dr.values.get(IS_INMEMORY))
    op = loaded.op()
    for dr in drs:
        mt = deferred_read_to_memtable(dr)
        op = op.replace(replace_from_to(dr, mt))
    return op.to_expr()


def memtables_to_deferred_reads(build_dir, expr):
    def memtable_to_read_op(builds_dir, mt):
        parquet_path = BuildManager._write_memtable(builds_dir, mt, "memtables")
        op = BuildManager._table_to_read_op(
            parquet_path=parquet_path,
            read_kwargs={
                "table_name": mt.name,
            },
            args_values={IS_INMEMORY: True},
        )
        return op

    op = expr.op()
    mts = walk_nodes((InMemoryTable,), expr)
    for mt in mts:
        dr_op = memtable_to_read_op(build_dir, mt)
        op = replace_nodes(replace_from_to(mt, dr_op), expr)
    new_expr = op.to_expr()
    return new_expr


def replace_inmemory_backend_tables(build_dir, expr):
    def database_table_to_read_op(builds_dir, mt, con):
        parquet_path = BuildManager._write_memtable(builds_dir, mt, "database_tables")
        op = BuildManager._table_to_read_op(
            parquet_path=parquet_path,
            read_kwargs={
                "table_name": mt.name,
                # we normalize based on content so we can reproducible hash
                "normalize_method": normalize_read_path_md5sum,
            },
            args_values={IS_DATABASE_TABLE: True},
            con=con,
        )
        return op

    op = expr.op()

    table_like_ops = tuple(o for o in opaque_ops if issubclass(o, DatabaseTable))
    tables = walk_nodes((DatabaseTable,), expr)
    for table in tables:
        if not isinstance(table, table_like_ops) and table.source.name in (
            "pandas",
            "duckdb",
            "datafusion",
            "xorq",
        ):
            dr_op = database_table_to_read_op(build_dir, table, con=table.source)
            op = replace_nodes(replace_from_to(table, dr_op), expr)
    new_expr = op.to_expr()
    return new_expr
