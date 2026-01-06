import contextlib
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
from attr.validators import instance_of, optional

import xorq
import xorq.common.utils.logging_utils as lu
import xorq.ibis_yaml.translate  #  noqa: F401
import xorq.vendor.ibis as ibis
import xorq.vendor.ibis.expr.types as ir
from xorq.caching import SnapshotStrategy
from xorq.common.utils.caching_utils import get_xorq_cache_dir
from xorq.common.utils.dask_normalize.dask_normalize_utils import (
    file_digest,
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
            expr_hash = dask.base.tokenize(expr)
        hash_length = config.hash_length
        return expr_hash[:hash_length]

    def save_yaml(self, yaml_dict: Dict[str, Any], expr_hash, filename) -> pathlib.Path:
        return self.write_yaml(yaml_dict, expr_hash, filename)

    def load_yaml(self, expr_hash: str, filename) -> Dict[str, Any]:
        return self.read_yaml(expr_hash, filename)

    def get_build_path(self, expr_hash: str) -> pathlib.Path:
        return self.ensure_dir(expr_hash)


class YamlExpressionTranslator:
    def __init__(self):
        pass

    def to_yaml(self, expr: ir.Expr, profiles=(), cache_dir=None) -> Dict[str, Any]:
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

    def from_yaml(
        self,
        yaml_dict: Dict[str, Any],
        profiles=(),
    ) -> ir.Expr:
        context = TranslationContext(
            registry=Registry(**yaml_dict.get("definitions", {})),
            profiles=freeze(dict(profiles)),
        )
        expr_dict = freeze(yaml_dict["expression"])
        return translate_from_yaml(expr_dict, context)


@frozen
class BuildManager:
    artifact_store = field(
        validator=instance_of(ArtifactStore), converter=ArtifactStore
    )
    cache_dir = field(validator=optional(instance_of(Path)), default=None)
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

    def _write_sql_file(self, sql: str, expr_hash: str, query_name: str) -> str:
        hash_length = config.hash_length
        sql_hash = dask.base.tokenize(sql)[:hash_length]
        filename = f"{sql_hash}.sql"
        sql_path = self.artifact_store.get_build_path(expr_hash) / filename
        sql_path.write_text(sql)
        return filename

    def _process_sql_plans(
        self, sql_plans: Dict[str, Any], expr_hash: str
    ) -> Dict[str, Any]:
        updated_plans = {"queries": {}}

        for query_name, query_info in sql_plans["queries"].items():
            sql_filename = self._write_sql_file(
                query_info["sql"], expr_hash, query_name
            )

            updated_query_info = query_info.copy()
            updated_query_info["sql_file"] = sql_filename
            updated_query_info.pop("sql")
            updated_plans["queries"][query_name] = updated_query_info

        return updated_plans

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

    def _process_deferred_reads(
        self, deferred_reads: Dict[str, Any], expr_hash: str
    ) -> Dict[str, Any]:
        updated_reads = {"reads": {}}

        for read_name, read_info in deferred_reads["reads"].items():
            sql_filename = self._write_sql_file(read_info["sql"], expr_hash, read_name)

            updated_read_info = read_info.copy()
            updated_read_info["sql_file"] = sql_filename
            updated_read_info.pop("sql")
            updated_reads["reads"][read_name] = updated_read_info

        return updated_reads

    def compile_expr(self, expr: ir.Expr) -> str:
        expr_hash = self.artifact_store.get_expr_hash(expr)
        expr_build_dir = self.artifact_store.root_path / expr_hash
        expr = replace_memtables(expr_build_dir, expr)
        expr = replace_database_tables(expr_build_dir, expr)

        backends = find_all_sources(expr)
        profiles = {
            backend._profile.hash_name: {
                **backend._profile.as_dict(),
                "kwargs_tuple": dict(backend._profile.as_dict()["kwargs_tuple"]),
            }
            for backend in backends
        }
        profiles = dict(sorted(profiles.items()))

        translator = YamlExpressionTranslator()
        yaml_dict = translator.to_yaml(expr, profiles, self.cache_dir)
        self.artifact_store.save_yaml(yaml_dict, expr_hash, "expr.yaml")
        self.artifact_store.save_yaml(profiles, expr_hash, "profiles.yaml")

        # write SQL plan and deferred-read artifacts if debug enabled
        if getattr(self, "debug", False):
            sql_plans, deferred_reads = generate_sql_plans(expr)
            updated_sql_plans = self._process_sql_plans(sql_plans, expr_hash)
            self.artifact_store.save_yaml(updated_sql_plans, expr_hash, "sql.yaml")
            updated_deferred_reads = self._process_deferred_reads(
                deferred_reads, expr_hash
            )
            self.artifact_store.save_yaml(
                updated_deferred_reads, expr_hash, "deferred_reads.yaml"
            )

        metadata_json = self._make_metadata()
        self.artifact_store.write_text(metadata_json, expr_hash, "metadata.json")

        return expr_hash

    def load_expr(self, expr_hash: str) -> ir.Expr:
        profiles_dict = self.artifact_store.load_yaml(expr_hash, "profiles.yaml")

        def f(values):
            dct = dict(values)
            if isinstance(dct["kwargs_tuple"], dict):
                dct["kwargs_tuple"] = tuple(dct["kwargs_tuple"].items())
            else:
                dct["kwargs_tuple"] = tuple(map(tuple, dct["kwargs_tuple"]))
            return dct

        profiles = {
            profile: Profile(**f(values)).get_con()
            for profile, values in profiles_dict.items()
        }
        translator = YamlExpressionTranslator()

        yaml_dict = self.artifact_store.load_yaml(expr_hash, "expr.yaml")
        expr = translator.from_yaml(yaml_dict, profiles=profiles)
        expr = replace_deferred_reads(expr)
        if self.cache_dir:
            expr = replace_base_path(expr, base_path=self.cache_dir)
        return expr

    # TODO: maybe change name
    def load_sql_plans(self, expr_hash: str) -> Dict[str, Any]:
        return self.artifact_store.load_yaml(expr_hash, "sql.yaml")

    def load_deferred_reads(self, expr_hash: str) -> Dict[str, Any]:
        return self.artifact_store.load_yaml(expr_hash, "deferred_reads.yaml")

    @staticmethod
    def validate_build(path) -> tuple:
        """
        Validate a build directory and extract build metadata for catalog.
        Returns: build_id, meta_digest, metadata_preview
        """

        def validate(path):
            build_path = Path(path)
            meta_file = build_path / "metadata.json"
            if not build_path.exists() or not build_path.is_dir():
                raise ValueError(f"Build path not found: {build_path}")
            if not meta_file.exists():
                raise ValueError(f"metadata.json not found in build path: {build_path}")
            return meta_file

        meta_file = validate(path)
        # The build_id is the directory name
        build_id = meta_file.parent.name
        # Compute meta_digest from metadata.json
        meta_digest = f"sha1:{file_digest(meta_file)}"
        return build_id, meta_digest


def load_expr(expr_path, cache_dir=None):
    expr_path = Path(expr_path)
    return BuildManager(expr_path.parent, cache_dir=cache_dir).load_expr(expr_path.name)


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


def replace_deferred_reads(loaded):
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


def replace_memtables(build_dir, expr):
    def memtable_to_read_op(builds_dir, mt, con=_backend_init()):
        memtables_dir = Path(builds_dir).joinpath("memtables")
        memtables_dir.mkdir(parents=True, exist_ok=True)
        df = mt.to_expr().execute()
        parquet_path = memtables_dir.joinpath(dask.base.tokenize(df)).with_suffix(
            ".parquet"
        )
        df.to_parquet(parquet_path)
        # FIXME: enable Path
        dr = deferred_read_parquet(parquet_path, con, table_name=mt.name)
        op = dr.op()
        args = dict(zip(op.__argnames__, op.__args__))
        args["values"] = {IS_INMEMORY: True}
        op = op.__recreate__(args)
        return op

    op = expr.op()
    mts = walk_nodes((InMemoryTable,), expr)
    for mt in mts:
        dr_op = memtable_to_read_op(build_dir, mt)
        op = replace_nodes(replace_from_to(mt, dr_op), expr)
    new_expr = op.to_expr()
    return new_expr


def replace_database_tables(build_dir, expr):
    def database_table_to_read_op(builds_dir, mt, con=_backend_init()):
        import pyarrow.parquet as pq

        database_tables_dir = Path(builds_dir).joinpath("database_tables")
        database_tables_dir.mkdir(parents=True, exist_ok=True)
        df = mt.to_expr().to_pyarrow()
        parquet_path = database_tables_dir.joinpath(dask.base.tokenize(df)).with_suffix(
            ".parquet"
        )
        pq.write_table(df, parquet_path)
        # we normalize based on content so we can reproducible hash
        dr = deferred_read_parquet(
            parquet_path,
            con,
            table_name=mt.name,
            normalize_method=normalize_read_path_md5sum,
        )
        op = dr.op()
        args = dict(zip(op.__argnames__, op.__args__))
        args["values"] = {IS_DATABASE_TABLE: True}
        op = op.__recreate__(args)
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
