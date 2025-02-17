import pathlib
from pathlib import Path
from typing import Any, Dict

import dask
import ibis.expr.types as ir
import yaml
from ibis.common.collections import FrozenOrderedDict

from letsql.ibis_yaml.sql import generate_sql_plans
from letsql.ibis_yaml.translate import (
    SchemaRegistry,
    translate_from_yaml,
    translate_to_yaml,
)
from letsql.ibis_yaml.utils import freeze


# is this the right way to handle this? or the right place
class CleanDictYAMLDumper(yaml.SafeDumper):
    def represent_frozenordereddict(self, data):
        return self.represent_dict(dict(data))


CleanDictYAMLDumper.add_representer(
    FrozenOrderedDict, CleanDictYAMLDumper.represent_frozenordereddict
)


class ArtifactStore:
    def __init__(self, root_path: pathlib.Path):
        self.root_path = (
            Path(root_path) if not isinstance(root_path, Path) else root_path
        )
        self.root_path.mkdir(parents=True, exist_ok=True)

    def get_path(self, *parts) -> pathlib.Path:
        return self.root_path.joinpath(*parts)

    def ensure_dir(self, *parts) -> pathlib.Path:
        path = self.get_path(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_yaml(self, data: Dict[str, Any], *path_parts) -> pathlib.Path:
        path = self.get_path(*path_parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.dump(
                data,
                f,
                Dumper=CleanDictYAMLDumper,
                default_flow_style=False,
                sort_keys=False,
            )
        return path

    def read_yaml(self, *path_parts) -> Dict[str, Any]:
        path = self.get_path(*path_parts)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with path.open("r") as f:
            return yaml.safe_load(f)

    def write_text(self, content: str, *path_parts) -> pathlib.Path:
        path = self.get_path(*path_parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            f.write(content)
        return path

    def read_text(self, *path_parts) -> str:
        path = self.get_path(*path_parts)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with path.open("r") as f:
            return f.read()

    def exists(self, *path_parts) -> bool:
        return self.get_path(*path_parts).exists()

    def get_expr_hash(self, expr) -> str:
        expr_hash = dask.base.tokenize(expr)
        return expr_hash[:12]  # TODO: make length of hash as a config

    def save_yaml(self, yaml_dict: Dict[str, Any], expr_hash, filename) -> pathlib.Path:
        return self.write_yaml(yaml_dict, expr_hash, filename)

    def load_yaml(self, expr_hash: str, filename) -> Dict[str, Any]:
        return self.read_yaml(expr_hash, filename)

    def get_build_path(self, expr_hash: str) -> pathlib.Path:
        return self.ensure_dir(expr_hash)


class YamlExpressionTranslator:
    def __init__(
        self,
        schema_registry: SchemaRegistry = None,
        profiles: Dict = None,
        current_path: Path = None,
    ):
        self.schema_registry = schema_registry or SchemaRegistry()
        self.definitions = {}
        self.profiles = profiles or {}
        self.current_path = current_path

    def to_yaml(self, expr: ir.Expr) -> Dict[str, Any]:
        schema_ref = self._register_expr_schema(expr)
        expr_dict = translate_to_yaml(expr, self)
        expr_dict = freeze({**dict(expr_dict), "schema_ref": schema_ref})

        return freeze(
            {
                "definitions": {"schemas": self.schema_registry.schemas},
                "expression": expr_dict,
            }
        )

    def from_yaml(self, yaml_dict: Dict[str, Any]) -> ir.Expr:
        self.definitions = yaml_dict.get("definitions", {})
        expr_dict = freeze(yaml_dict["expression"])
        return translate_from_yaml(expr_dict, self)

    def _register_expr_schema(self, expr: ir.Expr) -> str:
        if hasattr(expr, "schema"):
            schema = expr.schema()
            return self.schema_registry.register_schema(schema)
        return None


class BuildManager:
    def __init__(self, build_dir: pathlib.Path):
        self.artifact_store = ArtifactStore(build_dir)
        self.profiles = {}

    def compile_expr(self, expr: ir.Expr) -> None:
        expr_hash = self.artifact_store.get_expr_hash(expr)
        current_path = self.artifact_store.get_build_path(expr_hash)

        translator = YamlExpressionTranslator(
            profiles=self.profiles, current_path=current_path
        )
        # metadata.yaml (uv.lock, git commit version, version==xorq_internal_version, user, hostname, ip_address(host ip))
        yaml_dict = translator.to_yaml(expr)
        self.artifact_store.save_yaml(yaml_dict, expr_hash, "expr.yaml")

        sql_plans = generate_sql_plans(expr)
        self.artifact_store.save_yaml(sql_plans, expr_hash, "sql.yaml")

    def load_expr(self, expr_hash: str) -> ir.Expr:
        build_path = self.artifact_store.get_build_path(expr_hash)
        translator = YamlExpressionTranslator(
            current_path=build_path, profiles=self.profiles
        )

        yaml_dict = self.artifact_store.load_yaml(expr_hash, "expr.yaml")
        return translator.from_yaml(yaml_dict)

    # TODO: maybe change name
    def load_sql_plans(self, expr_hash: str) -> Dict[str, Any]:
        return self.artifact_store.load_yaml(expr_hash, "sql.yaml")
