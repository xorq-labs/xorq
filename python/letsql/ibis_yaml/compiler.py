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


class StorageHandler:
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


class BuildManager:
    def __init__(self, storage_path: pathlib.Path):
        self.storage = StorageHandler(storage_path)

    def get_expr_hash(self, expr) -> str:
        expr_hash = dask.base.tokenize(expr)
        return expr_hash[:12]  # TODO: make length of hash as a config

    def save_yaml(self, yaml_dict: Dict[str, Any], expr_hash) -> pathlib.Path:
        filename = (
            "sql.yaml"
            if isinstance(yaml_dict, dict) and "queries" in yaml_dict
            else "expr.yaml"
        )
        return self.storage.write_yaml(yaml_dict, expr_hash, filename)

    def load_yaml(self, expr_hash: str) -> Dict[str, Any]:
        return self.storage.read_yaml(expr_hash, "expr.yaml")

    def get_build_path(self, expr_hash: str) -> pathlib.Path:
        return self.storage.ensure_dir(expr_hash)


class IbisYamlCompiler:
    def __init__(self, build_dir, build_manager=BuildManager):
        self.build_manager = build_manager(build_dir)
        self.schema_registry = SchemaRegistry()
        self.current_path = None

    def compile(self, expr):
        yaml_dict = self.compile_to_yaml(expr)
        expr_hash = self.build_manager.get_expr_hash(expr)
        self.curent_path = self.build_manager.get_build_path(expr_hash)
        self.build_manager.save_yaml(yaml_dict, expr_hash)
        plans = generate_sql_plans(expr)
        self.build_manager.save_yaml(plans, expr_hash)

    def from_hash(self, expr_hash) -> ir.Expr:
        yaml_dict = self.build_manager.load_yaml(expr_hash)

        def convert_to_frozen(d):
            if isinstance(d, dict):
                items = []
                for k, v in d.items():
                    converted_v = convert_to_frozen(v)
                    if isinstance(converted_v, list):
                        converted_v = tuple(converted_v)
                    items.append((k, converted_v))
                return FrozenOrderedDict(items)
            elif isinstance(d, list):
                return [convert_to_frozen(x) for x in d]
            return d

        yaml_dict = convert_to_frozen(yaml_dict)
        return self.compile_from_yaml(yaml_dict)

    def compile_from_yaml(self, yaml_dict) -> ir.Expr:
        self.definitions = yaml_dict.get("definitions", {})
        return translate_from_yaml(yaml_dict["expression"], self)

    def compile_to_yaml(self, expr) -> Dict:
        expr_hash = self.build_manager.get_expr_hash(expr)
        self.current_path = self.build_manager.get_build_path(expr_hash)
        expr_yaml = translate_to_yaml(expr, self)

        schema_definitions = {}
        for schema_id, schema in self.schema_registry.schemas.items():
            schema_definitions[schema_id] = schema

        return freeze(
            {"definitions": {"schemas": schema_definitions}, "expression": expr_yaml}
        )
