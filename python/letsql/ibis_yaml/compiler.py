import pathlib
from pathlib import Path
from typing import Any, Dict

import dask
import yaml

from letsql.ibis_yaml.translate import translate_from_yaml, translate_to_yaml


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

    def save_yaml(self, yaml_dict: Dict[str, Any], expr) -> pathlib.Path:
        expr_hash = self.get_expr_hash(expr)
        return self.storage.write_yaml(yaml_dict, expr_hash, "expr.yaml")

    def load_yaml(self, expr_hash: str) -> Dict[str, Any]:
        return self.storage.read_yaml(expr_hash, "expr.yaml")

    def get_build_path(self, expr_hash: str) -> pathlib.Path:
        return self.storage.ensure_dir(expr_hash)


class IbisYamlCompiler:
    def __init__(self):
        pass

    def compile_to_yaml(self, expr):
        return translate_to_yaml(expr.op(), self)

    def compile_from_yaml(self, yaml_dict):
        return translate_from_yaml(yaml_dict, self)
