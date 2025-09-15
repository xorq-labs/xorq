import base64
import functools
import itertools
from pathlib import Path
from typing import Any

import attr
import cloudpickle
import toolz
from attr import (
    field,
    frozen,
)
from dask.base import tokenize

import xorq.expr.datatypes as dt
import xorq.vendor.ibis.expr.types as ir
from xorq.ibis_yaml.utils import freeze
from xorq.vendor.ibis.common.collections import FrozenOrderedDict


FROM_YAML_HANDLERS: dict[str, Any] = {}


def serialize_callable(fn: callable) -> str:
    pickled = cloudpickle.dumps(fn)
    encoded = base64.b64encode(pickled).decode("ascii")
    return encoded


def deserialize_callable(encoded_fn: str) -> callable:
    pickled = base64.b64decode(encoded_fn)
    return cloudpickle.loads(pickled)


class SchemaRegistry:
    def __init__(self):
        self.schemas = {}
        self.counter = itertools.count()
        self.nodes = {}

    def register_schema(self, schema):
        frozen_schema = freeze(
            {name: translate_to_yaml(dtype, None) for name, dtype in schema.items()}
        )
        for schema_id, existing_schema in self.schemas.items():
            if existing_schema == frozen_schema:
                return schema_id
        schema_id = f"schema_{next(self.counter)}"
        self.schemas[schema_id] = frozen_schema
        return schema_id

    def _register_expr_schema(self, expr: ir.Expr) -> str:
        if hasattr(expr, "schema"):
            schema = expr.schema()
            return self.register_schema(schema)
        return None

    def register_node(self, node_dict):
        frozen_node = freeze(node_dict)

        node_hash = tokenize(frozen_node)

        if node_hash not in self.nodes:
            self.nodes[node_hash] = frozen_node

        return node_hash


def _is_absolute_path(instance, attribute, value):
    if value and not Path(value).is_absolute():
        raise ValueError("cache_dir must be absolute")


@frozen
class TranslationContext:
    schema_registry: SchemaRegistry = field(factory=SchemaRegistry)
    profiles: FrozenOrderedDict = field(factory=FrozenOrderedDict)
    definitions: FrozenOrderedDict = field(
        factory=lambda: freeze({"schemas": {}, "nodes": {}})
    )
    cache_dir: Path = field(
        default=None,
        converter=toolz.excepts(TypeError, Path),
        validator=_is_absolute_path,
    )

    def update_definitions(self, new_definitions: FrozenOrderedDict):
        return attr.evolve(self, definitions=new_definitions)

    def finalize_definitions(self):
        updated_defs = dict(self.definitions)
        updated_defs["schemas"] = self.schema_registry.schemas
        updated_defs["nodes"] = self.schema_registry.nodes
        return attr.evolve(self, definitions=freeze(updated_defs))


def register_from_yaml_handler(*op_names: str):
    def decorator(func):
        for name in op_names:
            FROM_YAML_HANDLERS[name] = func
        return func

    return decorator


@functools.cache
@functools.singledispatch
def translate_from_yaml(yaml_dict: dict, context: TranslationContext) -> Any:
    if "node_ref" in yaml_dict:
        node_ref = yaml_dict["node_ref"]
        if "nodes" not in context.definitions:
            raise ValueError(f"Missing 'nodes' in definitions for reference {node_ref}")

        try:
            node_dict = context.definitions["nodes"][node_ref]
        except KeyError:
            raise ValueError(f"Node reference {node_ref} not found in definitions")
        return translate_from_yaml(node_dict, context)
    op_type = yaml_dict["op"]
    if op_type not in FROM_YAML_HANDLERS:
        raise NotImplementedError(f"No handler for operation {op_type}")
    return FROM_YAML_HANDLERS[op_type](yaml_dict, context)


@functools.cache
@functools.singledispatch
def translate_to_yaml(op: Any, context: TranslationContext) -> dict:
    raise NotImplementedError(f"No translation rule for {type(op)}")


@functools.singledispatch
def _translate_type(dtype: dt.DataType) -> dict:
    return freeze({"name": type(dtype).__name__, "nullable": dtype.nullable})
