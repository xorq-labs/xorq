import base64
import functools
import itertools
from pathlib import Path
from typing import Any

import cloudpickle
import toolz
from attr import (
    evolve,
    field,
    frozen,
)
from dask.base import tokenize

import xorq.expr.datatypes as dt
import xorq.vendor.ibis.expr.operations as ops
from xorq.caching.strategy import SnapshotStrategy
from xorq.expr.relations import Tag
from xorq.ibis_yaml.utils import freeze
from xorq.vendor.ibis.common.collections import FrozenOrderedDict
from xorq.vendor.ibis.expr.schema import Schema


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
        if (schema_id := self.schemas.get(frozen_schema)) is not None:
            return schema_id
        schema_id = f"schema_{next(self.counter)}"
        self.schemas[schema_id] = frozen_schema
        return schema_id

    def register_node(self, node, node_dict):
        """Register a node and return its name.

        Returns a name like '@read_{hash}', '@filter_{hash}', etc.
        """

        match node:
            case Tag():
                untagged_repr = ("Tag", node.parent.to_expr(), node_dict["metadata"])
                with SnapshotStrategy().normalization_context(node.to_expr()):
                    node_hash = tokenize(untagged_repr)
            case Schema():
                untagged_repr = ("Schema", tuple(node.items()))
                node_hash = tokenize(node)
            case _:
                untagged_repr = node.to_expr().ls.untagged
                with SnapshotStrategy().normalization_context(node.to_expr()):
                    node_hash = tokenize(untagged_repr)
        op_name = node_dict.get("op", "unknown").lower()
        node_name = f"@{op_name}_{node_hash[:16]}"
        node_dict_with_hash = freeze(dict(node_dict) | {"snapshot_hash": node_hash})
        self.nodes.setdefault(node_name, node_dict_with_hash)
        return node_name


def _is_absolute_path(instance, attribute, value):
    if value and not Path(value).is_absolute():
        raise ValueError("cache_dir must be absolute")


@frozen
class TranslationContext:
    schema_registry: SchemaRegistry = field(factory=SchemaRegistry)
    profiles: FrozenOrderedDict = field(factory=FrozenOrderedDict)
    definitions: FrozenOrderedDict = field(
        factory=functools.partial(freeze, {"schemas": {}, "nodes": {}}),
    )
    cache_dir: Path = field(
        default=None,
        converter=toolz.excepts(TypeError, Path),
        validator=_is_absolute_path,
    )

    def update_definitions(self, new_definitions: FrozenOrderedDict):
        return evolve(self, definitions=new_definitions)

    def finalize_definitions(self):
        updated_defs = dict(self.definitions)
        updated_defs["schemas"] = self.schema_registry.schemas
        updated_defs["nodes"] = self.schema_registry.nodes
        return evolve(self, definitions=freeze(updated_defs))

    def translate_from_yaml(self, yaml_dict: dict) -> Any:
        return translate_from_yaml(yaml_dict, self)

    def translate_to_yaml(self, op: Any) -> dict:
        return translate_to_yaml(op, self)


def register_from_yaml_handler(*op_names: str):
    def decorator(func):
        for name in op_names:
            FROM_YAML_HANDLERS[name] = func
        return func

    return decorator


def default_handler(yaml_dict: dict, context: TranslationContext):
    spec = dict(yaml_dict)
    cls = getattr(ops, spec["op"])
    return cls(
        **{
            key: translate_from_yaml(spec[key], context)
            for key in tuple(cls.__signature__.parameters)
        }
    ).to_expr()


@functools.cache
@functools.singledispatch
def translate_from_yaml(yaml_dict: dict, context: TranslationContext) -> Any:
    match yaml_dict:
        case None:
            return None
        case {"node_ref": node_ref, **_kwargs}:
            if "nodes" not in context.definitions:
                raise ValueError(
                    f"Missing 'nodes' in definitions for reference {node_ref}"
                )

            try:
                node_dict = context.definitions["nodes"][node_ref]
            except KeyError:
                raise ValueError(f"Node reference {node_ref} not found in definitions")
            return translate_from_yaml(node_dict, context)
        case {"op": op_type, **_kwargs}:
            return FROM_YAML_HANDLERS.get(op_type, default_handler)(yaml_dict, context)
        case _:
            raise ValueError


@functools.lru_cache(maxsize=None, typed=True)
@functools.singledispatch
def translate_to_yaml(op: Any, context: TranslationContext) -> dict:
    raise NotImplementedError(f"No translation rule for {type(op)}")


@functools.singledispatch
def _translate_type(dtype: dt.DataType) -> dict:
    return freeze({"name": type(dtype).__name__, "nullable": dtype.nullable})
