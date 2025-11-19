import base64
import functools
import itertools
import operator
from pathlib import Path
from typing import Any

import attr
import cloudpickle
import dask.base
import toolz
from attr import (
    field,
    frozen,
)

import xorq.expr.datatypes as dt
import xorq.vendor.ibis.expr.types as ir
from xorq.ibis_yaml.utils import freeze
from xorq.vendor.ibis.common.collections import FrozenOrderedDict


def _extract_parent_ref(node_dict):
    return toolz.pipe(
        node_dict,
        operator.methodcaller("get", "parent", {}),
        lambda p: p.get("node_ref") if isinstance(p, dict) else None,
    )


def _dict_to_sorted_tuple(obj):
    if isinstance(obj, (dict, FrozenOrderedDict)):
        sorted_items = sorted(obj.items(), key=lambda kv: str(kv[0]))
        return tuple((k, _dict_to_sorted_tuple(v)) for k, v in sorted_items)
    elif isinstance(obj, (list, tuple)):
        return tuple(_dict_to_sorted_tuple(item) for item in obj)
    else:
        return obj


@dask.base.normalize_token.register(FrozenOrderedDict)
def _normalize_frozen_ordered_dict(d):
    """Normalize FrozenOrderedDict using sorted tuple representation."""
    return ("FrozenOrderedDict", _dict_to_sorted_tuple(d))


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

    def register_node(self, node, node_dict):
        """Register a node and return its name.

        Returns a name like '@read_{hash}', '@filter_{hash}', etc.
        """
        import re

        from xorq.expr.relations import Tag

        if isinstance(node, Tag):
            parent_ref = _extract_parent_ref(node_dict)
            metadata = node_dict.get("metadata", FrozenOrderedDict())
            untagged_repr = ("Tag", parent_ref, metadata)
        else:
            untagged_repr = node.to_expr().ls.untagged

        node_hash = dask.base.tokenize(untagged_repr)

        op_type = node_dict.get("op", "unknown")
        short_hash = node_hash[:8]
        op_name = re.sub(r"[^a-zA-Z0-9_]", "_", op_type.lower())
        node_name = f"@{op_name}_{short_hash}"

        node_dict_with_hash = dict(node_dict) | {"snapshot_hash": node_hash}
        self.nodes.setdefault(node_name, freeze(node_dict_with_hash))

        return node_name


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
            if op_type not in FROM_YAML_HANDLERS:
                raise NotImplementedError(f"No handler for operation {op_type}")
            return FROM_YAML_HANDLERS[op_type](yaml_dict, context)
        case _:
            raise ValueError


@functools.cache
@functools.singledispatch
def translate_to_yaml(op: Any, context: TranslationContext) -> dict:
    raise NotImplementedError(f"No translation rule for {type(op)}")


@functools.singledispatch
def _translate_type(dtype: dt.DataType) -> dict:
    return freeze({"name": type(dtype).__name__, "nullable": dtype.nullable})
