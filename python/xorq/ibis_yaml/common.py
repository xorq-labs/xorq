import base64
import functools
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
from xorq.ibis_yaml.config import config
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


class Registry:
    def __init__(self):
        self.dtypes = {}
        self.nodes = {}
        self.schemas = {}

    def getstate(self):
        return {
            "dtypes": self.dtypes,
            "nodes": self.nodes,
            "schemas": self.schemas,
        }

    def register_dtype(self, dtype, dtype_dict):
        dtype_ref = tokenize(dtype_dict)
        self.dtypes.setdefault(dtype_ref, dtype_dict)
        frozen = freeze({"dtype_ref": dtype_ref})
        return frozen

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
        node_ref = f"@{op_name}_{node_hash[: config.hash_length]}"
        node_dict_with_hash = freeze(node_dict | {"snapshot_hash": node_hash})
        self.nodes.setdefault(node_ref, node_dict_with_hash)
        frozen = freeze({"node_ref": node_ref})
        return frozen

    def register_schema(self, schema):
        frozen_schema = freeze(
            toolz.valmap(
                functools.partial(translate_to_yaml, context=None),
                schema,
            )
        )
        schema_ref = f"schema_{tokenize(frozen_schema)[: config.hash_length]}"
        self.schemas.setdefault(schema_ref, frozen_schema)
        frozen = freeze({"schema_ref": schema_ref})
        return frozen


def _is_absolute_path(instance, attribute, value):
    if value and not Path(value).is_absolute():
        raise ValueError("cache_dir must be absolute")


@frozen
class TranslationContext:
    registry: Registry = field(factory=Registry)
    profiles: FrozenOrderedDict = field(factory=FrozenOrderedDict)
    definitions: FrozenOrderedDict = field(
        factory=functools.partial(freeze, Registry().getstate()),
    )
    cache_dir: Path = field(
        default=None,
        converter=toolz.excepts(TypeError, Path),
        validator=_is_absolute_path,
    )

    def finalize_definitions(self):
        definitions = freeze(self.definitions | self.registry.getstate())
        return evolve(self, definitions=definitions)

    def translate_from_yaml(self, yaml_dict: dict) -> Any:
        return translate_from_yaml(yaml_dict, self)

    def translate_to_yaml(self, op: Any) -> dict:
        return translate_to_yaml(op, self)

    def register(self, which, op, frozen=None):
        match which:
            case "dtypes":
                return self.registry.register_dtype(op, frozen)
            case "nodes":
                return self.registry.register_node(op, frozen)
            case "schemas":
                return self.registry.register_schema(op)
            case _:
                raise ValueError(f"don't know how to register {which}")

    def get_definition(self, which, ref):
        try:
            return self.definitions[which][ref]
        except KeyError:
            raise ValueError(f"ref {ref} not found in definitions for {which}")

    def get_dtype(self, dtype_ref):
        dtype_def = self.get_definition("dtypes", dtype_ref)
        return self.translate_from_yaml(dtype_def)

    def get_node(self, node_ref):
        node_def = self.get_definition("nodes", node_ref)
        return self.translate_from_yaml(node_def)

    def get_schema(self, schema_ref):
        schema_def = self.get_definition("schemas", schema_ref)
        schema = Schema(toolz.valmap(self.translate_from_yaml, schema_def))
        return schema


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
        case {"dtype_ref": dtype_ref, **rest}:
            if rest:
                raise ValueError(
                    f"don't know how to handle additional keys ({tuple(rest)}"
                )
            return context.get_dtype(dtype_ref)
        case {"node_ref": node_ref, "schema_ref": _schema_ref, **rest}:
            if rest:
                raise ValueError(
                    f"don't know how to handle additional keys ({tuple(rest)}"
                )
            return context.get_node(node_ref)
        case {"node_ref": node_ref, **rest}:
            if rest:
                raise ValueError(
                    f"don't know how to handle additional keys ({tuple(rest)}"
                )
            return context.get_node(node_ref)
        case {"op": op_type}:
            return FROM_YAML_HANDLERS.get(op_type, default_handler)(yaml_dict, context)
        case {"schema_ref": schema_ref, **rest}:
            if rest:
                raise ValueError(
                    f"don't know how to handle additional keys ({tuple(rest)}"
                )
            import pdb

            pdb.set_trace()  # noqa
            return context.get_schema(schema_ref)
        case _:
            raise ValueError(f"don't know how to handle keys ({tuple(yaml_dict)})")


@functools.lru_cache(maxsize=None, typed=True)
@functools.singledispatch
def translate_to_yaml(op: Any, context: TranslationContext) -> dict:
    raise NotImplementedError(f"No translation rule for {type(op)}")


@functools.singledispatch
def _translate_type(dtype: dt.DataType) -> dict:
    return freeze({"name": type(dtype).__name__, "nullable": dtype.nullable})
