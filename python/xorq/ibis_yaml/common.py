import base64
import functools
from pathlib import Path
from typing import Any

import cloudpickle
import toolz
from attr import (
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


try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum


class RefEnum(StrEnum):
    dtype_ref = "dtype_ref"
    node_ref = "node_ref"
    schema_ref = "schema_ref"


class RegistryEnum(StrEnum):
    dtypes = "dtypes"
    nodes = "nodes"
    schemas = "schemas"


FROM_YAML_HANDLERS: dict[str, Any] = {}


def serialize_callable(fn: callable) -> str:
    pickled = cloudpickle.dumps(fn)
    encoded = base64.b64encode(pickled).decode("ascii")
    return encoded


def deserialize_callable(encoded_fn: str) -> callable:
    pickled = base64.b64decode(encoded_fn)
    return cloudpickle.loads(pickled)


class Registry:
    def __init__(self, dtypes=(), nodes=(), schemas=()):
        self.dtypes = dict(dtypes)
        self.nodes = dict(nodes)
        self.schemas = dict(schemas)

    def getstate(self):
        return freeze(
            {
                RegistryEnum.dtypes: self.dtypes,
                RegistryEnum.nodes: self.nodes,
                RegistryEnum.schemas: self.schemas,
            }
        )

    def register_dtype(self, dtype, dtype_dict):
        dtype_ref = f"dtype_{tokenize(dtype_dict)[: config.hash_length]}"
        self.dtypes.setdefault(dtype_ref, dtype_dict)
        frozen = freeze({RefEnum.dtype_ref: dtype_ref})
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
        frozen = freeze({RefEnum.node_ref: node_ref})
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
        frozen = freeze({RefEnum.schema_ref: schema_ref})
        return frozen

    def get(self, which, ref):
        match which:
            case RegistryEnum.dtypes:
                dct = self.dtypes
            case RegistryEnum.nodes:
                dct = self.nodes
            case RegistryEnum.schemas:
                dct = self.schemas
            case _:
                raise ValueError(f"don't know how to handle which={which}")
        try:
            return freeze(dct[ref])
        except KeyError:
            raise ValueError(f"ref {ref} not found in definitions for which={which}")


def _is_absolute_path(instance, attribute, value):
    if value and not Path(value).is_absolute():
        raise ValueError("cache_dir must be absolute")


@frozen
class TranslationContext:
    registry: Registry = field(factory=Registry)
    profiles: FrozenOrderedDict = field(factory=FrozenOrderedDict)
    cache_dir: Path = field(
        default=None,
        converter=toolz.excepts(TypeError, Path),
        validator=_is_absolute_path,
    )

    @property
    def definitions(self):
        return self.registry.getstate()

    def translate_from_yaml(self, yaml_dict: dict) -> Any:
        return translate_from_yaml(yaml_dict, self)

    def translate_to_yaml(self, op: Any) -> dict:
        return translate_to_yaml(op, self)

    def register(self, which, op, frozen=None):
        match which:
            case RegistryEnum.dtypes:
                return self.registry.register_dtype(op, frozen)
            case RegistryEnum.nodes:
                return self.registry.register_node(op, frozen)
            case RegistryEnum.schemas:
                return self.registry.register_schema(op)
            case _:
                raise ValueError(f"don't know how to register {which}")

    def get_definition(self, which, ref):
        return self.registry.get(which, ref)

    def get_dtype(self, dtype_ref):
        dtype_def = self.get_definition(RegistryEnum.dtypes, dtype_ref)
        return self.translate_from_yaml(dtype_def)

    def get_node(self, node_ref):
        node_def = self.get_definition(RegistryEnum.nodes, node_ref)
        return self.translate_from_yaml(node_def)

    def get_schema(self, schema_ref):
        schema_def = self.get_definition(RegistryEnum.schemas, schema_ref)
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
        case {RefEnum.dtype_ref: dtype_ref, **rest}:
            if rest:
                raise ValueError(
                    f"don't know how to handle additional keys ({tuple(rest)}"
                )
            return context.get_dtype(dtype_ref)
        case {RefEnum.node_ref: node_ref, RefEnum.schema_ref: _schema_ref, **rest}:
            if rest:
                raise ValueError(
                    f"don't know how to handle additional keys ({tuple(rest)}"
                )
            return context.get_node(node_ref)
        case {RefEnum.node_ref: node_ref, **rest}:
            if rest:
                raise ValueError(
                    f"don't know how to handle additional keys ({tuple(rest)}"
                )
            return context.get_node(node_ref)
        case {"op": op_type}:
            return FROM_YAML_HANDLERS.get(op_type, default_handler)(yaml_dict, context)
        case {RefEnum.schema_ref: schema_ref, **rest}:
            if rest:
                raise ValueError(
                    f"don't know how to handle additional keys ({tuple(rest)}"
                )
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
