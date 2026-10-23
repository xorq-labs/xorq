from __future__ import annotations

import base64
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cloudpickle
import toolz
from attr import (
    field,
    frozen,
)
from attr.validators import instance_of

from xorq.common.utils.content_hash import content_hash
from xorq.common.utils.dasher import tokenize
from xorq.expr.relations import Read
from xorq.ibis_yaml.config import config
from xorq.ibis_yaml.enums import NodeKey, ReadKwarg, RefEnum, RegistryEnum
from xorq.ibis_yaml.utils import freeze
from xorq.vendor.ibis.common.collections import FrozenOrderedDict
from xorq.vendor.ibis.expr.schema import Schema


def serialize_callable(fn: callable) -> str:
    pickled = cloudpickle.dumps(fn)
    encoded = base64.b64encode(pickled).decode("ascii")
    return encoded


def deserialize_callable(encoded_fn: str) -> callable:
    pickled = base64.b64decode(encoded_fn)
    return cloudpickle.loads(pickled)


def tuplify(obj):
    # tokenize sorts dict items, but schema and struct order is meaningful
    match obj:
        case dict():
            return tuple((key, tuplify(value)) for key, value in obj.items())
        case tuple():
            return tuple(map(tuplify, obj))
        case _:
            return obj


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
        dtype_ref = f"dtype_{tokenize(tuplify(dtype_dict))[: config.hash_length]}"
        self.dtypes.setdefault(dtype_ref, dtype_dict)
        frozen = freeze({RefEnum.dtype_ref: dtype_ref})
        return frozen

    def register_node(self, node, node_dict):
        """Register a node and return its name.

        Returns a name like '@read_{hash}', '@filter_{hash}', etc.
        """

        node_hash = content_hash(node)
        op_name = node_dict.get(NodeKey.op, "unknown").lower()
        node_ref = f"@{op_name}_{node_hash[: config.hash_length]}"
        node_dict_with_hash = freeze(node_dict | {NodeKey.snapshot_hash: node_hash})
        if isinstance(node, Read) and ReadKwarg.read_path in dict(node.read_kwargs):
            # Reads whose parquet was materialized into the build bundle carry
            # a build-relative `read_path`. The absolute `hash_path` assigned
            # at build time embeds the tmpdir root, so serializing it verbatim
            # would make expr.yaml bytes non-deterministic across rebuilds.
            # The loader (ExprLoader.deferred_reads_to_memtables) overwrites
            # hash_path with expr_path.joinpath(read_path) anyway, so rewriting
            # the stored value to the relative read_path is lossless.
            from xorq.common.utils.node_utils import update_read_kwargs  # noqa: PLC0415

            old_read_kwargs = node_dict_with_hash[NodeKey.read_kwargs]
            new_path = Path(dict(old_read_kwargs)[ReadKwarg.read_path])
            modified_read_kwargs = update_read_kwargs(
                old_read_kwargs, ((ReadKwarg.hash_path, new_path),)
            )
            node_dict_with_hash = freeze(
                node_dict_with_hash | {NodeKey.read_kwargs: modified_read_kwargs}
            )
        self.nodes.setdefault(node_ref, node_dict_with_hash)
        frozen = freeze({RefEnum.node_ref: node_ref})
        return frozen

    def register_schema(self, frozen_schema):
        schema_ref = f"schema_{tokenize(tuplify(frozen_schema))[: config.hash_length]}"
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
        except KeyError as err:
            raise ValueError(
                f"ref {ref} not found in definitions for which={which}"
            ) from err


# a rule: one value to its yaml, or one yaml dict back to its value; to_yaml
# with context None writes the value inline instead of registering a ref
ToYaml = Callable[[Any, "TranslationContext | None"], Any]
FromYaml = Callable[[dict, "TranslationContext"], Any]


def _is_absolute_path(instance, attribute, value):
    if value and not Path(value).is_absolute():
        raise ValueError("cache_dir must be absolute")


@frozen
class TranslationContext:
    """The state one translation shares, and the translators it runs.

    ``to_yaml`` and ``from_yaml`` are the dispatchers ``translate.py`` builds
    from its rule tables; a context is made with them (``make_context``), so a
    context that cannot translate cannot be constructed.
    """

    to_yaml: ToYaml = field(kw_only=True, eq=False, repr=False)
    from_yaml: FromYaml = field(kw_only=True, eq=False, repr=False)
    registry: Registry = field(factory=Registry)
    profiles: FrozenOrderedDict = field(factory=FrozenOrderedDict)
    cache_dir: Path = field(
        default=None,
        converter=toolz.excepts(TypeError, Path),
        validator=_is_absolute_path,
    )
    remote_table_stack: list[str] = field(
        validator=instance_of(list), factory=list, eq=False
    )

    @property
    def definitions(self):
        return self.registry.getstate()

    @contextmanager
    def remote_table_scope(self, ident: str):
        self.remote_table_stack.append(ident)
        try:
            yield
        finally:
            self.remote_table_stack.pop()

    @property
    def current_remote_table(self) -> str | None:
        return self.remote_table_stack[-1] if self.remote_table_stack else None

    def translate_from_yaml(self, yaml_dict: dict) -> Any:
        return self.from_yaml(yaml_dict, self)

    def translate_to_yaml(self, op: Any) -> dict:
        return self.to_yaml(op, self)

    def register_schema(self, schema: Schema) -> dict:
        # dtypes inside a schema are written inline (context=None), not as refs
        return self.registry.register_schema(
            freeze({name: self.to_yaml(dtype, None) for name, dtype in schema.items()})
        )

    def register(self, which, op, frozen=None):
        match which:
            case RegistryEnum.dtypes:
                return self.registry.register_dtype(op, frozen)
            case RegistryEnum.nodes:
                return self.registry.register_node(op, frozen)
            case RegistryEnum.schemas:
                return self.register_schema(op)
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
