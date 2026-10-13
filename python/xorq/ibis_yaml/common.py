from __future__ import annotations

import base64
import functools
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any

import cloudpickle
import toolz
from attr import (
    field,
    frozen,
)
from attr.validators import instance_of

import xorq.expr.datatypes as dt
import xorq.expr.udf as udf
import xorq.vendor.ibis.expr.operations as ops
from xorq.common.utils.content_hash import content_hash
from xorq.common.utils.dasher import tokenize
from xorq.common.utils.graph_utils import postorder
from xorq.expr.relations import Read, RemoteTable
from xorq.ibis_yaml.config import config
from xorq.ibis_yaml.enums import NodeKey, ReadKwarg, RefEnum, RegistryEnum
from xorq.ibis_yaml.utils import freeze
from xorq.vendor.ibis.common.collections import FrozenOrderedDict
from xorq.vendor.ibis.common.graph import Node
from xorq.vendor.ibis.expr.schema import Schema
from xorq.vendor.ibis.expr.types import Expr


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
        except KeyError as err:
            raise ValueError(
                f"ref {ref} not found in definitions for which={which}"
            ) from err


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
        return translate_from_yaml(yaml_dict, self)

    def translate_to_yaml(self, op: Any) -> dict:
        return safe_translate_to_yaml(op, self)

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


@functools.lru_cache(maxsize=None, typed=True)
@functools.singledispatch
def translate_from_yaml(yaml_dict: dict, context: TranslationContext) -> Any:
    match yaml_dict:
        case None:
            return None
        case bool() | int() | float() | str():
            return yaml_dict
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


@wraps(translate_to_yaml)
def safe_translate_to_yaml(*args, **kwargs):
    try:
        return translate_to_yaml(*args, **kwargs)
    except Exception:
        return translate_to_yaml.__wrapped__(*args, **kwargs)


def _nested_nodes(value: Any) -> Iterator[Node]:
    """Yield the nodes in *value*: itself, an Expr's op, or a container's members.

    Contract
    --------
    Any other value (a scalar, a set, a dtype) yields nothing; no error.
    """
    match value:
        case Node():
            yield value
        case Expr():
            yield value.op()
        case Mapping():
            yield from (n for v in value.values() for n in _nested_nodes(v))
        case tuple() | list() | frozenset():
            yield from (n for v in value for n in _nested_nodes(v))
        case _:
            return


def _translated_values(node: Node) -> Iterator[Any]:
    """Yield the argument values *node*'s handler translates, in its order.

    Contract
    --------
    Raises AttributeError if *node* is not a ``Node`` (reads ``__args__``) and
    KeyError if an ``ExprScalarUDF`` has no ``computed_kwargs_expr`` in its
    ``__config__`` (``make_pandas_expr_udf`` always sets it).
    """
    match node:
        case RemoteTable():
            # remote_expr is translated inside remote_table_scope by the warm-up
            yield from (
                arg
                for name, arg in zip(node.__argnames__, node.__args__)
                if name != "remote_expr"
            )
        case udf.ExprScalarUDF():
            # the handler also serializes __config__, home of computed_kwargs_expr
            yield from node.__args__
            yield node.computed_kwargs_expr
        case ops.InSubquery():
            # the handler translates needle before rel
            yield node.needle
            yield node.rel
        case ops.WindowFunction():
            # the handler translates the keys before the frame bounds
            yield node.func
            yield node.group_by
            yield node.order_by
            yield node.start
            yield node.end
        case _:
            yield from node.__args__


def serialization_children(node: Node) -> tuple[Node, ...]:
    """Return every node *node*'s handler may ask the cache for.

    Returns
    -------
    tuple[Node, ...]
        In argument order; a node referenced twice appears two times.

    Contract
    --------
    Preconditions and errors are those of :func:`_translated_values`.
    """
    return tuple(n for value in _translated_values(node) for n in _nested_nodes(value))


def warm_translate_to_yaml(
    op: Node, context: TranslationContext, warmed: set[Node] | None = None
) -> None:
    """Translate every node under *op* children-first, into the cache.

    The handlers recurse into their inputs; with the inputs cached, the root
    translates one level deep however long the chain is. Post-order keeps the
    recursive walk's registration order (expr.yaml bytes, first-registered
    ref wins).

    Parameters
    ----------
    op
        Root of the graph to warm.
    context
        The translation context the root will be translated with.
    warmed
        Nodes already translated. Mutated: every node this call translates is
        added, and the scoped RemoteTable passes share it so a subtree reached
        through two RemoteTables is walked once, not once per path.

    Contract
    --------
    Raises whatever a handler raises (``NotImplementedError`` for a node type
    with no translation rule). Each level of RemoteTable nesting costs one
    nested call, so a few hundred nested ``into_backend`` boundaries still
    raise ``RecursionError``, as on ``main``.
    """
    warmed = set() if warmed is None else warmed

    def children(node: Node) -> tuple[Node, ...]:
        return () if node in warmed else serialization_children(node)

    for node in postorder(op, children=children):
        if node in warmed:
            continue
        if isinstance(node, RemoteTable):
            with context.remote_table_scope(node.name):
                warm_translate_to_yaml(node.remote_expr.op(), context, warmed)
        context.translate_to_yaml(node)
        warmed.add(node)


@functools.singledispatch
def _translate_type(dtype: dt.DataType) -> dict:
    return freeze({"name": type(dtype).__name__, "nullable": dtype.nullable})
