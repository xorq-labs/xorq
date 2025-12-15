from __future__ import annotations

import datetime
import decimal
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import toolz

import xorq.expr.datatypes as dt
import xorq.vendor.ibis as ibis
import xorq.vendor.ibis.expr.datashape as ds
import xorq.vendor.ibis.expr.operations as ops
import xorq.vendor.ibis.expr.operations.temporal as tm
import xorq.vendor.ibis.expr.types as ir
from xorq.expr.relations import (
    CachedNode,
    Read,
    RemoteTable,
    Tag,
    into_backend,
)
from xorq.ibis_yaml.common import (
    TranslationContext,
    _translate_type,
    deserialize_callable,
    register_from_yaml_handler,
    serialize_callable,
    translate_to_yaml,
)
from xorq.ibis_yaml.udf import _scalar_udf_from_yaml, _scalar_udf_to_yaml  # noqa: F401
from xorq.ibis_yaml.utils import (
    freeze,
    load_cache_from_yaml,
    translate_cache,
)
from xorq.vendor.ibis.common.collections import FrozenDict, FrozenOrderedDict
from xorq.vendor.ibis.expr.datashape import Columnar
from xorq.vendor.ibis.expr.operations.relations import Namespace
from xorq.vendor.ibis.util import normalize_filenames


def should_register_node(node_dict):
    return "parent" in node_dict and isinstance(node_dict["parent"], dict)


@translate_to_yaml.register(ops.Node)
def _object_to_yaml(obj: ops.Node, context: Any) -> dict:
    return freeze(
        {"op": obj.__class__.__name__}
        | {
            name: context.translate_to_yaml(arg)
            for name, arg in zip(obj.argnames, obj.args)
        }
        | {
            name: context.translate_to_yaml(getattr(obj, attribute))
            for name, attribute in (("type", "dtype"),)
            if hasattr(obj, attribute)
        }
    )


@translate_to_yaml.register(Callable)
def _callable_to_yaml(obj: Callable, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "Callable",
            "pickled_fn": serialize_callable(obj),
        }
    )


@register_from_yaml_handler("Callable")
def _callable_from_yaml(yaml_dict: dict, compiler: Any) -> Callable:
    return deserialize_callable(yaml_dict["pickled_fn"])


@translate_to_yaml.register(tuple)
def _tuple_to_yaml(tpl: tuple, context: TranslationContext) -> dict:
    return freeze(
        {"op": "tuple", "values": [context.translate_to_yaml(value) for value in tpl]}
    )


@register_from_yaml_handler("tuple")
def _tuple_from_yaml(yaml_dict: dict, context: TranslationContext) -> Any:
    return tuple(context.translate_from_yaml(value) for value in yaml_dict["values"])


@translate_to_yaml.register(tm.IntervalUnit)
def _interval_unit_to_yaml(
    interval_unit: tm.IntervalUnit, context: TranslationContext
) -> dict:
    if interval_unit.is_date():
        unit_name = "DateUnit"
    elif interval_unit.is_time():
        unit_name = "TimeUnit"
    else:
        unit_name = "IntervalUnit"

    return freeze(
        {"op": "IntervalUnit", "name": unit_name, "value": interval_unit.value}
    )


@register_from_yaml_handler("IntervalUnit")
def _interval_unit_from_yaml(yaml_dict: dict, context: TranslationContext) -> any:
    return tm.IntervalUnit(yaml_dict["value"])


@translate_to_yaml.register(bool)
def _bool_to_yaml(value: bool, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "bool",
            "value": value,
        }
    )


@register_from_yaml_handler("bool")
def _bool_from_yaml(yaml_dict: dict, context: TranslationContext) -> bool:
    return yaml_dict["value"]


@translate_to_yaml.register(int)
def _int_to_yaml(dct: int, context: TranslationContext) -> dict:
    return freeze({"op": "int", "value": str(dct)})


@register_from_yaml_handler("int")
def _int_from_yaml(yaml_dict: dict, context: TranslationContext) -> any:
    return int(yaml_dict["value"])


@translate_to_yaml.register(float)
def _float_to_yaml(dct: float, context: TranslationContext) -> dict:
    return freeze({"op": "float", "value": str(dct)})


@register_from_yaml_handler("float")
def _float_from_yaml(yaml_dict: dict, context: TranslationContext) -> any:
    return float(yaml_dict["value"])


@_translate_type.register(dt.Timestamp)
def _translate_timestamp_type(dtype: dt.Timestamp) -> dict:
    base = {"name": "Timestamp", "nullable": dtype.nullable}
    if dtype.timezone is not None:
        base["timezone"] = dtype.timezone
    return freeze(base)


@_translate_type.register(dt.Decimal)
def _translate_decimal_type(dtype: dt.Decimal) -> dict:
    base = {"name": "Decimal", "nullable": dtype.nullable}
    if dtype.precision is not None:
        base["precision"] = dtype.precision
    if dtype.scale is not None:
        base["scale"] = dtype.scale
    return freeze(base)


@_translate_type.register(dt.Array)
def _translate_array_type(dtype: dt.Array) -> dict:
    return freeze(
        {
            "name": "Array",
            "value_type": translate_to_yaml(dtype.value_type, None),
            "nullable": dtype.nullable,
        }
    )


@_translate_type.register(dt.Map)
def _translate_map_type(dtype: dt.Map) -> dict:
    return freeze(
        {
            "name": "Map",
            "key_type": translate_to_yaml(dtype.key_type, None),
            "value_type": translate_to_yaml(dtype.value_type, None),
            "nullable": dtype.nullable,
        }
    )


@_translate_type.register(dt.Interval)
def _translate_type_interval(dtype: dt.Interval) -> dict:
    return freeze(
        {
            "name": "Interval",
            "unit": _translate_temporal_unit(dtype.unit),
            "nullable": dtype.nullable,
        }
    )


@_translate_type.register(dt.Struct)
def _translate_struct_type(dtype: dt.Struct) -> dict:
    return freeze(
        {
            "name": "Struct",
            "fields": {
                name: _translate_type(field_type)
                for name, field_type in zip(dtype.names, dtype.types)
            },
            "nullable": dtype.nullable,
        }
    )


def _translate_temporal_unit(unit: tm.IntervalUnit) -> dict:
    if unit.is_date():
        unit_name = "DateUnit"
    elif unit.is_time():
        unit_name = "TimeUnit"
    else:
        unit_name = "IntervalUnit"
    return freeze({"name": unit_name, "value": unit.value})


def _translate_literal_value(value: Any, dtype: dt.DataType) -> Any:
    if value is None:
        return None
    elif isinstance(value, (bool, int, float, str)):
        return value
    elif isinstance(value, decimal.Decimal):
        return str(value)
    elif isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    elif isinstance(value, list):
        return [_translate_literal_value(v, dtype.value_type) for v in value]
    elif isinstance(value, dict):
        if isinstance(dtype, dt.Struct):
            raise NotImplementedError
        else:
            return {
                _translate_literal_value(k, dtype.key_type): _translate_literal_value(
                    v, dtype.value_type
                )
                for k, v in value.items()
            }
    else:
        return value


@translate_to_yaml.register(dt.DataType)
def _datatype_to_yaml(dtype: dt.DataType, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "DataType",
            "type": type(dtype).__name__,
        }
        | {
            argname: context.translate_to_yaml(arg)
            if context is not None
            else translate_to_yaml(arg, context)
            for argname, arg in zip(dtype.argnames, dtype.args)
        }
    )


@register_from_yaml_handler("DataType")
def _datatype_from_yaml(yaml_dict: dict, context: TranslationContext) -> any:
    typ = getattr(dt, yaml_dict["type"])
    dct = toolz.dissoc(yaml_dict, "op", "type")
    return typ(
        **{
            key: context.translate_from_yaml(value) if value is not None else None
            for key, value in dct.items()
        }
    )


@translate_to_yaml.register(ir.Expr)
def _expr_to_yaml(expr: ir.Expr, context: any) -> dict:
    return context.translate_to_yaml(expr.op())


@translate_to_yaml.register(ops.WindowFunction)
def _window_function_to_yaml(
    op: ops.WindowFunction, context: TranslationContext
) -> dict:
    result = {
        "op": "WindowFunction",
        "args": [context.translate_to_yaml(op.func)],
        "type": context.translate_to_yaml(op.dtype),
    }

    if op.group_by:
        result["group_by"] = [context.translate_to_yaml(expr) for expr in op.group_by]

    if op.order_by:
        result["order_by"] = [context.translate_to_yaml(expr) for expr in op.order_by]

    if op.start is not None:
        result["start"] = (
            context.translate_to_yaml(op.start.value)["value"]
            if isinstance(op.start, ops.WindowBoundary)
            else op.start
        )

    if op.end is not None:
        result["end"] = (
            context.translate_to_yaml(op.end.value)["value"]
            if isinstance(op.end, ops.WindowBoundary)
            else op.end
        )

    return freeze(result)


@register_from_yaml_handler("WindowFunction")
def _window_function_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    func = context.translate_from_yaml(yaml_dict["args"][0])
    group_by = [context.translate_from_yaml(g) for g in yaml_dict.get("group_by", [])]
    order_by = [context.translate_from_yaml(o) for o in yaml_dict.get("order_by", [])]
    start = ibis.literal(yaml_dict["start"]) if "start" in yaml_dict else None
    end = ibis.literal(yaml_dict["end"]) if "end" in yaml_dict else None
    window = ibis.window(
        group_by=group_by, order_by=order_by, preceding=start, following=end
    )
    return func.over(window)


@translate_to_yaml.register(ops.WindowBoundary)
def _window_boundary_to_yaml(
    op: ops.WindowBoundary, context: TranslationContext
) -> dict:
    return freeze(
        {
            "op": "WindowBoundary",
            "value": context.translate_to_yaml(op.value),
            "preceding": op.preceding,
            "type": context.translate_to_yaml(op.dtype),
        }
    )


@register_from_yaml_handler("WindowBoundary")
def _window_boundary_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    value = context.translate_from_yaml(yaml_dict["value"])
    return ops.WindowBoundary(value, preceding=yaml_dict["preceding"])


@translate_to_yaml.register(ops.StructField)
def _struct_field_to_yaml(op: ops.Node, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": type(op).__name__,
            "args": [context.translate_to_yaml(arg) for arg in op.args],
        }
    )


@translate_to_yaml.register(ops.UnboundTable)
def _unbound_table_to_yaml(op: ops.UnboundTable, context: TranslationContext) -> dict:
    schema_id = context.schema_registry.register_schema(op.schema)
    namespace_dict = freeze(
        {
            "catalog": op.namespace.catalog,
            "database": op.namespace.database,
        }
    )
    return freeze(
        {
            "op": "UnboundTable",
            "name": op.name,
            "schema_ref": schema_id,
            "namespace": namespace_dict,
        }
    )


@register_from_yaml_handler("UnboundTable")
def _unbound_table_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    table_name = yaml_dict["name"]

    schema_ref = yaml_dict["schema_ref"]
    try:
        schema_def = context.definitions["schemas"][schema_ref]
    except KeyError:
        raise ValueError(f"Schema {schema_ref} not found in definitions")
    namespace_dict = yaml_dict.get("namespace", {})
    catalog = namespace_dict.get("catalog")
    database = namespace_dict.get("database")
    schema = {
        name: context.translate_from_yaml(dtype_yaml)
        for name, dtype_yaml in schema_def.items()
    }
    # TODO: use UnboundTable node to construct instead of builder API
    return ibis.table(schema, name=table_name, catalog=catalog, database=database)


@translate_to_yaml.register(ops.DatabaseTable)
def _database_table_to_yaml(op: ops.DatabaseTable, context: TranslationContext) -> dict:
    profile_name = op.source._profile.hash_name
    schema_id = context.schema_registry.register_schema(op.schema)
    namespace_dict = freeze(
        {
            "catalog": op.namespace.catalog,
            "database": op.namespace.database,
        }
    )

    node_dict = freeze(
        {
            "op": "DatabaseTable",
            "table": op.name,
            "schema_ref": schema_id,
            "profile": profile_name,
            "namespace": namespace_dict,
        }
    )

    if should_register_node(node_dict):
        node_hash = context.schema_registry.register_node(node_dict)
        return freeze({"node_ref": node_hash})

    return node_dict


@register_from_yaml_handler("DatabaseTable")
def database_table_from_yaml(yaml_dict: dict, context: TranslationContext) -> ibis.Expr:
    profile_name = yaml_dict.get("profile")
    table_name = yaml_dict.get("table")
    namespace_dict = yaml_dict.get("namespace", {})
    catalog = namespace_dict.get("catalog")
    database = namespace_dict.get("database")
    # we should validate that schema is the same
    schema_ref = yaml_dict.get("schema_ref")
    schema_def = context.definitions["schemas"][schema_ref]
    fields = []
    for name, dtype_yaml in schema_def.items():
        dtype = context.translate_from_yaml(dtype_yaml)
        fields.append((name, dtype))
    schema = ibis.Schema.from_tuples(fields)

    try:
        con = context.profiles[profile_name]
    except KeyError:
        raise ValueError(f"Profile {profile_name!r} not found in context.profiles")
    return ops.DatabaseTable(
        schema=schema,
        source=con,
        name=table_name,
        namespace=Namespace(catalog=catalog, database=database),
    ).to_expr()


@translate_to_yaml.register(CachedNode)
def _cached_node_to_yaml(op: CachedNode, context: any) -> dict:
    schema_id = context.schema_registry.register_schema(op.schema)
    # source should be called profile_name

    return freeze(
        {
            "op": "CachedNode",
            "schema_ref": schema_id,
            "name": op.name,
            "parent": context.translate_to_yaml(op.parent),
            "source": op.source._profile.hash_name,
            "cache": translate_cache(op.cache, context),
        }
    )


@register_from_yaml_handler("CachedNode")
def _cached_node_from_yaml(yaml_dict: dict, context: any) -> ibis.Expr:
    schema_ref = yaml_dict["schema_ref"]
    try:
        schema_def = context.definitions["schemas"][schema_ref]
    except KeyError:
        raise ValueError(f"Schema {schema_ref} not found in definitions")

    schema = {
        name: context.translate_from_yaml(dtype_yaml)
        for name, dtype_yaml in schema_def.items()
    }

    name = yaml_dict["name"]

    parent_expr = context.translate_from_yaml(yaml_dict["parent"])
    profile_name = yaml_dict.get("source")
    try:
        source = context.profiles[profile_name]
    except KeyError:
        raise ValueError(f"Profile {profile_name!r} not found in context.profiles")
    cache = load_cache_from_yaml(yaml_dict["cache"], context)

    op = CachedNode(
        schema=schema,
        name=name,
        parent=parent_expr,
        source=source,
        cache=cache,
    )
    return op.to_expr()


@translate_to_yaml.register(RemoteTable)
def _remotetable_to_yaml(op: RemoteTable, context: TranslationContext) -> dict:
    profile_name = op.source._profile.hash_name
    remote_expr_yaml = context.translate_to_yaml(op.remote_expr)
    schema_id = context.schema_registry.register_schema(op.schema)
    # TODO: change profile to profile_name
    return freeze(
        {
            "op": "RemoteTable",
            "table": op.name,
            "schema_ref": schema_id,
            "profile": profile_name,
            "remote_expr": remote_expr_yaml,
        }
    )


@register_from_yaml_handler("RemoteTable")
def _remotetable_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    profile_name = yaml_dict.get("profile")
    table_name = yaml_dict.get("table")
    remote_expr_yaml = yaml_dict.get("remote_expr")
    if profile_name is None:
        raise ValueError(
            "Missing keys in RemoteTable YAML; ensure 'profile_name' are present."
        )
    try:
        con = context.profiles[profile_name]
    except KeyError:
        raise ValueError(f"Profile {profile_name!r} not found in context.profiles")

    remote_expr = context.translate_from_yaml(remote_expr_yaml)

    remote_table_expr = into_backend(remote_expr, con, table_name)
    return remote_table_expr


def warn_on_local_path(items: dict) -> None:
    from urllib.parse import urlparse

    def is_local_path(any: str | Path) -> bool:
        parsed = urlparse(any)
        return not parsed.scheme or parsed.scheme in ("file",)

    if path := next(
        (v for k, v in dict(items).items() if k in ("path", "source")), None
    ):
        f = toolz.excepts((ValueError, AttributeError), is_local_path)
        paths = normalize_filenames(path)
        if any(map(f, paths)):
            warnings.warn(
                "The Read op path is using a local filesystem path, running the build may not work in other environments."
            )


@translate_to_yaml.register(Read)
def _read_to_yaml(op: Read, context: TranslationContext) -> dict:
    schema_id = context.schema_registry.register_schema(op.schema)
    profile_hash_name = (
        op.source._profile.hash_name if hasattr(op.source, "_profile") else None
    )

    warn_on_local_path(op.read_kwargs)

    return freeze(
        {
            "op": "Read",
            "method_name": op.method_name,
            "name": op.name,
            "schema_ref": schema_id,
            "profile": profile_hash_name,
            "read_kwargs": freeze(op.read_kwargs if op.read_kwargs else {}),
            "normalize_method": serialize_callable(op.normalize_method),
        }
    )


@register_from_yaml_handler("Read")
def _read_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    schema_ref = yaml_dict["schema_ref"]
    schema_def = context.definitions["schemas"][schema_ref]
    schema = {
        name: context.translate_from_yaml(dtype_yaml)
        for name, dtype_yaml in schema_def.items()
    }

    source = context.profiles[yaml_dict["profile"]]
    read_kwargs = tuple(
        (k, ibis.schema(v)) if k == "schema" else (k, v)
        for k, v in yaml_dict.get("read_kwargs", ())
    )
    read_op = Read(
        method_name=yaml_dict["method_name"],
        name=yaml_dict["name"],
        schema=schema,
        source=source,
        read_kwargs=read_kwargs,
        normalize_method=deserialize_callable(yaml_dict["normalize_method"]),
    )

    return read_op.to_expr()


@translate_to_yaml.register(str)
def _str_to_yaml(string: str, context: TranslationContext) -> str:
    return {
        "op": "str",
        "value": string,
    }


@register_from_yaml_handler("str")
def _str_from_yaml(yaml_dict: dict, context: TranslationContext) -> str:
    return yaml_dict["value"]


@translate_to_yaml.register(ops.Alias)
def _alias_to_yaml(op: ops.Alias, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "Alias",
            "type": context.translate_to_yaml(op.dtype),
            "args": [context.translate_to_yaml(arg) for arg in op.args],
        }
    )


@register_from_yaml_handler("Alias")
def _alias_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    (arg, name) = (context.translate_from_yaml(arg) for arg in yaml_dict["args"])
    return arg.name(name)


@translate_to_yaml.register(ops.Round)
def _round_to_yaml(op: ops.Round, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": op.__class__.__name__,
            "type": context.translate_to_yaml(op.dtype),
            "args": [context.translate_to_yaml(arg) for arg in op.args],
        }
    )


@register_from_yaml_handler("Round")
def _round_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    (arg, digits) = (
        None if arg is None else context.translate_from_yaml(arg)
        for arg in yaml_dict["args"]
    )
    return arg.round(digits)


@translate_to_yaml.register(type(None))
def _none_to_yaml(value: None, context: TranslationContext) -> None:
    return None


@translate_to_yaml.register(ops.Literal)
def _literal_to_yaml(op: ops.Literal, context: TranslationContext) -> dict:
    value = _translate_literal_value(op.value, op.dtype)
    return freeze(
        {"op": "Literal", "value": value, "type": context.translate_to_yaml(op.dtype)}
    )


@register_from_yaml_handler("Literal")
def _literal_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    value = yaml_dict["value"]
    dtype = context.translate_from_yaml(yaml_dict["type"])
    return ibis.literal(value, type=dtype)


@translate_to_yaml.register(ops.Lag)
def _lag_to_yaml(op: ops.Lag, context: TranslationContext) -> dict:
    result = {
        "op": "Lag",
        "arg": context.translate_to_yaml(op.arg),
        "type": context.translate_to_yaml(op.dtype),
    }

    if op.offset is not None:
        result["offset"] = context.translate_to_yaml(op.offset)

    if op.default is not None:
        result["default"] = context.translate_to_yaml(op.default)

    node_dict = freeze(result)

    return node_dict


@register_from_yaml_handler("Lag")
def _lag_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    arg = context.translate_from_yaml(yaml_dict["arg"])

    offset = None
    if "offset" in yaml_dict:
        offset = context.translate_from_yaml(yaml_dict["offset"])

    default = None
    if "default" in yaml_dict:
        default = context.translate_from_yaml(yaml_dict["default"])

    return arg.lag(offset, default)


@translate_to_yaml.register(ops.Intersection)
@translate_to_yaml.register(ops.Union)
@translate_to_yaml.register(ops.Difference)
def _set_op_to_yaml(op: ops.Set, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": type(op).__name__,
            "left": context.translate_to_yaml(op.left),
            "right": context.translate_to_yaml(op.right),
            "distinct": context.translate_to_yaml(op.distinct),
            "values": {
                name: context.translate_to_yaml(val) for name, val in op.values.items()
            },
        }
    )


set_ops_map = {
    "Intersection": ops.Intersection,
    "Union": ops.Union,
    "Difference": ops.Difference,
}


@register_from_yaml_handler("Intersection", "Union", "Difference")
def _set_op(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    left = context.translate_from_yaml(yaml_dict["left"])
    right = context.translate_from_yaml(yaml_dict["right"])
    distinct = context.translate_from_yaml(yaml_dict["distinct"])
    set_op = set_ops_map[yaml_dict["op"]]
    return set_op(left, right, distinct=distinct).to_expr()


@translate_to_yaml.register(ops.Binary)
def _binary_op_to_yaml(op: ops.Binary, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": type(op).__name__,
            "left": context.translate_to_yaml(op.left),
            "right": context.translate_to_yaml(op.right),
            "type": context.translate_to_yaml(op.dtype),
        }
    )


@register_from_yaml_handler("Binary")
def _binary_op_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    left, right = [
        context.translate_from_yaml(yaml_dict[key]) for key in ("left", "right")
    ]
    op_name = yaml_dict["op"].lower()
    return getattr(left, op_name)(right)


@translate_to_yaml.register(ops.Filter)
def _filter_to_yaml(op: ops.Filter, context: TranslationContext) -> dict:
    node_dict = freeze(
        {
            "op": "Filter",
            "parent": context.translate_to_yaml(op.parent),
            "predicates": [context.translate_to_yaml(pred) for pred in op.predicates],
        }
    )

    if should_register_node(node_dict):
        node_hash = context.schema_registry.register_node(node_dict)
        return freeze({"node_ref": node_hash})

    return node_dict


@register_from_yaml_handler("Filter")
def _filter_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    parent = context.translate_from_yaml(yaml_dict["parent"])
    predicates = [context.translate_from_yaml(pred) for pred in yaml_dict["predicates"]]
    filter_op = ops.Filter(parent, predicates)
    return filter_op.to_expr()


@translate_to_yaml.register(ops.Project)
def _project_to_yaml(op: ops.Project, context: TranslationContext) -> dict:
    node_dict = {
        "op": "Project",
        "parent": context.translate_to_yaml(op.parent),
        "values": {
            name: context.translate_to_yaml(val) for name, val in op.values.items()
        },
    }

    if should_register_node(node_dict):
        node_hash = context.schema_registry.register_node(freeze(node_dict))
        return freeze({"node_ref": node_hash})

    return freeze(node_dict)


@register_from_yaml_handler("Project")
def _project_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    parent = context.translate_from_yaml(yaml_dict["parent"])

    values_dict = yaml_dict.get("values", {})

    values = {
        name: context.translate_from_yaml(val) for name, val in values_dict.items()
    }

    projected = parent.projection(values)
    return projected


@translate_to_yaml.register(ops.Aggregate)
def _aggregate_to_yaml(op: ops.Aggregate, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "Aggregate",
            "parent": context.translate_to_yaml(op.parent),
            "by": {
                name: context.translate_to_yaml(group)
                for name, group in op.groups.items()
            },
            "metrics": {
                name: context.translate_to_yaml(metric)
                for name, metric in op.metrics.items()
            },
        }
    )


@register_from_yaml_handler("Aggregate")
def _aggregate_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    parent = context.translate_from_yaml(yaml_dict["parent"])
    by = yaml_dict.get("by", {})
    if isinstance(by, dict):
        group_mapping = {
            name: context.translate_from_yaml(expr) for name, expr in by.items()
        }
    else:
        exprs = [context.translate_from_yaml(expr) for expr in by]
        group_mapping = {expr.get_name(): expr for expr in exprs}

    metrics = {
        name: context.translate_from_yaml(metric)
        for name, metric in yaml_dict.get("metrics", {}).items()
    }

    if group_mapping:
        result = parent.group_by(list(group_mapping.values())).aggregate(metrics)
        rename_map = {
            alias: expr.get_name()
            for alias, expr in group_mapping.items()
            if alias != expr.get_name()
        }
        if rename_map:
            result = result.rename(rename_map)
    else:
        result = parent.aggregate(metrics)

    return result


@translate_to_yaml.register(ops.JoinChain)
def _join_to_yaml(op: ops.JoinChain, context: TranslationContext) -> dict:
    node_dict = {
        "op": "JoinChain",
        "first": context.translate_to_yaml(op.first),
        "rest": [
            {
                "how": link.how,
                "table": context.translate_to_yaml(link.table),
                "predicates": [
                    context.translate_to_yaml(pred) for pred in link.predicates
                ],
            }
            for link in op.rest
        ],
        "values": {
            name: context.translate_to_yaml(val) for name, val in op.values.items()
        },
    }

    if should_register_node(node_dict):
        node_hash = context.schema_registry.register_node(freeze(node_dict))
        return freeze({"node_ref": node_hash})
    return freeze(node_dict)


@register_from_yaml_handler("JoinChain")
def _join_chain_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    first = context.translate_from_yaml(yaml_dict["first"])
    result = first

    for join in yaml_dict["rest"]:
        table = context.translate_from_yaml(join["table"])
        predicates = [context.translate_from_yaml(pred) for pred in join["predicates"]]
        result = result.join(table, predicates, how=join["how"])

    values = {
        name: context.translate_from_yaml(val)
        for name, val in yaml_dict["values"].items()
    }
    result = result.select(values)
    return result


@translate_to_yaml.register(ops.Limit)
def _limit_to_yaml(op: ops.Limit, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "Limit",
            "parent": context.translate_to_yaml(op.parent),
            "n": op.n,
            "offset": op.offset,
        }
    )


@register_from_yaml_handler("Limit")
def _limit_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    parent = context.translate_from_yaml(yaml_dict["parent"])
    return parent.limit(yaml_dict["n"], offset=yaml_dict["offset"])


@translate_to_yaml.register(ops.ScalarSubquery)
def _scalar_subquery_to_yaml(
    op: ops.ScalarSubquery, context: TranslationContext
) -> dict:
    return freeze(
        {
            "op": "ScalarSubquery",
            "args": [context.translate_to_yaml(arg) for arg in op.args],
            "type": context.translate_to_yaml(op.dtype),
        }
    )


@register_from_yaml_handler("ScalarSubquery")
def _scalar_subquery_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    subquery = context.translate_from_yaml(yaml_dict["args"][0])
    return ops.ScalarSubquery(subquery).to_expr()


@translate_to_yaml.register(ops.ExistsSubquery)
def _exists_subquery_to_yaml(
    op: ops.ExistsSubquery, context: TranslationContext
) -> dict:
    return freeze(
        {
            "op": "ExistsSubquery",
            "rel": context.translate_to_yaml(op.rel),
            "type": context.translate_to_yaml(op.dtype),
        }
    )


@register_from_yaml_handler("ExistsSubquery")
def _exists_subquery_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    rel = context.translate_from_yaml(yaml_dict["rel"])
    return ops.ExistsSubquery(rel).to_expr()


@translate_to_yaml.register(ops.InSubquery)
def _in_subquery_to_yaml(op: ops.InSubquery, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "InSubquery",
            "needle": context.translate_to_yaml(op.needle),
            "haystack": context.translate_to_yaml(op.rel),
            "type": context.translate_to_yaml(op.dtype),
        }
    )


@register_from_yaml_handler("InSubquery")
def _in_subquery_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    needle = context.translate_from_yaml(yaml_dict["needle"])
    haystack = context.translate_from_yaml(yaml_dict["haystack"])
    return ops.InSubquery(haystack, needle).to_expr()


@translate_to_yaml.register(ops.Field)
def _field_to_yaml(op: ops.Field, context: TranslationContext) -> dict:
    result = {
        "op": "Field",
        "name": op.name,
        "relation": context.translate_to_yaml(op.rel),
        "type": context.translate_to_yaml(op.dtype),
    }

    if op.args and len(op.args) >= 2 and isinstance(op.args[1], str):
        underlying_name = op.args[1]
        if underlying_name != op.name:
            result["original_name"] = underlying_name

    node_dict = freeze(result)

    if hasattr(context.schema_registry, "register_node"):
        node_hash = context.schema_registry.register_node(node_dict)
        return freeze({"node_ref": node_hash})

    return node_dict


@register_from_yaml_handler("Field")
def field_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    relation = context.translate_from_yaml(yaml_dict["relation"])
    target_name = yaml_dict["name"]
    source_name = yaml_dict.get("original_name", target_name)
    field = relation[source_name]
    if target_name != source_name:
        field = field.name(target_name)

    return freeze(field)


@translate_to_yaml.register(ops.InValues)
def _in_values_to_yaml(op: ops.InValues, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "InValues",
            "args": [
                context.translate_to_yaml(op.value),
                *[context.translate_to_yaml(opt) for opt in op.options],
            ],
            "type": context.translate_to_yaml(op.dtype),
        }
    )


@register_from_yaml_handler("InValues")
def _in_values_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    value = context.translate_from_yaml(yaml_dict["args"][0])
    options = tuple(context.translate_from_yaml(opt) for opt in yaml_dict["args"][1:])
    return ops.InValues(value, options).to_expr()


@translate_to_yaml.register(ops.SimpleCase)
def _simple_case_to_yaml(op: ops.SimpleCase, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "SimpleCase",
            "base": context.translate_to_yaml(op.base),
            "cases": [context.translate_to_yaml(case) for case in op.cases],
            "results": [context.translate_to_yaml(result) for result in op.results],
            "default": context.translate_to_yaml(op.default),
            "type": context.translate_to_yaml(op.dtype),
        }
    )


@register_from_yaml_handler("SimpleCase")
def _simple_case_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    base = context.translate_from_yaml(yaml_dict["base"])
    cases = tuple(context.translate_from_yaml(case) for case in yaml_dict["cases"])
    results = tuple(
        context.translate_from_yaml(result) for result in yaml_dict["results"]
    )
    default = context.translate_from_yaml(yaml_dict["default"])
    return ops.SimpleCase(base, cases, results, default).to_expr()


@translate_to_yaml.register(ops.IfElse)
def _if_else_to_yaml(op: ops.IfElse, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "IfElse",
            "bool_expr": context.translate_to_yaml(op.bool_expr),
            "true_expr": context.translate_to_yaml(op.true_expr),
            "false_null_expr": context.translate_to_yaml(op.false_null_expr),
            "type": context.translate_to_yaml(op.dtype),
        }
    )


@register_from_yaml_handler("IfElse")
def _if_else_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    bool_expr = context.translate_from_yaml(yaml_dict["bool_expr"])
    true_expr = context.translate_from_yaml(yaml_dict["true_expr"])
    false_null_expr = context.translate_from_yaml(yaml_dict["false_null_expr"])
    return ops.IfElse(bool_expr, true_expr, false_null_expr).to_expr()


@translate_to_yaml.register(ops.Coalesce)
def _coalesce_to_yaml(op: ops.Coalesce, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "Coalesce",
            "args": [context.translate_to_yaml(arg) for arg in op.arg],
            "type": context.translate_to_yaml(op.dtype),
        }
    )


@register_from_yaml_handler("Coalesce")
def _coalesce_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    args = [context.translate_from_yaml(arg) for arg in yaml_dict["args"]]
    return ibis.coalesce(*args)


@translate_to_yaml.register(ops.RankBase)
def _rank_base_to_yaml(op: ops.RankBase, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": type(op).__name__,
        }
    )


@register_from_yaml_handler("MinRank")
def _min_rank_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    return ibis.rank()


@register_from_yaml_handler("RowNumber")
def _row_number_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    return ibis.row_number()


@translate_to_yaml.register(ops.SelfReference)
def _self_reference_to_yaml(op: ops.SelfReference, context: TranslationContext) -> dict:
    result = {"op": "SelfReference", "identifier": op.identifier}
    if op.args:
        result["args"] = [context.translate_to_yaml(op.args[0])]
    return freeze(result)


@register_from_yaml_handler("SelfReference")
def _self_reference_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    underlying = None
    if "args" in yaml_dict and yaml_dict["args"]:
        underlying = context.translate_from_yaml(yaml_dict["args"][0])
    else:
        if underlying is None:
            raise NotImplementedError("No relation available for SelfReference")

    identifier = yaml_dict.get("identifier", 0)
    ref = ops.SelfReference(underlying, identifier=identifier)

    return ref.to_expr()


@translate_to_yaml.register(ops.DropColumns)
def _drop_columns_to_yaml(op: ops.DropColumns, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "DropColumns",
            "parent": context.translate_to_yaml(op.parent),
            "columns_to_drop": list(op.columns_to_drop),
        }
    )


@register_from_yaml_handler("DropColumns")
def _drop_columns_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    parent = context.translate_from_yaml(yaml_dict["parent"])
    columns = frozenset(yaml_dict["columns_to_drop"])
    op = ops.DropColumns(parent, columns)
    return op.to_expr()


@translate_to_yaml.register(ops.TableUnnest)
def _table_unnest_to_yaml(op: ops.TableUnnest, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "TableUnnest",
            "parent": context.translate_to_yaml(op.parent),
            "column": context.translate_to_yaml(op.column),
            "column_name": context.translate_to_yaml(op.column_name),
            "offset": context.translate_to_yaml(op.offset)
            if op.offset is not None
            else None,
            "keep_empty": context.translate_to_yaml(op.keep_empty),
        }
    )


@register_from_yaml_handler("TableUnnest")
def _table_unnest_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    parent = context.translate_from_yaml(yaml_dict["parent"])
    column = context.translate_from_yaml(yaml_dict["column"])
    column_name = context.translate_from_yaml(yaml_dict["column_name"])
    offset = (
        context.translate_from_yaml(yaml_dict["offset"])
        if yaml_dict["offset"] is not None
        else None
    )
    keep_empty = context.translate_from_yaml(yaml_dict["keep_empty"])

    op = ops.TableUnnest(
        parent=parent,
        column=column,
        column_name=column_name,
        offset=offset,
        keep_empty=keep_empty,
    )
    return op.to_expr()


@translate_to_yaml.register(ops.SearchedCase)
def _searched_case_to_yaml(op: ops.SearchedCase, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "SearchedCase",
            "cases": [context.translate_to_yaml(case) for case in op.cases],
            "results": [context.translate_to_yaml(result) for result in op.results],
            "default": context.translate_to_yaml(op.default),
            "dtype": context.translate_to_yaml(op.dtype),
        }
    )


@register_from_yaml_handler("SearchedCase")
def _searched_case_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    cases = [context.translate_from_yaml(case) for case in yaml_dict["cases"]]
    results = [context.translate_from_yaml(result) for result in yaml_dict["results"]]
    default = context.translate_from_yaml(yaml_dict["default"])
    op = ops.SearchedCase(cases, results, default)
    return op.to_expr()


@register_from_yaml_handler("JoinReference")
def _join_reference_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    table_yaml = yaml_dict["parent"]
    return context.translate_from_yaml(table_yaml)


@translate_to_yaml.register(ops.FillNull)
def _fill_null_to_yaml(op: ops.FillNull, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "FillNull",
            "parent": context.translate_to_yaml(op.parent),
            "replacements": context.translate_to_yaml(op.replacements),
        }
    )


@register_from_yaml_handler("FillNull")
def _fill_null_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    parent = context.translate_from_yaml(yaml_dict["parent"])
    replacements = context.translate_from_yaml(yaml_dict["replacements"])
    return ops.FillNull(parent=parent, replacements=replacements).to_expr()


@translate_to_yaml.register(ops.NullIf)
def _nullif_to_yaml(op: ops.NullIf, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "NullIf",
            "arg": context.translate_to_yaml(op.arg),
            "null_if_expr": context.translate_to_yaml(op.null_if_expr),
        }
    )


@register_from_yaml_handler("NullIf")
def _nullif_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    arg = context.translate_from_yaml(yaml_dict["arg"])
    null_if_expr = context.translate_from_yaml(yaml_dict["null_if_expr"])
    return arg.nullif(null_if_expr)


@translate_to_yaml.register(ops.DropNull)
def _drop_null_to_yaml(op: ops.DropNull, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "DropNull",
            "parent": context.translate_to_yaml(op.parent),
            "how": context.translate_to_yaml(op.how) if op.how is not None else None,
            "subset": context.translate_to_yaml(op.subset)
            if op.subset is not None
            else None,
        }
    )


@register_from_yaml_handler("DropNull")
def _drop_null_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    return ops.DropNull(
        parent=context.translate_from_yaml(yaml_dict["parent"]),
        how=context.translate_from_yaml(yaml_dict["how"]),
        subset=context.translate_from_yaml(yaml_dict["subset"])
        if yaml_dict["subset"] is not None
        else None,
    ).to_expr()


@translate_to_yaml.register(ops.StringJoin)
def _string_join_to_yaml(op: ops.StringJoin, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "StringJoin",
            "args": [context.translate_to_yaml(arg) for arg in op.arg],
            "sep": context.translate_to_yaml(op.sep),
        }
    )


@register_from_yaml_handler("StringJoin")
def _string_join_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    args = tuple(context.translate_from_yaml(arg) for arg in yaml_dict["args"])
    sep = context.translate_from_yaml(yaml_dict["sep"])

    return ops.StringJoin(args, sep).to_expr()


@register_from_yaml_handler("StructField")
def _structfield_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    args = tuple(context.translate_from_yaml(arg) for arg in yaml_dict["args"])
    return ops.StructField(*args).to_expr()


def _type_from_yaml(yaml_dict: dict) -> dt.DataType:
    type_name = yaml_dict["name"]
    base_type = REVERSE_TYPE_REGISTRY.get(type_name)
    if base_type is None:
        raise ValueError(f"Unknown type: {type_name}")
    if callable(base_type) and not isinstance(base_type, dt.DataType):
        base_type = base_type(yaml_dict)
    elif (
        "nullable" in yaml_dict
        and isinstance(base_type, dt.DataType)
        and not isinstance(base_type, (tm.IntervalUnit, dt.Timestamp))
    ):
        base_type = base_type.copy(nullable=yaml_dict["nullable"])
    return base_type


REVERSE_TYPE_REGISTRY = {
    "Int8": dt.Int8(),
    "Int16": dt.Int16(),
    "Int32": dt.Int32(),
    "Int64": dt.Int64(),
    "UInt8": dt.UInt8(),
    "UInt16": dt.UInt16(),
    "UInt32": dt.UInt32(),
    "UInt64": dt.UInt64(),
    "Float32": dt.Float32(),
    "Float64": dt.Float64(),
    "String": dt.String(),
    "Boolean": dt.Boolean(),
    "Date": dt.Date(),
    "Time": dt.Time(),
    "Binary": dt.Binary(),
    "JSON": dt.JSON(),
    "Null": dt.null,
    "Timestamp": lambda yaml_dict: dt.Timestamp(
        nullable=yaml_dict.get("nullable", True)
    ),
    "Decimal": lambda yaml_dict: dt.Decimal(
        precision=yaml_dict.get("precision"),
        scale=yaml_dict.get("scale"),
        nullable=yaml_dict.get("nullable", True),
    ),
    "IntervalUnit": lambda yaml_dict: tm.IntervalUnit(
        yaml_dict["value"] if isinstance(yaml_dict, dict) else yaml_dict
    ),
    "Interval": lambda yaml_dict: dt.Interval(
        unit=_type_from_yaml(yaml_dict["unit"]),
        nullable=yaml_dict.get("nullable", True),
    ),
    "DateUnit": lambda yaml_dict: tm.DateUnit(yaml_dict["value"]),
    "TimeUnit": lambda yaml_dict: tm.TimeUnit(yaml_dict["value"]),
    "TimestampUnit": lambda yaml_dict: tm.TimestampUnit(yaml_dict["value"]),
    "Array": lambda yaml_dict: dt.Array(
        _type_from_yaml(yaml_dict["value_type"]),
        nullable=yaml_dict.get("nullable", True),
    ),
    "Map": lambda yaml_dict: dt.Map(
        _type_from_yaml(yaml_dict["key_type"]),
        _type_from_yaml(yaml_dict["value_type"]),
        nullable=yaml_dict.get("nullable", True),
    ),
}


@translate_to_yaml.register(FrozenDict)
@translate_to_yaml.register(FrozenOrderedDict)
def _frozenordereddict_to_yaml(dct: dict, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": type(dct).__name__,
        }
        | {
            key: context.translate_to_yaml(value)
            if context is not None
            else translate_to_yaml(value, context)
            for key, value in dct.items()
        }
    )


@register_from_yaml_handler("FrozenOrderedDict")
def _frozenordereddict_from_yaml(
    yaml_dict: dict, context: TranslationContext
) -> FrozenOrderedDict:
    dct = FrozenOrderedDict(
        {
            key: context.translate_from_yaml(value)
            for key, value in toolz.dissoc(yaml_dict, "op").items()
        }
    )
    return dct


@register_from_yaml_handler("FrozenDict")
def _frozendict_from_yaml(yaml_dict: dict, context: TranslationContext) -> FrozenDict:
    dct = FrozenDict(
        {
            key: context.translate_from_yaml(value)
            for key, value in toolz.dissoc(yaml_dict, "op").items()
        }
    )
    return dct


@translate_to_yaml.register(Tag)
def _tag_to_yaml(op: Tag, context: Any) -> dict:
    schema_id = context.schema_registry.register_schema(op.schema)
    # source should be called profile_name

    return freeze(
        {
            "op": "Tag",
            "schema_ref": schema_id,
            # fixme: translate_to_yaml on Node should result in Node, not Expr
            "parent": context.translate_to_yaml(op.parent),
            "metadata": context.translate_to_yaml(op.metadata),
        }
    )


@register_from_yaml_handler("Tag")
def _tag_from_yaml(yaml_dict: dict, context: Any) -> ibis.Expr:
    schema_ref = yaml_dict["schema_ref"]
    try:
        schema_def = context.definitions["schemas"][schema_ref]
    except KeyError:
        raise ValueError(f"Schema {schema_ref} not found in definitions")

    schema = {
        name: context.translate_from_yaml(dtype_yaml)
        for name, dtype_yaml in schema_def.items()
    }

    # fixme: enable translation of nodes
    parent_expr = context.translate_from_yaml(yaml_dict["parent"])
    metadata = context.translate_from_yaml(yaml_dict["metadata"])
    op = Tag(
        schema=schema,
        parent=parent_expr.op(),
        metadata=metadata,
    )
    return op.to_expr()


@translate_to_yaml.register(ops.Argument)
def _array_filter_to_yaml(op: ops.Argument, context: Any) -> dict:
    return freeze(
        {
            "op": "Argument",
            "name": context.translate_to_yaml(op.name),
            "shape": context.translate_to_yaml(op.shape),
            "dtype": context.translate_to_yaml(op.dtype),
        }
    )


@register_from_yaml_handler("Argument")
def _array_filter_from_yaml(yaml_dict: dict, context: Any) -> ibis.Expr:
    name = context.translate_from_yaml(yaml_dict["name"])
    shape = context.translate_from_yaml(yaml_dict["shape"])
    dtype = context.translate_from_yaml(yaml_dict["dtype"])

    return ops.Argument(name, shape=shape, dtype=dtype).to_expr()


@translate_to_yaml.register(ds.DataShape)
def _columnar_to_yaml(op, context) -> dict:
    return freeze(
        {
            "op": op.__class__.__name__,
            "ndim": context.translate_to_yaml(op.ndim),
        }
    )


@register_from_yaml_handler("Columnar")
def _array_filter_from_yaml(yaml_dict: dict, context: Any) -> Any:
    return Columnar()
