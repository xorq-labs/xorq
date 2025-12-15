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
    translate_from_yaml,
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
def _node_to_yaml(obj: ops.Node, context: Any) -> dict:
    return freeze(
        {"op": obj.__class__.__name__}
        | {
            name: translate_to_yaml(arg, context)
            for name, arg in zip(obj.argnames, obj.args)
        }
        | {
            name: translate_to_yaml(getattr(obj, attribute), context)
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
        {"op": "tuple", "values": [translate_to_yaml(value, context) for value in tpl]}
    )


@register_from_yaml_handler("tuple")
def _tuple_from_yaml(yaml_dict: dict, context: TranslationContext) -> Any:
    return tuple(translate_from_yaml(value, context) for value in yaml_dict["values"])


@translate_to_yaml.register(frozenset)
def _frozenset_to_yaml(tpl: tuple, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "frozenset",
            "values": [translate_to_yaml(value, context) for value in tpl],
        }
    )


@register_from_yaml_handler("frozenset")
def _frozenset_from_yaml(yaml_dict: dict, context: TranslationContext) -> Any:
    return frozenset(
        translate_from_yaml(value, context) for value in yaml_dict["values"]
    )


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
            argname: translate_to_yaml(arg, context)
            for argname, arg in zip(dtype.argnames, dtype.args)
        }
    )


@register_from_yaml_handler("DataType")
def _datatype_from_yaml(yaml_dict: dict, context: TranslationContext) -> any:
    typ = getattr(dt, yaml_dict["type"])
    dct = toolz.dissoc(yaml_dict, "op", "type")
    return typ(
        **{
            key: translate_from_yaml(value, context) if value is not None else None
            for key, value in dct.items()
        }
    )


@translate_to_yaml.register(ir.Expr)
def _expr_to_yaml(expr: ir.Expr, context: any) -> dict:
    return translate_to_yaml(expr.op(), context)


@translate_to_yaml.register(ops.WindowFunction)
def _window_function_to_yaml(
    op: ops.WindowFunction, context: TranslationContext
) -> dict:
    result = {
        "op": "WindowFunction",
        "args": [translate_to_yaml(op.func, context)],
        "type": translate_to_yaml(op.dtype, context),
    }

    if op.group_by:
        result["group_by"] = [translate_to_yaml(expr, context) for expr in op.group_by]

    if op.order_by:
        result["order_by"] = [translate_to_yaml(expr, context) for expr in op.order_by]

    if op.start is not None:
        result["start"] = (
            translate_to_yaml(op.start.value, context)["value"]
            if isinstance(op.start, ops.WindowBoundary)
            else op.start
        )

    if op.end is not None:
        result["end"] = (
            translate_to_yaml(op.end.value, context)["value"]
            if isinstance(op.end, ops.WindowBoundary)
            else op.end
        )

    return freeze(result)


@register_from_yaml_handler("WindowFunction")
def _window_function_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    func = translate_from_yaml(yaml_dict["args"][0], context)
    group_by = [translate_from_yaml(g, context) for g in yaml_dict.get("group_by", [])]
    order_by = [translate_from_yaml(o, context) for o in yaml_dict.get("order_by", [])]
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
            "value": translate_to_yaml(op.value, context),
            "preceding": op.preceding,
            "type": translate_to_yaml(op.dtype, context),
        }
    )


@register_from_yaml_handler("WindowBoundary")
def _window_boundary_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    value = translate_from_yaml(yaml_dict["value"], context)
    return ops.WindowBoundary(value, preceding=yaml_dict["preceding"])


@translate_to_yaml.register(ops.StructField)
def _struct_field_to_yaml(op: ops.Node, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": type(op).__name__,
            "args": [translate_to_yaml(arg, context) for arg in op.args],
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
        name: translate_from_yaml(dtype_yaml, context)
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

    if should_register_node(node_dict) and hasattr(
        context.schema_registry, "register_node"
    ):
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
        dtype = translate_from_yaml(dtype_yaml, context)
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
def _cached_node_to_yaml(op: CachedNode, context: TranslationContext) -> dict:
    schema_id = context.schema_registry.register_schema(op.schema)
    # source should be called profile_name

    return freeze(
        {
            "op": "CachedNode",
            "schema_ref": schema_id,
            "name": op.name,
            "parent": translate_to_yaml(op.parent, context),
            "source": op.source._profile.hash_name,
            "cache": translate_cache(op.cache, context),
        }
    )


@register_from_yaml_handler("CachedNode")
def _cached_node_from_yaml(yaml_dict: dict, context: Any) -> ibis.Expr:
    schema_ref = yaml_dict["schema_ref"]
    try:
        schema_def = context.definitions["schemas"][schema_ref]
    except KeyError:
        raise ValueError(f"Schema {schema_ref} not found in definitions")

    schema = {
        name: translate_from_yaml(dtype_yaml, context)
        for name, dtype_yaml in schema_def.items()
    }

    name = yaml_dict["name"]

    parent_expr = translate_from_yaml(yaml_dict["parent"], context)
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
def _remote_table_to_yaml(op: RemoteTable, context: TranslationContext) -> dict:
    profile_name = op.source._profile.hash_name
    remote_expr_yaml = translate_to_yaml(op.remote_expr, context)
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
def _remotet_table_from_yaml(yaml_dict: dict, context: TranslationContext) -> dict:
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

    remote_expr = translate_from_yaml(remote_expr_yaml, context)

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
        name: translate_from_yaml(dtype_yaml, context)
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


@translate_to_yaml.register(type(None))
def _none_to_yaml(value: None, context: TranslationContext) -> None:
    return None


@translate_to_yaml.register(ops.Literal)
def _literal_to_yaml(op: ops.Literal, context: TranslationContext) -> dict:
    value = _translate_literal_value(op.value, op.dtype)
    return freeze(
        {"op": "Literal", "value": value, "type": translate_to_yaml(op.dtype, context)}
    )


@register_from_yaml_handler("Literal")
def _literal_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    value = yaml_dict["value"]
    dtype = translate_from_yaml(yaml_dict["type"], context)
    return ibis.literal(value, type=dtype)


@translate_to_yaml.register(ops.Binary)
def _binary_op_to_yaml(op: ops.Binary, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": type(op).__name__,
            "left": translate_to_yaml(op.left, context),
            "right": translate_to_yaml(op.right, context),
            "type": translate_to_yaml(op.dtype, context),
        }
    )


@translate_to_yaml.register(ops.Filter)
def _filter_to_yaml(op: ops.Filter, context: TranslationContext) -> dict:
    node_dict = freeze(
        {
            "op": "Filter",
            "parent": translate_to_yaml(op.parent, context),
            "predicates": [translate_to_yaml(pred, context) for pred in op.predicates],
        }
    )

    if should_register_node(node_dict) and hasattr(
        context.schema_registry, "register_node"
    ):
        node_hash = context.schema_registry.register_node(node_dict)
        return freeze({"node_ref": node_hash})

    return node_dict


@register_from_yaml_handler("Filter")
def _filter_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    parent = translate_from_yaml(yaml_dict["parent"], context)
    predicates = [
        translate_from_yaml(pred, context) for pred in yaml_dict["predicates"]
    ]
    filter_op = ops.Filter(parent, predicates)
    return filter_op.to_expr()


@translate_to_yaml.register(ops.Project)
def _project_to_yaml(op: ops.Project, context: TranslationContext) -> dict:
    node_dict = {
        "op": "Project",
        "parent": translate_to_yaml(op.parent, context),
        "values": {
            name: translate_to_yaml(val, context) for name, val in op.values.items()
        },
    }

    if should_register_node(node_dict):
        node_hash = context.schema_registry.register_node(freeze(node_dict))
        return freeze({"node_ref": node_hash})

    return freeze(node_dict)


@register_from_yaml_handler("Project")
def _project_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    parent = translate_from_yaml(yaml_dict["parent"], context)

    values_dict = yaml_dict.get("values", {})

    values = {
        name: translate_from_yaml(val, context) for name, val in values_dict.items()
    }

    projected = parent.projection(values)
    return projected


@translate_to_yaml.register(ops.Aggregate)
def _aggregate_to_yaml(op: ops.Aggregate, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "Aggregate",
            "parent": translate_to_yaml(op.parent, context),
            "by": {
                name: translate_to_yaml(group, context)
                for name, group in op.groups.items()
            },
            "metrics": {
                name: translate_to_yaml(metric, context)
                for name, metric in op.metrics.items()
            },
        }
    )


@register_from_yaml_handler("Aggregate")
def _aggregate_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    parent = translate_from_yaml(yaml_dict["parent"], context)
    by = yaml_dict.get("by", {})
    if isinstance(by, dict):
        group_mapping = {
            name: translate_from_yaml(expr, context) for name, expr in by.items()
        }
    else:
        exprs = [translate_from_yaml(expr, context) for expr in by]
        group_mapping = {expr.get_name(): expr for expr in exprs}

    metrics = {
        name: translate_from_yaml(metric, context)
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
        "first": translate_to_yaml(op.first, context),
        "rest": [
            {
                "how": link.how,
                "table": translate_to_yaml(link.table, context),
                "predicates": [
                    translate_to_yaml(pred, context) for pred in link.predicates
                ],
            }
            for link in op.rest
        ],
        "values": {
            name: translate_to_yaml(val, context) for name, val in op.values.items()
        },
    }

    if should_register_node(node_dict):
        node_hash = context.schema_registry.register_node(freeze(node_dict))
        return freeze({"node_ref": node_hash})
    return freeze(node_dict)


@register_from_yaml_handler("JoinChain")
def _join_chain_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    first = translate_from_yaml(yaml_dict["first"], context)
    result = first

    for join in yaml_dict["rest"]:
        table = translate_from_yaml(join["table"], context)
        predicates = [translate_from_yaml(pred, context) for pred in join["predicates"]]
        result = result.join(table, predicates, how=join["how"])

    values = {
        name: translate_from_yaml(val, context)
        for name, val in yaml_dict["values"].items()
    }
    result = result.select(values)
    return result


@translate_to_yaml.register(ops.Limit)
def _limit_to_yaml(op: ops.Limit, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "Limit",
            "parent": translate_to_yaml(op.parent, context),
            "n": op.n,
            "offset": op.offset,
        }
    )


@register_from_yaml_handler("Limit")
def _limit_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    parent = translate_from_yaml(yaml_dict["parent"], context)
    return parent.limit(yaml_dict["n"], offset=yaml_dict["offset"])


@translate_to_yaml.register(ops.ScalarSubquery)
def _scalar_subquery_to_yaml(
    op: ops.ScalarSubquery, context: TranslationContext
) -> dict:
    return freeze(
        {
            "op": "ScalarSubquery",
            "args": [translate_to_yaml(arg, context) for arg in op.args],
            "type": translate_to_yaml(op.dtype, context),
        }
    )


@register_from_yaml_handler("ScalarSubquery")
def _scalar_subquery_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    subquery = translate_from_yaml(yaml_dict["args"][0], context)
    return ops.ScalarSubquery(subquery).to_expr()


@translate_to_yaml.register(ops.ExistsSubquery)
def _exists_subquery_to_yaml(
    op: ops.ExistsSubquery, context: TranslationContext
) -> dict:
    return freeze(
        {
            "op": "ExistsSubquery",
            "rel": translate_to_yaml(op.rel, context),
            "type": translate_to_yaml(op.dtype, context),
        }
    )


@register_from_yaml_handler("ExistsSubquery")
def _exists_subquery_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    rel = translate_from_yaml(yaml_dict["rel"], context)
    return ops.ExistsSubquery(rel).to_expr()


@translate_to_yaml.register(ops.InSubquery)
def _in_subquery_to_yaml(op: ops.InSubquery, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": "InSubquery",
            "needle": translate_to_yaml(op.needle, context),
            "haystack": translate_to_yaml(op.rel, context),
            "type": translate_to_yaml(op.dtype, context),
        }
    )


@register_from_yaml_handler("InSubquery")
def _in_subquery_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    needle = translate_from_yaml(yaml_dict["needle"], context)
    haystack = translate_from_yaml(yaml_dict["haystack"], context)
    return ops.InSubquery(haystack, needle).to_expr()


@translate_to_yaml.register(ops.Field)
def _field_to_yaml(op: ops.Field, context: TranslationContext) -> dict:
    result = {
        "op": "Field",
        "name": op.name,
        "relation": translate_to_yaml(op.rel, context),
        "type": translate_to_yaml(op.dtype, context),
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
    relation = translate_from_yaml(yaml_dict["relation"], context)
    target_name = yaml_dict["name"]
    source_name = yaml_dict.get("original_name", target_name)
    field = relation[source_name]
    if target_name != source_name:
        field = field.name(target_name)

    return freeze(field)


@translate_to_yaml.register(ops.SelfReference)
def _self_reference_to_yaml(op: ops.SelfReference, context: TranslationContext) -> dict:
    result = {"op": "SelfReference", "identifier": op.identifier}
    if op.args:
        result["args"] = [translate_to_yaml(op.args[0], context)]
    return freeze(result)


@register_from_yaml_handler("SelfReference")
def _self_reference_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    underlying = None
    if "args" in yaml_dict and yaml_dict["args"]:
        underlying = translate_from_yaml(yaml_dict["args"][0], context)
    else:
        if underlying is None:
            raise NotImplementedError("No relation available for SelfReference")

    identifier = yaml_dict.get("identifier", 0)
    ref = ops.SelfReference(underlying, identifier=identifier)

    return ref.to_expr()


@register_from_yaml_handler("JoinReference")
def _join_reference_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    table_yaml = yaml_dict["parent"]
    return translate_from_yaml(table_yaml, context)


@register_from_yaml_handler("StructField")
def _structfield_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    args = tuple(translate_from_yaml(arg, context) for arg in yaml_dict["args"])
    return ops.StructField(*args).to_expr()


@translate_to_yaml.register(FrozenDict)
@translate_to_yaml.register(FrozenOrderedDict)
def _frozenordereddict_to_yaml(dct: dict, context: TranslationContext) -> dict:
    return freeze(
        {
            "op": type(dct).__name__,
        }
        | {key: translate_to_yaml(value, context) for key, value in dct.items()}
    )


@register_from_yaml_handler("FrozenOrderedDict")
def _frozenordereddict_from_yaml(
    yaml_dict: dict, context: TranslationContext
) -> FrozenOrderedDict:
    dct = FrozenOrderedDict(
        {
            key: translate_from_yaml(value, context)
            for key, value in toolz.dissoc(yaml_dict, "op").items()
        }
    )
    return dct


@register_from_yaml_handler("FrozenDict")
def _frozendict_from_yaml(yaml_dict: dict, context: TranslationContext) -> FrozenDict:
    dct = FrozenDict(
        {
            key: translate_from_yaml(value, context)
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
            "parent": translate_to_yaml(op.parent, context),
            "metadata": translate_to_yaml(op.metadata, context),
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
        name: translate_from_yaml(dtype_yaml, context)
        for name, dtype_yaml in schema_def.items()
    }

    # fixme: enable translation of nodes
    parent_expr = translate_from_yaml(yaml_dict["parent"], context)
    metadata = translate_from_yaml(yaml_dict["metadata"], context)
    op = Tag(
        schema=schema,
        parent=parent_expr.op(),
        metadata=metadata,
    )
    return op.to_expr()


@translate_to_yaml.register(ds.DataShape)
def _columnar_to_yaml(op, context) -> dict:
    return freeze(
        {
            "op": op.__class__.__name__,
            "ndim": translate_to_yaml(op.ndim, context),
        }
    )


@register_from_yaml_handler("Columnar")
def _array_filter_from_yaml(yaml_dict: dict, context: Any) -> Any:
    return Columnar()
