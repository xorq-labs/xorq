from __future__ import annotations

import datetime
import decimal
import functools
import pathlib
from typing import Any

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.operations.temporal as tm
import ibis.expr.rules as rlz
import ibis.expr.types as ir
import pyarrow.parquet as pq
from ibis.common.annotations import Argument
from ibis.common.exceptions import IbisTypeError

import letsql as ls
from letsql.expr.relations import CachedNode, RemoteTable, into_backend
from letsql.ibis_yaml.utils import (
    deserialize_udf_function,
    freeze,
    load_storage_from_yaml,
    serialize_udf_function,
    translate_storage,
)


FROM_YAML_HANDLERS: dict[str, Any] = {}


def register_from_yaml_handler(*op_names: str):
    def decorator(func):
        for name in op_names:
            FROM_YAML_HANDLERS[name] = func
        return func

    return decorator


@functools.cache
@functools.singledispatch
def translate_from_yaml(yaml_dict: dict, compiler: Any) -> Any:
    op_type = yaml_dict["op"]
    if op_type not in FROM_YAML_HANDLERS:
        raise NotImplementedError(f"No handler for operation {op_type}")
    return FROM_YAML_HANDLERS[op_type](yaml_dict, compiler)


@functools.cache
@functools.singledispatch
def translate_to_yaml(op: Any, compiler: Any) -> dict:
    raise NotImplementedError(f"No translation rule for {type(op)}")


@functools.singledispatch
def _translate_type(dtype: dt.DataType) -> dict:
    return freeze({"name": type(dtype).__name__, "nullable": dtype.nullable})


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
            "value_type": _translate_type(dtype.value_type),
            "nullable": dtype.nullable,
        }
    )


@_translate_type.register(dt.Map)
def _translate_map_type(dtype: dt.Map) -> dict:
    return freeze(
        {
            "name": "Map",
            "key_type": _translate_type(dtype.key_type),
            "value_type": _translate_type(dtype.value_type),
            "nullable": dtype.nullable,
        }
    )


@_translate_type.register(dt.Interval)
def _tranlate_type_interval(dtype: dt.Interval) -> dict:
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
        return {
            _translate_literal_value(k, dtype.key_type): _translate_literal_value(
                v, dtype.value_type
            )
            for k, v in value.items()
        }
    else:
        return value


@translate_to_yaml.register(ir.Expr)
def _expr_to_yaml(expr: ir.Expr, compiler: any) -> dict:
    return translate_to_yaml(expr.op(), compiler)


@translate_to_yaml.register(ops.WindowFunction)
def _window_function_to_yaml(op: ops.WindowFunction, compiler: Any) -> dict:
    result = {
        "op": "WindowFunction",
        "args": [translate_to_yaml(op.func, compiler)],
        "type": _translate_type(op.dtype),
    }

    if op.group_by:
        result["group_by"] = [translate_to_yaml(expr, compiler) for expr in op.group_by]

    if op.order_by:
        result["order_by"] = [translate_to_yaml(expr, compiler) for expr in op.order_by]

    if op.start is not None:
        result["start"] = (
            translate_to_yaml(op.start.value, compiler)["value"]
            if isinstance(op.start, ops.WindowBoundary)
            else op.start
        )

    if op.end is not None:
        result["end"] = (
            translate_to_yaml(op.end.value, compiler)["value"]
            if isinstance(op.end, ops.WindowBoundary)
            else op.end
        )

    return freeze(result)


@register_from_yaml_handler("WindowFunction")
def _window_function_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    func = translate_from_yaml(yaml_dict["args"][0], compiler)
    group_by = [translate_from_yaml(g, compiler) for g in yaml_dict.get("group_by", [])]
    order_by = [translate_from_yaml(o, compiler) for o in yaml_dict.get("order_by", [])]
    start = ibis.literal(yaml_dict["start"]) if "start" in yaml_dict else None
    end = ibis.literal(yaml_dict["end"]) if "end" in yaml_dict else None
    window = ibis.window(
        group_by=group_by, order_by=order_by, preceding=start, following=end
    )
    return func.over(window)


@translate_to_yaml.register(ops.WindowBoundary)
def _window_boundary_to_yaml(op: ops.WindowBoundary, compiler: Any) -> dict:
    return freeze(
        {
            "op": "WindowBoundary",
            "value": translate_to_yaml(op.value, compiler),
            "preceding": op.preceding,
            "type": _translate_type(op.dtype),
        }
    )


@register_from_yaml_handler("WindowBoundary")
def _window_boundary_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    value = translate_from_yaml(yaml_dict["value"], compiler)
    return ops.WindowBoundary(value, preceding=yaml_dict["preceding"])


@translate_to_yaml.register(ops.Node)
def _base_op_to_yaml(op: ops.Node, compiler: Any) -> dict:
    return freeze(
        {
            "op": type(op).__name__,
            "args": [
                translate_to_yaml(arg, compiler)
                for arg in op.args
                if isinstance(arg, (ops.Value, ops.Node))
            ],
        }
    )


@translate_to_yaml.register(ops.UnboundTable)
def _unbound_table_to_yaml(op: ops.UnboundTable, compiler: Any) -> dict:
    return freeze(
        {
            "op": "UnboundTable",
            "name": op.name,
            "schema": {
                name: _translate_type(dtype) for name, dtype in op.schema.items()
            },
        }
    )


@register_from_yaml_handler("UnboundTable")
def _unbound_table_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    table_name = yaml_dict["name"]
    schema = [
        (name, _type_from_yaml(dtype)) for name, dtype in yaml_dict["schema"].items()
    ]
    return ibis.table(schema, name=table_name)


@translate_to_yaml.register(ops.DatabaseTable)
def _database_table_to_yaml(op: ops.DatabaseTable, compiler: Any) -> dict:
    profile_name = getattr(op.source, "profile_name", None)
    return freeze(
        {
            "op": "DatabaseTable",
            "table": op.name,
            "schema": {
                name: _translate_type(dtype) for name, dtype in op.schema.items()
            },
            "profile": profile_name,
        }
    )


@register_from_yaml_handler("DatabaseTable")
def _database_table_from_yaml(yaml_dict: dict, compiler: Any) -> ibis.Expr:
    profile_name = yaml_dict.get("profile")
    table_name = yaml_dict.get("table")
    if not profile_name or not table_name:
        raise ValueError(
            "Missing 'profile' or 'table' information in YAML for DatabaseTable."
        )

    try:
        con = compiler.profiles[profile_name]
    except KeyError:
        raise ValueError(f"Profile {profile_name!r} not found in compiler.profiles")

    return con.table(table_name)


@translate_to_yaml.register(CachedNode)
def _cached_node_to_yaml(op: CachedNode, compiler: any) -> dict:
    return freeze(
        {
            "op": "CachedNode",
            "schema": {
                name: _translate_type(dtype) for name, dtype in op.schema.items()
            },
            "parent": translate_to_yaml(op.parent, compiler),
            "source": getattr(op.source, "profile_name", None),
            "storage": translate_storage(op.storage, compiler),
            "values": dict(op.values),
        }
    )


@register_from_yaml_handler("CachedNode")
def _cached_node_from_yaml(yaml_dict: dict, compiler: any) -> ibis.Expr:
    schema = {
        name: _type_from_yaml(dtype_yaml)
        for name, dtype_yaml in yaml_dict["schema"].items()
    }
    parent_expr = translate_from_yaml(yaml_dict["parent"], compiler)
    profile_name = yaml_dict.get("source")
    try:
        source = compiler.profiles[profile_name]
    except KeyError:
        raise ValueError(f"Profile {profile_name!r} not found in compiler.profiles")
    storage = load_storage_from_yaml(yaml_dict["storage"], compiler)

    op = CachedNode(
        schema=schema,
        parent=parent_expr,
        source=source,
        storage=storage,
    )
    return op.to_expr()


@translate_to_yaml.register(ops.InMemoryTable)
def _memtable_to_yaml(op: ops.InMemoryTable, compiler: Any) -> dict:
    if not hasattr(compiler, "tmp_path"):
        raise ValueError(
            "Compiler is missing the 'tmp_path' attribute for memtable serialization"
        )

    arrow_table = op.data.to_pyarrow(op.schema)

    file_path = compiler.tmp_path / f"memtable_{id(op)}.parquet"
    pq.write_table(arrow_table, str(file_path))

    return freeze(
        {
            "op": "InMemoryTable",
            "table": op.name,
            "schema": {
                name: _translate_type(dtype) for name, dtype in op.schema.items()
            },
            "file": str(file_path),
        }
    )


@register_from_yaml_handler("InMemoryTable")
def _memtable_from_yaml(yaml_dict: dict, compiler: Any) -> ibis.Expr:
    file_path = yaml_dict["file"]
    arrow_table = pq.read_table(file_path)
    df = arrow_table.to_pandas()

    table_name = yaml_dict.get("table", "memtable")

    memtable_expr = ls.memtable(df, columns=list(df.columns), name=table_name)
    return memtable_expr


@translate_to_yaml.register(RemoteTable)
def _remotetable_to_yaml(op: RemoteTable, compiler: any) -> dict:
    profile_name = getattr(op.source, "profile_name", None)
    remote_expr_yaml = translate_to_yaml(op.remote_expr, compiler)
    return freeze(
        {
            "op": "RemoteTable",
            "table": op.name,
            "schema": {
                name: _translate_type(dtype) for name, dtype in op.schema.items()
            },
            "profile": profile_name,
            "remote_expr": remote_expr_yaml,
        }
    )


@register_from_yaml_handler("RemoteTable")
def _remotetable_from_yaml(yaml_dict: dict, compiler: any) -> ibis.Expr:
    profile_name = yaml_dict.get("profile")
    table_name = yaml_dict.get("table")
    remote_expr_yaml = yaml_dict.get("remote_expr")
    if profile_name is None:
        raise ValueError(
            "Missing keys in RemoteTable YAML; ensure 'profile', 'table', and 'remote_expr' are present."
        )
    try:
        con = compiler.profiles[profile_name]
    except KeyError:
        raise ValueError(f"Profile {profile_name!r} not found in compiler.profiles")

    remote_expr = translate_from_yaml(remote_expr_yaml, compiler)

    remote_table_expr = into_backend(remote_expr, con, table_name)
    return remote_table_expr


@translate_to_yaml.register(ops.Literal)
def _literal_to_yaml(op: ops.Literal, compiler: Any) -> dict:
    value = _translate_literal_value(op.value, op.dtype)
    return freeze({"op": "Literal", "value": value, "type": _translate_type(op.dtype)})


@register_from_yaml_handler("Literal")
def _literal_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    value = yaml_dict["value"]
    dtype = _type_from_yaml(yaml_dict["type"])
    return ibis.literal(value, type=dtype)


@translate_to_yaml.register(ops.ValueOp)
def _value_op_to_yaml(op: ops.ValueOp, compiler: Any) -> dict:
    return freeze(
        {
            "op": type(op).__name__,
            "type": _translate_type(op.dtype),
            "args": [
                translate_to_yaml(arg, compiler)
                for arg in op.args
                if isinstance(arg, (ops.Value, ops.Node))
            ],
        }
    )


@register_from_yaml_handler("ValueOp")
def _value_op_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    args = [translate_from_yaml(arg, compiler) for arg in yaml_dict["args"]]
    method_name = yaml_dict["op"].lower()
    method = getattr(args[0], method_name)
    return method(*args[1:])


@translate_to_yaml.register(ops.StringUnary)
def _string_unary_to_yaml(op: ops.StringUnary, compiler: Any) -> dict:
    return freeze(
        {
            "op": type(op).__name__,
            "args": [translate_to_yaml(op.arg, compiler)],
            "type": _translate_type(op.dtype),
        }
    )


@register_from_yaml_handler("StringUnary")
def _string_unary_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    arg = translate_from_yaml(yaml_dict["args"][0], compiler)
    method_name = yaml_dict["op"].lower()
    return getattr(arg, method_name)()


@translate_to_yaml.register(ops.Substring)
def _substring_to_yaml(op: ops.Substring, compiler: Any) -> dict:
    args = [
        translate_to_yaml(op.arg, compiler),
        translate_to_yaml(op.start, compiler),
    ]
    if op.length is not None:
        args.append(translate_to_yaml(op.length, compiler))
    return freeze({"op": "Substring", "args": args, "type": _translate_type(op.dtype)})


@register_from_yaml_handler("Substring")
def _substring_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    args = [translate_from_yaml(arg, compiler) for arg in yaml_dict["args"]]
    return args[0].substr(args[1], args[2] if len(args) > 2 else None)


@translate_to_yaml.register(ops.StringLength)
def _string_length_to_yaml(op: ops.StringLength, compiler: Any) -> dict:
    return freeze(
        {
            "op": "StringLength",
            "args": [translate_to_yaml(op.arg, compiler)],
            "type": _translate_type(op.dtype),
        }
    )


@register_from_yaml_handler("StringLength")
def _string_length_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    arg = translate_from_yaml(yaml_dict["args"][0], compiler)
    return arg.length()


@translate_to_yaml.register(ops.StringConcat)
def _string_concat_to_yaml(op: ops.StringConcat, compiler: Any) -> dict:
    return freeze(
        {
            "op": "StringConcat",
            "args": [translate_to_yaml(arg, compiler) for arg in op.arg],
            "type": _translate_type(op.dtype),
        }
    )


@register_from_yaml_handler("StringConcat")
def _string_concat_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    args = [translate_from_yaml(arg, compiler) for arg in yaml_dict["args"]]
    return functools.reduce(lambda x, y: x.concat(y), args)


@translate_to_yaml.register(ops.BinaryOp)
def _binary_op_to_yaml(op: ops.BinaryOp, compiler: Any) -> dict:
    return freeze(
        {
            "op": type(op).__name__,
            "args": [
                translate_to_yaml(op.left, compiler),
                translate_to_yaml(op.right, compiler),
            ],
            "type": _translate_type(op.dtype),
        }
    )


@register_from_yaml_handler("BinaryOp")
def _binary_op_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    args = [translate_from_yaml(arg, compiler) for arg in yaml_dict["args"]]
    op_name = yaml_dict["op"].lower()
    return getattr(args[0], op_name)(args[1])


@translate_to_yaml.register(ops.Filter)
def _filter_to_yaml(op: ops.Filter, compiler: Any) -> dict:
    return freeze(
        {
            "op": "Filter",
            "parent": translate_to_yaml(op.parent, compiler),
            "predicates": [translate_to_yaml(pred, compiler) for pred in op.predicates],
        }
    )


@register_from_yaml_handler("Filter")
def _filter_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    parent = translate_from_yaml(yaml_dict["parent"], compiler)
    predicates = [
        translate_from_yaml(pred, compiler) for pred in yaml_dict["predicates"]
    ]
    filter_op = ops.Filter(parent, predicates)
    return filter_op.to_expr()


@translate_to_yaml.register(ops.Project)
def _project_to_yaml(op: ops.Project, compiler: Any) -> dict:
    return freeze(
        {
            "op": "Project",
            "parent": translate_to_yaml(op.parent, compiler),
            "values": {
                name: translate_to_yaml(val, compiler)
                for name, val in op.values.items()
            },
        }
    )


@register_from_yaml_handler("Project")
def _project_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    parent = translate_from_yaml(yaml_dict["parent"], compiler)
    values = {
        name: translate_from_yaml(val, compiler)
        for name, val in yaml_dict["values"].items()
    }
    projected = parent.projection(values)
    return projected


@translate_to_yaml.register(ops.Aggregate)
def _aggregate_to_yaml(op: ops.Aggregate, compiler: Any) -> dict:
    return freeze(
        {
            "op": "Aggregate",
            "parent": translate_to_yaml(op.parent, compiler),
            "by": [translate_to_yaml(group, compiler) for group in op.groups.values()],
            "metrics": {
                name: translate_to_yaml(metric, compiler)
                for name, metric in op.metrics.items()
            },
        }
    )


@register_from_yaml_handler("Aggregate")
def _aggregate_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    parent = translate_from_yaml(yaml_dict["parent"], compiler)
    groups = tuple(
        translate_from_yaml(group, compiler) for group in yaml_dict.get("by", [])
    )

    raw_metrics = {
        name: translate_from_yaml(metric, compiler)
        for name, metric in yaml_dict.get("metrics", {}).items()
    }
    metrics = raw_metrics

    if groups:
        return parent.group_by(list(groups)).aggregate(metrics)
    else:
        return parent.aggregate(metrics)


@translate_to_yaml.register(ops.JoinChain)
def _join_to_yaml(op: ops.JoinChain, compiler: Any) -> dict:
    result = {
        "op": "JoinChain",
        "first": translate_to_yaml(op.first, compiler),
        "rest": [
            {
                "how": link.how,
                "table": translate_to_yaml(link.table, compiler),
                "predicates": [
                    translate_to_yaml(pred, compiler) for pred in link.predicates
                ],
            }
            for link in op.rest
        ],
    }
    if hasattr(op, "values") and op.values:
        result["values"] = {
            name: translate_to_yaml(val, compiler) for name, val in op.values.items()
        }
    return freeze(result)


@register_from_yaml_handler("JoinChain")
def _join_chain_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    first = translate_from_yaml(yaml_dict["first"], compiler)
    result = first

    for join in yaml_dict["rest"]:
        table = translate_from_yaml(join["table"], compiler)
        predicates = [
            translate_from_yaml(pred, compiler) for pred in join["predicates"]
        ]
        result = result.join(table, predicates, how=join["how"])

    if "values" in yaml_dict:
        values = {
            name: translate_from_yaml(val, compiler)
            for name, val in yaml_dict["values"].items()
        }
        result = result.select(values)
    return result


@translate_to_yaml.register(ops.Sort)
def _sort_to_yaml(op: ops.Sort, compiler: Any) -> dict:
    return freeze(
        {
            "op": "Sort",
            "parent": translate_to_yaml(op.parent, compiler),
            "keys": [translate_to_yaml(key, compiler) for key in op.keys],
        }
    )


@register_from_yaml_handler("Sort")
def _sort_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    parent = translate_from_yaml(yaml_dict["parent"], compiler)
    keys = tuple(translate_from_yaml(key, compiler) for key in yaml_dict["keys"])
    sort_op = ops.Sort(parent, keys=keys)
    return sort_op.to_expr()


@translate_to_yaml.register(ops.SortKey)
def _sort_key_to_yaml(op: ops.SortKey, compiler: Any) -> dict:
    return freeze(
        {
            "op": "SortKey",
            "arg": translate_to_yaml(op.expr, compiler),
            "ascending": op.ascending,
            "nulls_first": op.nulls_first,
            "type": _translate_type(op.dtype),
        }
    )


@register_from_yaml_handler("SortKey")
def _sort_key_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    expr = translate_from_yaml(yaml_dict["arg"], compiler)
    ascending = yaml_dict.get("ascending", True)
    nulls_first = yaml_dict.get("nulls_first", False)
    return ops.SortKey(expr, ascending=ascending, nulls_first=nulls_first).to_expr()


@translate_to_yaml.register(ops.Limit)
def _limit_to_yaml(op: ops.Limit, compiler: Any) -> dict:
    return freeze(
        {
            "op": "Limit",
            "parent": translate_to_yaml(op.parent, compiler),
            "n": op.n,
            "offset": op.offset,
        }
    )


@register_from_yaml_handler("Limit")
def _limit_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    parent = translate_from_yaml(yaml_dict["parent"], compiler)
    return parent.limit(yaml_dict["n"], offset=yaml_dict["offset"])


@translate_to_yaml.register(ops.ScalarSubquery)
def _scalar_subquery_to_yaml(op: ops.ScalarSubquery, compiler: Any) -> dict:
    return freeze(
        {
            "op": "ScalarSubquery",
            "args": [translate_to_yaml(arg, compiler) for arg in op.args],
            "type": _translate_type(op.dtype),
        }
    )


@register_from_yaml_handler("ScalarSubquery")
def _scalar_subquery_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    subquery = translate_from_yaml(yaml_dict["args"][0], compiler)
    return ops.ScalarSubquery(subquery).to_expr()


@translate_to_yaml.register(ops.ExistsSubquery)
def _exists_subquery_to_yaml(op: ops.ExistsSubquery, compiler: Any) -> dict:
    return freeze(
        {
            "op": "ExistsSubquery",
            "rel": translate_to_yaml(op.rel, compiler),
            "type": _translate_type(op.dtype),
        }
    )


@register_from_yaml_handler("ExistsSubquery")
def _exists_subquery_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    rel = translate_from_yaml(yaml_dict["rel"], compiler)
    return ops.ExistsSubquery(rel).to_expr()


@translate_to_yaml.register(ops.InSubquery)
def _in_subquery_to_yaml(op: ops.InSubquery, compiler: Any) -> dict:
    return freeze(
        {
            "op": "InSubquery",
            "needle": translate_to_yaml(op.needle, compiler),
            "haystack": translate_to_yaml(op.rel, compiler),
            "type": _translate_type(op.dtype),
        }
    )


@register_from_yaml_handler("InSubquery")
def _in_subquery_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    needle = translate_from_yaml(yaml_dict["needle"], compiler)
    haystack = translate_from_yaml(yaml_dict["haystack"], compiler)
    return ops.InSubquery(haystack, needle).to_expr()


@translate_to_yaml.register(ops.Field)
def _field_to_yaml(op: ops.Field, compiler: Any) -> dict:
    result = {
        "op": "Field",
        "name": op.name,
        "relation": translate_to_yaml(op.rel, compiler),
        "type": _translate_type(op.dtype),
    }
    if op.args and len(op.args) >= 2 and isinstance(op.args[1], str):
        underlying_name = op.args[1]
        if underlying_name != op.name:
            result["original_name"] = underlying_name
    return freeze(result)


@register_from_yaml_handler("Field")
def field_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    relation = translate_from_yaml(yaml_dict["relation"], compiler)

    target_name = yaml_dict["name"]
    source_name = yaml_dict.get("original_name", target_name)

    schema = relation.schema() if callable(relation.schema) else relation.schema

    if source_name not in schema.names:
        if target_name in schema.names:
            source_name = target_name
        else:
            columns_formatted = ", ".join(schema.names)
            raise IbisTypeError(
                f"Column {source_name!r} not found in table. "
                f"Existing columns: {columns_formatted}."
            )
    field = relation[source_name]

    if target_name != source_name:
        field = field.name(target_name)

    return freeze(field)


@translate_to_yaml.register(ops.InValues)
def _in_values_to_yaml(op: ops.InValues, compiler: Any) -> dict:
    return freeze(
        {
            "op": "InValues",
            "args": [
                translate_to_yaml(op.value, compiler),
                *[translate_to_yaml(opt, compiler) for opt in op.options],
            ],
            "type": _translate_type(op.dtype),
        }
    )


@register_from_yaml_handler("InValues")
def _in_values_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    value = translate_from_yaml(yaml_dict["args"][0], compiler)
    options = tuple(translate_from_yaml(opt, compiler) for opt in yaml_dict["args"][1:])
    return ops.InValues(value, options).to_expr()


@translate_to_yaml.register(ops.SimpleCase)
def _simple_case_to_yaml(op: ops.SimpleCase, compiler: Any) -> dict:
    return freeze(
        {
            "op": "SimpleCase",
            "base": translate_to_yaml(op.base, compiler),
            "cases": [translate_to_yaml(case, compiler) for case in op.cases],
            "results": [translate_to_yaml(result, compiler) for result in op.results],
            "default": translate_to_yaml(op.default, compiler),
            "type": _translate_type(op.dtype),
        }
    )


@register_from_yaml_handler("SimpleCase")
def _simple_case_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    base = translate_from_yaml(yaml_dict["base"], compiler)
    cases = tuple(translate_from_yaml(case, compiler) for case in yaml_dict["cases"])
    results = tuple(
        translate_from_yaml(result, compiler) for result in yaml_dict["results"]
    )
    default = translate_from_yaml(yaml_dict["default"], compiler)
    return ops.SimpleCase(base, cases, results, default).to_expr()


@translate_to_yaml.register(ops.IfElse)
def _if_else_to_yaml(op: ops.IfElse, compiler: Any) -> dict:
    return freeze(
        {
            "op": "IfElse",
            "bool_expr": translate_to_yaml(op.bool_expr, compiler),
            "true_expr": translate_to_yaml(op.true_expr, compiler),
            "false_null_expr": translate_to_yaml(op.false_null_expr, compiler),
            "type": _translate_type(op.dtype),
        }
    )


@register_from_yaml_handler("IfElse")
def _if_else_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    bool_expr = translate_from_yaml(yaml_dict["bool_expr"], compiler)
    true_expr = translate_from_yaml(yaml_dict["true_expr"], compiler)
    false_null_expr = translate_from_yaml(yaml_dict["false_null_expr"], compiler)
    return ops.IfElse(bool_expr, true_expr, false_null_expr).to_expr()


@translate_to_yaml.register(ops.CountDistinct)
def _count_distinct_to_yaml(op: ops.CountDistinct, compiler: Any) -> dict:
    return freeze(
        {
            "op": "CountDistinct",
            "args": [translate_to_yaml(op.arg, compiler)],
            "type": _translate_type(op.dtype),
        }
    )


@register_from_yaml_handler("CountDistinct")
def _count_distinct_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    arg = translate_from_yaml(yaml_dict["args"][0], compiler)
    return arg.nunique()


@translate_to_yaml.register(ops.SelfReference)
def _self_reference_to_yaml(op: ops.SelfReference, compiler: Any) -> dict:
    result = {"op": "SelfReference", "identifier": op.identifier}
    if op.args:
        result["args"] = [translate_to_yaml(op.args[0], compiler)]
    return freeze(result)


@register_from_yaml_handler("SelfReference")
def _self_reference_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    if "args" in yaml_dict and yaml_dict["args"]:
        underlying = translate_from_yaml(yaml_dict["args"][0], compiler)
    else:
        if underlying is None:
            raise NotImplementedError("No relation available for SelfReference")

    identifier = yaml_dict.get("identifier", 0)
    ref = ops.SelfReference(underlying, identifier=identifier)

    return ref.to_expr()


@translate_to_yaml.register(ops.DropColumns)
def _drop_columns_to_yaml(op: ops.DropColumns, compiler: Any) -> dict:
    return freeze(
        {
            "op": "DropColumns",
            "parent": translate_to_yaml(op.parent, compiler),
            "columns_to_drop": list(op.columns_to_drop),
        }
    )


@register_from_yaml_handler("DropColumns")
def _drop_columns_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    parent = translate_from_yaml(yaml_dict["parent"], compiler)
    columns = frozenset(yaml_dict["columns_to_drop"])
    op = ops.DropColumns(parent, columns)
    return op.to_expr()


@translate_to_yaml.register(ops.SearchedCase)
def _searched_case_to_yaml(op: ops.SearchedCase, compiler: Any) -> dict:
    return freeze(
        {
            "op": "SearchedCase",
            "cases": [translate_to_yaml(case, compiler) for case in op.cases],
            "results": [translate_to_yaml(result, compiler) for result in op.results],
            "default": translate_to_yaml(op.default, compiler),
            "dtype": _translate_type(op.dtype),
        }
    )


@register_from_yaml_handler("SearchedCase")
def _searched_case_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    cases = [translate_from_yaml(case, compiler) for case in yaml_dict["cases"]]
    results = [translate_from_yaml(result, compiler) for result in yaml_dict["results"]]
    default = translate_from_yaml(yaml_dict["default"], compiler)
    op = ops.SearchedCase(cases, results, default)
    return op.to_expr()


@translate_to_yaml.register(ops.ScalarUDF)
def _scalar_udf_to_yaml(op: ops.ScalarUDF, compiler: Any) -> dict:
    arg_names = [
        name
        for name in dir(op)
        if not name.startswith("__") and name not in op.__class__.__slots__
    ]

    return freeze(
        {
            "op": "ScalarUDF",
            "unique_name": op.__func_name__,
            "input_type": "builtin",
            "args": [translate_to_yaml(arg, compiler) for arg in op.args],
            "type": _translate_type(op.dtype),
            "pickle": serialize_udf_function(op.__func__),
            "module": op.__module__,
            "class_name": op.__class__.__name__,
            "arg_names": arg_names,
        }
    )


@register_from_yaml_handler("ScalarUDF")
def _scalar_udf_from_yaml(yaml_dict: dict, compiler: any) -> any:
    encoded_fn = yaml_dict.get("pickle")
    if not encoded_fn:
        raise ValueError("Missing pickle data for ScalarUDF")
    fn = deserialize_udf_function(encoded_fn)

    args = tuple(
        translate_from_yaml(arg, compiler) for arg in yaml_dict.get("args", [])
    )
    if not args:
        raise ValueError("ScalarUDF requires at least one argument")

    arg_names = yaml_dict.get("arg_names", [f"arg{i}" for i in range(len(args))])

    fields = {
        name: Argument(pattern=rlz.ValueOf(arg.type()), typehint=arg.type())
        for name, arg in zip(arg_names, args)
    }

    bases = (ops.ScalarUDF,)
    meta = {
        "dtype": dt.dtype(yaml_dict["type"]["name"]),
        "__input_type__": ops.udf.InputType.BUILTIN,
        "__func__": property(fget=lambda _, f=fn: f),
        "__config__": {"volatility": "immutable"},
        "__udf_namespace__": None,
        "__module__": yaml_dict.get("module", "__main__"),
        "__func_name__": yaml_dict["unique_name"],
    }

    kwds = {**fields, **meta}
    class_name = yaml_dict.get("class_name", yaml_dict["unique_name"])

    node = type(
        class_name,
        bases,
        kwds,
    )

    return node(*args).to_expr()


@register_from_yaml_handler("View")
def _view_from_yaml(yaml_dict: dict, compiler: any) -> ir.Expr:
    underlying = translate_from_yaml(yaml_dict["args"][0], compiler)
    alias = yaml_dict.get("name")
    if alias:
        return underlying.alias(alias)
    return underlying


@register_from_yaml_handler("Mean")
def _mean_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    args = [translate_from_yaml(arg, compiler) for arg in yaml_dict["args"]]
    return args[0].mean()


@register_from_yaml_handler("Add", "Subtract", "Multiply", "Divide")
def _binary_arithmetic_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    left = translate_from_yaml(yaml_dict["args"][0], compiler)
    right = translate_from_yaml(yaml_dict["args"][1], compiler)
    op_map = {
        "Add": lambda left, right: left + right,
        "Subtract": lambda left, right: left - right,
        "Multiply": lambda left, right: left * right,
        "Divide": lambda left, right: left / right,
    }
    op_func = op_map.get(yaml_dict["op"])
    if op_func is None:
        raise ValueError(f"Unsupported arithmetic operation: {yaml_dict['op']}")
    return op_func(left, right)


@register_from_yaml_handler("Repeat")
def _repeat_from_yaml(yaml_dict: dict, compiler: Any) -> ibis.expr.types.Expr:
    arg = translate_from_yaml(yaml_dict["args"][0], compiler)
    times = translate_from_yaml(yaml_dict["args"][1], compiler)
    return ops.Repeat(arg, times).to_expr()


@register_from_yaml_handler("Sum")
def _sum_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    args = [translate_from_yaml(arg, compiler) for arg in yaml_dict["args"]]
    return args[0].sum()


@register_from_yaml_handler("Min")
def _min_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    args = [translate_from_yaml(arg, compiler) for arg in yaml_dict["args"]]
    return args[0].min()


@register_from_yaml_handler("Max")
def _max_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    args = [translate_from_yaml(arg, compiler) for arg in yaml_dict["args"]]
    return args[0].max()


@register_from_yaml_handler("Abs")
def _abs_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    arg = translate_from_yaml(yaml_dict["args"][0], compiler)
    return arg.abs()


@register_from_yaml_handler("Count")
def _count_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    arg = translate_from_yaml(yaml_dict["args"][0], compiler)
    return arg.count()


@register_from_yaml_handler("JoinReference")
def _join_reference_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    table_yaml = yaml_dict["args"][0]
    return translate_from_yaml(table_yaml, compiler)


@register_from_yaml_handler(
    "Equals", "NotEquals", "GreaterThan", "GreaterEqual", "LessThan", "LessEqual"
)
def _binary_compare_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    left = translate_from_yaml(yaml_dict["args"][0], compiler)
    right = translate_from_yaml(yaml_dict["args"][1], compiler)

    op_map = {
        "Equals": lambda left, right: left == right,
        "NotEquals": lambda left, right: left != right,
        "GreaterThan": lambda left, right: left > right,
        "GreaterEqual": lambda left, right: left >= right,
        "LessThan": lambda left, right: left < right,
        "LessEqual": lambda left, right: left <= right,
    }

    op_func = op_map.get(yaml_dict["op"])
    if op_func is None:
        raise ValueError(f"Unsupported comparison operation: {yaml_dict['op']}")
    return op_func(left, right)


@register_from_yaml_handler("Between")
def _between_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    args = [translate_from_yaml(arg, compiler) for arg in yaml_dict["args"]]
    return args[0].between(args[1], args[2])


@register_from_yaml_handler("Greater", "Less")
def _boolean_ops_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    args = [translate_from_yaml(arg, compiler) for arg in yaml_dict["args"]]
    op_name = yaml_dict["op"]
    op_map = {
        "Greater": lambda left, right: left > right,
        "Less": lambda left, right: left < right,
    }
    return op_map[op_name](*args)


@register_from_yaml_handler("And")
def _boolean_and_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    args = [translate_from_yaml(arg, compiler) for arg in yaml_dict.get("args", [])]
    if not args:
        raise ValueError("And operator requires at least one argument")
    return functools.reduce(lambda x, y: x & y, args)


@register_from_yaml_handler("Or")
def _boolean_or_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    args = [translate_from_yaml(arg, compiler) for arg in yaml_dict.get("args", [])]
    if not args:
        raise ValueError("Or operator requires at least one argument")
    return functools.reduce(lambda x, y: x | y, args)


@register_from_yaml_handler("Not")
def _not_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    arg = translate_from_yaml(yaml_dict["args"][0], compiler)
    return ~arg


@register_from_yaml_handler("IsNull")
def _is_null_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    arg = translate_from_yaml(yaml_dict["args"][0], compiler)
    return arg.isnull()


@register_from_yaml_handler(
    "ExtractYear",
    "ExtractMonth",
    "ExtractDay",
    "ExtractHour",
    "ExtractMinute",
    "ExtractSecond",
)
def _extract_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    arg = translate_from_yaml(yaml_dict["args"][0], compiler)
    op_map = {
        "ExtractYear": lambda x: x.year(),
        "ExtractMonth": lambda x: x.month(),
        "ExtractDay": lambda x: x.day(),
        "ExtractHour": lambda x: x.hour(),
        "ExtractMinute": lambda x: x.minute(),
        "ExtractSecond": lambda x: x.second(),
    }
    return op_map[yaml_dict["op"]](arg)


@register_from_yaml_handler("TimestampDiff")
def _timestamp_diff_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    left = translate_from_yaml(yaml_dict["args"][0], compiler)
    right = translate_from_yaml(yaml_dict["args"][1], compiler)
    return left - right


@register_from_yaml_handler("TimestampAdd", "TimestampSub")
def _timestamp_arithmetic_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    timestamp = translate_from_yaml(yaml_dict["args"][0], compiler)
    interval = translate_from_yaml(yaml_dict["args"][1], compiler)
    if yaml_dict["op"] == "TimestampAdd":
        return timestamp + interval
    else:
        return timestamp - interval


@register_from_yaml_handler("Cast")
def _cast_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    arg = translate_from_yaml(yaml_dict["args"][0], compiler)
    target_dtype = _type_from_yaml(yaml_dict["type"])
    return arg.cast(target_dtype)


@register_from_yaml_handler("CountStar")
def _count_star_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    arg = translate_from_yaml(yaml_dict["args"][0], compiler)

    return ops.CountStar(arg).to_expr()


@register_from_yaml_handler("StringSQLLike")
def _string_sql_like_from_yaml(yaml_dict: dict, compiler: Any) -> ir.Expr:
    args = yaml_dict.get("args", [])
    if not args:
        raise ValueError("Missing arguments for StringSQLLike operator")

    col = translate_from_yaml(args[0], compiler)

    if len(args) >= 2:
        pattern_expr = translate_from_yaml(args[1], compiler)
    else:
        pattern_value = args[0].get("value")
        if pattern_value is None:
            pattern_value = yaml_dict.get("value")
        if pattern_value is None:
            raise ValueError("Missing pattern for StringSQLLike operator")
        pattern_expr = ibis.literal(pattern_value, type=dt.String())

    escape = yaml_dict.get("escape")

    return ops.StringSQLLike(col, pattern_expr, escape=escape).to_expr()


def _type_from_yaml(yaml_dict: dict) -> dt.DataType:
    if isinstance(yaml_dict, str):
        raise ValueError(
            f"Unexpected string value '{yaml_dict}' - type definitions should be dictionaries"
        )
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

# === Helper functions for translating cache storage ===


def translate_storage(storage, compiler: any) -> dict:
    from letsql.common.caching import ParquetStorage, SourceStorage

    if isinstance(storage, ParquetStorage):
        return {"type": "ParquetStorage", "path": str(storage.path)}
    elif isinstance(storage, SourceStorage):
        return {
            "type": "SourceStorage",
            "source": getattr(storage.source, "profile_name", None),
        }
    else:
        raise NotImplementedError(f"Unknown storage type: {type(storage)}")


def load_storage_from_yaml(storage_yaml: dict, compiler: any):
    from letsql.expr.relations import ParquetStorage, _SourceStorage

    if storage_yaml["type"] == "ParquetStorage":
        default_profile = list(compiler.profiles.values())[0]
        return ParquetStorage(
            source=default_profile, path=pathlib.Path(storage_yaml["path"])
        )
    elif storage_yaml["type"] == "SourceStorage":
        source_profile_name = storage_yaml["source"]
        try:
            source = compiler.profiles[source_profile_name]
        except KeyError:
            raise ValueError(
                f"Source profile {source_profile_name!r} not found in compiler.profiles"
            )
        return _SourceStorage(source=source)
    else:
        raise NotImplementedError(f"Unknown storage type: {storage_yaml['type']}")
