from __future__ import annotations

import hashlib
import operator
from functools import singledispatch
from typing import Any

import pyarrow as pa

import xorq.vendor.ibis.expr.operations as ops
from xorq.expr import (
    Aggregate,
    AggregateFunction,
    Alias,
    BinaryExpr,
    Case,
    Cast,
    Column,
    EmptyRelation,
    Filter,
    IsFalse,
    IsNotNull,
    IsNull,
    IsTrue,
    Join,
    Limit,
    Literal,
    Not,
    Ordered,
    Projection,
    Sort,
    SubqueryAlias,
    TableScan,
    Wildcard,
)
from xorq.internal import ContextProvider
from xorq.sql import parser
from xorq.vendor import ibis
from xorq.vendor.ibis.formats.pyarrow import PyArrowTableProxy


class Catalog(dict[str, Any]):
    """A catalog of tables and their schemas."""


@singledispatch
def convert(step, catalog, *args):
    raise TypeError(type(step))


def _generate_unique_id(schema: pa.Schema) -> str:
    return hashlib.md5("".join(str(schema).split()).encode()).hexdigest()


@convert.register(EmptyRelation)
def convert_empty_relation(empty_relation, catalog):
    arrow_schema = empty_relation.arrow_schema()
    schema = ibis.Schema.from_pyarrow(arrow_schema)
    data = pa.Table.from_pydict(
        dict.fromkeys(arrow_schema.names, []), schema=arrow_schema
    )
    name = f"empty_relation_{_generate_unique_id(arrow_schema)}"

    table = ops.InMemoryTable(
        name=name, schema=schema, data=PyArrowTableProxy(data)
    ).to_expr()

    catalog[name] = table

    return table


@convert.register(Limit)
def convert_limit(limit, catalog):
    name, table = None, None
    for i in limit.input():
        table = convert(i.to_variant(), catalog=catalog)

    try:
        name = table.get_name()
    except AttributeError:
        pass

    fetch = getattr(limit.fetch(), "real", None)
    skip = limit.skip() or 0

    table = table.limit(fetch, offset=skip)

    if name:
        catalog[name] = table

    return table


@convert.register(Wildcard)
def convert_wildcard(wildcard, catalog, table):
    return table


@convert.register(Projection)
def convert_projection(projection, catalog):
    name, table = None, None
    for i in projection.input():
        table = convert(i.to_variant(), catalog=catalog)

    try:
        name = table.get_name()
    except AttributeError:
        pass

    projections = [
        pr
        for p in projection.projections()
        if (pr := convert(p.to_variant(), catalog=catalog)) is not None
    ]
    if projections:
        table = table.select(projections)

    if name:
        catalog[name] = table

    return table


@convert.register(Sort)
def convert_sort(sort, catalog):
    name, table = None, None
    for i in sort.input():
        table = convert(i.to_variant(), catalog)

    if table.get_name():
        name = table.get_name()

    if expressions := sort.sort_exprs():
        keys = [convert(expr.to_variant(), catalog) for expr in expressions]
        table = table.order_by(keys)

    if name:
        catalog[name] = table

    return table


@convert.register(Case)
def convert_case(case, catalog):
    expr = convert(case.expr().to_variant(), catalog=catalog)
    when_then_expr = [
        (
            convert(i.to_variant(), catalog=catalog),
            convert(v.to_variant(), catalog=catalog),
        )
        for i, v in case.when_then_expr()
    ]
    else_expr = convert(case.else_expr().to_variant(), catalog=catalog)

    result = expr.case()

    for condition, value in when_then_expr:
        result = result.when(condition, value)

    result = result.else_(else_expr).end()

    return result


@convert.register(Filter)
def convert_filter(_filter, catalog):
    name, table = None, None
    for i in _filter.input():
        table = convert(i.to_variant(), catalog)

    try:
        name = table.get_name()
    except AttributeError:
        pass

    if predicate := _filter.predicate():
        predicate = convert(predicate.to_variant(), catalog)
        table = table.filter(predicate)

    if name:
        catalog[name] = table

    return table


@convert.register(TableScan)
def convert_table_scan(scan, catalog):
    table = catalog[scan.table_name()]

    if filters := getattr(scan, "filters", None):
        fs = filters()
        if fs:
            fs = [convert(f.to_variant(), catalog=catalog) for f in fs]
            table = table.filter(fs)

    if projections := getattr(scan, "projections", None):
        ps = [convert(p.to_variant(), catalog=catalog) for p in projections()]
        if ps:
            table = table.select(ps)
    elif projection := getattr(scan, "projection", None):
        ps = projection()
        if ps:
            table = table[[name for i, name in ps]]

    return table


@convert.register(Cast)
def convert_cast(cast, catalog):
    value = convert(cast.expr().to_variant(), catalog=catalog)
    return value.cast(cast.data_type().to_arrow_str())


@convert.register(Aggregate)
def convert_aggregate(aggregate, catalog):
    name, table = None, None
    for i in aggregate.input():
        table = convert(i.to_variant(), catalog)

    try:
        name = table.get_name()
    except AttributeError:
        pass

    metrics = []
    if aggregates := aggregate.aggregate_exprs():
        metrics.extend(convert(agg.to_variant(), catalog, table) for agg in aggregates)

    groups = [
        convert(group.to_variant(), catalog) for group in aggregate.group_by_exprs()
    ]

    table = table.aggregate(metrics, by=groups)

    for metric_name, metric in table.op().metrics.items():
        catalog[metric_name.lower()] = metric.to_expr()

    if name:
        catalog[name] = table

    return table


@convert.register(SubqueryAlias)
def convert_subquery_alias(subquery, catalog):
    alias = subquery.alias()

    name, table = None, None
    for i in subquery.input():
        table = convert(i.to_variant(), catalog=catalog)

    try:
        name = table.get_name()
    except AttributeError:
        pass

    if alias is not None:
        catalog[alias] = table
    elif name:
        catalog[name] = table

    return table


_reduction_methods = {"SUM": "sum", "AVG": "mean", "COUNT": "count"}


@convert.register(AggregateFunction)
def convert_aggregate_function(agg_fun, catalog, table):
    method = _reduction_methods[agg_fun.aggregate_type().upper()]
    args = []
    for arg in agg_fun.args():
        variant = arg.to_variant()
        if isinstance(variant, Wildcard):
            converted = convert(variant, catalog, table)
        else:
            converted = convert(variant, catalog)
        args.append(converted)

    this, *rest = args
    ibis_agg_expr = getattr(this, method)(*rest)
    catalog[str(agg_fun).lower()] = ibis_agg_expr
    return ibis_agg_expr


_scalar_functions = {"abs": "abs"}


@convert.register(Join)
def convert_join(join, catalog):
    left = convert(join.left().to_variant(), catalog=catalog)
    right = convert(join.right().to_variant(), catalog=catalog)

    left_name, right_name = None, None
    try:
        left_name, right_name = left.get_name(), right.get_name()
    except AttributeError:
        pass

    how = str(join.join_type()).lower()

    if join.filter():
        predicates = convert(join.filter().to_variant(), catalog=catalog)
        left = left.join(right, predicates=predicates, how=how)
    elif join.on():
        predicates = [
            operator.eq(
                convert(le.to_variant(), catalog=catalog),
                convert(ri.to_variant(), catalog=catalog),
            )
            for le, ri in join.on()
        ]
        left = left.join(right, predicates, how=how)

    if left_name:
        catalog[left_name] = left

    if right_name:
        catalog[right_name] = left

    return left


_binary_operations = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "=": operator.eq,
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "OR": operator.or_,
}


@convert.register(BinaryExpr)
def convert_binary(binary, catalog):
    op = _binary_operations[binary.op()]
    left = convert(binary.left().to_variant(), catalog=catalog)
    right = convert(binary.right().to_variant(), catalog=catalog)

    return op(left, right)


@convert.register(Column)
def convert_column(column, catalog):
    if column.relation():
        table = catalog[column.relation()]
        return table[column.name()]
    elif (expr := catalog.get(str(column).lower(), None)) is not None:
        return expr
    return None


@convert.register(Literal)
def convert_literal(literal, catalog):
    dtype = literal.data_type()
    value = None
    if dtype == "Int64":
        value = literal.value_i64()
    elif dtype == "Int8":
        value = literal.value_i8()
    elif dtype == "Int32":
        value = literal.value_i32()
    elif dtype == "Float64":
        value = literal.value_f64()
    elif dtype == "Utf8":
        value = literal.value_string()
    return ibis.literal(value)


@convert.register(Alias)
def convert_alias(alias, catalog):
    expr = convert(alias.expr().to_variant(), catalog=catalog)
    result = expr.name(alias.alias())
    catalog[alias.alias()] = result
    return result


@convert.register(Ordered)
def convert_ordered(ordered, catalog):
    this = convert(ordered.expr().to_variant(), catalog=catalog)
    asc = ordered.asc()
    return ibis.asc(this) if asc else ibis.desc(this)


@convert.register(IsTrue)
def convert_is_true(is_true, catalog):
    expr = convert(is_true.expr().to_variant(), catalog=catalog)
    return operator.eq(expr, True)


@convert.register(IsFalse)
def convert_is_false(is_false, catalog):
    expr = convert(is_false.expr().to_variant(), catalog=catalog)
    return operator.eq(expr, False)


@convert.register(IsNull)
def convert_is_null(is_null, catalog):
    expr = convert(is_null.expr().to_variant(), catalog=catalog)
    return expr.isnull()


@convert.register(IsNotNull)
def convert_is_not_null(is_not_null, catalog):
    expr = convert(is_not_null.expr().to_variant(), catalog=catalog)
    return expr.notnull()


@convert.register(Not)
def convert_not(is_not, catalog):
    expr = convert(is_not.expr().to_variant(), catalog=catalog)
    return expr.negate()


def plan_to_ibis(plan, catalog):
    """Parse a DataFusion logical plan into an Ibis expression.

    Parameters
    ----------
    plan : LogicalPlan
        DataFusion LogicalPlan
    catalog : dict
        A dictionary mapping table names to either schemas or ibis table expressions.
        If a schema is passed, a table expression will be created using the schema.

    Returns
    -------
    expr : ir.Expr

    """
    catalog = Catalog(
        {name: ibis.table(schema, name=name) for name, schema in catalog.items()}
    )

    return convert(plan.to_variant(), catalog=catalog)


def sql_to_ibis(
    sql: str, catalog: dict[str, ibis.ir.Table], dialect: str = None
) -> ibis.Expr:
    plan = parser.parse_sql(
        sql,
        ContextProvider({k: v.schema().to_pyarrow() for k, v in catalog.items()}),
        dialect=dialect,
    )

    catalog = Catalog({name: table for name, table in catalog.items()})

    return convert(plan.to_variant(), catalog=catalog)
