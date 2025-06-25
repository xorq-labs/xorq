"""This file is used to generate the Ibis API Expression implemented by LETSQL"""

from __future__ import annotations

import inspect
import re
from collections import defaultdict

import ibis.expr.operations as ops
import yaml
from ibis.backends.sql.compilers.base import ALL_OPERATIONS
from ibis.expr.types.arrays import ArrayValue
from ibis.expr.types.binary import BinaryValue
from ibis.expr.types.collections import SetValue
from ibis.expr.types.core import Expr
from ibis.expr.types.generic import Column, Scalar, Value
from ibis.expr.types.geospatial import GeoSpatialValue
from ibis.expr.types.inet import INETValue, MACADDRValue
from ibis.expr.types.joins import Join
from ibis.expr.types.json import JSONValue
from ibis.expr.types.logical import BooleanColumn, BooleanValue
from ibis.expr.types.maps import MapValue
from ibis.expr.types.numeric import (
    DecimalColumn,
    FloatingColumn,
    IntegerColumn,
    NumericColumn,
)
from ibis.expr.types.relations import Table
from ibis.expr.types.strings import StringValue
from ibis.expr.types.structs import StructValue
from ibis.expr.types.temporal import (
    DateValue,
    DayOfWeek,
    IntervalValue,
    TimestampValue,
    TimeValue,
)
from ibis.expr.types.uuid import UUIDValue

from xorq.backends.let import Backend as LETSQLBackend


support_matrix_ignored_operations = (ops.ScalarParameter,)

public_ops = ALL_OPERATIONS.difference(support_matrix_ignored_operations)

letsql_ops = {op.__name__ for op in public_ops if LETSQLBackend.has_operation(op)}

values = [
    ArrayValue,
    BinaryValue,
    SetValue,
    Expr,
    Value,
    Scalar,
    GeoSpatialValue,
    MACADDRValue,
    INETValue,
    JSONValue,
    BooleanValue,
    BooleanColumn,
    MapValue,
    NumericColumn,
    IntegerColumn,
    FloatingColumn,
    DecimalColumn,
    Table,
    Join,
    StringValue,
    StructValue,
    TimeValue,
    DateValue,
    Column,
    DayOfWeek,
    TimestampValue,
    IntervalValue,
    UUIDValue,
]

matching_result = {}
immediate = re.compile(r"return (_binop\()?ops\.(\w+)")
saving = re.compile(r"(node|op) = ops\.(\w+)")
node_ret = re.compile(r"return (_binop\()?(node|op)")

groups = defaultdict(list)
for value in values:
    if hasattr(value, "__module__"):
        groups[value.__module__].append(value)
    else:
        groups[value.__name__].append(value)


def order_key(v, order=None):
    if order is None:
        order = [
            "relations",
            "joins",
            "generic",
            "numeric",
            "strings",
            "temporal",
            "collections",
            "geospatial",
            "json",
            "maps",
            "uuid",
        ]
    module_name = v.split(".")[-1]
    return order.index(module_name) if module_name in order else len(order)


def extract_members(class_value, implemented_ops):
    class_members = set()
    for m in dir(class_value):
        at = getattr(class_value, m)
        if inspect.isfunction(at) and not at.__name__.startswith("__"):
            source_code = inspect.getsource(at)
            matches = list(immediate.finditer(source_code))
            for match in matches:
                if match:
                    op = match.groups()[-1]
                    if op in implemented_ops:
                        implemented_ops.remove(op)
                        class_members.add(at.__name__)

            if not matches:
                if node_ret.search(source_code) and (
                    matches := list(saving.finditer(source_code))
                ):
                    for match in matches:
                        if match:
                            op = match.groups()[-1]
                            if op in implemented_ops:
                                implemented_ops.remove(op)
                                class_members.add(at.__name__)
    return sorted(class_members)


pages = []
for key in sorted(groups, key=order_key):
    first = True
    page = {}
    for value in groups[key]:
        members = extract_members(value, letsql_ops)
        if members:
            if first:
                path = key.split(".")[-1]
                page = {
                    "kind": "page",
                    "path": f"expression-{path}",
                    "package": key,
                    "contents": [],
                }
                first = False
            page["contents"].append({"name": value.__name__, "members": members})
    if "contents" in page and page["contents"]:
        pages.append(page)

print(yaml.dump(pages, sort_keys=False))
