import functools

from ibis import Schema as IbisSchema
from ibis.expr.datatypes import DataType as IbisDataType
from ibis.formats.pandas import PandasDataFrameProxy as IbisPandasDataFrameProxy

import xorq.vendor.ibis.expr.operations as ops
from xorq.vendor.ibis import Schema
from xorq.vendor.ibis.expr.datatypes import DataType
from xorq.vendor.ibis.formats.pandas import PandasDataFrameProxy


@functools.singledispatch
def map_op(op, kwargs):
    try:
        klass_name = op.__class__.__name__
        cls = getattr(ops, klass_name)

        return cls(
            **dict(zip(op.argnames, tuple(map_op(arg, kwargs) for arg in op.args)))
        )
    except AttributeError:
        raise ValueError(f"Cannot map: {type(op)}")


@map_op.register(int)
@map_op.register(str)
@map_op.register(float)
def map_builtins(op, kwargs):
    return op


@map_op.register(IbisSchema)
def map_schema(schema, kwargs):
    return Schema(
        dict(zip(schema.names, tuple(map_op(typ, kwargs) for typ in schema.types)))
    )


@map_op.register(IbisDataType)
def map_datatype(datatype, kwargs):
    return DataType.from_pyarrow(datatype.to_pyarrow())


@map_op.register(IbisPandasDataFrameProxy)
def map_pandas_dataframe_proxy(proxy, kwargs):
    return PandasDataFrameProxy(proxy.obj)


def from_ibis(ibis_expr):
    return ibis_expr.op().replace(map_op).to_expr()
