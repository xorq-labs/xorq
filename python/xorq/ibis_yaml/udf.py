import base64
from typing import Any, Dict

import cloudpickle
import dask

import xorq.vendor.ibis.expr.datatypes as dt
import xorq.vendor.ibis.expr.operations as ops
import xorq.vendor.ibis.expr.rules as rlz
from xorq.common.utils.dask_normalize.dask_normalize_utils import (
    normalize_seq_with_caller,
)
from xorq.expr.relations import FlightExpr, FlightUDXF
from xorq.ibis_yaml.common import (
    _translate_type,
    register_from_yaml_handler,
    translate_from_yaml,
    translate_to_yaml,
)
from xorq.ibis_yaml.utils import freeze
from xorq.vendor.ibis.common.annotations import Argument
from xorq.vendor.ibis.expr.operations.udf import InputType


@dask.base.normalize_token.register(InputType)
def tokenize_input_type(obj):
    return normalize_seq_with_caller(
        obj.__class__.__module__, obj.__class__.__name__, obj.name, obj.value
    )


def serialize_udf_function(fn: callable) -> str:
    pickled = cloudpickle.dumps(fn)
    encoded = base64.b64encode(pickled).decode("ascii")
    return encoded


def deserialize_udf_function(encoded_fn: str) -> callable:
    pickled = base64.b64decode(encoded_fn)
    return cloudpickle.loads(pickled)


@translate_to_yaml.register(ops.ScalarUDF)
def _scalar_udf_to_yaml(op: ops.ScalarUDF, compiler: Any) -> dict:
    input_type = getattr(op.__class__, "__input_type__", None)

    if input_type not in [ops.udf.InputType.BUILTIN, ops.udf.InputType.PYARROW]:
        raise NotImplementedError(
            f"Translation of UDFs with input type {getattr(op.__class__, '__input_type__', None)} is not supported"
        )
    arg_names = [
        name
        for name in dir(op)
        if not name.startswith("__") and name not in op.__class__.__slots__
    ]

    return freeze(
        {
            "op": "ScalarUDF",
            "func_name": op.__func_name__,
            "input_type": input_type,
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
    # TODO: use make_pandas_udf (expr/udf.py)
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
    input_type = yaml_dict.get("input_type")

    if input_type == "BUILTIN":
        input_type = ops.udf.InputType.BUILTIN
    elif input_type == "PYARROW":
        input_type = ops.udf.InputType.PYARROW

    fields = {
        name: Argument(pattern=rlz.ValueOf(arg.type()), typehint=arg.type())
        for name, arg in zip(arg_names, args)
    }

    bases = (ops.ScalarUDF,)
    meta = {
        "dtype": dt.dtype(yaml_dict["type"]["name"]),
        "__input_type__": input_type,
        "__func__": property(fget=lambda _, f=fn: f),
        "__config__": {"volatility": "immutable"},
        "__udf_namespace__": None,
        "__module__": yaml_dict.get("module", "__main__"),
        "__func_name__": yaml_dict["func_name"],
    }

    kwds = {**fields, **meta}
    class_name = yaml_dict.get("class_name", yaml_dict["func_name"])

    node = type(
        class_name,
        bases,
        kwds,
    )

    return node(*args).to_expr()


@translate_to_yaml.register(FlightExpr)
def flight_expr_to_yaml(op: FlightExpr, context: any) -> dict:
    input_expr_yaml = translate_to_yaml(op.input_expr, context)
    unbound_expr_yaml = translate_to_yaml(op.unbound_expr, context)

    schema_id = context.schema_registry.register_schema(op.schema)

    make_server_pickle = serialize_udf_function(op.make_server)
    make_connection_pickle = serialize_udf_function(op.make_connection)

    return freeze(
        {
            "op": "FlightExpr",
            "name": op.name,
            "schema_ref": schema_id,
            "input_expr": input_expr_yaml,
            "unbound_expr": unbound_expr_yaml,
            "make_server": make_server_pickle,
            "make_connection": make_connection_pickle,
            "do_instrument_reader": op.do_instrument_reader,
        }
    )


@register_from_yaml_handler("FlightExpr")
def flight_expr_from_yaml(yaml_dict: Dict, context: Any) -> Any:
    name = yaml_dict.get("name")
    input_expr_yaml = yaml_dict.get("input_expr")
    unbound_expr_yaml = yaml_dict.get("unbound_expr")
    make_server_pickle = yaml_dict.get("make_server")
    make_connection_pickle = yaml_dict.get("make_connection")
    do_instrument_reader = yaml_dict.get("do_instrument_reader", False)

    input_expr = translate_from_yaml(input_expr_yaml, context)
    unbound_expr = translate_from_yaml(unbound_expr_yaml, context)

    make_server = (
        deserialize_udf_function(make_server_pickle) if make_server_pickle else None
    )
    make_connection = (
        deserialize_udf_function(make_connection_pickle)
        if make_connection_pickle
        else None
    )

    return FlightExpr.from_exprs(
        input_expr=input_expr,
        unbound_expr=unbound_expr,
        make_server=make_server,
        make_connection=make_connection,
        name=name,
        do_instrument_reader=do_instrument_reader,
    ).to_expr()


@translate_to_yaml.register(FlightUDXF)
def flight_udxf_to_yaml(op: FlightUDXF, context: any) -> dict:
    input_expr_yaml = translate_to_yaml(op.input_expr, context)
    schema_id = context.schema_registry.register_schema(op.schema)
    udxf_pickle = serialize_udf_function(op.udxf)
    make_server_pickle = serialize_udf_function(op.make_server)
    make_connection_pickle = serialize_udf_function(op.make_connection)

    return freeze(
        {
            "op": "FlightUDXF",
            "name": op.name,
            "schema_ref": schema_id,
            "input_expr": input_expr_yaml,
            "udxf": udxf_pickle,
            "make_server": make_server_pickle,
            "make_connection": make_connection_pickle,
            "do_instrument_reader": op.do_instrument_reader,
        }
    )


@register_from_yaml_handler("FlightUDXF")
def flight_udxf_from_yaml(yaml_dict: Dict, context: Any) -> Any:
    name = yaml_dict.get("name")
    input_expr_yaml = yaml_dict.get("input_expr")
    udxf_pickle = yaml_dict.get("udxf")
    make_server_pickle = yaml_dict.get("make_server")
    make_connection_pickle = yaml_dict.get("make_connection")
    do_instrument_reader = yaml_dict.get("do_instrument_reader", False)

    input_expr = translate_from_yaml(input_expr_yaml, context)
    udxf = deserialize_udf_function(udxf_pickle)
    make_server = (
        deserialize_udf_function(make_server_pickle) if make_server_pickle else None
    )
    make_connection = (
        deserialize_udf_function(make_connection_pickle)
        if make_connection_pickle
        else None
    )

    return FlightUDXF.from_expr(
        input_expr=input_expr,
        udxf=udxf,
        make_server=make_server,
        make_connection=make_connection,
        name=name,
        do_instrument_reader=do_instrument_reader,
    ).to_expr()
