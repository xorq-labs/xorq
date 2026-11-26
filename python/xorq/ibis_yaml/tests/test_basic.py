import datetime
import decimal
import pickle
import subprocess
import sys

import pytest

import xorq.vendor.ibis as ibis
import xorq.vendor.ibis.expr.datatypes as dt
from xorq.ibis_yaml.common import TranslationContext
from xorq.ibis_yaml.enums import RefEnum, RegistryEnum
from xorq.ibis_yaml.tests.conftest import get_dtype_yaml
from xorq.ibis_yaml.translate import make_context, translate_to_yaml


def test_unbound_table(t, compiler):
    yaml_dict = compiler.to_yaml(t)
    node_ref = yaml_dict["expression"][RefEnum.node_ref]
    expression = yaml_dict["definitions"][RegistryEnum.nodes][node_ref]
    assert expression["op"] == "UnboundTable"
    assert expression["name"] == "test_table"
    assert expression[RefEnum.schema_ref]

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.schema() == t.schema()
    assert roundtrip_expr.op().name == t.op().name


def test_field(t, compiler):
    expr = t.a
    yaml_dict = compiler.to_yaml(expr)

    expression = yaml_dict["expression"]
    assert expression["op"] == "Field"
    assert expression["name"] == "a"
    assert get_dtype_yaml(yaml_dict, expression) == {
        "op": "DataType",
        "type": "Int64",
        "nullable": True,
    }

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)
    assert roundtrip_expr.get_name() == expr.get_name()


def test_literal(compiler):
    lit = ibis.literal(42)
    yaml_dict = compiler.to_yaml(lit)

    expression = yaml_dict["expression"]
    assert expression["op"] == "Literal"
    assert expression["value"] == 42
    dtype_yaml = yaml_dict["definitions"][RegistryEnum.dtypes][
        expression["type"][RefEnum.dtype_ref]
    ]
    assert dtype_yaml == {
        "op": "DataType",
        "type": "Int8",
        "nullable": True,
    }

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(lit)


def test_binary_op(t, compiler):
    expr = t.a + 1
    yaml_dict = compiler.to_yaml(expr)
    expression = yaml_dict["expression"]
    assert expression["op"] == "Add"
    assert expression["right"]["op"] == "Literal"

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


@pytest.mark.parametrize(
    "primitive",
    [
        (ibis.literal(True), "Boolean"),
        (ibis.literal(1), "Int8"),
        (ibis.literal(1000), "Int16"),
        (ibis.literal(1.0), "Float64"),
        (ibis.literal("hello"), "String"),
        (ibis.literal(None), "Null"),
    ],
)
def test_primitive_types(compiler, primitive):
    lit, expected_type = primitive
    yaml_dict = compiler.to_yaml(lit)

    expression = yaml_dict["expression"]
    assert expression["op"] == "Literal"
    assert get_dtype_yaml(yaml_dict, expression)["type"] == expected_type

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(lit)
    assert roundtrip_expr.type().name == lit.type().name


def test_temporal_types(compiler):
    now = datetime.datetime.now()
    today = datetime.date.today()
    time = datetime.time(12, 0)
    temporals = [
        (ibis.literal(now), "Timestamp"),
        (ibis.literal(today), "Date"),
        (ibis.literal(time), "Time"),
    ]
    for lit, expected_type in temporals:
        yaml_dict = compiler.to_yaml(lit)
        expression = yaml_dict["expression"]
        assert expression["op"] == "Literal"
        assert get_dtype_yaml(yaml_dict, expression)["type"] == expected_type

        roundtrip_expr = compiler.from_yaml(yaml_dict)
        assert roundtrip_expr.equals(lit)
        assert roundtrip_expr.type().name == lit.type().name


def test_decimal_type(compiler):
    dec = decimal.Decimal("123.45")
    lit = ibis.literal(dec)
    yaml_dict = compiler.to_yaml(lit)
    expression = yaml_dict["expression"]
    assert expression["op"] == "Literal"
    dtype_yaml = get_dtype_yaml(yaml_dict, expression)
    assert dtype_yaml["type"] == "Decimal"
    assert dtype_yaml["nullable"]

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(lit)
    assert roundtrip_expr.type().name == lit.type().name


def test_array_type(compiler):
    lit = ibis.literal([1, 2, 3])
    yaml_dict = compiler.to_yaml(lit)
    expression = yaml_dict["expression"]
    dtype_yaml = get_dtype_yaml(yaml_dict, expression)
    inner = get_dtype_yaml(yaml_dict, dtype_yaml, key="value_type")
    assert expression["op"] == "Literal"
    assert expression["value"] == (1, 2, 3)
    assert dtype_yaml["type"] == "Array"
    assert inner["type"] == "Int8"

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(lit)
    assert roundtrip_expr.type().value_type == lit.type().value_type


def test_map_type(compiler):
    lit = ibis.literal({"a": 1, "b": 2})
    yaml_dict = compiler.to_yaml(lit)
    expression = yaml_dict["expression"]
    dtype_yaml = get_dtype_yaml(yaml_dict, expression)
    key_dtype_yaml = get_dtype_yaml(yaml_dict, dtype_yaml, key="key_type")
    value_dtype_yaml = get_dtype_yaml(yaml_dict, dtype_yaml, key="value_type")
    assert expression["op"] == "Literal"
    assert dtype_yaml["type"] == "Map"
    assert key_dtype_yaml["type"] == "String"
    assert value_dtype_yaml["type"] == "Int8"

    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(lit)
    assert roundtrip_expr.type().key_type == lit.type().key_type
    assert roundtrip_expr.type().value_type == lit.type().value_type


def test_complex_expression_roundtrip(t, compiler):
    expr = (t.a + 1).abs() * 2
    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_window_function_roundtrip(t, compiler):
    expr = t.a.sum().over(ibis.window(group_by=t.a))
    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.equals(expr)


def test_join_roundtrip(t, compiler, con):
    t2 = con.create_table("test_table_2", schema={"b": "int64"}, overwrite=True)
    expr = t.join(t2, t.a == t2.b)
    yaml_dict = compiler.to_yaml(expr)
    profiles = {con._profile.hash_name: con}
    roundtrip_expr = compiler.from_yaml(yaml_dict, profiles)
    assert roundtrip_expr.schema() == expr.schema()


def test_aggregation_roundtrip(t, compiler):
    expr = t.group_by(t.a).aggregate(count=t.a.count())
    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.schema() == expr.schema()


def test_aggregation_with_alias_roundtrip(t, compiler):
    expr = t.group_by(t.a.name("cust_id")).aggregate(count=t.a.count())
    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.schema() == expr.schema()


def test_table_perserves_namespace(compiler):
    expr = ibis.table({"b": "int64"}, name="t2", database="db", catalog="catalog")
    yaml_dict = compiler.to_yaml(expr)
    roundtrip_expr = compiler.from_yaml(yaml_dict)
    assert roundtrip_expr.op().namespace == expr.op().namespace


def test_issue_1546_snippet_cannot_reach_an_empty_dispatcher():
    # the snippet from #1546, verbatim, in an interpreter that has imported
    # nothing else: common must offer no dispatcher to call before the rules
    # exist, so it fails at the import, not with "No translation rule"
    code = """
import xorq.expr.datatypes as dt
from xorq.ibis_yaml.common import safe_translate_to_yaml
safe_translate_to_yaml(dt.Boolean(), None)
"""
    result = subprocess.run(
        (sys.executable, "-c", code), capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert "ImportError" in result.stderr, result.stderr
    assert "No translation rule" not in result.stderr


def test_context_round_trips_a_dtype():
    context = make_context()
    dtype = dt.Boolean()
    assert context.translate_from_yaml(context.translate_to_yaml(dtype)) == dtype


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="no-translators"),
        pytest.param({"to_yaml": None, "from_yaml": None}, id="none-translators"),
    ],
)
def test_context_without_translators_is_refused(kwargs: dict) -> None:
    with pytest.raises(TypeError):
        TranslationContext(**kwargs)


def test_context_translates_after_a_pickle_round_trip():
    context = pickle.loads(pickle.dumps(make_context()))
    dtype = dt.Int64()
    assert context.translate_from_yaml(context.translate_to_yaml(dtype)) == dtype


def test_no_rule_can_be_added_after_import():
    with pytest.raises(AttributeError):
        translate_to_yaml.register(complex, lambda value, context: {})
