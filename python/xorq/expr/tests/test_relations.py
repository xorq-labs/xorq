import pytest

import xorq.vendor.ibis as ibis
from xorq.expr.relations import FlightUDXF


@pytest.mark.parametrize(
    "schema_val,pattern",
    [
        (ibis.schema({"x": "float64"}), "Schema validation failed, expected"),
        (None, "^Schema validation failed$"),
    ],
)
def test_flight_udxf_validate_schema_fail(schema_val, pattern):
    schema = ibis.schema({"x": "int64"})
    input_expr = ibis.table(schema, name="t")

    class DummyUDXF:
        schema_in_required = schema_val

        @staticmethod
        def schema_in_condition(sch):
            return False

        @staticmethod
        def calc_schema_out(sch):
            return None

    with pytest.raises(ValueError, match=pattern):
        FlightUDXF.validate_schema(input_expr, DummyUDXF)
