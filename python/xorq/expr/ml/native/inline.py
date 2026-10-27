import ibis
import orbital
from orbital import types as orbital_types

import xorq.expr.datatypes as dt
import xorq.vendor.ibis.expr.operations as ops
from xorq.common.utils.ibis_utils import from_ibis


ORBITAL_COLUMN_TYPES = {
    dt.Float64: orbital_types.DoubleColumnType,
    dt.Float32: orbital_types.FloatColumnType,
    dt.Int64: orbital_types.Int64ColumnType,
    dt.Int32: orbital_types.Int32ColumnType,
    dt.String: orbital_types.StringColumnType,
    dt.Boolean: orbital_types.BooleanColumnType,
}


def orbital_features(schema):
    return {name: ORBITAL_COLUMN_TYPES[type(typ)]() for name, typ in schema.items()}


def inline_pipeline(pipeline, expr, features):
    """Translate a fitted sklearn pipeline into a xorq expression over `expr`.

    orbital emits upstream ibis over a placeholder table; `from_ibis` maps it
    to xorq ops, and the placeholder is swapped for `expr`'s feature columns.
    """
    inputs = expr.select(features)
    schema = inputs.schema()
    parsed = orbital.parse_pipeline(pipeline, features=orbital_features(schema))
    placeholder = ibis.table(ibis.Schema.from_pyarrow(schema.to_pyarrow()), name="t")
    translated = from_ibis(orbital.translate(placeholder, parsed)).op()
    (unbound,) = translated.find(ops.UnboundTable)
    return translated.replace({unbound: inputs.op()}).to_expr()
