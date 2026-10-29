import functools

import ibis
import orbital
from orbital import types as orbital_types

import xorq.expr.datatypes as dt
import xorq.vendor.ibis.expr.operations as ops
from xorq.common.utils.ibis_utils import from_ibis
from xorq.vendor.ibis import Schema


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


def translate(pipeline, schema):
    """orbital's translation of `pipeline`, as xorq ops over an UnboundTable."""
    parsed = orbital.parse_pipeline(pipeline, features=orbital_features(schema))
    placeholder = ibis.table(ibis.Schema.from_pyarrow(schema.to_pyarrow()), name="t")
    return from_ibis(orbital.translate(placeholder, parsed)).op()


def inline_pipeline(pipeline, expr, features):
    """Translate a fitted sklearn pipeline into a xorq expression over `expr`.

    orbital emits upstream ibis over a placeholder table; `from_ibis` maps it
    to xorq ops, and the placeholder is swapped for `expr`'s feature columns.
    """
    inputs = expr.select(features)
    translated = translate(pipeline, inputs.schema())
    (unbound,) = translated.find(ops.UnboundTable)
    return translated.replace({unbound: inputs.op()}).to_expr()


@functools.lru_cache(maxsize=128)
def translated_output(estimator, schema):
    """The pipeline's first output as one value over the placeholder's fields.

    orbital chains a Project per step; each Field reading a Project is
    replaced by that Project's value until only placeholder Fields remain.
    """
    translated = translate(estimator.instance, schema)

    @functools.cache
    def resolve(value):
        fields = value.find(ops.Field, filter=ops.Value)
        return value.replace(
            {field: dereference(field) for field in fields}, filter=ops.Value
        )

    def dereference(field):
        match field.rel:
            case ops.Project(values=values):
                return resolve(values[field.name])
            case _:
                return field

    return resolve(next(iter(translated.values.values())))


def inline_predict(node, features):
    """`node`'s scores as SQL expressions over `features`, no Python UDF.

    None for a constant model: its translation is a scalar with no row to
    score.
    """
    schema = Schema({feature.name: feature.dtype for feature in features})
    output = translated_output(node.model.estimator, schema)
    if output.shape.is_scalar():
        return None
    by_name = {feature.name: feature for feature in features}
    fields = output.find(ops.Field, filter=ops.Value)
    return ops.Cast(
        output.replace(
            {field: by_name[field.name] for field in fields}, filter=ops.Value
        ),
        to=node.return_type,
    )
