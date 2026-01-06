import pyarrow as pa

from xorq.vendor.ibis import Schema
from xorq.vendor.ibis.expr import datatypes as dt
from xorq.vendor.ibis.formats.pyarrow import PyArrowSchema, PyArrowType


class DataFusionPyArrowType(PyArrowType):
    @classmethod
    def from_ibis(cls, dtype: dt.DataType) -> pa.DataType:
        if dtype.is_interval() and dtype.unit.short == "D":
            return pa.int64()
        elif dtype.is_struct():
            fields = [
                pa.field(name, cls.from_ibis(dtype), nullable=dtype.nullable)
                for name, dtype in dtype.items()
            ]
            return pa.struct(fields)
        else:
            return PyArrowType.from_ibis(dtype)


class DataFusionPyArrowSchema(PyArrowSchema):
    @classmethod
    def from_ibis(cls, schema: Schema) -> pa.Schema:
        """Convert a schema to a pyarrow schema."""
        fields = [
            pa.field(
                name, DataFusionPyArrowType.from_ibis(dtype), nullable=dtype.nullable
            )
            for name, dtype in schema.items()
        ]
        return pa.schema(fields)

    @classmethod
    def to_ibis(cls, schema: pa.Schema) -> Schema:
        """Convert a pyarrow schema to a schema."""
        fields = [
            (f.name, DataFusionPyArrowType.to_ibis(f.type, f.nullable)) for f in schema
        ]
        return Schema.from_tuples(fields)
