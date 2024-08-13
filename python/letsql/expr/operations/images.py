import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.expr.operations.core import Value


class SegmentAnything(Value):
    arg: Value[dt.Binary]
    model_name: Value[dt.String]
    seed: Value[dt.Array[dt.Float64]]

    dtype = dt.binary
    shape = rlz.shape_like("arg")


class Rotate90(Value):
    arg: Value[dt.Binary]

    dtype = dt.binary
    shape = rlz.shape_like("arg")
