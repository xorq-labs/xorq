import ibis.expr.datatypes as dt
import ibis.expr.rules as rlz
from ibis.expr.operations.core import Value


class Rotate90(Value):
    arg: Value[dt.Binary]

    dtype = dt.binary
    shape = rlz.shape_like("arg")
