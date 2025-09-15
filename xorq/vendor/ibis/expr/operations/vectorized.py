from __future__ import annotations

from types import FunctionType, LambdaType  # noqa: TCH003
from typing import Union

from public import public

import xorq.vendor.ibis.expr.datashape as ds
import xorq.vendor.ibis.expr.datatypes as dt
from xorq.vendor.ibis.common.typing import VarTuple  # noqa: TCH001
from xorq.vendor.ibis.expr.operations.analytic import Analytic
from xorq.vendor.ibis.expr.operations.core import Column, Value
from xorq.vendor.ibis.expr.operations.reductions import Reduction


class VectorizedUDF(Value):
    func: Union[FunctionType, LambdaType]
    func_args: VarTuple[Column]
    # TODO(kszucs): should rename these arguments to input_dtypes and return_dtype
    input_type: VarTuple[dt.DataType]
    return_type: dt.DataType

    @property
    def dtype(self):
        return self.return_type


@public
class ElementWiseVectorizedUDF(VectorizedUDF):
    """Node for element wise UDF."""

    shape = ds.columnar


@public
class ReductionVectorizedUDF(VectorizedUDF, Reduction):
    """Node for reduction UDF."""

    shape = ds.scalar


# TODO(kszucs): revisit
@public
class AnalyticVectorizedUDF(VectorizedUDF, Analytic):
    """Node for analytics UDF."""

    shape = ds.columnar
