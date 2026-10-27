import cloudpickle
from attr import field, frozen
from attr.validators import instance_of

import xorq.expr.datatypes as dt
import xorq.vendor.ibis.expr.datashape as ds
import xorq.vendor.ibis.expr.operations as ops
import xorq.vendor.ibis.expr.rules as rlz
from xorq.expr.ml.enums import ResponseMethod
from xorq.vendor.ibis.common.annotations import attribute
from xorq.vendor.ibis.common.typing import VarTuple


@frozen
class PickledEstimator:
    """A sklearn estimator, hashable and tokenized by its pickled bytes."""

    pickled = field(validator=instance_of(bytes))

    def __dasher_tokenize__(self):
        return self.pickled

    @property
    def instance(self):
        return cloudpickle.loads(self.pickled)


class Fit(ops.Reduction):
    """Fit an estimator over a relation's rows; the value is the pickled model."""

    features: VarTuple[ops.Column]
    estimator: PickledEstimator
    target: ops.Column | None = None

    dtype = dt.binary

    @property
    def feature_names(self):
        return tuple(feature.name for feature in self.features)


class FittedModel(ops.Value):
    """A model already fitted: a literal standing in for its Fit subquery."""

    estimator: PickledEstimator

    shape = ds.scalar
    dtype = dt.binary


class Predict(ops.Value):
    """Score rows with a fitted model: one value per row of `features`."""

    model: ops.Value[dt.Binary]
    features: VarTuple[ops.Value]
    return_type: dt.DataType
    method: ResponseMethod = ResponseMethod.PREDICT

    shape = rlz.shape_like("features")

    @attribute
    def dtype(self):
        return self.return_type
