import cloudpickle
from sklearn.base import clone, is_classifier, is_regressor

import xorq.expr.datatypes as dt
from xorq.expr.api import to_sql
from xorq.expr.ml.enums import ResponseMethod
from xorq.expr.ml.native.lower import model_fit
from xorq.expr.ml.native.ops import Fit, PickledEstimator, Predict
from xorq.expr.ml.native.rewrite import rewrite


def fit(expr, features, target, estimator):
    """A scalar expression whose value is `estimator` fitted on `expr`."""
    node = Fit(
        features=tuple(expr[name].op() for name in features),
        estimator=PickledEstimator(cloudpickle.dumps(clone(estimator))),
        target=expr[target].op() if target is not None else None,
    )
    return expr.aggregate(model=node.to_expr()).as_scalar()


def predict_dtype(fit_node):
    """Regressors score floats, classifiers their target's labels, others ints."""
    estimator = fit_node.estimator.instance
    match (is_regressor(estimator), is_classifier(estimator), fit_node.target):
        case (True, _, _):
            return dt.float64
        case (_, True, target) if target is not None:
            return target.dtype
        case _:
            return dt.int64


def predict(model, expr, method=ResponseMethod.PREDICT):
    """Score `expr`'s rows with `model`, reading the features it was fitted on."""
    fit_node, _ = model_fit(model.op())
    method = ResponseMethod(method)
    return_type = {
        ResponseMethod.PREDICT: predict_dtype(fit_node),
        ResponseMethod.PREDICT_PROBA: dt.Array(dt.float64),
    }[method]
    return Predict(
        model=model.op(),
        features=tuple(expr[name].op() for name in fit_node.feature_names),
        return_type=return_type,
        method=method,
    ).to_expr()


def explain(expr):
    """The SQL `expr` runs after ML rewrites; fits its models to get there."""
    return to_sql(rewrite(expr))
