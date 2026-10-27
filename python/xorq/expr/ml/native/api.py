import cloudpickle
from sklearn.base import clone, is_classifier, is_regressor

import xorq.expr.datatypes as dt
from xorq.expr.api import to_sql
from xorq.expr.ml.enums import ResponseMethod
from xorq.expr.ml.native.lower import model_fit
from xorq.expr.ml.native.ops import Fit, PickledEstimator, Predict
from xorq.expr.ml.native.rewrite import rewrite


def fit_node(expr, features, target, estimator):
    return Fit(
        features=tuple(expr[name].op() for name in features),
        estimator=PickledEstimator(cloudpickle.dumps(clone(estimator))),
        target=expr[target].op() if target is not None else None,
    )


def fit(expr, features, target, estimator):
    """A scalar expression whose value is `estimator` fitted on `expr`."""
    node = fit_node(expr, features, target, estimator)
    return expr.aggregate(model=node.to_expr()).as_scalar()


def fit_by(expr, by, features, target, estimator):
    """A table of `by` keys and, per key, `estimator` fitted on that group.

    `by` names columns of `expr`: groups are fitted from the rows read back,
    so computed keys are not supported.
    """
    match by:
        case str():
            keys = (by,)
        case list() | tuple() if all(isinstance(key, str) for key in by):
            keys = tuple(by)
        case _:
            raise TypeError(f"fit_by groups by column names, got {type(by).__name__}")
    node = fit_node(expr, features, target, estimator)
    return expr.group_by(keys).aggregate(model=node.to_expr())


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
