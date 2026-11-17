import toolz

import xorq.vendor.ibis.expr.operations as ops
from xorq.common.utils.name_utils import make_name
from xorq.expr.ml.enums import ResponseMethod
from xorq.expr.ml.fit_lib import (
    DeferredFitOther,
    predict_proba_sklearn,
    predict_sklearn,
)
from xorq.expr.ml.native.ops import Fit, FittedModel, Predict
from xorq.expr.udf import make_pandas_udf
from xorq.vendor import ibis


METHOD_FNS = {
    ResponseMethod.PREDICT: predict_sklearn,
    ResponseMethod.PREDICT_PROBA: predict_proba_sklearn,
}


def fit_estimator(X, y=None, *, spec):
    return spec.instance.fit(X, y)


def deferred_fit(node, parent):
    return DeferredFitOther(
        expr=parent.to_expr(),
        target=node.target.name if node.target is not None else None,
        features=node.feature_names,
        fit=toolz.curry(fit_estimator, spec=node.estimator),
        other=predict_sklearn,
        return_type=node.dtype,
        name_infix="native",
    )


def model_fit(model):
    """The (Fit, parent relation) behind a Predict's model argument."""
    match model:
        case ops.ScalarSubquery(rel=ops.Aggregate(parent=parent, metrics=metrics)):
            ((_, fit),) = metrics.items()
            return fit, parent
        case _:
            raise TypeError(f"unsupported model argument: {type(model).__name__}")


def lower_predict(node, features):
    fit, parent = model_fit(node.model)
    construct = deferred_fit(fit, parent).make_deferred_other(
        fn=METHOD_FNS[node.method],
        return_type=node.return_type,
        name_infix=str(node.method),
    )
    return construct(*(feature.to_expr() for feature in features)).op()


def lower_fitted_predict(node, features):
    """Score with a FittedModel: a pandas UDF closing over the fitted estimator."""
    model = node.model.estimator.instance
    fn = METHOD_FNS[node.method]
    construct = make_pandas_udf(
        fn=lambda df: fn(model, df),
        schema=ibis.schema({feature.name: feature.dtype for feature in features}),
        return_type=node.return_type,
        name=make_name(f"fitted_{node.method}", (node.model.estimator,)),
    )
    return construct(*(feature.to_expr() for feature in features)).op()


def lower_fit(node):
    (parent,) = {feature.rel for feature in node.features}
    udaf = deferred_fit(node, parent).model_udaf
    columns = (*node.features, *((node.target,) if node.target is not None else ()))
    return udaf(*(column.to_expr() for column in columns)).op()


def make_lower_replacer():
    """Replace Predict and Fit with today's ExprScalarUDF and pandas UDAF.

    A Predict lowers from its original node: by the time the bottom-up walk
    reaches it, the Fit inside its model argument is already lowered.
    """

    def replacer(node, kwargs):
        match node:
            case Predict(model=FittedModel()):
                return lower_fitted_predict(
                    node, (kwargs or {}).get("features", node.features)
                )
            case Predict():
                return lower_predict(
                    node, (kwargs or {}).get("features", node.features)
                )
            case Fit():
                return lower_fit(node.__recreate__(kwargs) if kwargs else node)
            case _ if kwargs:
                return node.__recreate__(kwargs)
            case _:
                return node

    return replacer


def has_ml_nodes(expr):
    return bool(expr.op().find((Predict, Fit)))
