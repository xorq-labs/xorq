import functools
from importlib.util import find_spec

import cloudpickle

import xorq.vendor.ibis.expr.operations as ops
from xorq.config import options
from xorq.expr.enums import Traversal
from xorq.expr.ml.enums import ResponseMethod
from xorq.expr.ml.native.lower import has_ml_nodes, make_lower_replacer, model_fit
from xorq.expr.ml.native.ops import FittedModel, PickledEstimator, Predict
from xorq.expr.transform import TransformPass


@functools.lru_cache(maxsize=128)
def materialize(subquery):
    """Fit the model behind a Predict's model argument, once per subquery."""
    fit, parent = model_fit(subquery)
    columns = (*fit.features, *((fit.target,) if fit.target is not None else ()))
    df = parent.to_expr().select(*(column.name for column in columns)).execute()
    target = (df[fit.target.name],) if fit.target is not None else ()
    model = fit.estimator.instance.fit(df[list(fit.feature_names)], *target)
    return FittedModel(PickledEstimator(cloudpickle.dumps(model)))


def try_inline(node, features):
    """orbital's SQL for `node`, or None when orbital is absent or refuses.

    Only labels and regression values inline; orbital's probability outputs
    are per-class columns, not the array Predict declares.
    """
    if node.method != ResponseMethod.PREDICT or find_spec("orbital") is None:
        return None
    from skl2onnx.common.exceptions import MissingShapeCalculator  # noqa: PLC0415

    from xorq.expr.ml.native.inline import inline_predict  # noqa: PLC0415

    try:
        return inline_predict(node, features)
    # no converter for an estimator, no orbital type for a feature, or inputs
    # orbital cannot combine (mixed numeric types into a bare model)
    except (MissingShapeCalculator, NotImplementedError, KeyError, ValueError):
        return None


def materialize_models(expr):
    return {
        node.model: materialize(node.model)
        for node in expr.op().find(Predict)
        if isinstance(node.model, ops.ScalarSubquery)
    }


def make_ml_replacer(expr):
    """Swap each Predict's model for its fit, then lower what is left.

    With ``options.ml.rewrite`` off nothing is fitted ahead of time: Fit and
    Predict lower straight to the UDF path.
    """
    lower = make_lower_replacer()
    if not options.ml.rewrite:
        return lower
    models = materialize_models(expr)
    rules = options.ml.rules

    def replacer(node, kwargs):
        match node:
            case Predict(model=model) if model in models:
                fitted = node.copy(model=models[model])
                features = (kwargs or {}).get("features", node.features)
                inlined = "inline" in rules and try_inline(fitted, features)
                return inlined or lower(fitted, kwargs)
            case _:
                return lower(node, kwargs)

    return replacer


def rewrite(expr):
    return expr.op().replace(make_ml_replacer(expr)).to_expr()


ML_PASS = TransformPass(
    name="ml",
    traversal=Traversal.BOUNDARY,
    build=lambda expr, ctx: make_ml_replacer(expr),
    when=lambda expr, ctx: has_ml_nodes(expr),
    after=("cache",),
)
