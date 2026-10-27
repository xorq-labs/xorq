import functools

import xorq.vendor.ibis.expr.operations as ops
from xorq.config import options
from xorq.expr.enums import Traversal
from xorq.expr.ml.native.lower import has_ml_nodes, make_lower_replacer
from xorq.expr.ml.native.ops import FittedModel, PickledEstimator, Predict
from xorq.expr.transform import TransformPass


@functools.lru_cache(maxsize=128)
def materialize(subquery):
    """Fit the model behind a Predict's model argument, once per subquery."""
    (pickled,) = subquery.rel.to_expr().execute().iloc[0]
    return FittedModel(PickledEstimator(pickled))


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

    def replacer(node, kwargs):
        match node:
            case Predict(model=model) if model in models:
                return lower(node.copy(model=models[model]), kwargs)
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
