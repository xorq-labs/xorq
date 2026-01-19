import functools
from functools import wraps
from typing import Callable

import cloudpickle
import numpy as np
import pyarrow as pa
import toolz
from attr import (
    field,
    frozen,
)
from attr.validators import (
    deep_iterable,
    instance_of,
    optional,
)

import xorq.expr.datatypes as dt
import xorq.expr.udf as udf
from xorq.common.utils.name_utils import make_name
from xorq.expr.ml.structer import Structer
from xorq.expr.udf import make_pandas_expr_udf
from xorq.vendor import ibis


@toolz.curry
def fit_sklearn(df, target=None, *, cls, params):
    obj = cls(**dict(params))
    obj.fit(df, target)
    return obj


@toolz.curry
def fit_sklearn_series(df, col, cls, params):
    model = cls(**dict(params))
    model.fit(df[col])
    return model


@toolz.curry
def transform_sklearn(model, df):
    transformed = model.transform(df)
    return transformed


@toolz.curry
def transform_sklearn_series(model, df, col):
    return model.transform(df[col]).toarray().tolist()


@toolz.curry
def transform_sklearn_feature_names_out(model, df):
    import pandas as pd

    names = model.get_feature_names_out()
    return pd.Series(
        (
            tuple({"key": key, "value": float(value)} for key, value in zip(names, row))
            for row in model.transform(df).toarray()
        )
    )


@toolz.curry
def fit_sklearn_struct(df, *args, cls, params):
    instance = cls(**dict(params))
    # if args exists, is likely (target,): see TfidfVectorizer
    instance.fit(df, *args)
    return instance


@toolz.curry
def transform_sklearn_struct(convert_array, model, df):
    return convert_array(model.transform(df))


@toolz.curry
def predict_sklearn(model, df):
    predicted = model.predict(df)
    return predicted


@toolz.curry
def predict_proba_sklearn(model, df):
    """Predict class probabilities using sklearn model."""
    proba = model.predict_proba(df)
    return [row for row in proba]


@toolz.curry
def decision_function_sklearn(model, df):
    """Compute decision function scores with consistent array output."""
    scores = np.asarray(model.decision_function(df))
    if scores.ndim == 1:
        return [[float(value)] for value in scores]
    return [row.tolist() for row in scores]


@toolz.curry
def feature_importances_sklearn(model, df):
    """Extract feature importances from sklearn model as a single row."""
    importances = model.feature_importances_
    return [importances.tolist()]


@frozen
class DeferredFitOther:
    expr = field(validator=instance_of(ibis.Expr))
    target = field(validator=optional(instance_of(str)))
    features = field(
        validator=optional(deep_iterable(instance_of(str), instance_of(tuple))),
        converter=tuple,
    )
    fit = field(validator=instance_of(Callable))
    other = field(validator=instance_of(Callable))
    return_type = field(validator=instance_of(dt.DataType))
    name_infix = field(validator=instance_of(str))
    cache = field(default=None)

    def __attrs_post_init__(self):
        from xorq.caching import Cache

        if not isinstance(self.cache, (Cache, type(None))):
            raise ValueError(
                f"cache must be of type Optional[Cache], is of type {type(self.cache)}"
            )
        if self.features is None:
            object.__setattr__(self, "features", tuple(self.expr.schema()))

    def make_name(self, prefix):
        to_tokenize = (self.fit, self.other)
        return make_name(prefix, to_tokenize)

    @property
    def schema(self):
        schema = self.expr.select(self.features).schema()
        return schema

    @property
    def model_udaf(self):
        fit_schema = self.schema | (
            ibis.schema({self.target: self.expr[self.target].type()})
            if self.target
            else {}
        )
        model_udaf = udf.agg.pandas_df(
            fn=toolz.compose(
                cloudpickle.dumps,
                self.inner_fit(
                    fit=self.fit, target=self.target, features=self.features
                ),
            ),
            schema=fit_schema,
            return_type=dt.binary,
            name=self.make_name(f"fit_{self.name_infix}"),
        )
        return model_udaf

    @property
    def deferred_model(self):
        deferred_model = self.model_udaf.on_expr(self.expr)
        if self.cache:
            deferred_model = deferred_model.as_table().cache(cache=self.cache)
        return deferred_model

    @functools.cache
    # if we don't cache this, we get extra tags
    def make_deferred_other(self, fn, return_type, name_infix):
        deferred_other = make_pandas_expr_udf(
            computed_kwargs_expr=self.deferred_model,
            fn=fn,
            schema=self.schema,
            return_type=return_type,
            name=self.make_name(name_infix),
        )
        return deferred_other

    @property
    def deferred_other(self):
        return self.make_deferred_other(
            fn=self.inner_other(
                other=self.other, features=self.features, return_type=self.return_type
            ),
            return_type=self.return_type,
            name_infix=self.name_infix,
        )

    @property
    def deferred_model_udaf_other(self):
        return (self.deferred_model, self.model_udaf, self.deferred_other)

    @staticmethod
    @toolz.curry
    def inner_fit(df, fit, target, features):
        # fixme: use inspect to ensure that `fit`'s signature has `features` and `target`/`*args` as arg names
        args = (df[list(features)],) + ((df[target],) if target else ())
        obj = fit(*args)
        return obj

    @staticmethod
    @toolz.curry
    def inner_other(model, df, other, features, return_type):
        return pa.array(
            other(model, df[list(features)]),
            type=return_type.to_pyarrow(),
        )

    @classmethod
    def from_fitted_step(cls, fitted_step):
        kwargs = {
            "expr": fitted_step.expr,
            "target": fitted_step.target,
            "features": fitted_step.features,
            # missing: fit, other, return_type, name_infix
            "cache": fitted_step.cache,
        }
        sklearn_cls, params = fitted_step.step.typ, fitted_step.step.params_tuple

        if fitted_step.is_transform:
            # at the end of each of these conditions, we should have the kwargs for a DeferredFitOther
            match fitted_step.step.typ:
                # FIXME: formalize registration of non-Structer handling
                case type(__name__="TfidfVectorizer"):
                    # features must be length 1
                    (col,) = kwargs["features"]
                    kwargs = kwargs | {
                        "fit": fit_sklearn_series(
                            col=col, cls=sklearn_cls, params=params
                        ),
                        "other": transform_sklearn_series(col=col),
                        "return_type": dt.Array(dt.float64),
                        "name_infix": "transformed",
                    }
                case type(__name__="SelectKBest"):
                    # SelectKBest is a Structer special case that needs target
                    structer = Structer.from_instance_expr(
                        sklearn_cls(**dict(params)),
                        fitted_step.expr,
                        features=fitted_step.features,
                    )
                    kwargs = kwargs | {
                        "fit": fit_sklearn_struct(cls=sklearn_cls, params=params),
                        "other": transform_sklearn_struct(structer.get_convert_array()),
                        "return_type": structer.return_type,
                        "name_infix": "transformed",
                    }
                case type(get_step_kwargs=get_step_kwargs):
                    # FIXME: create abstract class for BaseEstimator with get_step_kwargs
                    kwargs = (
                        kwargs
                        | {
                            "fit": fit_sklearn(cls=sklearn_cls, params=params),
                        }
                        | get_step_kwargs()
                    )
                case _:
                    structer = Structer.from_instance_expr(
                        sklearn_cls(**dict(params)),
                        fitted_step.expr,
                        features=fitted_step.features,
                    )
                    kwargs = kwargs | {
                        "fit": fit_sklearn_struct(
                            cls=sklearn_cls,
                            params=params,
                        ),
                        "other": transform_sklearn_struct(structer.get_convert_array()),
                        "return_type": structer.return_type,
                        "name_infix": "transformed",
                    }
        elif fitted_step.is_predict:
            from xorq.expr.ml.pipeline_lib import get_predict_return_type

            kwargs = kwargs | {
                "fit": fit_sklearn(cls=sklearn_cls, params=params),
                "other": predict_sklearn,
                "return_type": get_predict_return_type(
                    instance=fitted_step.step.instance,
                    step=fitted_step.step,
                    expr=fitted_step.expr,
                    features=fitted_step.features,
                    target=fitted_step.target,
                ),
                "name_infix": "predicted",
            }
        else:
            raise ValueError("fitted_step is neither transform nor predict")
        instance = cls(**kwargs)
        return instance


@wraps(DeferredFitOther)
@toolz.curry
def deferred_fit_other(
    expr,
    target,
    features,
    fit,
    other,
    return_type,
    name_infix,
    cache=None,
):
    return DeferredFitOther(
        expr=expr,
        target=target,
        features=features,
        fit=fit,
        other=other,
        return_type=return_type,
        name_infix=name_infix,
        cache=cache,
    )


@toolz.curry
def deferred_fit_other_sklearn(
    expr,
    target,
    features,
    cls,
    other,
    return_type,
    name_infix,
    params=(),
    cache=None,
):
    return DeferredFitOther(
        expr=expr,
        target=target,
        features=features,
        fit=fit_sklearn(cls=cls, params=params),
        other=other,
        return_type=return_type,
        name_infix=name_infix,
        cache=cache,
    )


deferred_fit_transform = deferred_fit_other(target=None, name_infix="transform")
deferred_fit_predict = deferred_fit_other_sklearn(name_infix="predict")
deferred_fit_transform_sklearn = deferred_fit_other_sklearn(
    other=transform_sklearn,
    name_infix="transformed",
)
deferred_fit_predict_sklearn = deferred_fit_other_sklearn(
    other=predict_sklearn,
    name_infix="predicted",
)
deferred_fit_predict_proba_sklearn = deferred_fit_other_sklearn(
    other=predict_proba_sklearn,
    name_infix="predicted_proba",
)
deferred_fit_decision_function_sklearn = deferred_fit_other_sklearn(
    other=decision_function_sklearn,
    name_infix="decision_function",
)
deferred_fit_feature_importances_sklearn = deferred_fit_other_sklearn(
    other=feature_importances_sklearn,
    name_infix="feature_importances_",
)


@toolz.curry
def deferred_fit_transform_series_sklearn(
    expr, col, cls, return_type, params=(), name="predicted", cache=None
):
    return DeferredFitOther(
        expr=expr,
        target=None,
        features=(col,),
        fit=fit_sklearn_series(col=col, cls=cls, params=params),
        other=transform_sklearn_series(col=col),
        return_type=return_type,
        name_infix=name,
        cache=cache,
    )


@toolz.curry
def deferred_fit_transform_sklearn_struct(
    expr, features, cls, params=(), target=None, name_infix="transformed", cache=None
):
    structer = Structer.from_instance_expr(cls(**dict(params)), expr, features=features)
    return DeferredFitOther(
        expr=expr,
        features=list(features),
        fit=fit_sklearn_struct(cls=cls, params=params),
        other=transform_sklearn_struct(structer.get_convert_array()),
        return_type=structer.return_type,
        target=target,
        name_infix=name_infix,
        cache=cache,
    )
