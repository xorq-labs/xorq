import functools
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
from xorq.expr.ml.structer import KV_ENCODED_TYPE, KVEncoder, Structer
from xorq.expr.udf import make_pandas_expr_udf
from xorq.vendor import ibis


@toolz.curry
def fit_sklearn(df, target=None, *, cls, params):
    obj = cls(**dict(params))
    obj.fit(df, target)
    return obj


@toolz.curry
def fit_sklearn_args(df, *args, cls, params):
    instance = cls(**dict(params))
    instance.fit(df, *args)
    return instance


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
def transform_sklearn_series_kv(model, df, col):
    return KVEncoder.encode(model, df[col])


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


@toolz.curry
def kv_encode_output(model, df):
    """
    Convert sklearn transform output to Array[Struct{key, value}] encoded format.

    Enables fully deferred execution by using a fixed return type regardless
    of the actual output schema, which is resolved at execution time via
    get_feature_names_out().
    """
    return KVEncoder.encode(model, df)


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
        if self.features is None or len(self.features) == 0:
            object.__setattr__(self, "features", tuple(self.expr.schema()))

    def make_name(self, prefix):
        to_tokenize = (self.fit, self.other)
        return make_name(prefix, to_tokenize)

    @property
    def schema(self):
        schema = self.expr.select(self.features).schema()
        return schema

    @property
    def encoded_cols(self):
        return KVEncoder.get_kv_encoded_cols(self.expr, self.features)

    @property
    def fit_schema(self):
        base = self.schema
        if self.target:
            return base | ibis.schema({self.target: self.expr[self.target].type()})
        return base

    @property
    def model_udaf(self):
        return udf.agg.pandas_df(
            fn=toolz.compose(
                cloudpickle.dumps,
                self._inner_fit(
                    fit=self.fit,
                    target=self.target,
                    features=self.features,
                    encoded_cols=self.encoded_cols,
                ),
            ),
            schema=self.fit_schema,
            return_type=dt.binary,
            name=self.make_name(f"fit_{self.name_infix}"),
        )

    @property
    def deferred_model(self):
        deferred_model = self.model_udaf.on_expr(self.expr)
        if self.cache:
            deferred_model = deferred_model.as_table().cache(cache=self.cache)
        return deferred_model

    @functools.cache
    # if we don't cache this, we get extra tags
    def make_deferred_other(self, fn, return_type, name_infix):
        wrapped_fn = self._inner_other(
            other=fn,
            features=self.features,
            return_type=return_type,
            encoded_cols=self.encoded_cols,
        )
        deferred_other = make_pandas_expr_udf(
            computed_kwargs_expr=self.deferred_model,
            fn=wrapped_fn,
            schema=self.schema,
            return_type=return_type,
            name=self.make_name(name_infix),
        )
        return deferred_other

    @property
    def deferred_other(self):
        return self.make_deferred_other(
            fn=self.other,
            return_type=self.return_type,
            name_infix=self.name_infix,
        )

    @property
    def deferred_model_udaf_other(self):
        return (self.deferred_model, self.model_udaf, self.deferred_other)

    @staticmethod
    @toolz.curry
    def _inner_fit(df, fit, target, features, encoded_cols):
        df_decoded, features_decoded = KVEncoder.decode_encoded_columns(
            df, features, encoded_cols
        )
        args = (df_decoded[list(features_decoded)],) + (
            (df_decoded[target],) if target else ()
        )
        return fit(*args)

    @staticmethod
    @toolz.curry
    def _inner_other(model, df, other, features, return_type, encoded_cols):
        df_decoded, features_decoded = KVEncoder.decode_encoded_columns(
            df, features, encoded_cols
        )
        return pa.array(
            other(model, df_decoded[list(features_decoded)]),
            type=return_type.to_pyarrow(),
        )

    @classmethod
    def from_fitted_step(cls, fitted_step, mode=None):
        # Auto-detect mode if not specified (backward compat)
        if mode is None:
            if fitted_step.is_predict and not fitted_step.is_transform:
                mode = "predict"
            else:
                mode = "transform"  # Default to transform

        kwargs = {
            # still need to add: fit, other, return_type, name_infix
            "expr": fitted_step.expr,
            "target": fitted_step.target,
            "features": fitted_step.features,
            "cache": fitted_step.cache,
        }
        if mode == "predict":
            return cls(
                fit=fit_sklearn(
                    cls=fitted_step.step.typ, params=fitted_step.step.params_tuple
                ),
                other=predict_sklearn,
                return_type=fitted_step.predict_return_type,
                name_infix="predict",
                **kwargs,
            )
        elif mode == "transform":
            structer = Structer.from_instance_expr(
                fitted_step.instance, fitted_step.expr, features=fitted_step.features
            )
            target = fitted_step.target if structer.needs_target else None
            match (
                structer.is_series,
                structer.is_kv_encoded,
                structer.struct_has_kv_fields,
            ):
                case (True, True, _):
                    (col,) = fitted_step.features
                    return cls(
                        fit=fit_sklearn_series(
                            col=col,
                            cls=fitted_step.step.typ,
                            params=fitted_step.step.params_tuple,
                        ),
                        other=transform_sklearn_series_kv(col=col),
                        return_type=KV_ENCODED_TYPE,
                        name_infix="transformed_encoded",
                        **kwargs
                        | {
                            "target": target,
                        },
                    )
                case (False, True, _):
                    # Pure KV-encoded output (struct is None)
                    return cls(
                        fit=fit_sklearn_args(
                            cls=fitted_step.step.typ,
                            params=fitted_step.step.params_tuple,
                        ),
                        other=kv_encode_output,
                        return_type=KV_ENCODED_TYPE,
                        name_infix="transformed_encoded",
                        **kwargs,
                    )
                case (False, False, True):
                    # Struct with KV-encoded fields
                    return cls(
                        fit=fit_sklearn_args(
                            cls=fitted_step.step.typ,
                            params=fitted_step.step.params_tuple,
                        ),
                        other=structer.get_convert_struct_with_kv(),
                        return_type=structer.return_type,
                        name_infix="transformed_struct_kv",
                        **kwargs,
                    )
                case _:
                    # Pure struct output (no KV columns)
                    return cls(
                        fit=fit_sklearn_args(
                            cls=fitted_step.step.typ,
                            params=fitted_step.step.params_tuple,
                        ),
                        other=transform_sklearn_struct(structer.get_convert_array()),
                        return_type=structer.return_type,
                        name_infix="transformed",
                        **kwargs,
                    )
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'transform' or 'predict'")


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


# Begin specific deferred
@toolz.curry
def deferred_fit_transform(
    expr,
    features,
    fit,
    other,
    return_type,
    target=None,
    name_infix="transform",
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
def deferred_fit_predict(
    expr,
    target,
    features,
    cls,
    return_type,
    params=(),
    name_infix="predict",
    cache=None,
):
    return DeferredFitOther(
        expr=expr,
        target=target,
        features=features,
        fit=fit_sklearn(cls=cls, params=params),
        other=predict_sklearn,
        return_type=return_type,
        name_infix=name_infix,
        cache=cache,
    )


@toolz.curry
def deferred_fit_transform_sklearn(
    expr,
    target,
    features,
    cls,
    return_type,
    params=(),
    name_infix="transformed",
    cache=None,
):
    return DeferredFitOther(
        expr=expr,
        target=target,
        features=features,
        fit=fit_sklearn(cls=cls, params=params),
        other=transform_sklearn,
        return_type=return_type,
        name_infix=name_infix,
        cache=cache,
    )


@toolz.curry
def deferred_fit_predict_sklearn(
    expr,
    target,
    features,
    cls,
    return_type,
    params=(),
    name_infix="predict",
    cache=None,
):
    return DeferredFitOther(
        expr=expr,
        target=target,
        features=features,
        fit=fit_sklearn(cls=cls, params=params),
        other=predict_sklearn,
        return_type=return_type,
        name_infix=name_infix,
        cache=cache,
    )


@toolz.curry
def deferred_fit_predict_proba_sklearn(
    expr,
    target,
    features,
    cls,
    return_type,
    params=(),
    name_infix="predict_proba",
    cache=None,
):
    return DeferredFitOther(
        expr=expr,
        target=target,
        features=features,
        fit=fit_sklearn(cls=cls, params=params),
        other=predict_proba_sklearn,
        return_type=return_type,
        name_infix=name_infix,
        cache=cache,
    )


@toolz.curry
def deferred_fit_decision_function_sklearn(
    expr,
    target,
    features,
    cls,
    return_type,
    params=(),
    name_infix="decision_function",
    cache=None,
):
    return DeferredFitOther(
        expr=expr,
        target=target,
        features=features,
        fit=fit_sklearn(cls=cls, params=params),
        other=decision_function_sklearn,
        return_type=return_type,
        name_infix=name_infix,
        cache=cache,
    )


@toolz.curry
def deferred_fit_feature_importances_sklearn(
    expr,
    target,
    features,
    cls,
    return_type,
    params=(),
    name_infix="feature_importances_",
    cache=None,
):
    return DeferredFitOther(
        expr=expr,
        target=target,
        features=features,
        fit=fit_sklearn(cls=cls, params=params),
        other=feature_importances_sklearn,
        return_type=return_type,
        name_infix=name_infix,
        cache=cache,
    )


@toolz.curry
def deferred_fit_transform_series_sklearn(
    expr, col, cls, return_type, params=(), name="predict", cache=None
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
def deferred_fit_transform_series_sklearn_encoded(
    expr, col, cls, params=(), name="transformed_encoded", cache=None
):
    return DeferredFitOther(
        expr=expr,
        target=None,
        features=(col,),
        fit=fit_sklearn_series(col=col, cls=cls, params=params),
        other=transform_sklearn_series_kv(col=col),
        return_type=KV_ENCODED_TYPE,
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
        fit=fit_sklearn_args(cls=cls, params=params),
        other=transform_sklearn_struct(structer.get_convert_array()),
        return_type=structer.return_type,
        target=target,
        name_infix=name_infix,
        cache=cache,
    )


@toolz.curry
def deferred_fit_transform_sklearn_encoded(
    expr,
    features,
    cls,
    params=(),
    target=None,
    name_infix="transformed_encoded",
    cache=None,
):
    return DeferredFitOther(
        expr=expr,
        features=list(features),
        fit=fit_sklearn_args(cls=cls, params=params),
        other=kv_encode_output,
        return_type=KV_ENCODED_TYPE,
        target=target,
        name_infix=name_infix,
        cache=cache,
    )
