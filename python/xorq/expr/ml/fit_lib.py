import functools
from typing import Callable

import cloudpickle
import dask.base
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
from xorq.vendor.ibis.expr.types.core import Expr


@toolz.curry
def fit_sklearn(df, target=None, *, cls, params_pickled):
    params = cloudpickle.loads(params_pickled)
    obj = cls(**dict(params))
    obj.fit(df, target)
    return obj


@toolz.curry
def fit_sklearn_series(df, col, cls, params_pickled):
    params = cloudpickle.loads(params_pickled)
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
def transform_sklearn_feature_names_out(model, df):
    """Transform and encode output using KVEncoder."""
    return KVEncoder.encode(model, df)


@toolz.curry
def fit_sklearn_struct(df, *args, cls, params_pickled):
    params = cloudpickle.loads(params_pickled)
    instance = cls(**dict(params))
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


@toolz.curry
def kv_encode_output(model, df):
    """
    Convert sklearn transform output to Array[Struct{key, value}] encoded format.

    Enables fully deferred execution by using a fixed return type regardless
    of the actual output schema, which is resolved at execution time via
    get_feature_names_out().
    """
    return KVEncoder.encode(model, df)


def _maybe_decode_encoded_columns(df, features, encoded_cols):
    """
    Decode specified KV-encoded columns.

    Enables Pipeline.from_instance() to work seamlessly with KV-encoded
    transformers by decoding intermediate encoded columns before fitting
    subsequent steps.
    """
    if not encoded_cols:
        return df, features

    result_df = df.copy()
    new_features = list(features)

    for col in encoded_cols:
        if col not in df.columns:
            continue
        first_row = result_df[col].iloc[0]
        decoded_names = [item["key"] for item in first_row]
        result_df = KVEncoder.decode(result_df, col)
        new_features = [f for f in new_features if f != col]
        new_features.extend(decoded_names)

    return result_df, tuple(new_features)


def _make_name(prefix, to_tokenize, n=32):
    tokenized = dask.base.tokenize(to_tokenize)
    return ("_" + prefix + "_" + tokenized)[:n].lower()


@frozen
class DeferredFitOther:
    expr = field(validator=instance_of(Expr))
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
            name=_make_name(f"fit_{self.name_infix}", (self.fit, self.other)),
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
        deferred_other = make_pandas_expr_udf(
            computed_kwargs_expr=self.deferred_model,
            fn=fn,
            schema=self.schema,
            return_type=return_type,
            name=_make_name(self.name_infix, (self.fit, self.other)),
        )
        return deferred_other

    @property
    def deferred_other(self):
        return self.make_deferred_other(
            fn=self._inner_other(
                other=self.other,
                features=self.features,
                return_type=self.return_type,
                encoded_cols=self.encoded_cols,
            ),
            return_type=self.return_type,
            name_infix=self.name_infix,
        )

    @property
    def deferred_model_udaf_other(self):
        return (self.deferred_model, self.model_udaf, self.deferred_other)

    @staticmethod
    @toolz.curry
    def _inner_fit(df, fit, target, features, encoded_cols):
        df_decoded, features_decoded = _maybe_decode_encoded_columns(
            df, features, encoded_cols
        )
        args = (df_decoded[list(features_decoded)],) + (
            (df_decoded[target],) if target else ()
        )
        return fit(*args)

    @staticmethod
    @toolz.curry
    def _inner_other(model, df, other, features, return_type, encoded_cols):
        df_decoded, features_decoded = _maybe_decode_encoded_columns(
            df, features, encoded_cols
        )
        return pa.array(
            other(model, df_decoded[list(features_decoded)]),
            type=return_type.to_pyarrow(),
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
    params_pickled = cloudpickle.dumps(params)
    return DeferredFitOther(
        expr=expr,
        target=target,
        features=features,
        fit=fit_sklearn(cls=cls, params_pickled=params_pickled),
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
    params_pickled = cloudpickle.dumps(params)
    return DeferredFitOther(
        expr=expr,
        target=target,
        features=features,
        fit=fit_sklearn(cls=cls, params_pickled=params_pickled),
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
    params_pickled = cloudpickle.dumps(params)
    return DeferredFitOther(
        expr=expr,
        target=target,
        features=features,
        fit=fit_sklearn(cls=cls, params_pickled=params_pickled),
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
    name_infix="predicted",
    cache=None,
):
    params_pickled = cloudpickle.dumps(params)
    return DeferredFitOther(
        expr=expr,
        target=target,
        features=features,
        fit=fit_sklearn(cls=cls, params_pickled=params_pickled),
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
    name_infix="predicted_proba",
    cache=None,
):
    params_pickled = cloudpickle.dumps(params)
    return DeferredFitOther(
        expr=expr,
        target=target,
        features=features,
        fit=fit_sklearn(cls=cls, params_pickled=params_pickled),
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
    params_pickled = cloudpickle.dumps(params)
    return DeferredFitOther(
        expr=expr,
        target=target,
        features=features,
        fit=fit_sklearn(cls=cls, params_pickled=params_pickled),
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
    params_pickled = cloudpickle.dumps(params)
    return DeferredFitOther(
        expr=expr,
        target=target,
        features=features,
        fit=fit_sklearn(cls=cls, params_pickled=params_pickled),
        other=feature_importances_sklearn,
        return_type=return_type,
        name_infix=name_infix,
        cache=cache,
    )


@toolz.curry
def deferred_fit_transform_series_sklearn(
    expr, col, cls, return_type, params=(), name="predicted", cache=None
):
    params_pickled = cloudpickle.dumps(params)
    return DeferredFitOther(
        expr=expr,
        target=None,
        features=(col,),
        fit=fit_sklearn_series(col=col, cls=cls, params_pickled=params_pickled),
        other=transform_sklearn_series(col=col),
        return_type=return_type,
        name_infix=name,
        cache=cache,
    )


@toolz.curry
def deferred_fit_transform_series_sklearn_packed(
    expr, col, cls, params=(), name="transformed_packed", cache=None
):
    params_pickled = cloudpickle.dumps(params)
    return DeferredFitOther(
        expr=expr,
        target=None,
        features=(col,),
        fit=fit_sklearn_series(col=col, cls=cls, params_pickled=params_pickled),
        other=transform_sklearn_series_kv(col=col),
        return_type=KV_ENCODED_TYPE,
        name_infix=name,
        cache=cache,
    )


@toolz.curry
def deferred_fit_transform_sklearn_struct(
    expr, features, cls, params=(), target=None, name_infix="transformed", cache=None
):
    params_pickled = cloudpickle.dumps(params)
    structer = Structer.from_instance_expr(cls(**dict(params)), expr, features=features)
    return DeferredFitOther(
        expr=expr,
        features=list(features),
        fit=fit_sklearn_struct(cls=cls, params_pickled=params_pickled),
        other=transform_sklearn_struct(structer.get_convert_array()),
        return_type=structer.return_type,
        target=target,
        name_infix=name_infix,
        cache=cache,
    )


@toolz.curry
def deferred_fit_transform_sklearn_packed(
    expr,
    features,
    cls,
    params=(),
    target=None,
    name_infix="transformed_packed",
    cache=None,
):
    """
    Generic wrapper for ANY sklearn transformer using packed format.

    Uses Array[Struct{key, value}] as output type, which allows fully deferred
    execution without needing to know the output schema at graph construction time.
    Feature names are resolved at execution time via get_feature_names_out().
    """

    @toolz.curry
    def fit(df, *args, cls, params_pickled):
        params = cloudpickle.loads(params_pickled)
        instance = cls(**dict(params))
        instance.fit(df, *args)
        return instance

    params_pickled = cloudpickle.dumps(params)

    return deferred_fit_transform(
        expr=expr,
        features=list(features),
        fit=fit(cls=cls, params_pickled=params_pickled),
        other=kv_encode_output,
        return_type=KV_ENCODED_TYPE,
        target=target,
        name_infix=name_infix,
        cache=cache,
    )
