import cloudpickle
import dask
import pandas as pd
import pyarrow as pa
import toolz

import xorq as xo
import xorq.expr.datatypes as dt
import xorq.expr.udf as udf
from xorq.expr.ml.structer import Structer
from xorq.expr.relations import TagType
from xorq.expr.udf import make_pandas_expr_udf


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
    names = model.get_feature_names_out()
    return pd.Series(
        (
            tuple({"key": key, "value": float(value)} for key, value in zip(names, row))
            for row in model.transform(df).toarray()
        )
    )


@toolz.curry
def predict_sklearn(model, df):
    predicted = model.predict(df)
    return predicted


@toolz.curry
def _deferred_fit_other(
    expr,
    target,
    features,
    fit,
    other,
    return_type,
    name_infix,
    storage=None,
):
    @toolz.curry
    def inner_fit(df, fit, target, features):
        # fixme: use inspect to ensure that `fit`'s signature has `features` and `target`/`*args` as arg names
        args = (df[list(features)],) + ((df[target],) if target else ())
        obj = fit(*args)
        return obj

    @toolz.curry
    def inner_other(model, df, other, features, return_type):
        return pa.array(
            other(model, df[list(features)]),
            type=return_type.to_pyarrow(),
        )

    def make_name(prefix, to_tokenize, n=32):
        tokenized = dask.base.tokenize(to_tokenize)
        return ("_" + prefix + "_" + tokenized)[:n].lower()

    features = tuple(features or expr.schema())
    schema = expr.select(features).schema()
    fit_schema = schema | (xo.schema({target: expr[target].type()}) if target else {})
    model_udaf = udf.agg.pandas_df(
        fn=toolz.compose(
            cloudpickle.dumps, inner_fit(fit=fit, target=target, features=features)
        ),
        schema=fit_schema,
        return_type=dt.binary,
        name=make_name(f"fit_{name_infix}", (fit, other)),
    )
    deferred_model = model_udaf.on_expr(expr)
    if storage:
        # cache the fitted model for MODEL metadata
        deferred_model = deferred_model.as_table().cache(storage=storage)
    # tag the model aggregation node for MODEL metadata
    deferred_model = deferred_model.tag(f"model_{name_infix}", type=TagType.MODEL)

    deferred_predict = make_pandas_expr_udf(
        computed_kwargs_expr=deferred_model,
        fn=inner_other(other=other, features=features, return_type=return_type),
        schema=schema,
        return_type=return_type,
        name=make_name(name_infix, (fit, other)),
    )

    return deferred_model, model_udaf, deferred_predict


@toolz.curry
def deferred_fit_predict(
    expr,
    target,
    features,
    fit,
    predict,
    return_type,
    name_infix="predict",
    storage=None,
):
    deferred_model, model_udaf, deferred_predict = _deferred_fit_other(
        expr=expr,
        target=target,
        features=features,
        fit=fit,
        other=predict,
        return_type=return_type,
        name_infix=name_infix,
        storage=storage,
    )
    return deferred_model, model_udaf, deferred_predict


@toolz.curry
def deferred_fit_transform(
    expr,
    features,
    fit,
    transform,
    return_type,
    target=None,
    name_infix="transform",
    storage=None,
):
    deferred_model, model_udaf, deferred_transform = _deferred_fit_other(
        expr=expr,
        target=target,
        features=features,
        fit=fit,
        other=transform,
        return_type=return_type,
        name_infix=name_infix,
        storage=storage,
    )
    return deferred_model, model_udaf, deferred_transform


@toolz.curry
def deferred_fit_transform_sklearn(
    expr,
    target,
    features,
    cls,
    return_type,
    params=(),
    name_infix="transformed",
    storage=None,
):
    deferred_model, model_udaf, deferred_transform = _deferred_fit_other(
        expr=expr,
        target=target,
        features=features,
        fit=fit_sklearn(cls=cls, params=params),
        other=transform_sklearn,
        return_type=return_type,
        name_infix=name_infix,
        storage=storage,
    )
    return deferred_model, model_udaf, deferred_transform


@toolz.curry
def deferred_fit_predict_sklearn(
    expr,
    target,
    features,
    cls,
    return_type,
    params=(),
    name_infix="predicted",
    storage=None,
):
    deferred_model, model_udaf, deferred_predict = _deferred_fit_other(
        expr=expr,
        target=target,
        features=features,
        fit=fit_sklearn(cls=cls, params=params),
        other=predict_sklearn,
        return_type=return_type,
        name_infix=name_infix,
        storage=storage,
    )
    return deferred_model, model_udaf, deferred_predict


@toolz.curry
def deferred_fit_transform_series_sklearn(
    expr, col, cls, return_type, params=(), name="predicted", storage=None
):
    deferred_model, model_udaf, deferred_transform = _deferred_fit_other(
        expr=expr,
        target=None,
        features=(col,),
        fit=fit_sklearn_series(col=col, cls=cls, params=params),
        other=transform_sklearn_series(col=col),
        return_type=return_type,
        name_infix=name,
        storage=storage,
    )
    return deferred_model, model_udaf, deferred_transform


@toolz.curry
def deferred_fit_transform_sklearn_struct(
    expr, features, cls, params=(), target=None, name_infix="transformed", storage=None
):
    @toolz.curry
    def fit(df, *args, cls, params):
        instance = cls(**dict(params))
        # if args exists, is likely (target,): see TfidfVectorizer
        instance.fit(df, *args)
        return instance

    @toolz.curry
    def transform(convert_array, model, df):
        return convert_array(model.transform(df))

    structer = Structer.from_instance_expr(cls(**dict(params)), expr, features=features)
    return deferred_fit_transform(
        expr=expr,
        features=list(features),
        fit=fit(cls=cls, params=params),
        transform=transform(structer.get_convert_array()),
        return_type=structer.return_type,
        target=target,
        name_infix=name_infix,
        storage=storage,
    )
