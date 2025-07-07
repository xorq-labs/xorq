import cloudpickle
import dask
import pyarrow as pa
import toolz

import xorq as xo
import xorq.expr.datatypes as dt
import xorq.expr.udf as udf
from xorq.expr.ml.structer import Structer
from xorq.expr.udf import make_pandas_expr_udf


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
        deferred_model = deferred_model.as_table().cache(storage=storage)

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
    @toolz.curry
    def fit(df, target, cls, params):
        obj = cls(**dict(params))
        obj.fit(df, target)
        return obj

    @toolz.curry
    def transform(model, df):
        transformed = model.transform(df)
        return transformed

    deferred_model, model_udaf, deferred_transform = _deferred_fit_other(
        expr=expr,
        target=target,
        features=features,
        fit=fit(cls=cls, params=params),
        other=transform,
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
    @toolz.curry
    def fit(df, target, cls, params):
        obj = cls(**dict(params))
        obj.fit(df, target)
        return obj

    @toolz.curry
    def predict(model, df):
        predicted = model.predict(df)
        return predicted

    deferred_model, model_udaf, deferred_predict = _deferred_fit_other(
        expr=expr,
        target=target,
        features=features,
        fit=fit(cls=cls, params=params),
        other=predict,
        return_type=return_type,
        name_infix=name_infix,
        storage=storage,
    )
    return deferred_model, model_udaf, deferred_predict


@toolz.curry
def deferred_fit_transform_series_sklearn(
    expr, col, cls, return_type, params=(), name="predicted", storage=None
):
    @toolz.curry
    def fit(df, cls, col, params):
        model = cls(**dict(params))
        model.fit(df[col])
        return model

    @toolz.curry
    def transform(model, df, col):
        return model.transform(df[col]).toarray().tolist()

    deferred_model, model_udaf, deferred_transform = _deferred_fit_other(
        expr=expr,
        target=None,
        features=(col,),
        fit=fit(cls=cls, col=col, params=params),
        other=transform(col=col),
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
