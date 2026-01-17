import cloudpickle
import dask
import numpy as np
import pyarrow as pa
import toolz

import xorq.expr.datatypes as dt
import xorq.expr.udf as udf
from xorq.expr.ml.structer import Structer
from xorq.expr.udf import make_pandas_expr_udf
from xorq.vendor import ibis


def _make_name(prefix, to_tokenize, n=32):
    tokenized = dask.base.tokenize(to_tokenize)
    return ("_" + prefix + "_" + tokenized)[:n].lower()


def _make_other_fn(*, other, features, return_type):
    """Create a function that applies a model operation to features.

    Parameters
    ----------
    other : callable
        Function to apply (e.g., predict_sklearn, predict_proba_sklearn)
    features : tuple or list
        Feature column names (converted to tuple for immutability)
    return_type : DataType
        Return type for the operation

    Returns
    -------
    callable
        Function that applies the operation to a model and DataFrame
    """
    features = tuple(features)

    def inner(model, df):
        return pa.array(
            other(model, df[list(features)]),
            type=return_type.to_pyarrow(),
        )

    return inner


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
def _deferred_fit_other(
    expr,
    target,
    features,
    fit,
    other,
    return_type,
    name_infix,
    cache=None,
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
    fit_schema = schema | (ibis.schema({target: expr[target].type()}) if target else {})
    model_udaf = udf.agg.pandas_df(
        fn=toolz.compose(
            cloudpickle.dumps, inner_fit(fit=fit, target=target, features=features)
        ),
        schema=fit_schema,
        return_type=dt.binary,
        name=make_name(f"fit_{name_infix}", (fit, other)),
    )
    deferred_model = model_udaf.on_expr(expr)
    if cache:
        deferred_model = deferred_model.as_table().cache(cache=cache)

    deferred_predict = make_pandas_expr_udf(
        computed_kwargs_expr=deferred_model,
        fn=inner_other(other=other, features=features, return_type=return_type),
        schema=schema,
        return_type=return_type,
        name=make_name(name_infix, (fit, other)),
    )

    return deferred_model, model_udaf, deferred_predict


def deferred_other_from_trained_model(
    *,
    schema,
    features,
    deferred_model,
    other,
    return_type,
    name_infix,
):
    """Create a deferred operation from a trained model.

    Parameters
    ----------
    schema : ibis.Schema
        Schema of the input features
    features : tuple or list
        Feature column names (converted to tuple for immutability)
    deferred_model : ibis.Expr
        Expression containing the trained model
    other : callable
        Function to apply to the model (e.g., predict_proba_sklearn)
    return_type : DataType
        Return type of the operation
    name_infix : str
        Name component for the UDF

    Returns
    -------
    ibis.Expr
        UDF expression that applies the operation
    """
    features = tuple(features)
    other_fn = _make_other_fn(
        other=other,
        features=features,
        return_type=return_type,
    )
    return make_pandas_expr_udf(
        computed_kwargs_expr=deferred_model,
        fn=other_fn,
        schema=schema,
        return_type=return_type,
        name=_make_name(
            name_infix,
            (features, other),
        ),
    )


@toolz.curry
def deferred_fit_predict(
    expr,
    target,
    features,
    fit,
    predict,
    return_type,
    name_infix="predict",
    cache=None,
):
    deferred_model, model_udaf, deferred_predict = _deferred_fit_other(
        expr=expr,
        target=target,
        features=features,
        fit=fit,
        other=predict,
        return_type=return_type,
        name_infix=name_infix,
        cache=cache,
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
    cache=None,
):
    deferred_model, model_udaf, deferred_transform = _deferred_fit_other(
        expr=expr,
        target=target,
        features=features,
        fit=fit,
        other=transform,
        return_type=return_type,
        name_infix=name_infix,
        cache=cache,
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
    cache=None,
):
    deferred_model, model_udaf, deferred_transform = _deferred_fit_other(
        expr=expr,
        target=target,
        features=features,
        fit=fit_sklearn(cls=cls, params=params),
        other=transform_sklearn,
        return_type=return_type,
        name_infix=name_infix,
        cache=cache,
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
    cache=None,
):
    deferred_model, model_udaf, deferred_predict = _deferred_fit_other(
        expr=expr,
        target=target,
        features=features,
        fit=fit_sklearn(cls=cls, params=params),
        other=predict_sklearn,
        return_type=return_type,
        name_infix=name_infix,
        cache=cache,
    )
    return deferred_model, model_udaf, deferred_predict


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
    """Create deferred predict_proba operation for sklearn models."""
    deferred_model, model_udaf, deferred_predict_proba = _deferred_fit_other(
        expr=expr,
        target=target,
        features=features,
        fit=fit_sklearn(cls=cls, params=params),
        other=predict_proba_sklearn,
        return_type=return_type,
        name_infix=name_infix,
        cache=cache,
    )
    return deferred_model, model_udaf, deferred_predict_proba


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
    """Create deferred decision_function operation for sklearn models."""
    deferred_model, model_udaf, deferred_decision = _deferred_fit_other(
        expr=expr,
        target=target,
        features=features,
        fit=fit_sklearn(cls=cls, params=params),
        other=decision_function_sklearn,
        return_type=return_type,
        name_infix=name_infix,
        cache=cache,
    )
    return deferred_model, model_udaf, deferred_decision


def deferred_fit_feature_importances_sklearn(
    expr,
    target,
    features,
    cls,
    return_type,
    params=(),
    name_infix="feature_importances",
    cache=None,
):
    """Create deferred feature_importances operation for sklearn models."""
    deferred_model, model_udaf, deferred_importances = _deferred_fit_other(
        expr=expr,
        target=target,
        features=features,
        fit=fit_sklearn(cls=cls, params=params),
        other=feature_importances_sklearn,
        return_type=return_type,
        name_infix=name_infix,
        cache=cache,
    )
    return deferred_model, model_udaf, deferred_importances


def deferred_predict_proba_from_trained_model(
    *,
    schema,
    features,
    deferred_model,
    return_type,
    name_infix="predicted_proba",
):
    """Create deferred predict_proba from a trained model."""
    return deferred_other_from_trained_model(
        schema=schema,
        features=features,
        deferred_model=deferred_model,
        other=predict_proba_sklearn,
        return_type=return_type,
        name_infix=name_infix,
    )


def deferred_decision_function_from_trained_model(
    *,
    schema,
    features,
    deferred_model,
    return_type,
    name_infix="decision_function",
):
    """Create deferred decision_function from a trained model."""
    return deferred_other_from_trained_model(
        schema=schema,
        features=features,
        deferred_model=deferred_model,
        other=decision_function_sklearn,
        return_type=return_type,
        name_infix=name_infix,
    )


def deferred_feature_importances_from_trained_model(
    *,
    schema,
    features,
    deferred_model,
    return_type,
    name_infix="feature_importances",
):
    """Create deferred feature_importances from a trained model."""
    return deferred_other_from_trained_model(
        schema=schema,
        features=features,
        deferred_model=deferred_model,
        other=feature_importances_sklearn,
        return_type=return_type,
        name_infix=name_infix,
    )


@toolz.curry
def deferred_fit_transform_series_sklearn(
    expr, col, cls, return_type, params=(), name="predicted", cache=None
):
    deferred_model, model_udaf, deferred_transform = _deferred_fit_other(
        expr=expr,
        target=None,
        features=(col,),
        fit=fit_sklearn_series(col=col, cls=cls, params=params),
        other=transform_sklearn_series(col=col),
        return_type=return_type,
        name_infix=name,
        cache=cache,
    )
    return deferred_model, model_udaf, deferred_transform


@toolz.curry
def deferred_fit_transform_sklearn_struct(
    expr, features, cls, params=(), target=None, name_infix="transformed", cache=None
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
        cache=cache,
    )
