import cloudpickle
import dask
import pyarrow as pa
import toolz

import xorq.expr.datatypes as dt
import xorq.expr.udf as udf
from xorq.expr.ml.structer import (
    KV_ENCODED_TYPE,
    KVEncoder,
    Structer,
    get_kv_encoded_cols,
)
from xorq.expr.udf import make_pandas_expr_udf
from xorq.vendor import ibis


def decode_encoded_column(df, col_name="transformed"):
    """
    Decode Array[Struct{key, value}] column to named columns in a pandas DataFrame.

    Used at pipeline boundaries or before predictors that need decoded features.
    """
    return KVEncoder.decode(df, col_name)


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
    return model.transform(df)


@toolz.curry
def transform_sklearn_series(model, df, col):
    return model.transform(df[col]).toarray().tolist()


@toolz.curry
def transform_sklearn_feature_names_out(model, df):
    """Transform and encode output using KVEncoder."""
    return KVEncoder.encode(model, df)


@toolz.curry
def fit_sklearn_struct(df, *args, cls, params):
    instance = cls(**dict(params))
    instance.fit(df, *args)
    return instance


@toolz.curry
def transform_sklearn_struct(convert_array, model, df):
    return convert_array(model.transform(df))


@toolz.curry
def predict_sklearn(model, df):
    return model.predict(df)


@toolz.curry
def predict_proba_sklearn_packed(model, df):
    """
    Call predict_proba and return as packed format Array[Struct{key, value}].

    The keys are the class labels from model.classes_.
    """
    import pandas as pd

    proba = model.predict_proba(df)
    classes = model.classes_

    return pd.Series(
        [
            tuple({"key": str(cls), "value": float(p)} for cls, p in zip(classes, row))
            for row in proba
        ]
    )


@toolz.curry
def decision_function_sklearn_packed(model, df):
    """
    Call decision_function and return as packed format Array[Struct{key, value}].

    For binary classification, returns single value; for multiclass, one per class.
    """
    import pandas as pd

    decision = model.decision_function(df)

    if decision.ndim == 1:
        return pd.Series([({"key": "decision", "value": float(d)},) for d in decision])
    else:
        classes = model.classes_
        return pd.Series(
            [
                tuple(
                    {"key": str(cls), "value": float(d)} for cls, d in zip(classes, row)
                )
                for row in decision
            ]
        )


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
        result_df = decode_encoded_column(result_df, col)
        new_features = [f for f in new_features if f != col]
        new_features.extend(decoded_names)

    return result_df, tuple(new_features)


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
    def inner_fit(df, fit, target, features, encoded_cols):
        df_decoded, features_decoded = _maybe_decode_encoded_columns(
            df, features, encoded_cols
        )
        args = (df_decoded[list(features_decoded)],) + (
            (df_decoded[target],) if target else ()
        )
        return fit(*args)

    @toolz.curry
    def inner_other(model, df, other, features, return_type, encoded_cols):
        df_decoded, features_decoded = _maybe_decode_encoded_columns(
            df, features, encoded_cols
        )
        return pa.array(
            other(model, df_decoded[list(features_decoded)]),
            type=return_type.to_pyarrow(),
        )

    def make_name(prefix, to_tokenize, n=32):
        tokenized = dask.base.tokenize(to_tokenize)
        return ("_" + prefix + "_" + tokenized)[:n].lower()

    features = tuple(features or expr.schema())
    schema = expr.select(features).schema()
    encoded_cols = get_kv_encoded_cols(expr, features)
    fit_schema = schema | (ibis.schema({target: expr[target].type()}) if target else {})

    model_udaf = udf.agg.pandas_df(
        fn=toolz.compose(
            cloudpickle.dumps,
            inner_fit(
                fit=fit, target=target, features=features, encoded_cols=encoded_cols
            ),
        ),
        schema=fit_schema,
        return_type=dt.binary,
        name=make_name(f"fit_{name_infix}", (fit, other)),
    )
    deferred_model = model_udaf.on_expr(expr)
    if cache:
        deferred_model = deferred_model.as_table().cache(cache=cache)

    deferred_other = make_pandas_expr_udf(
        computed_kwargs_expr=deferred_model,
        fn=inner_other(
            other=other,
            features=features,
            return_type=return_type,
            encoded_cols=encoded_cols,
        ),
        schema=schema,
        return_type=return_type,
        name=make_name(name_infix, (fit, other)),
    )

    return deferred_model, model_udaf, deferred_other


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
    return _deferred_fit_other(
        expr=expr,
        target=target,
        features=features,
        fit=fit,
        other=predict,
        return_type=return_type,
        name_infix=name_infix,
        cache=cache,
    )


deferred_fit_transform = _deferred_fit_other(target=None, name_infix="transform")


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
    return _deferred_fit_other(
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
    return _deferred_fit_other(
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
def deferred_fit_transform_series_sklearn(
    expr, col, cls, return_type, params=(), name="predicted", cache=None
):
    params_pickled = cloudpickle.dumps(params)
    return _deferred_fit_other(
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
def deferred_fit_transform_sklearn_struct(
    expr, features, cls, params=(), target=None, name_infix="transformed", cache=None
):
    @toolz.curry
    def fit(X, *args, cls, params_pickled):
        params = cloudpickle.loads(params_pickled)
        instance = cls(**dict(params))
        instance.fit(X, *args)
        return instance

    @toolz.curry
    def transform(convert_array, model, df):
        return convert_array(model.transform(df))

    params_pickled = cloudpickle.dumps(params)
    structer = Structer.from_instance_expr(cls(**dict(params)), expr, features=features)

    return _deferred_fit_other(
        expr=expr,
        target=target,
        features=list(features),
        fit=fit(cls=cls, params_pickled=params_pickled),
        other=transform(structer.get_convert_array()),
        return_type=structer.return_type,
        name_infix=name_infix,
        cache=cache,
    )


@toolz.curry
def kv_encode_output(model, df):
    """
    Convert sklearn transform output to Array[Struct{key, value}] encoded format.

    Enables fully deferred execution by using a fixed return type regardless
    of the actual output schema, which is resolved at execution time via
    get_feature_names_out().
    """
    return KVEncoder.encode(model, df)


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


@toolz.curry
def deferred_fit_transform_only_sklearn_packed(
    expr,
    features,
    cls,
    params=(),
    name_infix="fit_transform_only",
    cache=None,
):
    """
    Wrapper for estimators that only have fit_transform (TSNE, MDS, etc.).

    These estimators cannot transform new data - they only produce embeddings
    for the training data.
    """

    @toolz.curry
    def fit_and_pack(df, cls, params_pickled, features):
        import pandas as pd

        params = cloudpickle.loads(params_pickled)
        instance = cls(**dict(params))
        result = instance.fit_transform(df)

        if hasattr(result, "toarray"):
            result = result.toarray()

        if hasattr(instance, "get_feature_names_out"):
            names = instance.get_feature_names_out()
        else:
            names = [f"component_{i}" for i in range(result.shape[1])]

        packed = pd.Series(
            [
                tuple(
                    {"key": str(name), "value": float(val)}
                    for name, val in zip(names, row)
                )
                for row in result
            ]
        )
        return (instance, packed)

    @toolz.curry
    def extract_packed(model_and_result, df, features):
        _, packed = model_and_result
        return packed

    params_pickled = cloudpickle.dumps(params)

    return deferred_fit_transform(
        expr=expr,
        features=list(features),
        fit=fit_and_pack(
            cls=cls, params_pickled=params_pickled, features=tuple(features)
        ),
        other=extract_packed(features=tuple(features)),
        return_type=KV_ENCODED_TYPE,
        target=None,
        name_infix=name_infix,
        cache=cache,
    )


@toolz.curry
def deferred_fit_predict_only_sklearn(
    expr,
    features,
    cls,
    params=(),
    name_infix="fit_predict_only",
    cache=None,
):
    """
    Wrapper for transductive clusterers that only have fit_predict (DBSCAN, etc.).

    These estimators cannot predict on new data - they only produce cluster labels
    for the training data.
    """

    @toolz.curry
    def fit_predict(df, cls, params_pickled):
        params = cloudpickle.loads(params_pickled)
        instance = cls(**dict(params))
        labels = instance.fit_predict(df)
        return (instance, labels)

    @toolz.curry
    def extract_labels(model_and_labels, df, features):
        _, labels = model_and_labels
        return labels

    params_pickled = cloudpickle.dumps(params)

    return deferred_fit_predict(
        expr=expr,
        target=None,
        features=list(features),
        fit=fit_predict(cls=cls, params_pickled=params_pickled),
        predict=extract_labels(features=tuple(features)),
        return_type=dt.int64,
        name_infix=name_infix,
        cache=cache,
    )
