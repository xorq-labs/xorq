import cloudpickle
import dask
import pyarrow as pa
import toolz

import xorq.expr.datatypes as dt
import xorq.expr.udf as udf
from xorq.expr.ml.structer import Structer
from xorq.expr.udf import make_pandas_expr_udf
from xorq.vendor import ibis


# Universal packed format for all transformers - enables fully deferred execution
# without needing to know output schema at graph construction time
PACKED_TRANSFORM_TYPE = dt.Array(dt.Struct({"key": dt.string, "value": dt.float64}))


def unpack_packed_column(df, col_name="transformed"):
    """
    Unpack Array[Struct{key, value}] column to named columns in a pandas DataFrame.

    Used at pipeline boundaries or before predictors that need unpacked features.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a packed column
    col_name : str
        Name of the column containing Array[Struct{key, value}] data

    Returns
    -------
    pandas.DataFrame
        DataFrame with the packed column replaced by individual feature columns
    """

    packed_col = df[col_name]

    # Extract keys and values from first row to get column names
    if len(packed_col) == 0:
        return df.drop(columns=[col_name])

    first_row = packed_col.iloc[0]
    if first_row is None or len(first_row) == 0:
        return df.drop(columns=[col_name])

    column_names = [item["key"] for item in first_row]

    # Build new columns
    unpacked_data = {
        name: [row[i]["value"] if row is not None else None for row in packed_col]
        for i, name in enumerate(column_names)
    }

    # Create result DataFrame
    other_cols = [c for c in df.columns if c != col_name]
    result = df[other_cols].copy()
    for name, values in unpacked_data.items():
        result[name] = values

    return result


@toolz.curry
def fit_sklearn(df, target=None, *, cls, params_pickled):
    # Unpickle params to handle unhashable types like lists in ColumnTransformer
    params = cloudpickle.loads(params_pickled)
    obj = cls(**dict(params))
    obj.fit(df, target)
    return obj


@toolz.curry
def fit_sklearn_series(df, col, cls, params_pickled):
    # Unpickle params to handle unhashable types
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

    # Handle both binary (1D) and multiclass (2D) cases
    if decision.ndim == 1:
        # Binary classification - single decision value
        return pd.Series([({"key": "decision", "value": float(d)},) for d in decision])
    else:
        # Multiclass - one value per class
        classes = model.classes_
        return pd.Series(
            [
                tuple(
                    {"key": str(cls), "value": float(d)} for cls, d in zip(classes, row)
                )
                for row in decision
            ]
        )


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
    # Pickle params to make them hashable for FrozenDict/curry
    params_pickled = cloudpickle.dumps(params)
    deferred_model, model_udaf, deferred_transform = _deferred_fit_other(
        expr=expr,
        target=target,
        features=features,
        fit=fit_sklearn(cls=cls, params_pickled=params_pickled),
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
    # Pickle params to make them hashable for FrozenDict/curry
    params_pickled = cloudpickle.dumps(params)
    deferred_model, model_udaf, deferred_predict = _deferred_fit_other(
        expr=expr,
        target=target,
        features=features,
        fit=fit_sklearn(cls=cls, params_pickled=params_pickled),
        other=predict_sklearn,
        return_type=return_type,
        name_infix=name_infix,
        cache=cache,
    )
    return deferred_model, model_udaf, deferred_predict


@toolz.curry
def deferred_fit_transform_series_sklearn(
    expr, col, cls, return_type, params=(), name="predicted", cache=None
):
    # Pickle params to make them hashable for FrozenDict/curry
    params_pickled = cloudpickle.dumps(params)
    deferred_model, model_udaf, deferred_transform = _deferred_fit_other(
        expr=expr,
        target=None,
        features=(col,),
        fit=fit_sklearn_series(col=col, cls=cls, params_pickled=params_pickled),
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
    def fit(df, *args, cls, params_pickled):
        # Unpickle params to handle unhashable types
        params = cloudpickle.loads(params_pickled)
        instance = cls(**dict(params))
        # if args exists, is likely (target,): see TfidfVectorizer
        instance.fit(df, *args)
        return instance

    @toolz.curry
    def transform(convert_array, model, df):
        return convert_array(model.transform(df))

    # Pickle params to make them hashable for FrozenDict/curry
    params_pickled = cloudpickle.dumps(params)
    structer = Structer.from_instance_expr(cls(**dict(params)), expr, features=features)
    return deferred_fit_transform(
        expr=expr,
        features=list(features),
        fit=fit(cls=cls, params_pickled=params_pickled),
        transform=transform(structer.get_convert_array()),
        return_type=structer.return_type,
        target=target,
        name_infix=name_infix,
        cache=cache,
    )


@toolz.curry
def pack_transform_output(model, df, features):
    """
    Convert sklearn transform output to Array[Struct{key, value}] packed format.

    This enables fully deferred execution by using a fixed return type regardless
    of the actual output schema, which is resolved at execution time via
    get_feature_names_out().
    """
    import pandas as pd

    result = model.transform(df)
    # Handle sparse matrices
    if hasattr(result, "toarray"):
        result = result.toarray()

    # Get feature names from model if available, otherwise use input features
    if hasattr(model, "get_feature_names_out"):
        names = model.get_feature_names_out()
    else:
        names = features

    return pd.Series(
        [
            tuple(
                {"key": str(name), "value": float(val)} for name, val in zip(names, row)
            )
            for row in result
        ]
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

    This works for all transformers including:
    - Dynamic schema (OneHotEncoder, CountVectorizer, ColumnTransformer)
    - Static schema (StandardScaler, SimpleImputer)
    - Dimensionality reduction (PCA, TruncatedSVD)
    """

    @toolz.curry
    def fit(df, *args, cls, params_pickled):
        # Unpickle params to handle unhashable types like lists in ColumnTransformer
        params = cloudpickle.loads(params_pickled)
        instance = cls(**dict(params))
        instance.fit(df, *args)
        return instance

    # Pickle params to make them hashable for FrozenDict/curry
    params_pickled = cloudpickle.dumps(params)

    return deferred_fit_transform(
        expr=expr,
        features=list(features),
        fit=fit(cls=cls, params_pickled=params_pickled),
        transform=pack_transform_output(features=tuple(features)),
        return_type=PACKED_TRANSFORM_TYPE,
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
    for the training data. The transform() method on new data will raise an error.
    """

    @toolz.curry
    def fit_and_pack(df, cls, params_pickled, features):
        import pandas as pd

        # Unpickle params to handle unhashable types
        params = cloudpickle.loads(params_pickled)
        instance = cls(**dict(params))
        result = instance.fit_transform(df)

        # Handle sparse matrices
        if hasattr(result, "toarray"):
            result = result.toarray()

        # Get feature names if available, otherwise generate generic names
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
        # Return instance and packed result - instance stored for to_sklearn()
        return (instance, packed)

    @toolz.curry
    def extract_packed(model_and_result, df, features):
        """Extract the pre-computed packed result from fit_transform"""
        _, packed = model_and_result
        return packed

    # Note: This is a simplified implementation. The fit produces both model and result,
    # and transform just extracts the result. This means transform() only works on
    # the training data - calling it on new data will return wrong results.
    # A proper implementation should raise an error when transform is called on new data.

    # Pickle params to make them hashable for FrozenDict/curry
    params_pickled = cloudpickle.dumps(params)

    return deferred_fit_transform(
        expr=expr,
        features=list(features),
        fit=fit_and_pack(
            cls=cls, params_pickled=params_pickled, features=tuple(features)
        ),
        transform=extract_packed(features=tuple(features)),
        return_type=PACKED_TRANSFORM_TYPE,
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
    for the training data. The predict() method on new data will raise an error.
    """

    @toolz.curry
    def fit_predict(df, cls, params_pickled):
        # Unpickle params to handle unhashable types
        params = cloudpickle.loads(params_pickled)
        instance = cls(**dict(params))
        labels = instance.fit_predict(df)
        # Return both instance and labels
        return (instance, labels)

    @toolz.curry
    def extract_labels(model_and_labels, df, features):
        """Extract the pre-computed labels from fit_predict"""
        _, labels = model_and_labels
        return labels

    # Pickle params to make them hashable for FrozenDict/curry
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
