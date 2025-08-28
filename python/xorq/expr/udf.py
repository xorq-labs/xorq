import functools
from typing import Any

import cloudpickle
import pyarrow as pa
import toolz

import xorq.expr.datatypes as dt
import xorq.vendor.ibis.expr.rules as rlz
import xorq.vendor.ibis.expr.types as ir
from xorq.vendor.ibis.common.annotations import Argument
from xorq.vendor.ibis.common.collections import FrozenDict
from xorq.vendor.ibis.expr.operations import Namespace
from xorq.vendor.ibis.expr.operations.udf import (
    AggUDF,
    InputType,
    ScalarUDF,
    _make_udf_name,
    _wrap,
    restore_udf,
    scalar,
)
from xorq.vendor.ibis.expr.operations.udf import (
    agg as _agg,
)


def property_wrap_fn(fn):
    return property(fget=lambda _, fn=fn: fn)


def arrays_to_df(names, *arrays):
    import pandas as pd

    return pd.DataFrame(
        {name: array.to_pandas() for (name, array) in zip(names, arrays)}
    )


def make_pyarrow_array(return_type, series):
    return pa.Array.from_pandas(series, type=return_type.to_pyarrow())


def make_dunder_func(fn, schema, return_type=None):
    def fn_from_arrays(*arrays):
        df = arrays_to_df(schema, *arrays)
        value = fn(df)
        if return_type is not None:
            value = make_pyarrow_array(return_type, value)
        return value

    return property_wrap_fn(fn_from_arrays)


def make_expr_scalar_udf_dunder_func(fn, schema, return_type):
    def fn_from_arrays(*arrays, computed_arg=None, **kwargs):
        if computed_arg is None:
            raise ValueError(
                "Caller must bind computed_arg to the output of computed_kwargs_expr"
            )
        df = arrays_to_df(schema, *arrays)
        value = fn(computed_arg, df, **kwargs)
        return make_pyarrow_array(
            return_type,
            value,
        )

    return property_wrap_fn(fn_from_arrays)


@toolz.curry
def wrap_model(value, model_key="model"):
    return cloudpickle.dumps({model_key: value})


unwrap_model = cloudpickle.loads


class ExprScalarUDF(ScalarUDF):
    @property
    def computed_kwargs_expr(self):
        # must push the expr into __config__ so that it doesn't get turned into a window function
        return self.__config__["computed_kwargs_expr"]

    @property
    def post_process_fn(self):
        return self.__config__["post_process_fn"]

    @property
    def schema(self):
        return self.__config__["schema"]

    def on_expr(self, e, **kwargs):
        # rebind deferred_model (computed_kwargs_expr) to a new expr
        return type(self)(*(e[c] for c in self.schema), **kwargs)

    def with_computed_kwargs_expr(self, computed_kwargs_expr):
        # we must create a new typ to set __config__
        # from xorq.ibis_yaml.udf import make_op_kwargs
        kwargs = dict(zip(self.argnames, self.args))
        fields = {
            argname: Argument(pattern=rlz.ValueOf(typ), typehint=typ)
            for (argname, typ) in (
                (argname, arg.dtype) for argname, arg in kwargs.items()
            )
        }
        meta = {
            name: getattr(self, name)
            for name in (
                "dtype",
                "__input_type__",
                "__func_name__",
                "__udf_namespace__",
                "__module__",
            )
        } | {
            "__func__": staticmethod(self.__func__),
        }
        config = FrozenDict(
            self.__config__ | {"computed_kwargs_expr": computed_kwargs_expr}
        )
        new_typ = type(
            self.__class__.__name__,
            self.__class__.__bases__,
            fields | meta | {"__config__": config},
        )
        new_op = new_typ(**kwargs)
        return new_op

    def __reduce__(self):
        state = dict(zip(self.__argnames__, self.__args__))

        meta = {
            k: getattr(self.__class__, k)
            for k in (
                "dtype",
                "__input_type__",
                "__func__",
                "__config__",
                "__udf_namespace__",
                "__module__",
                "__func_name__",
            )
        }
        return restore_udf, (
            self.__class__.__name__,
            self.__class__.__bases__,
            meta,
            state,
        )


@toolz.curry
def make_pandas_expr_udf(
    computed_kwargs_expr,
    fn,
    schema,
    return_type=dt.binary,
    database=None,
    catalog=None,
    name=None,
    *,
    post_process_fn=unwrap_model,
    **kwargs,
):
    """
    Create an expression-based scalar UDF that incorporates pre-computed values.

    This function creates a special type of scalar UDF that can access pre-computed
    values (like trained machine learning models) during execution. The pre-computed
    value is generated from a separate expression and passed to the UDF function,
    enabling complex workflows like model training and inference within the same
    query pipeline.

    Parameters
    ----------
    computed_kwargs_expr : Expression
        An expression that computes a value to be passed to the UDF function.
        This is typically an aggregation that produces a model or other computed value.
    fn : callable
        The function to be executed. Should accept (computed_arg, df, **kwargs)
        where computed_arg is the result of computed_kwargs_expr and df is a
        pandas DataFrame containing the input columns.
    schema : Schema
        The input schema defining column names and their data types.
    return_type : DataType, default dt.binary
        The return data type of the UDF.
    database : str, optional
        Database name for the UDF namespace.
    catalog : str, optional
        Catalog name for the UDF namespace.
    name : str, optional
        Name of the UDF. If None, uses the function name.
    post_process_fn : callable, default unwrap_model
        Function to post-process the computed_kwargs_expr result before passing
        to the main function.
    **kwargs
        Additional configuration parameters.

    Returns
    -------
    callable
        A UDF constructor that can be used in expressions.

    Examples
    --------
    Machine learning workflow with penguin species classification:

    >>> import pickle
    >>> import pandas as pd
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from xorq.expr.udf import make_pandas_expr_udf, agg
    >>> import xorq.expr.datatypes as dt
    >>> import xorq.api as xo
    >>>
    >>> # Load penguins dataset
    >>> penguins = xo.examples.penguins.fetch(backend=xo.connect())
    >>> features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    >>>
    >>> # Split data
    >>> train_data = penguins.filter(penguins.year < 2009)
    >>> test_data = penguins.filter(penguins.year >= 2009)
    >>>
    >>> # Define training function
    >>> def train_penguin_model(df):
    ...     df_clean = df.dropna(subset=features + ['species'])
    ...     X = df_clean[features]
    ...     y = df_clean['species']
    ...
    ...     model = KNeighborsClassifier(n_neighbors=5)
    ...     model.fit(X, y)
    ...     return pickle.dumps(model)
    >>>
    >>> # Define prediction function
    >>> def predict_penguin_species(model, df):
    ...     df_clean = df.dropna(subset=features)
    ...     X = df_clean[features]
    ...     predictions = model.predict(X)
    ...     # Return predictions for all rows (fill NaN for missing data)
    ...     result = pd.Series(index=df.index, dtype='object')
    ...     result.loc[df_clean.index] = predictions
    ...     return result.fillna('Unknown')
    >>>
    >>> # Create schemas for training and prediction
    >>> train_schema = train_data.select(features + ['species']).schema()
    >>> test_schema = test_data.select(features).schema()
    >>>
    >>> # Create model training UDAF
    >>> model_udaf = agg.pandas_df(
    ...     fn=train_penguin_model,
    ...     schema=train_schema,
    ...     return_type=dt.binary,
    ...     name="train_penguin_model"
    >>> )
    >>>
    >>> # Create prediction UDF that uses trained model
    >>> predict_udf = make_pandas_expr_udf(
    ...     computed_kwargs_expr=model_udaf.on_expr(train_data),
    ...     fn=predict_penguin_species,
    ...     schema=test_schema,
    ...     return_type=dt.string,
    ...     name="predict_species"
    >>> )
    >>>
    >>> # Apply predictions to test data
    >>> result = test_data.mutate(
    ...     predicted_species=predict_udf.on_expr(test_data)
    >>> ).execute()

    Penguin size classification with pre-computed thresholds:

    >>> def compute_size_thresholds(df):
    ...     df_clean = df.dropna(subset=['body_mass_g'])
    ...     return {
    ...         'small_threshold': df_clean['body_mass_g'].quantile(0.33),
    ...         'large_threshold': df_clean['body_mass_g'].quantile(0.67)
    ...     }
    >>>
    >>> def classify_penguin_size(thresholds, df):
    ...     def classify_size(mass):
    ...         if pd.isna(mass):
    ...             return 'Unknown'
    ...         elif mass < thresholds['small_threshold']:
    ...             return 'Small'
    ...         elif mass > thresholds['large_threshold']:
    ...             return 'Large'
    ...         else:
    ...             return 'Medium'
    ...
    ...     return df['body_mass_g'].apply(classify_size)
    >>>
    >>> # Create threshold computation UDAF
    >>> threshold_udaf = agg.pandas_df(
    ...     fn=compute_size_thresholds,
    ...     schema=penguins.select(['body_mass_g']).schema(),
    ...     return_type=dt.Struct({
    ...         'small_threshold': dt.float64,
    ...         'large_threshold': dt.float64
    ...     }),
    ...     name="compute_thresholds"
    >>> )
    >>>
    >>> # Create size classification UDF
    >>> size_classify_udf = make_pandas_expr_udf(
    ...     computed_kwargs_expr=threshold_udaf.on_expr(penguins),
    ...     fn=classify_penguin_size,
    ...     schema=penguins.select(['body_mass_g']).schema(),
    ...     return_type=dt.string,
    ...     name="classify_size",
    ...     post_process_fn=lambda x: x  # thresholds are already a dict
    >>> )
    >>>
    >>> # Apply size classification
    >>> result = penguins.mutate(
    ...     size_category=size_classify_udf.on_expr(penguins)
    >>> ).execute()

    Notes
    -----
    This UDF type is particularly powerful for ML workflows where you need to:
    1. Train a model on aggregated data
    2. Serialize the trained model
    3. Use the model for predictions on new data

    The computed_kwargs_expr is evaluated once and its result is passed to every
    invocation of the main function, enabling efficient model reuse.

    See Also
    --------
    make_pandas_udf : For standard pandas-based scalar UDFs
    agg : For aggregation functions
    """

    name = name if name is not None else _make_udf_name(fn)
    bases = (ExprScalarUDF,)
    fields = {
        arg_name: Argument(pattern=rlz.ValueOf(typ), typehint=typ)
        for (arg_name, typ) in schema.items()
    }
    meta = {
        "dtype": return_type,
        "__input_type__": InputType.PYARROW,
        "__func__": make_expr_scalar_udf_dunder_func(fn, schema, return_type),
        # valid config keys: computed_kwargs_expr, post_process_fn, volatility
        "__config__": FrozenDict(
            computed_kwargs_expr=computed_kwargs_expr,
            post_process_fn=post_process_fn,
            schema=schema,
            fn=fn,
            **kwargs,
        ),
        "__udf_namespace__": Namespace(database=database, catalog=catalog),
        "__module__": fn.__module__,
        # FIXME: determine why this fails with case mismatch by default
        "__func_name__": name,
    }
    kwds = {
        **fields,
        **meta,
    }

    node = type(
        name,
        bases,
        kwds,
    )

    # FIXME: enable working with deferred like _wrap enables
    @functools.wraps(fn)
    def construct(*args: Any, **kwargs: Any) -> ir.Value:
        return node(*args, **kwargs).to_expr()

    def on_expr(e, **kwargs):
        return construct(*(e[c] for c in schema), **kwargs)

    construct.on_expr = on_expr
    construct.fn = fn
    return construct


@toolz.curry
def make_pandas_udf(
    fn, schema, return_type, database=None, catalog=None, name=None, **kwargs
):
    """
    Create a scalar User-Defined Function (UDF) that operates on pandas DataFrames.

    This function creates a scalar UDF that processes data row-by-row, converting
    PyArrow arrays to pandas DataFrames for processing. It's ideal for operations
    that benefit from pandas' rich functionality and are easier to express with
    DataFrame operations.

    Parameters
    ----------
    fn : callable
        The function to be executed. Should accept a pandas DataFrame and return
        a pandas Series or scalar value.
    schema : Schema
        The input schema defining column names and their data types.
    return_type : DataType
        The return data type of the UDF.
    database : str, optional
        Database name for the UDF namespace.
    catalog : str, optional
        Catalog name for the UDF namespace.
    name : str, optional
        Name of the UDF. If None, generates a name from the function.
    **kwargs
        Additional configuration parameters (e.g., volatility settings).

    Returns
    -------
    callable
        A UDF constructor that can be used in expressions with `.on_expr()` method.

    Examples
    --------
    Creating a UDF that calculates penguin bill ratio:

    >>> import pandas as pd
    >>> from xorq.expr.udf import make_pandas_udf
    >>> import xorq.expr.datatypes as dt
    >>> import xorq.api as xo
    >>>
    >>> # Load penguins dataset
    >>> penguins = xo.examples.penguins.fetch(backend=xo.connect())
    >>>
    >>> # Define the function
    >>> def bill_ratio(df):
    ...     return df['bill_length_mm'] / df['bill_depth_mm']
    >>>
    >>> # Create UDF
    >>> schema = penguins.select(['bill_length_mm', 'bill_depth_mm']).schema()
    >>> bill_ratio_udf = make_pandas_udf(
    ...     fn=bill_ratio,
    ...     schema=schema,
    ...     return_type=dt.float64,
    ...     name="bill_ratio"
    >>> )
    >>>
    >>> # Apply to table
    >>> result = penguins.mutate(
    ...     bill_ratio=bill_ratio_udf.on_expr(penguins)
    >>> ).execute()

    Creating a UDF for penguin size classification:

    >>> def classify_penguin_size(df):
    ...     def size_category(row):
    ...         mass = row['body_mass_g']
    ...         flipper = row['flipper_length_mm']
    ...
    ...         if pd.isna(mass) or pd.isna(flipper):
    ...             return 'Unknown'
    ...
    ...         # Simple size classification based on body mass and flipper length
    ...         if mass > 4500 and flipper > 210:
    ...             return 'Large'
    ...         elif mass < 3500 and flipper < 190:
    ...             return 'Small'
    ...         else:
    ...             return 'Medium'
    ...
    ...     return df.apply(size_category, axis=1)
    >>>
    >>> size_schema = penguins.select(['body_mass_g', 'flipper_length_mm']).schema()
    >>> size_udf = make_pandas_udf(
    ...     fn=classify_penguin_size,
    ...     schema=size_schema,
    ...     return_type=dt.string,
    ...     name="classify_size"
    >>> )
    >>>
    >>> # Apply size classification
    >>> result = penguins.mutate(
    ...     size_category=size_udf.on_expr(penguins)
    >>> ).execute()

    Creating a UDF for complex penguin feature engineering:

    >>> def penguin_features(df):
    ...     # Create multiple derived features
    ...     features = pd.DataFrame(index=df.index)
    ...
    ...     # Bill area
    ...     features['bill_area'] = df['bill_length_mm'] * df['bill_depth_mm']
    ...
    ...     # Body condition index
    ...     features['body_condition'] = df['body_mass_g'] / (df['flipper_length_mm'] ** 2)
    ...
    ...     # Aspect ratio of bill
    ...     features['bill_aspect_ratio'] = df['bill_length_mm'] / df['bill_depth_mm']
    ...
    ...     # Return as concatenated string for this example
    ...     return features.apply(lambda row: f"area:{row['bill_area']:.1f}_bci:{row['body_condition']:.4f}_ratio:{row['bill_aspect_ratio']:.2f}", axis=1)
    >>>
    >>> all_measurements = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    >>> features_schema = penguins.select(all_measurements).schema()
    >>> features_udf = make_pandas_udf(
    ...     fn=penguin_features,
    ...     schema=features_schema,
    ...     return_type=dt.string,
    ...     name="penguin_features"
    >>> )
    >>>
    >>> # Apply feature engineering
    >>> result = penguins.mutate(
    ...     derived_features=features_udf.on_expr(penguins)
    >>> ).execute()

    Notes
    -----
    - The function receives a pandas DataFrame where columns correspond to the schema keys
    - The function should return a pandas Series or scalar value compatible with return_type
    - PyArrow arrays are automatically converted to pandas and back for seamless integration
    - Use this when you need pandas-specific functionality like string operations,
      datetime handling, or complex data manipulations

    See Also
    --------
    scalar : For PyArrow-based scalar UDFs with potentially better performance
    make_pandas_expr_udf : For UDFs that need pre-computed values
    agg : For aggregation functions
    """

    from xorq.vendor.ibis.expr.operations.udf import ScalarUDF

    name = name if name is not None else _make_udf_name(fn)
    bases = (ScalarUDF,)
    fields = {
        arg_name: Argument(pattern=rlz.ValueOf(typ), typehint=typ)
        for (arg_name, typ) in schema.items()
    }
    meta = {
        "dtype": return_type,
        "__input_type__": InputType.PYARROW,
        "__func__": make_dunder_func(fn, schema, return_type),
        # valid config keys: volatility
        "__config__": FrozenDict(**kwargs),
        "__udf_namespace__": Namespace(database=database, catalog=catalog),
        "__module__": fn.__module__,
        # FIXME: determine why this fails with case mismatch by default
        "__func_name__": name,
    }
    kwds = {
        **fields,
        **meta,
    }

    node = type(
        name,
        bases,
        kwds,
    )

    # FIXME: enable working with deferred like _wrap enables
    @functools.wraps(fn)
    def construct(*args: Any, **kwargs: Any) -> ir.Value:
        return node(*args, **kwargs).to_expr()

    def on_expr(e, **kwargs):
        return construct(*(e[c] for c in schema), **kwargs)

    construct.on_expr = on_expr
    construct.fn = fn
    return construct


class agg(_agg):
    __slots__ = ()

    _base = AggUDF

    @classmethod
    def pyarrow(
        cls,
        fn=None,
        name=None,
        signature=None,
        **kwargs,
    ):
        """
        Decorator for creating PyArrow-based aggregation functions.

        This method creates high-performance aggregation UDFs that operate directly
        on PyArrow arrays using PyArrow compute functions. It's ideal for numerical
        computations and operations that benefit from vectorized processing.

        Parameters
        ----------
        fn : callable, optional
            The aggregation function. Should accept PyArrow arrays and return a scalar
            or array result using PyArrow compute functions.
        name : str, optional
            Name of the UDF. If None, uses the function name.
        signature : Signature, optional
            Function signature specification for type checking.
        **kwargs
            Additional configuration parameters like volatility settings.

        Returns
        -------
        callable
            A UDF decorator that can be applied to functions, or if fn is provided,
            the wrapped UDF function.

        Examples
        --------
        Creating a PyArrow aggregation for penguin bill measurements:

        >>> import pyarrow.compute as pc
        >>> from xorq.expr.udf import agg
        >>> import xorq.expr.datatypes as dt
        >>> import xorq.api as xo
        >>>
        >>> # Load penguins dataset
        >>> penguins = xo.examples.penguins.fetch(backend=xo.connect())
        >>>
        >>> @agg.pyarrow
        >>> def bill_length_range(arr: dt.float64) -> dt.float64:
        ...     return pc.subtract(pc.max(arr), pc.min(arr))
        >>>
        >>> # Calculate bill length range by species
        >>> result = penguins.group_by("species").agg(
        ...     length_range=bill_length_range(penguins.bill_length_mm)
        >>> ).execute()

        Creating a weighted average for penguin measurements:

        >>> @agg.pyarrow
        >>> def weighted_avg_flipper(lengths: dt.float64, weights: dt.float64) -> dt.float64:
        ...     return pc.divide(
        ...         pc.sum(pc.multiply(lengths, weights)),
        ...         pc.sum(weights)
        ...     )
        >>>
        >>> # Weighted average flipper length by body mass
        >>> result = penguins.group_by("species").agg(
        ...     weighted_flipper=weighted_avg_flipper(
        ...         penguins.flipper_length_mm,
        ...         penguins.body_mass_g
        ...     )
        >>> ).execute()

        Notes
        -----
        - PyArrow UDAFs typically offer the best performance for numerical operations
        - Functions receive PyArrow arrays and should use PyArrow compute functions
        - The return type should be compatible with PyArrow scalar types
        - Use this for high-performance aggregations on large datasets

        See Also
        --------
        agg.pandas_df : For pandas DataFrame-based aggregations
        agg.builtin : For database-native aggregate functions
        """
        result = _wrap(
            cls._make_wrapper,
            InputType.PYARROW,
            fn,
            name=name,
            signature=signature,
            **kwargs,
        )
        return result

    @classmethod
    @toolz.curry
    def pandas_df(
        cls,
        fn,
        schema,
        return_type,
        database=None,
        catalog=None,
        name=None,
        **kwargs,
    ):
        """
        Create a pandas DataFrame-based aggregation function.

        This method creates aggregation UDFs that operate on pandas DataFrames,
        providing access to the full pandas ecosystem for complex aggregations.
        It's particularly useful for statistical operations, machine learning
        model training, and complex data transformations that are easier to
        express with pandas.

        Parameters
        ----------
        fn : callable
            The aggregation function. Should accept a pandas DataFrame and return
            a value compatible with the return_type. The DataFrame contains all
            columns specified in the schema for each group.
        schema : Schema or dict
            Input schema defining column names and their data types.
        return_type : DataType
            The return data type of the aggregation.
        database : str, optional
            Database name for the UDF namespace.
        catalog : str, optional
            Catalog name for the UDF namespace.
        name : str, optional
            Name of the UDF. If None, generates a name from the function.
        **kwargs
            Additional configuration parameters (e.g., volatility settings).

        Returns
        -------
        callable
            A UDF constructor that can be used in aggregation expressions.

        Examples
        --------
        Training a KNN classifier on penguin data as an aggregation:

        >>> import pickle
        >>> from sklearn.neighbors import KNeighborsClassifier
        >>> from xorq.expr.udf import agg
        >>> import xorq.expr.datatypes as dt
        >>> import xorq.api as xo
        >>>
        >>> # Load penguins dataset
        >>> penguins = xo.examples.penguins.fetch(backend=xo.connect())
        >>> features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
        >>>
        >>> def train_penguin_classifier(df):
        ...     # Remove rows with missing values
        ...     df_clean = df.dropna(subset=features + ['species'])
        ...     X = df_clean[features]
        ...     y = df_clean['species']
        ...
        ...     model = KNeighborsClassifier(n_neighbors=3)
        ...     model.fit(X, y)
        ...     return pickle.dumps(model)
        >>>
        >>> # Create the aggregation UDF
        >>> penguin_schema = penguins.select(features + ['species']).schema()
        >>> train_model_udf = agg.pandas_df(
        ...     fn=train_penguin_classifier,
        ...     schema=penguin_schema,
        ...     return_type=dt.binary,
        ...     name="train_penguin_classifier"
        >>> )
        >>>
        >>> # Train one model per island
        >>> trained_models = penguins.group_by("island").agg(
        ...     model=train_model_udf.on_expr(penguins)
        >>> ).execute()

        Complex statistical aggregation for penguin measurements:

        >>> def penguin_stats(df):
        ...     return {
        ...         'bill_ratio_mean': (df['bill_length_mm'] / df['bill_depth_mm']).mean(),
        ...         'mass_flipper_corr': df['body_mass_g'].corr(df['flipper_length_mm']),
        ...         'count': len(df),
        ...         'size_score': (df['body_mass_g'] * df['flipper_length_mm']).mean()
        ...     }
        >>>
        >>> stats_schema = penguins.select([
        ...     'bill_length_mm', 'bill_depth_mm', 'body_mass_g', 'flipper_length_mm'
        >>> ]).schema()
        >>>
        >>> stats_udf = agg.pandas_df(
        ...     fn=penguin_stats,
        ...     schema=stats_schema,
        ...     return_type=dt.Struct({
        ...         'bill_ratio_mean': dt.float64,
        ...         'mass_flipper_corr': dt.float64,
        ...         'count': dt.int64,
        ...         'size_score': dt.float64
        ...     }),
        ...     name="penguin_stats"
        >>> )
        >>>
        >>> # Calculate statistics by species
        >>> result = penguins.group_by("species").agg(
        ...     stats=stats_udf.on_expr(penguins)
        >>> ).execute()

        Feature selection for penguin classification:

        >>> def select_best_penguin_features(df, n_features=2):
        ...     from sklearn.feature_selection import mutual_info_classif
        ...     import pandas as pd
        ...
        ...     df_clean = df.dropna()
        ...     features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
        ...     X = df_clean[features]
        ...     y = df_clean['species']
        ...
        ...     scores = mutual_info_classif(X, y)
        ...     return list(pd.Series(scores, index=features).nlargest(n_features).index)
        >>>
        >>> feature_selector = agg.pandas_df(
        ...     fn=select_best_penguin_features,
        ...     schema=penguins.schema(),
        ...     return_type=dt.Array(dt.string),
        ...     name="select_penguin_features"
        >>> )
        >>>
        >>> # Find best features by island
        >>> best_features = penguins.group_by("island").agg(
        ...     top_features=feature_selector.on_expr(penguins)
        >>> ).execute()

        Notes
        -----
        - The function receives a pandas DataFrame containing all rows in each group
        - PyArrow arrays are automatically converted to pandas for processing
        - The function can return complex data structures (dicts, lists) if return_type supports it
        - Use this when you need pandas-specific functionality or ML libraries
        - Performance may be lower than PyArrow UDAFs for simple numerical operations
        - Particularly powerful for ML model training workflows

        See Also
        --------
        agg.pyarrow : For high-performance PyArrow-based aggregations
        make_pandas_expr_udf : For using trained models in prediction UDFs
        make_pandas_udf : For scalar pandas operations
        """
        name = name if name is not None else _make_udf_name(fn)
        bases = (cls._base,)
        fields = {
            arg_name: Argument(pattern=rlz.ValueOf(typ), typehint=typ)
            for (arg_name, typ) in schema.items()
        }
        meta = {
            "dtype": return_type,
            "__input_type__": InputType.PYARROW,
            "__func__": make_dunder_func(fn, schema),
            # valid config keys: volatility
            "__config__": FrozenDict(fn=fn, **kwargs),
            "__udf_namespace__": Namespace(database=database, catalog=catalog),
            "__module__": fn.__module__,
            # FIXME: determine why this fails with case mismatch by default
            "__func_name__": name,
        }
        kwds = {
            **fields,
            **meta,
        }

        node = type(
            name,
            bases,
            kwds,
        )

        # FIXME: enable working with deferred like _wrap enables
        @functools.wraps(fn)
        def construct(*args: Any, **kwargs: Any) -> ir.Value:
            return node(*args, **kwargs).to_expr()

        def on_expr(e, **kwargs):
            return construct(*(e[c] for c in schema), **kwargs)

        construct.on_expr = on_expr
        construct.fn = fn
        return construct


def arbitrate_evaluate(
    uses_window_frame=False,
    supports_bounded_execution=False,
    include_rank=False,
    **config_kwargs,
):
    match (uses_window_frame, supports_bounded_execution, include_rank):
        case (False, False, False):
            return "evaluate_all"
        case (False, True, False):
            return "evaluate"
        case (False, _, True):
            return "evaluate_all_with_rank"
        case (True, _, _):
            return "evaluate"
        case _:
            raise RuntimeError


@toolz.curry
def pyarrow_udwf(
    fn,
    schema,
    return_type,
    name=None,
    namespace=Namespace(database=None, catalog=None),
    base=AggUDF,
    **config_kwargs,
):
    """
    Create a User-Defined Window Function (UDWF) using PyArrow.

    This decorator creates window functions that can process partitions of data with
    support for ordering, framing, and ranking. UDWFs are powerful for implementing
    custom analytics functions that need to operate over ordered sets of data within
    partitions.

    Parameters
    ----------
    fn : callable
        The window function implementation. The signature depends on config_kwargs:

        - Basic window function: `fn(self, values: list[pa.Array], num_rows: int) -> pa.Array`
        - With window frame: `fn(self, values: list[pa.Array], eval_range: tuple[int, int]) -> pa.Scalar`
        - With ranking: `fn(self, num_rows: int, ranks_in_partition: list[tuple[int, int]]) -> pa.Array`

    schema : Schema
        Input schema defining column names and data types.
    return_type : DataType
        The return data type of the window function.
    name : str, optional
        Name of the UDWF. If None, uses the function name.
    namespace : Namespace, optional
        Database and catalog namespace for the function.
    base : class, default AggUDF
        Base class for the UDWF (typically AggUDF).
    **config_kwargs
        Configuration options:

        - `uses_window_frame` (bool): Whether function uses window framing
        - `supports_bounded_execution` (bool): Whether function supports bounded execution
        - `include_rank` (bool): Whether function uses ranking information
        - Custom parameters: Additional parameters accessible via `self` in the function

    Returns
    -------
    callable
        A UDWF constructor that can be used in window expressions.

    Examples
    --------
    Exponential smoothing for penguin body mass by species:

    >>> from xorq.expr.udf import pyarrow_udwf
    >>> import pyarrow as pa
    >>> import xorq.api as xo
    >>> import xorq.expr.datatypes as dt
    >>> from xorq.vendor import ibis
    >>>
    >>> # Load penguins dataset
    >>> penguins = xo.examples.penguins.fetch(backend=xo.connect())
    >>>
    >>> @pyarrow_udwf(
    ...     schema=ibis.schema({"body_mass_g": dt.float64}),
    ...     return_type=dt.float64,
    ...     alpha=0.8  # Custom smoothing parameter
    ... )
    >>> def smooth_body_mass(self, values: list[pa.Array], num_rows: int) -> pa.Array:
    ...     results = []
    ...     curr_value = 0.0
    ...     mass_values = values[0]  # body_mass_g column
    ...
    ...     for idx in range(num_rows):
    ...         if idx == 0:
    ...             curr_value = float(mass_values[idx].as_py() or 0)
    ...         else:
    ...             new_val = float(mass_values[idx].as_py() or curr_value)
    ...             curr_value = new_val * self.alpha + curr_value * (1.0 - self.alpha)
    ...         results.append(curr_value)
    ...
    ...     return pa.array(results)
    >>>
    >>> # Apply smoothing within each species, ordered by year
    >>> result = penguins.mutate(
    ...     smooth_mass=smooth_body_mass.on_expr(penguins).over(
    ...         ibis.window(group_by="species", order_by="year")
    ...     )
    >>> ).execute()

    Running difference in penguin bill measurements:

    >>> @pyarrow_udwf(
    ...     schema=ibis.schema({"bill_length_mm": dt.float64}),
    ...     return_type=dt.float64,
    ...     uses_window_frame=True
    ... )
    >>> def bill_length_diff(self, values: list[pa.Array], eval_range: tuple[int, int]) -> pa.Scalar:
    ...     start, stop = eval_range
    ...     bill_values = values[0]
    ...
    ...     if start == stop - 1:  # Single row
    ...         return pa.scalar(0.0)
    ...
    ...     current_val = bill_values[stop - 1].as_py() or 0
    ...     previous_val = bill_values[start].as_py() or 0
    ...     return pa.scalar(float(current_val - previous_val))
    >>>
    >>> # Calculate difference from previous measurement within species
    >>> result = penguins.mutate(
    ...     bill_diff=bill_length_diff.on_expr(penguins).over(
    ...         ibis.window(
    ...             group_by="species",
    ...             order_by="year",
    ...             preceding=1,
    ...             following=0
    ...         )
    ...     )
    >>> ).execute()

    Penguin ranking within species by body mass:

    >>> @pyarrow_udwf(
    ...     schema=ibis.schema({"body_mass_g": dt.float64}),
    ...     return_type=dt.float64,
    ...     include_rank=True
    ... )
    >>> def mass_rank_score(self, num_rows: int, ranks_in_partition: list[tuple[int, int]]) -> pa.Array:
    ...     results = []
    ...     for idx in range(num_rows):
    ...         # Find rank for current row
    ...         rank = next(
    ...             i + 1 for i, (start, end) in enumerate(ranks_in_partition)
    ...             if start <= idx < end
    ...         )
    ...         # Convert rank to score (higher rank = higher score)
    ...         score = 1.0 - (rank - 1) / len(ranks_in_partition)
    ...         results.append(score)
    ...     return pa.array(results)
    >>>
    >>> # Calculate mass rank score within each species
    >>> result = penguins.mutate(
    ...     mass_rank_score=mass_rank_score.on_expr(penguins).over(
    ...         ibis.window(group_by="species", order_by="body_mass_g")
    ...     )
    >>> ).execute()

    Complex penguin feature calculation across measurements:

    >>> @pyarrow_udwf(
    ...     schema=ibis.schema({
    ...         "bill_length_mm": dt.float64,
    ...         "bill_depth_mm": dt.float64,
    ...         "flipper_length_mm": dt.float64
    ...     }),
    ...     return_type=dt.float64,
    ...     window_size=3  # Custom parameter for moving average
    ... )
    >>> def penguin_size_trend(self, values: list[pa.Array], num_rows: int) -> pa.Array:
    ...     bill_length = values[0]
    ...     bill_depth = values[1]
    ...     flipper_length = values[2]
    ...
    ...     results = []
    ...     window_size = self.window_size
    ...
    ...     for idx in range(num_rows):
    ...         # Calculate size metric for current and surrounding rows
    ...         start_idx = max(0, idx - window_size // 2)
    ...         end_idx = min(num_rows, idx + window_size // 2 + 1)
    ...
    ...         size_metrics = []
    ...         for i in range(start_idx, end_idx):
    ...             if (bill_length[i].is_valid and bill_depth[i].is_valid and
    ...                 flipper_length[i].is_valid):
    ...                 # Composite size metric
    ...                 bill_area = bill_length[i].as_py() * bill_depth[i].as_py()
    ...                 size_metric = bill_area * flipper_length[i].as_py()
    ...                 size_metrics.append(size_metric)
    ...
    ...         # Average size metric in window
    ...         avg_size = sum(size_metrics) / len(size_metrics) if size_metrics else 0.0
    ...         results.append(avg_size)
    ...
    ...     return pa.array(results)
    >>>
    >>> # Apply size trend calculation within each species
    >>> result = penguins.mutate(
    ...     size_trend=penguin_size_trend.on_expr(penguins).over(
    ...         ibis.window(group_by="species", order_by="year")
    ...     )
    >>> ).execute()

    Notes
    -----
    The function signature and behavior changes based on configuration:

    - **Standard window function**: Processes all rows in partition, returns array
    - **Frame-based**: Processes specific row ranges, returns scalar per invocation
    - **Rank-aware**: Has access to ranking information within partition

    Custom parameters passed in config_kwargs are accessible as `self.parameter_name`
    in the function implementation.

    See Also
    --------
    scalar : For row-by-row processing
    agg : For aggregation across groups
    make_pandas_udf : For pandas-based scalar operations
    """

    fields = {
        arg_name: Argument(pattern=rlz.ValueOf(typ), typehint=typ)
        for (arg_name, typ) in schema.items()
    }
    # which_evaluate = arbitrate_evaluate(**config_kwargs)
    name = name or fn.__name__
    meta = {
        "dtype": return_type,
        "__input_type__": InputType.PYARROW,
        "__func__": property(fget=toolz.functoolz.return_none),
        "__config__": FrozenDict(
            input_types=tuple(datatype for datatype in schema.fields.values()),
            return_type=return_type,
            name=name,
            **config_kwargs,
            # assert which_evaluate in ("evaluate", "evaluate_all", "evaluate_all_with_rank")
            # **{which_evaluate: fn},
            **{
                which_evaluate: fn
                for which_evaluate in (
                    "evaluate",
                    "evaluate_all",
                    "evaluate_all_with_rank",
                )
            },
        ),
        "__udf_namespace__": namespace,
        "__module__": __name__,
        "__func_name__": name,
    }
    node = type(
        name,
        (base,),
        {
            **fields,
            **meta,
        },
    )

    def construct(*args: Any, **kwargs: Any) -> ir.Value:
        return node(*args, **kwargs).to_expr()

    def on_expr(e, **kwargs):
        return construct(*(e[c] for c in schema), **kwargs)

    construct.on_expr = on_expr
    return construct


__all__ = [
    "pyarrow_udwf",
    "make_pandas_expr_udf",
    "make_pandas_udf",
    "scalar",
    "agg",
]
