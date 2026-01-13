import pickle

import dask
from attr import (
    field,
    frozen,
)
from attr.validators import (
    deep_iterable,
    instance_of,
    optional,
)
from dask.utils import Dispatch
from toolz.curried import (
    excepts as cexcepts,
)

import xorq.expr.datatypes as dt
from xorq.backends.xorq import connect
from xorq.caching import (
    ParquetCache,
    ParquetSnapshotCache,
    ParquetTTLSnapshotCache,
    SourceCache,
)
from xorq.common.utils.dask_normalize.dask_normalize_utils import (
    normalize_attrs,
)
from xorq.common.utils.func_utils import (
    return_constant,
)
from xorq.expr.ml.fit_lib import (
    deferred_fit_predict_only_sklearn,
    deferred_fit_predict_sklearn,
    deferred_fit_transform_only_sklearn_packed,
    deferred_fit_transform_series_sklearn,
    deferred_fit_transform_sklearn_packed,
    deferred_fit_transform_sklearn_struct,
)
from xorq.expr.ml.structer import (
    Structer,
    structer_from_instance,
)
from xorq.vendor.ibis.expr.types.core import Expr


def has_structer_registration(instance, expr, features):
    """
    Check if an sklearn instance has a Structer registration.

    Returns True if there's a registered Structer handler, False otherwise.
    """
    try:
        structer_from_instance(instance, expr, features=features)
        return True
    except ValueError:
        return False


def is_fit_transform_only(typ):
    """
    Check if an estimator type only supports fit_transform (not separate transform).

    These are transductive estimators like TSNE, MDS, SpectralEmbedding, etc.
    They can only produce embeddings for the training data.
    """
    # Estimators that have fit_transform but not transform, or where transform
    # raises NotImplementedError
    fit_transform_only_types = {
        "TSNE",
        "MDS",
        "SpectralEmbedding",
        "Isomap",
        "LocallyLinearEmbedding",
    }
    return typ.__name__ in fit_transform_only_types


def is_fit_predict_only(typ):
    """
    Check if an estimator type only supports fit_predict (not separate predict).

    These are transductive clusterers like DBSCAN, OPTICS, etc.
    They can only produce cluster labels for the training data.
    """
    fit_predict_only_types = {
        "DBSCAN",
        "OPTICS",
        "HDBSCAN",
        "AgglomerativeClustering",
        "SpectralClustering",
        "Birch",
        "AffinityPropagation",
        "MeanShift",
    }
    return typ.__name__ in fit_predict_only_types


def do_into_backend(expr, con=None):
    return expr.into_backend(con or connect())


def make_estimator_typ(fit, return_type, name=None, *, transform=None, predict=None):
    from sklearn.base import BaseEstimator

    def arbitrate_transform_predict(transform, predict):
        match (transform, predict):
            case [None, None]:
                raise ValueError
            case [other, None]:
                return other, "transform"
            case [None, other]:
                return other, "predict"
            case [other0, other1]:
                raise ValueError(other0, other1)
            case _:
                raise ValueError

    assert isinstance(return_type, dt.DataType)
    other, which = arbitrate_transform_predict(transform, predict)
    assert hasattr(fit, "__call__") and hasattr(other, "__call__")

    def make_name(prefix, to_tokenize, n=32):
        tokenized = dask.base.tokenize(to_tokenize)
        return ("_" + prefix + "_" + tokenized)[:n].lower()

    def wrapped_fit(self, *args, **kwargs):
        self._model = fit(*args, **kwargs)

    def wrapped_other(self, *args, **kwargs):
        return other(self._model, *args, **kwargs)

    name = name or make_name("estimator", (fit, other))
    typ = type(
        name,
        (BaseEstimator,),
        {
            "fit": wrapped_fit,
            which: wrapped_other,
            "return_type": return_type,
            "_fit": fit,
            f"_{which}": other,
        },
    )
    return typ


@frozen
class Step:
    """
    A single step in a machine learning pipeline that wraps a scikit-learn estimator.

    This class represents an individual processing step that can either transform data
    (transformers like StandardScaler, SelectKBest) or make predictions (classifiers
    like KNeighborsClassifier, LinearSVC). Steps can be combined into Pipeline objects
    to create complex ML workflows.

    Parameters
    ----------
    typ : type
        The scikit-learn estimator class (must inherit from BaseEstimator).
    name : str, optional
        A unique name for this step. If None, generates a name from the class name and ID.
    params_tuple : tuple, optional
        Tuple of (parameter_name, parameter_value) pairs for the estimator.
        Parameters are automatically sorted for consistency.

    Attributes
    ----------
    typ : type
        The scikit-learn estimator class.
    name : str
        The unique name for this step in the pipeline.
    params_tuple : tuple
        Sorted tuple of parameter key-value pairs.

    Examples
    --------
    Create a scaler step:

    >>> from xorq.ml import Step
    >>> from sklearn.preprocessing import StandardScaler
    >>> scaler_step = Step(typ=StandardScaler, name="scaler")
    >>> scaler_step.instance
    StandardScaler()

    Create a classifier step with parameters:

    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> knn_step = Step(
    ...     typ=KNeighborsClassifier,
    ...     name="knn",
    ...     params_tuple=(("n_neighbors", 5), ("weights", "uniform"))
    ... )
    >>> knn_step.instance
    KNeighborsClassifier(n_neighbors=5)

    Notes
    -----
    - The Step class is frozen (immutable) using attrs.
    - All estimators must inherit from sklearn.base.BaseEstimator.
    - Parameter tuples are automatically sorted for hash consistency.
    - Steps can be fitted to data using the fit() method which returns a FittedStep.
    """

    typ = field(validator=instance_of(type))
    name = field(validator=optional(instance_of(str)), default=None)
    params_tuple = field(validator=instance_of(tuple), default=(), converter=tuple)

    def __attrs_post_init__(self):
        from sklearn.base import BaseEstimator

        assert BaseEstimator in self.typ.mro()
        if self.name is None:
            object.__setattr__(self, "name", f"{self.typ.__name__.lower()}_{id(self)}")
        # param order invariant
        object.__setattr__(self, "params_tuple", tuple(sorted(self.params_tuple)))

    @property
    def tag_kwargs(self):
        def make_hashable(v):
            """Convert unhashable types (lists, dicts) to hashable tuples."""
            if isinstance(v, list):
                return tuple(make_hashable(x) for x in v)
            elif isinstance(v, dict):
                return tuple(sorted((k, make_hashable(val)) for k, val in v.items()))
            elif isinstance(v, tuple):
                return tuple(make_hashable(x) for x in v)
            return v

        # Convert params_tuple to fully hashable form
        params_hashable = make_hashable(self.params_tuple)
        return {"typ": self.typ, "name": self.name, "params_tuple": params_hashable}

    @property
    def instance(self):
        """
        Create an instance of the estimator with the configured parameters.

        Returns
        -------
        object
            An instantiated scikit-learn estimator.
        """

        return self.typ(**dict(self.params_tuple))

    def to_sklearn(self):
        """
        Return the unfitted sklearn estimator instance.

        This provides round-trip conversion from xorq Step back to sklearn.
        The returned estimator is equivalent to the original but unfitted.

        Returns
        -------
        object
            An unfitted scikit-learn estimator instance.

        Examples
        --------
        >>> from sklearn.preprocessing import StandardScaler
        >>> step = Step.from_instance_name(StandardScaler(), name="scaler")
        >>> sklearn_scaler = step.to_sklearn()
        >>> isinstance(sklearn_scaler, StandardScaler)
        True
        """
        return self.instance

    def fit(self, expr, features=None, target=None, cache=None, dest_col=None):
        """
        Fit this step to the given expression data.

        Parameters
        ----------
        expr : Expr
            The xorq expression containing the training data.
        features : tuple of str, optional
            Column names to use as features. If None, infers from expr.columns.
        target : str, optional
            Target column name. Required for prediction steps.
        cache : Cache, optional
            Storage backend for caching fitted models.
        dest_col : str, optional
            Destination column name for transformed output.

        Returns
        -------
        FittedStep
            A fitted step that can transform or predict on new data.
        """

        # how does ColumnTransformer interact with features/target?
        # features = features or self.features or tuple(expr.columns)
        features = features or tuple(expr.columns)
        return FittedStep(
            step=self,
            expr=expr,
            features=features,
            target=target,
            cache=cache,
            dest_col=dest_col,
        )

    def set_params(self, **kwargs):
        """
        Create a new Step with updated parameters.

        Parameters
        ----------
        **kwargs
            Parameter names and values to update.

        Returns
        -------
        Step
            A new Step instance with updated parameters.

        Examples
        --------
        >>> knn_step = Step(typ=KNeighborsClassifier, name="knn")
        >>> updated_step = knn_step.set_params(n_neighbors=10, weights="distance")
        """
        return self.__class__.from_instance_name(
            self.instance.set_params(**kwargs),
            name=self.name,
        )

    @classmethod
    def from_instance_name(cls, instance, name=None, deep=False):
        """
        Create a Step from an existing scikit-learn estimator instance.

        Parameters
        ----------
        instance : object
            A scikit-learn estimator instance.
        name : str, optional
            Name for the step. If None, generates from instance class name.

        Returns
        -------
        Step
            A new Step wrapping the estimator instance.
        """
        params_tuple = tuple(instance.get_params(deep=deep).items())
        return cls(typ=instance.__class__, name=name, params_tuple=params_tuple)

    @classmethod
    def from_name_instance(cls, name, instance, deep=False):
        """
        Create a Step from a name and estimator instance.

        Parameters
        ----------
        name : str
            Name for the step.
        instance : object
            A scikit-learn estimator instance.

        Returns
        -------
        Step
            A new Step wrapping the estimator instance.
        """
        return cls.from_instance_name(instance, name, deep=deep)

    @classmethod
    def from_fit_transform(
        cls, fit, transform, return_type, klass_name=None, name=None
    ):
        """
        Create a Step from custom fit and transform functions.

        Parameters
        ----------
        fit : callable
            Function to fit the model.
        transform : callable
            Function to transform with.
        return_type : DataType
            The return type for the transformation.
        klass_name : str, optional
            Name for the generated estimator class.
        name : str, optional
            Name for the step.

        Returns
        -------
        Step
            A new Step with a dynamically created transform type.
        """

        typ = make_estimator_typ(
            fit=fit, transform=transform, return_type=return_type, name=klass_name
        )
        return cls(typ=typ, name=name)

    @classmethod
    def from_fit_predict(cls, fit, predict, return_type, klass_name=None, name=None):
        """
        Create a Step from custom fit and predict functions.

        Parameters
        ----------
        fit : callable
            Function to fit the model.
        predict : callable
            Function to make predictions.
        return_type : DataType
            The return type for predictions.
        klass_name : str, optional
            Name for the generated estimator class.
        name : str, optional
            Name for the step.

        Returns
        -------
        Step
            A new Step with a dynamically created estimator type.
        """

        typ = make_estimator_typ(
            fit=fit, predict=predict, return_type=return_type, name=klass_name
        )
        return cls(typ=typ, name=name)

    __dask_tokenize__ = normalize_attrs


@frozen
class FittedStep:
    step = field(validator=instance_of(Step))
    expr = field(validator=instance_of(Expr))
    features = field(
        validator=optional(deep_iterable(instance_of(str), instance_of(tuple))),
        default=None,
        converter=tuple,
    )
    target = field(validator=optional(instance_of(str)), default=None)
    cache = field(
        validator=optional(
            instance_of(
                (
                    ParquetCache,
                    SourceCache,
                    ParquetSnapshotCache,
                    ParquetTTLSnapshotCache,
                )
            )
        ),
        default=None,
    )
    dest_col = field(validator=optional(instance_of(str)), default=None)
    # Cache for _pieces computation - excluded from hash/eq, not part of init
    _pieces_cache = field(default=None, init=False, eq=False, hash=False, repr=False)

    def __attrs_post_init__(self):
        # we are either transform or predict
        assert self.is_transform ^ self.is_predict
        # if we are predict, we must have target
        if self.target is None and self.is_predict:
            raise ValueError("Can't infer target")
        # we can do very simple feature inference
        if self.features is None:
            features = tuple(col for col in self.expr.columns if col != self.target)
            object.__setattr__(self, "features", features)
        # we should now have everything fixed

    @property
    def is_transform(self):
        # For sklearn Pipeline (has both transform and predict), prioritize predict
        has_transform = hasattr(self.step.typ, "transform")
        has_predict = hasattr(self.step.typ, "predict")
        if has_transform and has_predict:
            # If it has predict, treat it as a predictor (not transformer)
            return False
        return has_transform

    @property
    def is_predict(self):
        return hasattr(self.step.typ, "predict")

    @property
    def is_fit_transform_only(self):
        """
        Check if this step is a fit_transform-only estimator (TSNE, MDS, etc.).

        These estimators can only produce embeddings for training data and cannot
        transform new data.
        """
        return is_fit_transform_only(self.step.typ)

    @property
    def is_fit_predict_only(self):
        """
        Check if this step is a fit_predict-only estimator (DBSCAN, OPTICS, etc.).

        These are transductive clusterers that can only produce cluster labels
        for training data and cannot predict on new data.
        """
        return is_fit_predict_only(self.step.typ)

    @property
    def is_transductive(self):
        """
        Check if this step is a transductive estimator.

        Transductive estimators (TSNE, DBSCAN, etc.) cannot generalize to new data.
        They can only produce results for the training data.
        """
        return self.is_fit_transform_only or self.is_fit_predict_only

    @property
    def _pieces(self):
        # Use cached value if available (set via object.__setattr__ since class is frozen)
        if self._pieces_cache is not None:
            return self._pieces_cache

        kwargs = {
            "expr": self.expr,
            "features": self.features,
            "cls": self.step.typ,
            "params": self.step.params_tuple,
            "cache": self.cache,
        }
        (deferred_transform, deferred_predict) = (None, None)
        if self.is_transform:
            match self.step.typ:
                # FIXME: formalize registration of non-Structer handling
                case type(__name__="TfidfVectorizer"):
                    f = deferred_fit_transform_series_sklearn
                    # features must be length 1
                    (col,) = kwargs.pop("features")
                    kwargs = kwargs | {
                        "col": col,
                        "return_type": dt.Array(dt.float64),
                    }
                case type(__name__="SelectKBest"):
                    # SelectKBest is a Structer special case that needs target
                    f = deferred_fit_transform_sklearn_struct
                    kwargs = kwargs | {
                        "target": self.target,
                    }
                case typ:
                    # FIXME: create abstract class for BaseEstimator with get_step_f_kwargs
                    if get_step_f_kwargs := getattr(typ, "get_step_f_kwargs", None):
                        (f, kwargs) = get_step_f_kwargs(kwargs)
                    elif is_fit_transform_only(typ):
                        # Transductive estimators like TSNE, MDS that only have fit_transform
                        f = deferred_fit_transform_only_sklearn_packed
                    elif has_structer_registration(
                        self.step.instance, self.expr, self.features
                    ):
                        # Use Structer-based approach for registered types
                        f = deferred_fit_transform_sklearn_struct
                    else:
                        # Fallback to packed format for unregistered types
                        # This handles OneHotEncoder, CountVectorizer, ColumnTransformer, etc.
                        f = deferred_fit_transform_sklearn_packed
            (deferred_model, model_udf, deferred_transform) = f(**kwargs)
        elif self.is_predict:
            if is_fit_predict_only(self.step.typ):
                # Transductive clusterers like DBSCAN that only have fit_predict
                (deferred_model, model_udf, deferred_predict) = (
                    deferred_fit_predict_only_sklearn(**kwargs)
                )
            else:
                predict_kwargs = {
                    "target": self.target,
                    "return_type": get_predict_return_type(
                        instance=self.step.instance,
                        step=self.step,
                        expr=self.expr,
                        features=self.features,
                        target=self.target,
                    ),
                }
                (deferred_model, model_udf, deferred_predict) = (
                    deferred_fit_predict_sklearn(**kwargs, **predict_kwargs)
                )
        else:
            raise ValueError
        result = {
            "deferred_model": deferred_model,
            "model_udf": model_udf,
            "deferred_transform": deferred_transform,
            "deferred_predict": deferred_predict,
        }
        # Cache the result (using object.__setattr__ since class is frozen)
        object.__setattr__(self, "_pieces_cache", result)
        return result

    @property
    def deferred_model(self):
        return self._pieces["deferred_model"]

    @property
    def model_udf(self):
        return self._pieces["model_udf"]

    @property
    def deferred_transform(self):
        return self._pieces["deferred_transform"]

    @property
    def deferred_predict(self):
        return self._pieces["deferred_predict"]

    @property
    def model(self):
        # Note: This property is not cached due to FittedStep being @frozen
        # Consider using execute() sparingly as it triggers computation
        import pandas as pd

        match obj := self.deferred_model.execute():
            case pd.DataFrame():
                ((obj,),) = obj.values
            case bytes():
                pass
            case _:
                raise ValueError
        return pickle.loads(obj)

    def to_sklearn(self):
        """
        Return the fitted sklearn estimator.

        This triggers execution of the deferred model if not already executed,
        and returns the unpickled fitted sklearn estimator.

        Returns
        -------
        object
            A fitted scikit-learn estimator that can be used directly.

        Examples
        --------
        >>> fitted_step = step.fit(expr, target="y")
        >>> sklearn_model = fitted_step.to_sklearn()
        >>> sklearn_model.predict(new_df)  # Use sklearn directly
        """
        return self.model

    @property
    @cexcepts(ValueError)
    def structer(self):
        # Note: This property is not cached due to FittedStep being @frozen
        return Structer.from_instance_expr(
            self.step.instance, self.expr, features=self.features
        )

    def get_others(self, expr):
        others = tuple(other for other in expr.columns if other not in self.features)
        return others

    def transform_unpack(self, expr, retain_others=True, name="to_unpack"):
        struct_col = self.transform_raw(expr).name(name)
        if retain_others:
            expr = expr.select(*self.get_others(expr), struct_col)
        else:
            expr = struct_col.as_table()
        return expr.unpack(name)

    def transform_raw(self, expr, name=None):
        # when you use expr.mutate, you want transform_raw
        transformed = self.deferred_transform.on_expr(expr).name(
            name or self.dest_col or "transformed"
        )
        return transformed

    @property
    def tag_kwargs(self):
        return {
            "tag": "FittedStep-transform"
            if self.is_transform
            else "FittedStep-predict",
            **self.step.tag_kwargs,
            "features": self.features,
        }

    def transform(self, expr, retain_others=True):
        # Check for transductive estimators that can't transform new data
        if self.is_fit_transform_only:
            raise TypeError(
                f"{self.step.typ.__name__} is a transductive estimator that only supports "
                f"fit_transform(). It cannot transform new data. "
                f"Use the training data results from fit() or re-fit on the new data."
            )
        if self.structer is None:
            col = self.transform_raw(expr)
            if retain_others:
                expr = expr.select(*self.get_others(expr), col)
            else:
                return col
        else:
            expr = self.transform_unpack(expr, retain_others=retain_others)
        return expr.tag(
            **self.tag_kwargs,
        )

    @property
    def transformed(self):
        return self.transform(self.expr)

    def predict_raw(self, expr, name=None):
        # Check for transductive clusterers that can't predict new data
        if self.is_fit_predict_only:
            raise TypeError(
                f"{self.step.typ.__name__} is a transductive clusterer that only supports "
                f"fit_predict(). It cannot predict on new data. "
                f"Use the training data results from fit() or re-fit on the new data."
            )
        col = self.deferred_predict.on_expr(expr).name(
            name or self.dest_col or "predicted"
        )
        return col

    def predict(self, expr, retain_others=True, name=None):
        col = self.predict_raw(expr, name=name)
        if retain_others and (
            others := tuple(
                other for other in expr.columns if other not in self.features
            )
        ):
            expr = expr.select(*others, col)
        else:
            expr = col.as_table()
        return expr.tag(
            **self.tag_kwargs,
        )

    def mutate(self, expr, name=None):
        if self.is_predict:
            return self.predict_raw(expr, name=name)
        elif self.is_transform:
            return self.transform_raw(expr, name=name)
        else:
            raise ValueError

    @property
    def predicted(self):
        return self.predict(self.expr)

    def _make_proba_udf(self, method_name):
        """
        Create a deferred UDF for predict_proba or decision_function.

        Returns packed format Array[Struct{key, value}] where keys are class labels.
        """
        from xorq.expr.ml.fit_lib import (
            PACKED_TRANSFORM_TYPE,
            decision_function_sklearn_packed,
            predict_proba_sklearn_packed,
        )
        from xorq.expr.udf import make_pandas_expr_udf

        if method_name == "predict_proba":
            fn = predict_proba_sklearn_packed
        elif method_name == "decision_function":
            fn = decision_function_sklearn_packed
        else:
            raise ValueError(f"Unknown method: {method_name}")

        schema = self.expr.select(self.features).schema()
        return make_pandas_expr_udf(
            computed_kwargs_expr=self.deferred_model,
            fn=fn,
            schema=schema,
            return_type=PACKED_TRANSFORM_TYPE,
            name=f"_{method_name}_{self.step.name}",
        )

    def predict_proba_raw(self, expr, name=None):
        """
        Return class probabilities as packed format Array[Struct{key, value}].

        Keys are class labels from model.classes_, values are probabilities.

        Parameters
        ----------
        expr : Expr
            Input expression with feature columns.
        name : str, optional
            Name for the output column.

        Returns
        -------
        Column
            Deferred column with packed probability values.
        """
        if not hasattr(self.step.typ, "predict_proba"):
            raise AttributeError(
                f"{self.step.typ.__name__} does not have predict_proba method"
            )
        udf = self._make_proba_udf("predict_proba")
        return udf.on_expr(expr).name(name or "proba")

    def predict_proba(self, expr, retain_others=True, name=None):
        """
        Return class probabilities.

        Parameters
        ----------
        expr : Expr
            Input expression with feature columns.
        retain_others : bool, default True
            If True, retain non-feature columns in output.
        name : str, optional
            Name for the output column.

        Returns
        -------
        Expr
            Expression with probability column in packed format.
        """
        col = self.predict_proba_raw(expr, name=name)
        if retain_others:
            others = self.get_others(expr)
            if others:
                expr = expr.select(*others, col)
            else:
                expr = col.as_table()
        else:
            expr = col.as_table()
        return expr.tag(**self.tag_kwargs)

    def decision_function_raw(self, expr, name=None):
        """
        Return decision function values as packed format Array[Struct{key, value}].

        For binary classification, single "decision" key.
        For multiclass, keys are class labels.

        Parameters
        ----------
        expr : Expr
            Input expression with feature columns.
        name : str, optional
            Name for the output column.

        Returns
        -------
        Column
            Deferred column with packed decision values.
        """
        if not hasattr(self.step.typ, "decision_function"):
            raise AttributeError(
                f"{self.step.typ.__name__} does not have decision_function method"
            )
        udf = self._make_proba_udf("decision_function")
        return udf.on_expr(expr).name(name or "decision")

    def decision_function(self, expr, retain_others=True, name=None):
        """
        Return decision function values.

        Parameters
        ----------
        expr : Expr
            Input expression with feature columns.
        retain_others : bool, default True
            If True, retain non-feature columns in output.
        name : str, optional
            Name for the output column.

        Returns
        -------
        Expr
            Expression with decision function column in packed format.
        """
        col = self.decision_function_raw(expr, name=name)
        if retain_others:
            others = self.get_others(expr)
            if others:
                expr = expr.select(*others, col)
            else:
                expr = col.as_table()
        else:
            expr = col.as_table()
        return expr.tag(**self.tag_kwargs)


@frozen
class Pipeline:
    """
    A machine learning pipeline that chains multiple processing steps together.

    This class provides a xorq-native implementation that wraps scikit-learn pipelines,
    enabling deferred execution and integration with xorq expressions. The pipeline
    can contain both transform steps (data preprocessing) and a final prediction step.

    Parameters
    ----------
    steps : tuple of Step
        Sequence of Step objects that make up the pipeline.

    Attributes
    ----------
    steps : tuple of Step
        The sequence of processing steps.
    instance : sklearn.pipeline.Pipeline
        The equivalent scikit-learn Pipeline instance.
    transform_steps : tuple of Step
        All steps except the final prediction step (if any).
    predict_step : Step or None
        The final step if it has a predict method, otherwise None.

    Examples
    --------
    Create a pipeline from scikit-learn estimators:

    >>> from xorq.ml import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> import sklearn.pipeline
    >>>
    >>> sklearn_pipeline = sklearn.pipeline.Pipeline([
    ...     ("scaler", StandardScaler()),
    ...     ("knn", KNeighborsClassifier(n_neighbors=5))
    ... ])
    >>> xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)

    Fit and predict with xorq expressions:

    >>> # Assuming train and test are xorq expressions
    >>> fitted = xorq_pipeline.fit(train, features=("feature1", "feature2"), target="target")  # quartodoc: +SKIP
    >>> predictions = fitted.predict(test)  # quartodoc: +SKIP

    Update pipeline parameters:

    >>> updated_pipeline = xorq_pipeline.set_params(knn__n_neighbors=10)  # quartodoc: +SKIP

    Notes
    -----
    - The Pipeline class is frozen (immutable) using attrs.
    - Pipelines automatically detect transform vs predict steps based on method availability.
    - The fit() method returns a FittedPipeline that can transform and predict on new data.
    - Parameter updates use sklearn's parameter naming convention (step__parameter).
    """

    steps = field(
        validator=deep_iterable(instance_of(Step), instance_of(tuple)),
        converter=tuple,
    )

    def __attrs_post_init__(self):
        assert self.steps

    @property
    def instance(self):
        """
        Create an equivalent scikit-learn Pipeline instance.

        Returns
        -------
        sklearn.pipeline.Pipeline
            A scikit-learn pipeline with the same steps and parameters.
        """

        import sklearn

        return sklearn.pipeline.Pipeline(
            tuple((step.name, step.instance) for step in self.steps)
        )

    @property
    def transform_steps(self):
        """
        Get all transformation steps (excluding final prediction step).

        Returns
        -------
        tuple of Step
            All steps that transform data but don't make final predictions.
        """

        (*steps, last_step) = self.steps
        if hasattr(last_step.instance, "predict"):
            return steps
        else:
            return self.steps

    @property
    def predict_step(self):
        """
        Get the final prediction step if it exists.

        Returns
        -------
        Step or None
            The final step if it has a predict method, otherwise None.
        """

        (*_, last_step) = self.steps
        return last_step if hasattr(last_step.instance, "predict") else None

    def fit(self, expr, features=None, target=None, cache=None):
        """
        Fit the pipeline to training data.

        This method sequentially fits each step in the pipeline, using the output
        of each transform step as input to the next step.

        Parameters
        ----------
        expr : Expr
            The xorq expression containing training data.
        features : tuple of str, optional
            Column names to use as features. If None, infers from expr columns
            excluding the target.
        target : str, optional
            Target column name. Required if pipeline has a prediction step.
        cache : Cache, optional
            Storage backend for caching fitted models.

        Returns
        -------
        FittedPipeline
            A fitted pipeline that can transform and predict on new data.

        Raises
        ------
        ValueError
            If target is not provided but pipeline has a prediction step.

        Examples
        --------
        >>> fitted = pipeline.fit(
        ...     train_data,
        ...     features=("sepal_length", "sepal_width"),
        ...     target="species"
        ... )  # quartodoc: +SKIP
        """

        if not target and self.predict_step:
            raise ValueError("Can't infer target for a prediction step")
        features = features or tuple(col for col in expr.columns if col != target)
        fitted_steps = ()
        transformed = expr
        for step in self.transform_steps:
            fitted_step = step.fit(
                transformed,
                features=features,
                target=target,
                cache=cache,
            )
            fitted_steps += (fitted_step,)
            transformed = fitted_step.transform(transformed)
            # hack: unclear why we need to do this, but we do
            transformed = transformed.pipe(do_into_backend)
            # Update features for next step - use structer if available, otherwise
            # extract from transformed schema (for packed format transformers)
            if fitted_step.structer is not None:
                features = tuple(fitted_step.structer.dtype)
            else:
                # For packed format (ColumnTransformer, etc.), the output is a single
                # 'transformed' column. The next step needs all columns except retained others.
                features = tuple(
                    col
                    for col in transformed.columns
                    if col not in fitted_step.get_others(expr) or col == "transformed"
                )
        if step := self.predict_step:
            fitted_step = step.fit(
                transformed,
                features=features,
                target=target,
                cache=cache,
            )
            fitted_steps += (fitted_step,)
            # transformed = fitted_step.transform(transformed)
        return FittedPipeline(fitted_steps, expr)

    @classmethod
    def from_instance(cls, instance, deep=False):
        """
        Create a Pipeline from an existing scikit-learn Pipeline.

        Parameters
        ----------
        instance : sklearn.pipeline.Pipeline
            A fitted or unfitted scikit-learn pipeline.

        Returns
        -------
        Pipeline
            A new xorq Pipeline wrapping the scikit-learn pipeline.

        Examples
        --------
        >>> import sklearn.pipeline
        >>> from sklearn.preprocessing import StandardScaler
        >>> from sklearn.svm import SVC
        >>>
        >>> sklearn_pipe = sklearn.pipeline.Pipeline([
        ...     ("scaler", StandardScaler()),
        ...     ("svc", SVC())
        ... ])
        >>> xorq_pipe = Pipeline.from_instance(sklearn_pipe)
        """
        # https://github.com/scikit-learn/scikit-learn/issues/18272#issuecomment-682180783
        steps = tuple(
            Step.from_instance_name(step, name, deep=deep)
            for name, step in instance.steps
        )
        return cls(steps)

    def set_params(self, **kwargs):
        return self.__class__.from_instance(self.instance.set_params(**kwargs))

    def to_sklearn(self):
        """
        Return the unfitted sklearn Pipeline.

        This provides round-trip conversion from xorq Pipeline back to sklearn.
        The returned pipeline is equivalent to the original but unfitted.

        Returns
        -------
        sklearn.pipeline.Pipeline
            An unfitted scikit-learn Pipeline instance.

        Examples
        --------
        >>> import sklearn.pipeline
        >>> from sklearn.preprocessing import StandardScaler
        >>> from sklearn.svm import SVC
        >>>
        >>> sklearn_pipe = sklearn.pipeline.Pipeline([
        ...     ("scaler", StandardScaler()),
        ...     ("svc", SVC())
        ... ])
        >>> xorq_pipe = Pipeline.from_instance(sklearn_pipe)
        >>> sklearn_pipe_back = xorq_pipe.to_sklearn()
        >>> # sklearn_pipe_back is equivalent to sklearn_pipe
        """
        return self.instance


@frozen
class FittedPipeline:
    fitted_steps = field(
        validator=deep_iterable(instance_of(FittedStep), instance_of(tuple)),
        converter=tuple,
    )
    expr = field(validator=instance_of(Expr))

    def __attrs_post_init__(self):
        assert self.fitted_steps

    @property
    def is_predict(self):
        (*_, last_step) = self.fitted_steps
        return hasattr(last_step.step.instance, "predict")

    @property
    def transform_steps(self):
        return self.fitted_steps[:-1] if self.is_predict else self.fitted_steps

    @property
    def predict_step(self):
        return self.fitted_steps[-1] if self.is_predict else None

    @property
    def tag_kwargs(self):
        return {
            "transforms_tags": tuple(
                tuple(fitted_step.tag_kwargs.items())
                for fitted_step in self.transform_steps
            ),
        }

    def transform(self, expr, tag=True):
        transformed = expr
        for fitted_step in self.transform_steps:
            transformed = fitted_step.transform(transformed).pipe(do_into_backend)
        if tag:
            transformed = transformed.tag(
                "FittedPipeline-transform",
                **self.tag_kwargs,
            )
        return transformed

    def predict(self, expr):
        transformed = self.transform(expr, tag=False)
        return (
            self.predict_step.predict(transformed)
            .pipe(do_into_backend)
            .tag(
                "FittedPipeline-predict",
                predict_tags=tuple(self.predict_step.tag_kwargs.items()),
            )
        )

    def predict_proba(self, expr):
        """
        Return class probabilities for the pipeline.

        Transforms input through all transform steps, then calls predict_proba
        on the final prediction step.

        Parameters
        ----------
        expr : Expr
            Input expression with feature columns.

        Returns
        -------
        Expr
            Expression with probability column in packed format.

        Raises
        ------
        AttributeError
            If the final step doesn't have predict_proba method.
        """
        if not self.predict_step:
            raise ValueError("Pipeline has no prediction step")
        transformed = self.transform(expr, tag=False)
        return (
            self.predict_step.predict_proba(transformed)
            .pipe(do_into_backend)
            .tag(
                "FittedPipeline-predict_proba",
                predict_tags=tuple(self.predict_step.tag_kwargs.items()),
            )
        )

    def decision_function(self, expr):
        """
        Return decision function values for the pipeline.

        Transforms input through all transform steps, then calls decision_function
        on the final prediction step.

        Parameters
        ----------
        expr : Expr
            Input expression with feature columns.

        Returns
        -------
        Expr
            Expression with decision function column in packed format.

        Raises
        ------
        AttributeError
            If the final step doesn't have decision_function method.
        """
        if not self.predict_step:
            raise ValueError("Pipeline has no prediction step")
        transformed = self.transform(expr, tag=False)
        return (
            self.predict_step.decision_function(transformed)
            .pipe(do_into_backend)
            .tag(
                "FittedPipeline-decision_function",
                predict_tags=tuple(self.predict_step.tag_kwargs.items()),
            )
        )

    def score_expr(self, expr, **kwargs):
        # NOTE: this is non-deferred
        clf = self.predict_step.model
        df = self.transform(expr).execute()
        X = df[list(self.predict_step.features)]
        y = df[self.predict_step.target]
        return clf.score(X, y, **kwargs)

    def score(self, X, y, **kwargs):
        import numpy as np
        import pandas as pd

        from xorq.expr import api

        df = pd.DataFrame(np.array(X), columns=self.features).assign(**{self.target: y})
        expr = api.register(df, "t")
        return self.score_expr(expr, **kwargs)

    def to_sklearn(self):
        """
        Return a fitted sklearn Pipeline.

        This triggers execution of all deferred models if not already executed,
        and reconstructs a fitted sklearn Pipeline from the fitted steps.

        Returns
        -------
        sklearn.pipeline.Pipeline
            A fitted scikit-learn Pipeline that can be used directly.

        Examples
        --------
        >>> fitted_pipeline = pipeline.fit(expr, target="y")
        >>> sklearn_pipeline = fitted_pipeline.to_sklearn()
        >>> sklearn_pipeline.predict(new_df)  # Use sklearn directly
        """
        import sklearn.pipeline

        return sklearn.pipeline.Pipeline(
            [
                (fitted_step.step.name, fitted_step.to_sklearn())
                for fitted_step in self.fitted_steps
            ]
        )


def get_target_type(step_instance, step, expr, features, target):
    return expr[target].type()


registry = Dispatch()


def get_predict_return_type(instance, step, expr, features, target):
    assert isinstance(instance, step.typ)
    if return_type := getattr(instance, "return_type", None):
        return return_type
    else:
        return registry(instance, step, expr, features, target)


@registry.register(object)
def raise_on_unregistered(instance, step, expr, features, target):
    raise ValueError(f"Can't handle {instance.__class__.__name__}")


@registry.register_lazy("sklearn")
def lazy_register_sklearn():
    from sklearn.base import (
        ClassifierMixin,
        ClusterMixin,
        RegressorMixin,
    )
    from sklearn.linear_model import (
        LinearRegression,
        LogisticRegression,
    )
    from sklearn.neighbors import (
        KNeighborsClassifier,
    )
    from sklearn.pipeline import Pipeline as SklearnPipeline

    def get_pipeline_return_type(instance, step, expr, features, target):
        """Infer return type from sklearn Pipeline's final step."""
        # Get the final estimator in the pipeline
        final_step = instance.steps[-1][1]
        # Recursively determine return type based on final step's type
        if isinstance(final_step, ClassifierMixin):
            return get_target_type(instance, step, expr, features, target)
        elif isinstance(final_step, RegressorMixin):
            return dt.float64
        elif isinstance(final_step, ClusterMixin):
            return dt.int64
        else:
            # Default to target type if classifier-like, else float64
            return get_target_type(instance, step, expr, features, target)

    registry.register(LinearRegression, return_constant(dt.float64))
    registry.register(LogisticRegression, get_target_type)
    registry.register(KNeighborsClassifier, get_target_type)
    registry.register(ClassifierMixin, get_target_type)
    # Mixin catch-alls for generic sklearn support
    registry.register(RegressorMixin, return_constant(dt.float64))
    registry.register(ClusterMixin, return_constant(dt.int64))
    # sklearn Pipeline support
    registry.register(SklearnPipeline, get_pipeline_return_type)


get_predict_return_type.register = registry.register
