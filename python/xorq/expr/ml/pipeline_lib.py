import functools
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
    deferred_decision_function_from_trained_model,
    deferred_feature_importances_from_trained_model,
    deferred_fit_predict_sklearn,
    deferred_fit_transform_series_sklearn,
    deferred_fit_transform_sklearn_struct,
    deferred_predict_proba_from_trained_model,
)
from xorq.expr.ml.structer import (
    Structer,
)
from xorq.vendor.ibis.expr.types.core import Expr


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
        return {name: getattr(self, name) for name in ("typ", "name", "params_tuple")}

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


def _has_feature_importances(estimator_class):
    """Check if estimator class defines feature_importances_ property."""
    return any(
        "feature_importances_" in cls.__dict__ for cls in estimator_class.__mro__
    )


def _create_deferred_method(method_name, factory_fn, schema, features, deferred_model):
    """Create a deferred method if the estimator supports it.

    Returns
    -------
    callable or None
        The deferred method if supported, None otherwise
    """
    return factory_fn(
        schema=schema,
        features=features,
        deferred_model=deferred_model,
        return_type=dt.Array(dt.float64),
        name_infix=method_name,
    )


def _get_non_feature_columns(expr, features):
    """Get columns that are not in features as a tuple."""
    return tuple(col for col in expr.columns if col not in features)


def _apply_prediction_method(
    deferred_method, expr, features, tag_kwargs, name, default_name, retain_others=True
):
    """Apply a prediction method with optional column retention.

    Parameters
    ----------
    deferred_method : callable
        The deferred method to apply
    expr : ibis.Expr
        The expression to apply the method to
    features : tuple
        Feature columns used by the model
    tag_kwargs : dict
        Tagging kwargs for the expression
    name : str or None
        Custom name for the result column
    default_name : str
        Default name to use if name is None
    retain_others : bool
        Whether to retain non-feature columns

    Returns
    -------
    ibis.Expr
        Tagged expression with results
    """
    col = deferred_method.on_expr(expr).name(name or default_name)

    if retain_others and (others := _get_non_feature_columns(expr, features)):
        expr = expr.select(*others, col)
    else:
        expr = col.as_table()

    return expr.tag(**tag_kwargs)


def _apply_pipeline_prediction_method(
    fitted_pipeline, method_name, expr, tag_name, tag_key
):
    """Apply a prediction method through the fitted pipeline.

    Parameters
    ----------
    fitted_pipeline : FittedPipeline
        The fitted pipeline instance
    method_name : str
        Name of the method to call on predict_step
    expr : ibis.Expr
        The expression to predict on
    tag_name : str
        Tag name for the resulting expression
    tag_key : str
        Tag key for storing the predict step tags

    Returns
    -------
    ibis.Expr
        Tagged expression with predictions
    """
    if not fitted_pipeline.is_predict:
        raise ValueError("Pipeline does not have a predict step")

    transformed = fitted_pipeline.transform(expr, tag=False)
    method = getattr(fitted_pipeline.predict_step, method_name)

    return (
        method(transformed)
        .pipe(do_into_backend)
        .tag(
            tag_name,
            **{tag_key: tuple(fitted_pipeline.predict_step.tag_kwargs.items())},
        )
    )


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
        return hasattr(self.step.typ, "transform")

    @property
    def is_predict(self):
        return hasattr(self.step.typ, "predict")

    @property
    @functools.cache
    def _pieces(self):
        kwargs = {
            "expr": self.expr,
            "features": self.features,
            "cls": self.step.typ,
            "params": self.step.params_tuple,
            "cache": self.cache,
        }
        (
            deferred_transform,
            deferred_predict,
            deferred_predict_proba,
            deferred_decision_function,
            deferred_feature_importances,
        ) = (None, None, None, None, None)

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
                    else:
                        f = deferred_fit_transform_sklearn_struct
            (deferred_model, model_udf, deferred_transform) = f(**kwargs)
        elif self.is_predict:
            schema = self.expr.select(self.features).schema()
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

            # Create deferred methods if the model supports them
            deferred_predict_proba = (
                _create_deferred_method(
                    "predicted_proba",
                    deferred_predict_proba_from_trained_model,
                    schema,
                    self.features,
                    deferred_model,
                )
                if hasattr(self.step.instance, "predict_proba")
                else None
            )

            deferred_decision_function = (
                _create_deferred_method(
                    "decision_function",
                    deferred_decision_function_from_trained_model,
                    schema,
                    self.features,
                    deferred_model,
                )
                if hasattr(self.step.instance, "decision_function")
                else None
            )

            # feature_importances_ only exists after fitting, so we check the class hierarchy
            deferred_feature_importances = (
                _create_deferred_method(
                    "feature_importances",
                    deferred_feature_importances_from_trained_model,
                    schema,
                    self.features,
                    deferred_model,
                )
                if _has_feature_importances(self.step.instance.__class__)
                else None
            )
        else:
            raise ValueError
        return {
            "deferred_model": deferred_model,
            "model_udf": model_udf,
            "deferred_transform": deferred_transform,
            "deferred_predict": deferred_predict,
            "deferred_predict_proba": deferred_predict_proba,
            "deferred_decision_function": deferred_decision_function,
            "deferred_feature_importances": deferred_feature_importances,
        }

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
    def deferred_predict_proba(self):
        return self._pieces.get("deferred_predict_proba")

    @property
    def deferred_decision_function(self):
        return self._pieces.get("deferred_decision_function")

    @property
    def deferred_feature_importances(self):
        return self._pieces.get("deferred_feature_importances")

    @property
    @functools.cache
    def model(self):
        import pandas as pd

        match obj := self.deferred_model.execute():
            case pd.DataFrame():
                ((obj,),) = obj.values
            case bytes():
                pass
            case _:
                raise ValueError
        return pickle.loads(obj)

    @property
    @functools.cache
    @cexcepts(ValueError)
    def structer(self):
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

    def predict_proba_raw(self, expr, name=None):
        """Get raw probability predictions as a column expression."""
        if not self.deferred_predict_proba:
            raise AttributeError(
                f"'{self.step.typ.__name__}' object has no attribute 'predict_proba'"
            )
        return self.deferred_predict_proba.on_expr(expr).name(name or "predicted_proba")

    def predict_proba(self, expr, retain_others=True, name=None):
        """Get probability predictions with optional other columns."""
        if not self.deferred_predict_proba:
            raise AttributeError(
                f"'{self.step.typ.__name__}' object has no attribute 'predict_proba'"
            )
        return _apply_prediction_method(
            self.deferred_predict_proba,
            expr,
            self.features,
            self.tag_kwargs,
            name,
            "predicted_proba",
            retain_others,
        )

    def decision_function_raw(self, expr, name=None):
        """Get raw decision function scores as a column expression."""
        if not self.deferred_decision_function:
            raise AttributeError(
                f"'{self.step.typ.__name__}' object has no attribute 'decision_function'"
            )
        return self.deferred_decision_function.on_expr(expr).name(
            name or "decision_function"
        )

    def decision_function(self, expr, retain_others=True, name=None):
        """Get decision function scores with optional other columns."""
        if not self.deferred_decision_function:
            raise AttributeError(
                f"'{self.step.typ.__name__}' object has no attribute 'decision_function'"
            )
        return _apply_prediction_method(
            self.deferred_decision_function,
            expr,
            self.features,
            self.tag_kwargs,
            name,
            "decision_function",
            retain_others,
        )

    def feature_importances(self, expr=None, name=None):
        # FIXME
        import xorq.api as xo

        schema = self.expr.select(self.features).schema()
        empty_table = xo.memtable(
            [{name: None for name in schema.names}], schema=schema
        )
        col = self.deferred_feature_importances.on_expr(empty_table).name(
            name or "feature_importances"
        )
        return col.as_table().tag(**self.tag_kwargs)


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
            features = tuple(fitted_step.structer.dtype)
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
        """Predict class probabilities for classification problems.

        This method applies transformations and then uses the predictor's
        predict_proba method to get probability estimates.

        Parameters
        ----------
        expr : ibis.Expr
            The expression to predict on

        Returns
        -------
        ibis.Expr
            Expression with predicted probabilities column
        """
        return _apply_pipeline_prediction_method(
            self,
            "predict_proba",
            expr,
            "FittedPipeline-predict_proba",
            "predict_proba_tags",
        )

    def decision_function(self, expr):
        """Compute decision function scores for classifiers.

        Parameters
        ----------
        expr : ibis.Expr
            The expression to predict on

        Returns
        -------
        ibis.Expr
            Expression with decision function scores
        """
        return _apply_pipeline_prediction_method(
            self,
            "decision_function",
            expr,
            "FittedPipeline-decision_function",
            "decision_function_tags",
        )

    def feature_importances(self, expr):
        """Get feature importances for the model.

        This method applies transformations and then extracts feature importances
        from the predictor model.

        Parameters
        ----------
        expr : ibis.Expr
            The expression to compute feature importances on

        Returns
        -------
        ibis.Expr
            Expression with feature importances column
        """
        return _apply_pipeline_prediction_method(
            self,
            "feature_importances",
            expr,
            "FittedPipeline-feature_importances",
            "feature_importances_tags",
        )

    def score_expr(self, expr, metric_fn=None, use_proba=False, **kwargs):
        """Compute metrics using deferred execution.

        Parameters
        ----------
        expr : ibis.Expr
            Expression containing test data
        metric_fn : callable, optional
            Metric function from sklearn.metrics. If None, uses model's default
        use_proba : bool, optional
            If True, use predict_proba for metric computation (default: False)
        **kwargs : dict
            Additional arguments passed to the metric function

        Returns
        -------
        ibis.Expr
            Deferred metric expression
        """
        from sklearn.base import is_classifier
        from sklearn.metrics import accuracy_score, r2_score

        from xorq.expr.ml.metrics import deferred_sklearn_metric

        # Default metric based on model type
        if metric_fn is None:
            metric_fn = (
                accuracy_score if is_classifier(self.predict_step.model) else r2_score
            )

        # Get predictions with appropriate method
        expr_with_preds, pred_col = (
            (self.predict_proba(expr), "predicted_proba")
            if use_proba
            else (self.predict(expr), "predicted")
        )

        return deferred_sklearn_metric(
            expr=expr_with_preds,
            target=self.predict_step.target,
            pred_col=pred_col,
            metric_fn=metric_fn,
            metric_kwargs=kwargs,
        )

    def score(self, X, y, **kwargs):
        import numpy as np
        import pandas as pd

        from xorq.expr import api

        if not self.is_predict:
            raise ValueError("Pipeline does not have a predict step")

        df = pd.DataFrame(
            np.array(X),
            columns=self.predict_step.features,
        ).assign(**{self.predict_step.target: y})
        expr = api.register(df, "t")
        return self.score_expr(expr, **kwargs).execute()


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
        RegressorMixin,
    )
    from sklearn.ensemble import (
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.linear_model import (
        LinearRegression,
        LogisticRegression,
    )
    from sklearn.neighbors import (
        KNeighborsClassifier,
    )

    registry.register(LinearRegression, return_constant(dt.float))
    registry.register(LogisticRegression, get_target_type)
    registry.register(KNeighborsClassifier, get_target_type)
    registry.register(ClassifierMixin, get_target_type)
    registry.register(RandomForestRegressor, return_constant(dt.float))
    registry.register(RandomForestClassifier, get_target_type)
    registry.register(
        RegressorMixin, return_constant(dt.float)
    )  # General fallback for regressors


get_predict_return_type.register = registry.register
