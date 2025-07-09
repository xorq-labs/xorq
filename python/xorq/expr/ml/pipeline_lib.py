import functools
import pickle
from typing import Callable

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
from sklearn.base import (
    BaseEstimator,
)
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)
from sklearn.feature_selection import (
    SelectKBest,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)
from sklearn.neighbors import (
    KNeighborsClassifier,
)
from sklearn.svm import (
    LinearSVC,
)

import xorq as xo
import xorq.expr.datatypes as dt
from xorq.caching import (
    ParquetStorage,
    SourceStorage,
)
from xorq.common.utils.dask_normalize.dask_normalize_utils import (
    normalize_attrs,
)
from xorq.common.utils.func_utils import (
    return_constant,
)
from xorq.expr.ml.fit_lib import (
    deferred_fit_predict_sklearn,
    deferred_fit_transform_series_sklearn,
    deferred_fit_transform_sklearn_struct,
)
from xorq.expr.ml.structer import (
    Structer,
)
from xorq.vendor.ibis.expr.types.core import Expr


def do_into_backend(expr, con=None):
    return expr.into_backend(con or xo.connect())


def make_estimator_typ(fit, predict, return_type, name=None):
    assert hasattr(fit, "__call__") and hasattr(predict, "__call__")
    assert isinstance(return_type, dt.DataType)

    def make_name(prefix, to_tokenize, n=32):
        tokenized = dask.base.tokenize(to_tokenize)
        return ("_" + prefix + "_" + tokenized)[:n].lower()

    def wrapped_fit(self, *args, **kwargs):
        self._model = fit(*args, **kwargs)

    def wrapped_predict(self, *args, **kwargs):
        return predict(self._model, *args, **kwargs)

    name = name or make_name("estimator", (fit, predict))
    typ = type(
        name,
        (BaseEstimator,),
        {
            "fit": wrapped_fit,
            "predict": wrapped_predict,
            "return_type": return_type,
            "_fit": fit,
            "_predict": predict,
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
        assert BaseEstimator in self.typ.mro()
        if self.name is None:
            object.__setattr__(self, "name", f"{self.typ.__name__.lower()}_{id(self)}")
        # param order invariant
        object.__setattr__(self, "params_tuple", tuple(sorted(self.params_tuple)))

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

    def fit(self, expr, features=None, target=None, storage=None, dest_col=None):
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
        storage : Storage, optional
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
            storage=storage,
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
    def from_instance_name(cls, instance, name=None):
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
        params_tuple = tuple(instance.get_params().items())
        return cls(typ=instance.__class__, name=name, params_tuple=params_tuple)

    @classmethod
    def from_name_instance(cls, name, instance):
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
        return cls.from_instance_name(instance, name)

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
    storage = field(
        validator=optional(instance_of((ParquetStorage, SourceStorage))), default=None
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
            "storage": self.storage,
        }
        (deferred_transform, deferred_predict) = (None, None)
        # this should be in lock step with Structer.from_instance_expr
        if self.is_transform:
            f = deferred_fit_transform_sklearn_struct
            if self.step.typ in (TfidfVectorizer,):
                (kwargs["col"],) = kwargs.pop("features")
                kwargs["return_type"] = dt.Array(dt.float64)
                f = deferred_fit_transform_series_sklearn
            elif self.step.typ in (SelectKBest,):
                kwargs |= {"target": self.target}
            (deferred_model, model_udf, deferred_transform) = f(**kwargs)
        elif self.is_predict:
            predict_kwargs = {
                "target": self.target,
                "return_type": get_predict_return_type(
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
        return {
            "deferred_model": deferred_model,
            "model_udf": model_udf,
            "deferred_transform": deferred_transform,
            "deferred_predict": deferred_predict,
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
    @functools.cache
    def model(self):
        return pickle.loads(self.deferred_model.execute())

    @property
    @functools.cache
    def structer(self):
        return Structer.from_instance_expr(
            self.step.instance, self.expr, features=self.features
        )

    def transform_unpack(self, expr, retain_others=True, name="to_unpack"):
        struct_col = self.transform_raw(expr).name(name)
        if retain_others and (
            others := tuple(
                other for other in expr.columns if other not in self.features
            )
        ):
            expr = expr.select(*others, struct_col)
        else:
            expr = struct_col.as_table()
        return expr.unpack(name)

    def transform_raw(self, expr):
        transformed = self.deferred_transform.on_expr(expr)
        if self.dest_col is not None:
            transformed = transformed.name(self.dest_col)
        return transformed

    def transform(self, expr, retain_others=True):
        if self.step.typ in (TfidfVectorizer,):
            col = self.transform_raw(expr)
            if retain_others and (
                others := tuple(
                    other for other in expr.columns if other not in self.features
                )
            ):
                return expr.select(*others, col)
            else:
                return col
        else:
            return self.transform_unpack(expr, retain_others=retain_others)

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
        return expr

    @property
    def predicted(self):
        return self.predict(self.expr)


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

    def fit(self, expr, features=None, target=None):
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
        features = features or tuple(col for col in expr.features if col != target)
        fitted_steps = ()
        transformed = expr
        for step in self.transform_steps:
            fitted_step = step.fit(
                transformed,
                features=features,
                target=target,
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
            )
            fitted_steps += (fitted_step,)
            # transformed = fitted_step.transform(transformed)
        return FittedPipeline(fitted_steps, expr)

    @classmethod
    def from_instance(cls, instance):
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
        steps = tuple(
            Step.from_instance_name(step, name) for name, step in instance.steps
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

    def transform(self, expr):
        transformed = expr
        for fitted_step in self.transform_steps:
            transformed = fitted_step.transform(transformed).pipe(do_into_backend)
        return transformed

    def predict(self, expr):
        transformed = self.transform(expr)
        return self.predict_step.predict(transformed).pipe(do_into_backend)

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

        df = pd.DataFrame(np.array(X), columns=self.features).assign(**{self.target: y})
        expr = xo.register(df, "t")
        return self.score_expr(expr, **kwargs)


def get_target_type(step, expr, features, target):
    return expr[target].type()


step_typ_to_f = {
    LinearRegression: return_constant(dt.float),
    LogisticRegression: return_constant(dt.float),
    KNeighborsClassifier: get_target_type,
    LinearSVC: get_target_type,
}


def get_predict_return_type(step, expr, features, target):
    if return_type := getattr(step.typ, "return_type", None):
        return return_type
    elif f := step_typ_to_f.get(step.typ):
        return_type = f(step, expr, features, target)
        return return_type
    else:
        raise ValueError(f"Can't handle {step.typ.__name__}")


def register(typ, f, clobber=False):
    if not isinstance(typ, type):
        raise ValueError
    if not isinstance(f, Callable):
        raise ValueError
    # FIXME: check that signature matches
    if typ in step_typ_to_f and not clobber:
        raise ValueError
    step_typ_to_f[typ] = f


get_predict_return_type.register = register
