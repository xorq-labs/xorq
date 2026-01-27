import functools
import pickle

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
from xorq.common.utils.name_utils import (
    make_name,
)
from xorq.expr.ml.fit_lib import (
    DeferredFitOther,
    decision_function_sklearn,
    feature_importances_sklearn,
    predict_proba_sklearn,
)
from xorq.expr.ml.structer import (
    Structer,
)
from xorq.ibis_yaml.utils import freeze
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
    params_tuple = field(validator=instance_of(tuple), default=(), converter=freeze)

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
    def instance(self):
        return self.step.instance

    @property
    def is_transform(self):
        return hasattr(self.step.typ, "transform")

    @property
    def is_predict(self):
        return hasattr(self.step.typ, "predict")

    @property
    def predict_return_type(self):
        return get_predict_return_type(self)

    @property
    @functools.cache
    def _deferred_fit_other(self):
        return DeferredFitOther.from_fitted_step(self)

    @property
    def deferred_model(self):
        return self._deferred_fit_other.deferred_model

    @property
    def model_udf(self):
        return self._deferred_fit_other.deferred_other

    @property
    def deferred_transform(self):
        if self.is_transform:
            return self._deferred_fit_other.deferred_other
        else:
            return None

    @property
    def deferred_predict(self):
        if self.is_predict:
            return self._deferred_fit_other.deferred_other
        else:
            return None

    @property
    def deferred_predict_proba(self):
        (attrname, fn) = ("predict_proba", predict_proba_sklearn)
        if hasattr(self.step.instance.__class__, attrname):
            return self._deferred_fit_other.make_deferred_other(
                fn=fn,
                return_type=dt.Array(dt.float64),
                name_infix=attrname.rstrip("_"),
            )
        else:
            return None

    @property
    def deferred_decision_function(self):
        (attrname, fn) = ("decision_function", decision_function_sklearn)
        if hasattr(self.step.instance.__class__, attrname):
            return self._deferred_fit_other.make_deferred_other(
                fn=fn,
                return_type=dt.Array(dt.float64),
                name_infix=attrname.rstrip("_"),
            )
        else:
            return None

    @property
    def deferred_feature_importances(self):
        (attrname, fn) = ("feature_importances_", feature_importances_sklearn)
        if hasattr(self.step.instance.__class__, attrname):
            return self._deferred_fit_other.make_deferred_other(
                fn=fn,
                return_type=dt.Array(dt.float64),
                name_infix=attrname.rstrip("_"),
            )
        else:
            return None

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
        if retain_others and (others := self.get_others(expr)):
            expr = expr.select(*others, struct_col)
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
        col = self.transform_raw(expr)
        others = self.get_others(expr) if retain_others else ()
        expr = expr.select(*others, col) if others else col.as_table()
        return self.structer.maybe_unpack(expr, col.get_name()).tag(
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
        if retain_others and (others := self.get_others(expr)):
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

    @toolz.curry
    def invoke_method_raw(self, expr, name=None, *, methodname):
        if not (method := getattr(self, f"deferred_{methodname}", None)):
            raise AttributeError(
                f"'{self.step.typ.__name__}' object has no attribute '{methodname}'"
            )
        return method.on_expr(expr).name(name or methodname)

    @toolz.curry
    def invoke_method(self, expr, retain_others=True, name=None, *, methodname):
        col = self.invoke_method_raw(expr=expr, name=name, methodname=methodname)
        if retain_others and (others := self.get_others(expr)):
            expr = expr.select(*others, col)
        else:
            expr = col.as_table()

        return expr.tag(**self.tag_kwargs)

    predict_proba_raw = invoke_method_raw(
        methodname="predict_proba", name="predicted_proba"
    )

    predict_proba = invoke_method(methodname="predict_proba", name="predicted_proba")

    decision_function_raw = invoke_method_raw(methodname="decision_function")

    decision_function = invoke_method(methodname="decision_function")

    feature_importances_raw = invoke_method_raw(methodname="feature_importances")

    def feature_importances(self, expr=None, name=None):
        import xorq.api as xo

        schema = self.expr.select(self.features).schema()
        empty_table = xo.memtable(
            [{name: None for name in schema.names}], schema=schema
        )
        col = self.feature_importances_raw(empty_table, name)
        return col.as_table().tag(**self.tag_kwargs)


def _is_container_transformer(instance):
    """Check if instance is a container transformer (ColumnTransformer, FeatureUnion, Pipeline)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import FeatureUnion
    from sklearn.pipeline import Pipeline as SklearnPipeline

    return isinstance(instance, (ColumnTransformer, FeatureUnion, SklearnPipeline))


def _analyze_child_structer(transformer, expr, columns):
    """Analyze a child transformer to determine if it's KV-encoded."""
    if transformer in ("drop", "passthrough"):
        return None

    # Recursively check containers
    if _is_container_transformer(transformer):
        child_info = _analyze_container(transformer, expr, columns)
        # Container is KV-encoded if any child is KV-encoded
        any_kv = any(c["is_kv_encoded"] for c in child_info if c is not None)
        return {"is_kv_encoded": any_kv, "child_info": child_info}

    # For simple transformers, use Structer to determine
    try:
        structer = Structer.from_instance_expr(transformer, expr, features=columns)
        return {"is_kv_encoded": structer.is_kv_encoded, "structer": structer}
    except ValueError:
        # Unregistered transformer - assume KV-encoded for safety
        return {"is_kv_encoded": True}


def _analyze_container(instance, expr, features=None):
    """
    Analyze a container transformer to identify child structure.

    Returns list of child info dicts with:
    - name: transformer name
    - transformer: the transformer instance
    - columns: columns this transformer operates on
    - is_kv_encoded: whether this child produces KV-encoded output
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import FeatureUnion
    from sklearn.pipeline import Pipeline as SklearnPipeline

    features = features or tuple(expr.columns)
    children = []

    if isinstance(instance, ColumnTransformer):
        transformers = getattr(instance, "transformers_", None) or instance.transformers
        for name, transformer, columns in transformers:
            if transformer == "drop":
                continue
            # Normalize columns
            if isinstance(columns, str):
                columns = (columns,)
            elif isinstance(columns, list):
                columns = tuple(columns)
            elif columns is None:
                columns = ()

            child_analysis = _analyze_child_structer(transformer, expr, columns)
            if child_analysis is not None:
                children.append(
                    freeze(
                        {
                            "name": name,
                            "columns": columns,
                            "is_kv_encoded": child_analysis["is_kv_encoded"],
                        }
                    )
                )

    elif isinstance(instance, FeatureUnion):
        for name, transformer in instance.transformer_list:
            child_analysis = _analyze_child_structer(transformer, expr, features)
            if child_analysis is not None:
                children.append(
                    freeze(
                        {
                            "name": name,
                            "columns": features,
                            "is_kv_encoded": child_analysis["is_kv_encoded"],
                        }
                    )
                )

    elif isinstance(instance, SklearnPipeline):
        # For Pipeline, we care about the final step's output
        # but we track all steps for completeness
        current_features = features
        for name, transformer in instance.steps:
            if transformer == "passthrough":
                children.append(
                    freeze(
                        {
                            "name": name,
                            "columns": current_features,
                            "is_kv_encoded": False,
                        }
                    )
                )
                continue

            child_analysis = _analyze_child_structer(
                transformer, expr, current_features
            )
            if child_analysis is not None:
                children.append(
                    freeze(
                        {
                            "name": name,
                            "columns": current_features,
                            "is_kv_encoded": child_analysis["is_kv_encoded"],
                        }
                    )
                )
                # If not KV-encoded, we might know the output columns
                # For simplicity, keep tracking original features
                # (accurate tracking would require structer.output_columns)

    return children


@frozen
class ComplexStep:
    """
    A step for container transformers (ColumnTransformer, FeatureUnion, Pipeline).

    Unlike Step which wraps simple sklearn estimators, ComplexStep handles
    containers that may have mixed known-schema and KV-encoded children.
    The bookkeeping for hybrid output is managed by ComplexFittedStep.

    Parameters
    ----------
    typ : type
        The sklearn container class (ColumnTransformer, FeatureUnion, or Pipeline).
    name : str
        A unique name for this step.
    params_tuple : tuple
        Tuple of (parameter_name, parameter_value) pairs.
    child_info : tuple
        Analyzed child structure from _analyze_container.
    """

    typ = field(validator=instance_of(type))
    name = field(validator=optional(instance_of(str)), default=None)
    params_tuple = field(validator=instance_of(tuple), default=(), converter=freeze)
    child_info = field(validator=instance_of(tuple), default=())

    def __attrs_post_init__(self):
        from sklearn.base import BaseEstimator

        assert BaseEstimator in self.typ.mro()
        if self.name is None:
            object.__setattr__(self, "name", f"{self.typ.__name__.lower()}_{id(self)}")
        object.__setattr__(self, "params_tuple", tuple(sorted(self.params_tuple)))

    @property
    def tag_kwargs(self):
        return {name: getattr(self, name) for name in ("typ", "name", "params_tuple")}

    @property
    def instance(self):
        """Create an instance of the container with configured parameters."""
        return self.typ(**dict(self.params_tuple))

    @property
    def kv_child_names(self):
        """Names of children that produce KV-encoded output."""
        return tuple(c["name"] for c in self.child_info if c["is_kv_encoded"])

    @property
    def known_child_names(self):
        """Names of children that produce known-schema output."""
        return tuple(c["name"] for c in self.child_info if not c["is_kv_encoded"])

    @property
    def is_hybrid(self):
        """True if container has both KV-encoded and known-schema children."""
        return bool(self.kv_child_names) and bool(self.known_child_names)

    @property
    def is_all_kv(self):
        """True if all children are KV-encoded."""
        return bool(self.kv_child_names) and not self.known_child_names

    @property
    def is_all_known(self):
        """True if all children have known schema."""
        return bool(self.known_child_names) and not self.kv_child_names

    def fit(self, expr, features=None, target=None, cache=None, dest_col=None):
        """
        Fit this container step to the given expression data.

        Returns
        -------
        ComplexFittedStep
            A fitted step that can transform data with proper hybrid handling.
        """
        features = features or tuple(expr.columns)
        return ComplexFittedStep(
            step=self,
            expr=expr,
            features=features,
            target=target,
            cache=cache,
            dest_col=dest_col,
        )

    @classmethod
    def from_instance_name(cls, instance, name=None, expr=None, features=None):
        """
        Create a ComplexStep from an sklearn container instance.

        Parameters
        ----------
        instance : ColumnTransformer, FeatureUnion, or Pipeline
            The sklearn container instance.
        name : str, optional
            Name for the step.
        expr : Expr, optional
            Expression for analyzing child structure. If None, child analysis is deferred.
        features : tuple, optional
            Feature columns for analysis.

        Returns
        -------
        ComplexStep
        """
        if not _is_container_transformer(instance):
            raise ValueError(
                f"ComplexStep requires a container transformer, got {type(instance)}"
            )

        params_tuple = tuple(instance.get_params(deep=False).items())

        # Analyze children if expr is provided
        if expr is not None:
            child_info = tuple(_analyze_container(instance, expr, features))
        else:
            child_info = ()

        return cls(
            typ=type(instance),
            name=name,
            params_tuple=params_tuple,
            child_info=child_info,
        )

    __dask_tokenize__ = normalize_attrs


@frozen
class ComplexFittedStep:
    """
    A fitted container step with hybrid transform support.

    This class handles the bookkeeping for containers with mixed known-schema
    and KV-encoded children. The transform methods properly remap sklearn's
    prefixed output columns to the hybrid struct format.

    The key insight is that sklearn containers use naming conventions like
    'transformer_name__feature_name' for output columns. This class uses
    kv_child_names to determine which columns should be grouped into KV arrays
    vs extracted as known struct fields.
    """

    step = field(validator=instance_of(ComplexStep))
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
        if self.features is None:
            features = tuple(col for col in self.expr.columns if col != self.target)
            object.__setattr__(self, "features", features)

        # Re-analyze children now that we have expr
        if not self.step.child_info:
            child_info = tuple(
                _analyze_container(self.step.instance, self.expr, self.features)
            )
            # Update step's child_info (need to create new step since it's frozen)
            new_step = ComplexStep(
                typ=self.step.typ,
                name=self.step.name,
                params_tuple=self.step.params_tuple,
                child_info=child_info,
            )
            object.__setattr__(self, "step", new_step)

    @property
    def instance(self):
        return self.step.instance

    @property
    def kv_child_names(self):
        return self.step.kv_child_names

    @property
    def is_hybrid(self):
        return self.step.is_hybrid

    @property
    def is_transform(self):
        return hasattr(self.step.typ, "transform")

    @property
    @functools.cache
    def structer(self):
        """Build Structer with hybrid/KV info based on child analysis."""
        return Structer.from_instance_expr(
            self.step.instance, self.expr, features=self.features
        )

    @property
    @functools.cache
    def _deferred_fit_other(self):
        """Create DeferredFitOther with hybrid support."""
        from xorq.expr.ml.fit_lib import (
            DeferredFitOther,
            fit_sklearn_args,
            kv_encode_output,
            transform_sklearn_hybrid,
            transform_sklearn_struct,
        )

        kwargs = {
            "expr": self.expr,
            "target": self.target,
            "features": self.features,
            "cache": self.cache,
        }

        structer = self.structer

        if structer.is_hybrid:
            # Hybrid: use transform_sklearn_hybrid with kv_child_names
            return DeferredFitOther(
                fit=fit_sklearn_args(
                    cls=self.step.typ,
                    params=self.step.params_tuple,
                ),
                other=transform_sklearn_hybrid(self.kv_child_names),
                return_type=structer.return_type,
                name_infix="transformed_hybrid",
                **kwargs,
            )
        elif structer.is_kv_encoded:
            # All KV: use standard KV encoding
            return DeferredFitOther(
                fit=fit_sklearn_args(
                    cls=self.step.typ,
                    params=self.step.params_tuple,
                ),
                other=kv_encode_output,
                return_type=structer.return_type,
                name_infix="transformed_encoded",
                **kwargs,
            )
        else:
            # All known schema: use struct conversion
            return DeferredFitOther(
                fit=fit_sklearn_args(
                    cls=self.step.typ,
                    params=self.step.params_tuple,
                ),
                other=transform_sklearn_struct(structer.get_convert_array()),
                return_type=structer.return_type,
                name_infix="transformed",
                **kwargs,
            )

    @property
    def deferred_model(self):
        return self._deferred_fit_other.deferred_model

    @property
    def deferred_transform(self):
        if self.is_transform:
            return self._deferred_fit_other.deferred_other
        return None

    @property
    def tag_kwargs(self):
        return {
            "tag": "ComplexFittedStep-transform",
            **self.step.tag_kwargs,
            "features": self.features,
        }

    def get_others(self, expr):
        return tuple(other for other in expr.columns if other not in self.features)

    def transform_raw(self, expr, name=None):
        """
        Raw transform - returns the UDF output column.

        For hybrid containers, this returns a struct with known fields
        and named KV array fields, as produced by transform_sklearn_hybrid.
        """
        transformed = self.deferred_transform.on_expr(expr).name(
            name or self.dest_col or "transformed"
        )
        return transformed

    def transform(self, expr, retain_others=True):
        """
        Transform with proper unpacking.

        For hybrid containers, unpacks the struct to expose known fields
        while keeping KV array fields intact.
        """
        col = self.transform_raw(expr)
        others = self.get_others(expr) if retain_others else ()
        expr = expr.select(*others, col) if others else col.as_table()
        return self.structer.maybe_unpack(expr, col.get_name()).tag(
            **self.tag_kwargs,
        )

    @property
    def transformed(self):
        return self.transform(self.expr)


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
            features = fitted_step.structer.get_output_columns(
                fitted_step.dest_col or "transformed"
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

    @toolz.curry
    def invoke_predict_method(self, expr, tag_name, tag_key, *, methodname):
        if not self.is_predict:
            raise ValueError("Pipeline does not have a predict step")
        if not (method := getattr(self.predict_step, methodname, None)):
            raise ValueError(f"predict step does not have a method named {methodname}")
        transformed = self.transform(expr, tag=False)
        predicted = (
            method(transformed)
            .pipe(do_into_backend)
            .tag(
                tag_name,
                **{tag_key: tuple(self.predict_step.tag_kwargs.items())},
            )
        )
        return predicted

    predict_proba = invoke_predict_method(
        tag_name="FittedPipeline-predict_proba",
        tag_key="predict_proba_tags",
        methodname="predict_proba",
    )

    decision_function = invoke_predict_method(
        tag_name="FittedPipeline-decision_function",
        tag_key="decision_function_tags",
        methodname="decision_function",
    )

    feature_importances = invoke_predict_method(
        tag_name="FittedPipeline-feature_importances",
        tag_key="feature_importances_tags",
        methodname="feature_importances",
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


def get_predict_return_type(fitted_step):
    instance = fitted_step.step.instance
    if return_type := getattr(instance, "return_type", None):
        return return_type
    else:
        return registry(
            instance,
            fitted_step.step,
            fitted_step.expr,
            fitted_step.features,
            fitted_step.target,
        )


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
