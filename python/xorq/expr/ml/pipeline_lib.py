import pickle

import dask
import pyarrow as pa
import toolz
from attr import field, frozen
from attr.validators import deep_iterable, instance_of, optional
from dask.utils import Dispatch
from toolz.curried import excepts as cexcepts

import xorq.expr.datatypes as dt
from xorq.backends.xorq import connect
from xorq.caching import (
    ParquetCache,
    ParquetSnapshotCache,
    ParquetTTLSnapshotCache,
    SourceCache,
)
from xorq.common.utils.dask_normalize.dask_normalize_utils import normalize_attrs
from xorq.common.utils.func_utils import return_constant
from xorq.expr.ml.fit_lib import (
    deferred_fit_predict_only_sklearn,
    deferred_fit_predict_sklearn,
    deferred_fit_transform_only_sklearn_packed,
    deferred_fit_transform_series_sklearn,
    deferred_fit_transform_sklearn_packed,
    deferred_fit_transform_sklearn_struct,
)
from xorq.expr.ml.structer import Structer, structer_from_instance
from xorq.vendor.ibis.expr.types.core import Expr


def has_structer_registration(instance, expr, features):
    """Check if an sklearn instance has a Structer registration."""
    try:
        structer_from_instance(instance, expr, features=features)
        return True
    except ValueError:
        return False


def is_fit_transform_only(typ):
    """Check if an estimator type only supports fit_transform (not separate transform)."""
    fit_transform_only_types = {
        "TSNE",
        "MDS",
        "SpectralEmbedding",
        "Isomap",
        "LocallyLinearEmbedding",
    }
    return typ.__name__ in fit_transform_only_types


def is_fit_predict_only(typ):
    """Check if an estimator type only supports fit_predict (not separate predict)."""
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


def _is_cluster_mixin(typ):
    """Check if an estimator type inherits from ClusterMixin."""
    try:
        from sklearn.base import ClusterMixin

        return issubclass(typ, ClusterMixin)
    except (ImportError, TypeError):
        return False


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
    """A single step in a machine learning pipeline that wraps a scikit-learn estimator."""

    typ = field(validator=instance_of(type))
    name = field(validator=optional(instance_of(str)), default=None)
    params_tuple = field(validator=instance_of(tuple), default=(), converter=tuple)

    def __attrs_post_init__(self):
        from sklearn.base import BaseEstimator

        assert BaseEstimator in self.typ.mro()
        if self.name is None:
            object.__setattr__(self, "name", f"{self.typ.__name__.lower()}_{id(self)}")
        object.__setattr__(self, "params_tuple", tuple(sorted(self.params_tuple)))

    @property
    def tag_kwargs(self):
        def make_hashable(v):
            if isinstance(v, list):
                return tuple(make_hashable(x) for x in v)
            elif isinstance(v, dict):
                return tuple(sorted((k, make_hashable(val)) for k, val in v.items()))
            elif isinstance(v, tuple):
                return tuple(make_hashable(x) for x in v)
            return v

        params_hashable = make_hashable(self.params_tuple)
        return {"typ": self.typ, "name": self.name, "params_tuple": params_hashable}

    @property
    def instance(self):
        return self.typ(**dict(self.params_tuple))

    def to_sklearn(self):
        return self.instance

    def fit(self, expr, features=None, target=None, cache=None, dest_col=None):
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
        return self.__class__._from_instance_name(
            self.instance.set_params(**kwargs),
            name=self.name,
        )

    @classmethod
    def from_instance(cls, instance, name=None, deep=False, allow_unregistered=False):
        return cls._from_instance_name(
            instance, name=name, deep=deep, allow_unregistered=allow_unregistered
        )

    @classmethod
    def _from_instance_name(
        cls, instance, name=None, deep=False, allow_unregistered=False
    ):
        typ = instance.__class__

        if not allow_unregistered:
            is_predictor = hasattr(instance, "predict") and not hasattr(
                instance, "transform"
            )
            is_transductive = is_fit_transform_only(typ) or is_fit_predict_only(typ)
            is_clusterer = _is_cluster_mixin(typ)

            if not (is_predictor or is_transductive or is_clusterer):
                dispatch_func = structer_from_instance.dispatch(typ)
                if dispatch_func is structer_from_instance.dispatch(object):
                    raise ValueError(
                        f"No Structer registration for {typ.__name__}. "
                        f"Use allow_unregistered=True to wrap unregistered estimators."
                    )

        params_tuple = tuple(instance.get_params(deep=deep).items())
        return cls(typ=typ, name=name, params_tuple=params_tuple)

    @classmethod
    def _from_name_instance(cls, name, instance, deep=False, allow_unregistered=False):
        return cls._from_instance_name(
            instance, name, deep=deep, allow_unregistered=allow_unregistered
        )

    @classmethod
    def from_fit_transform(
        cls, fit, transform, return_type, klass_name=None, name=None
    ):
        typ = make_estimator_typ(
            fit=fit, transform=transform, return_type=return_type, name=klass_name
        )
        return cls(typ=typ, name=name)

    @classmethod
    def from_fit_predict(cls, fit, predict, return_type, klass_name=None, name=None):
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
    _pieces_cache = field(default=None, init=False, eq=False, hash=False, repr=False)

    def __attrs_post_init__(self):
        assert self.is_transform ^ self.is_predict
        if self.target is None and self.is_predict:
            if not is_fit_predict_only(self.step.typ) and not _is_cluster_mixin(
                self.step.typ
            ):
                raise ValueError("Can't infer target")
        if self.features is None:
            features = tuple(col for col in self.expr.columns if col != self.target)
            object.__setattr__(self, "features", features)

    @property
    def is_transform(self):
        if self.step.typ.__name__ == "Pipeline":
            instance = self.step.instance
            if hasattr(instance, "steps") and len(instance.steps) > 0:
                final_step = instance.steps[-1][1]
                if hasattr(final_step, "predict") and hasattr(final_step, "fit"):
                    return False
            return True

        has_transform = hasattr(self.step.typ, "transform")
        has_predict = hasattr(self.step.typ, "predict")
        if is_fit_transform_only(self.step.typ):
            return True
        if is_fit_predict_only(self.step.typ):
            return False
        if has_transform and has_predict:
            return False
        return has_transform

    @property
    def is_predict(self):
        if self.step.typ.__name__ == "Pipeline":
            instance = self.step.instance
            if hasattr(instance, "steps") and len(instance.steps) > 0:
                final_step = instance.steps[-1][1]
                if hasattr(final_step, "predict") and hasattr(final_step, "fit"):
                    return True
            return False

        return hasattr(self.step.typ, "predict") or is_fit_predict_only(self.step.typ)

    @property
    def is_fit_transform_only(self):
        return is_fit_transform_only(self.step.typ)

    @property
    def is_fit_predict_only(self):
        return is_fit_predict_only(self.step.typ)

    @property
    def is_transductive(self):
        return self.is_fit_transform_only or self.is_fit_predict_only

    @property
    def _pieces(self):
        if self._pieces_cache is not None:
            return self._pieces_cache

        kwargs = {
            "expr": self.expr,
            "features": self.features,
            "cls": self.step.typ,
            "params": self.step.params_tuple,
            "cache": self.cache,
        }
        deferred_transform, deferred_predict = None, None

        if self.is_transform:
            match self.step.typ:
                case type(__name__="Pipeline"):
                    f = deferred_fit_transform_sklearn_packed
                    instance = self.step.instance
                    if hasattr(instance, "steps"):
                        for step_name, step in instance.steps:
                            if step.__class__.__name__ in [
                                "SelectKBest",
                                "SelectPercentile",
                                "SelectFpr",
                                "SelectFdr",
                                "SelectFwe",
                                "RFE",
                                "RFECV",
                            ]:
                                kwargs = kwargs | {"target": self.target}
                                break
                case type(__name__="TfidfVectorizer"):
                    f = deferred_fit_transform_series_sklearn
                    (col,) = kwargs.pop("features")
                    kwargs = kwargs | {
                        "col": col,
                        "return_type": dt.Array(dt.float64),
                    }
                case type(__name__="SelectKBest"):
                    f = deferred_fit_transform_sklearn_struct
                    kwargs = kwargs | {"target": self.target}
                case typ:
                    if get_step_f_kwargs := getattr(typ, "get_step_f_kwargs", None):
                        f, kwargs = get_step_f_kwargs(kwargs)
                    elif is_fit_transform_only(typ):
                        f = deferred_fit_transform_only_sklearn_packed
                    elif has_structer_registration(
                        self.step.instance, self.expr, self.features
                    ):
                        structer = Structer.from_instance_expr(
                            self.step.instance, self.expr, features=self.features
                        )
                        if structer.is_kv_encoded:
                            f = deferred_fit_transform_sklearn_packed
                        else:
                            f = deferred_fit_transform_sklearn_struct
                    else:
                        f = deferred_fit_transform_sklearn_packed
            deferred_model, model_udf, deferred_transform = f(**kwargs)
        elif self.is_predict:
            if is_fit_predict_only(self.step.typ):
                deferred_model, model_udf, deferred_predict = (
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
                deferred_model, model_udf, deferred_predict = deferred_fit_predict_sklearn(
                    **kwargs, **predict_kwargs
                )
        else:
            raise ValueError

        result = {
            "deferred_model": deferred_model,
            "model_udf": model_udf,
            "deferred_transform": deferred_transform,
            "deferred_predict": deferred_predict,
        }
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
        return self.model

    @property
    @cexcepts(ValueError)
    def structer(self):
        return Structer.from_instance_expr(
            self.step.instance, self.expr, features=self.features
        )

    def get_others(self, expr):
        return tuple(other for other in expr.columns if other not in self.features)

    def transform_unpack(self, expr, retain_others=True, name="to_unpack"):
        struct_col = self.transform_raw(expr).name(name)
        if retain_others and (others := self.get_others(expr)):
            expr = expr.select(*others, struct_col)
        else:
            expr = struct_col.as_table()
        return expr.unpack(name)

    def transform_raw(self, expr, name=None):
        return self.deferred_transform.on_expr(expr).name(
            name or self.dest_col or "transformed"
        )

    @property
    def tag_kwargs(self):
        return {
            "tag": "FittedStep-transform" if self.is_transform else "FittedStep-predict",
            **self.step.tag_kwargs,
            "features": self.features,
        }

    def transform(self, expr, retain_others=True):
        if self.is_fit_transform_only:
            raise TypeError(
                f"{self.step.typ.__name__} is a transductive estimator that only supports "
                f"fit_transform(). It cannot transform new data."
            )
        if self.structer is None or self.structer.is_kv_encoded:
            col = self.transform_raw(expr)
            if retain_others and (others := self.get_others(expr)):
                expr = expr.select(*others, col)
            else:
                return col
        else:
            expr = self.transform_unpack(expr, retain_others=retain_others)
        return expr.tag(**self.tag_kwargs)

    @property
    def transformed(self):
        return self.transform(self.expr)

    def predict_raw(self, expr, name=None):
        if self.is_fit_predict_only:
            raise TypeError(
                f"{self.step.typ.__name__} is a transductive clusterer that only supports "
                f"fit_predict(). It cannot predict on new data."
            )
        return self.deferred_predict.on_expr(expr).name(
            name or self.dest_col or "predicted"
        )

    def predict(self, expr, retain_others=True, name=None):
        col = self.predict_raw(expr, name=name)
        if retain_others and (others := self.get_others(expr)):
            expr = expr.select(*others, col)
        else:
            expr = col.as_table()
        return expr.tag(**self.tag_kwargs)

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
        from xorq.expr.ml.fit_lib import (
            _maybe_decode_encoded_columns,
            decision_function_sklearn_packed,
            predict_proba_sklearn_packed,
        )
        from xorq.expr.ml.structer import KV_ENCODED_TYPE, get_kv_encoded_cols
        from xorq.expr.udf import make_pandas_expr_udf

        if method_name == "predict_proba":
            base_fn = predict_proba_sklearn_packed
        elif method_name == "decision_function":
            base_fn = decision_function_sklearn_packed
        else:
            raise ValueError(f"Unknown method: {method_name}")

        encoded_cols = get_kv_encoded_cols(self.expr, self.features)

        @toolz.curry
        def fn_with_decoding(model, df, base_fn, features, encoded_cols):
            df_decoded, features_decoded = _maybe_decode_encoded_columns(
                df, features, encoded_cols
            )
            return pa.array(
                base_fn(model, df_decoded[list(features_decoded)]),
                type=KV_ENCODED_TYPE.to_pyarrow(),
            )

        schema = self.expr.select(self.features).schema()
        return make_pandas_expr_udf(
            computed_kwargs_expr=self.deferred_model,
            fn=fn_with_decoding(
                base_fn=base_fn,
                features=self.features,
                encoded_cols=encoded_cols,
            ),
            schema=schema,
            return_type=KV_ENCODED_TYPE,
            name=f"_{method_name}_{self.step.name}",
        )

    def predict_proba_raw(self, expr, name=None):
        if not hasattr(self.step.typ, "predict_proba"):
            raise AttributeError(
                f"{self.step.typ.__name__} does not have predict_proba method"
            )
        udf = self._make_proba_udf("predict_proba")
        return udf.on_expr(expr).name(name or "proba")

    def predict_proba(self, expr, retain_others=True, name=None):
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
        if not hasattr(self.step.typ, "decision_function"):
            raise AttributeError(
                f"{self.step.typ.__name__} does not have decision_function method"
            )
        udf = self._make_proba_udf("decision_function")
        return udf.on_expr(expr).name(name or "decision")

    def decision_function(self, expr, retain_others=True, name=None):
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


def _validate_pipeline_step(instance, attribute, value):
    for step in value:
        if not isinstance(step, Step):
            raise TypeError(f"Pipeline steps must be Step, got {type(step).__name__}")


@frozen
class Pipeline:
    """A machine learning pipeline that chains multiple processing steps together."""

    steps = field(validator=_validate_pipeline_step, converter=tuple)

    def __attrs_post_init__(self):
        assert self.steps

    @property
    def instance(self):
        import sklearn

        return sklearn.pipeline.Pipeline(
            tuple((step.name, step.instance) for step in self.steps)
        )

    def _is_predict_step(self, step):
        return hasattr(step.instance, "predict")

    @property
    def transform_steps(self):
        (*steps, last_step) = self.steps
        if self._is_predict_step(last_step):
            return tuple(steps)
        else:
            return self.steps

    @property
    def predict_step(self):
        (*_, last_step) = self.steps
        if self._is_predict_step(last_step):
            return last_step
        return None

    def fit(self, expr, features=None, target=None, cache=None):
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
            transformed = transformed.pipe(do_into_backend)
            if (
                fitted_step.structer is not None
                and not fitted_step.structer.is_kv_encoded
            ):
                features = tuple(fitted_step.structer.dtype)
            else:
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
        return FittedPipeline(fitted_steps, expr)

    @classmethod
    def from_instance(cls, instance, deep=False, allow_unregistered=False):
        from sklearn.pipeline import Pipeline as SklearnPipeline

        def is_known_xorq_step(step):
            if hasattr(step, "predict") and not hasattr(step, "transform"):
                return True
            try:
                structer_from_instance.dispatch(type(step))
                return structer_from_instance.dispatch(
                    type(step)
                ) is not structer_from_instance.dispatch(object)
            except (ValueError, TypeError, KeyError):
                return False

        grouped_steps = []
        i = 0
        sklearn_steps = instance.steps

        while i < len(sklearn_steps):
            name, step = sklearn_steps[i]

            if is_known_xorq_step(step):
                grouped_steps.append(
                    Step._from_instance_name(
                        step, name, deep=deep, allow_unregistered=allow_unregistered
                    )
                )
                i += 1
            else:
                if not allow_unregistered:
                    raise ValueError(
                        f"No Structer registration for {type(step).__name__} (step '{name}'). "
                        f"Use allow_unregistered=True to wrap unregistered estimators."
                    )

                group = [(name, step)]
                group_names = [name]
                i += 1

                while i < len(sklearn_steps):
                    next_name, next_step = sklearn_steps[i]
                    if is_known_xorq_step(next_step):
                        break
                    group.append((next_name, next_step))
                    group_names.append(next_name)
                    i += 1

                if len(group) == 1:
                    grouped_steps.append(
                        Step._from_instance_name(
                            step, name, deep=deep, allow_unregistered=allow_unregistered
                        )
                    )
                else:
                    wrapped_pipe = SklearnPipeline(group)
                    wrapped_name = "_".join(group_names)
                    grouped_steps.append(
                        Step.from_instance(
                            wrapped_pipe,
                            name=wrapped_name,
                            deep=deep,
                            allow_unregistered=allow_unregistered,
                        )
                    )

        return cls(tuple(grouped_steps))

    def set_params(self, **kwargs):
        return self.__class__.from_instance(self.instance.set_params(**kwargs))

    def to_sklearn(self):
        return self.instance


def _validate_fitted_pipeline_step(instance, attribute, value):
    for step in value:
        if not isinstance(step, FittedStep):
            raise TypeError(
                f"FittedPipeline steps must be FittedStep, got {type(step).__name__}"
            )


@frozen
class FittedPipeline:
    fitted_steps = field(validator=_validate_fitted_pipeline_step, converter=tuple)
    expr = field(validator=instance_of(Expr))

    def __attrs_post_init__(self):
        assert self.fitted_steps

    def _is_fitted_predict_step(self, fitted_step):
        return hasattr(fitted_step.step.instance, "predict")

    @property
    def is_predict(self):
        (*_, last_step) = self.fitted_steps
        return self._is_fitted_predict_step(last_step)

    @property
    def transform_steps(self):
        return self.fitted_steps[:-1] if self.is_predict else self.fitted_steps

    @property
    def predict_step(self):
        return self.fitted_steps[-1] if self.is_predict else None

    @property
    def tag_kwargs(self):
        tags = []
        for fitted_step in self.transform_steps:
            tags.append(tuple(fitted_step.tag_kwargs.items()))
        return {"transforms_tags": tuple(tags)}

    def transform(self, expr, tag=True):
        transformed = expr
        for fitted_step in self.transform_steps:
            transformed = fitted_step.transform(transformed).pipe(do_into_backend)
        if tag:
            transformed = transformed.tag("FittedPipeline-transform", **self.tag_kwargs)
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
        clf = self.predict_step.model
        df = self.transform(expr).execute()
        X = df[list(self.predict_step.features)]
        y = df[self.predict_step.target]
        return clf.score(X, y, **kwargs)

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
        return self.score_expr(expr, **kwargs)

    def to_sklearn(self):
        import sklearn.pipeline

        steps = []
        for fitted_step in self.fitted_steps:
            steps.append((fitted_step.step.name, fitted_step.to_sklearn()))

        return sklearn.pipeline.Pipeline(steps)


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
    from sklearn.base import ClassifierMixin, ClusterMixin, RegressorMixin
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline as SklearnPipeline

    def get_pipeline_return_type(instance, step, expr, features, target):
        final_step = instance.steps[-1][1]
        if isinstance(final_step, ClassifierMixin):
            return get_target_type(instance, step, expr, features, target)
        elif isinstance(final_step, RegressorMixin):
            return dt.float64
        elif isinstance(final_step, ClusterMixin):
            return dt.int64
        else:
            return get_target_type(instance, step, expr, features, target)

    registry.register(LinearRegression, return_constant(dt.float64))
    registry.register(LogisticRegression, get_target_type)
    registry.register(KNeighborsClassifier, get_target_type)
    registry.register(ClassifierMixin, get_target_type)
    registry.register(RegressorMixin, return_constant(dt.float64))
    registry.register(ClusterMixin, return_constant(dt.int64))
    registry.register(SklearnPipeline, get_pipeline_return_type)


get_predict_return_type.register = registry.register
