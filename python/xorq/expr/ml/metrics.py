"""Deferred scikit-learn metrics evaluation for xorq."""

import functools
from typing import Callable

import numpy as np
from attr import (
    field,
    frozen,
)
from attr.validators import (
    instance_of,
    optional,
)
from toolz import compose

import xorq.expr.datatypes as dt
import xorq.expr.udf as udf
from xorq.common.utils.name_utils import make_name
from xorq.expr.ml.enums import ResponseMethod


@functools.lru_cache(maxsize=1)
def _build_known_scorer_funcs():
    """Set of known scorer functions.

    Used to check whether a bare callable is a known sklearn scorer function
    (e.g. accuracy_score, mean_squared_error) vs an unknown callable
    (e.g. confusion_matrix, custom fn).
    """
    from sklearn.metrics import get_scorer, get_scorer_names

    return frozenset(get_scorer(name)._score_func for name in get_scorer_names())


def _default_scorer_for_model(model):
    """Get the default scorer based on model type.

    Parameters
    ----------
    model : estimator
        A fitted sklearn estimator.

    Returns
    -------
    scorer : _BaseScorer
        An sklearn scorer object appropriate for the model type.

    Raises
    ------
    ValueError
        If model type is not recognized.
    """
    from sklearn.base import ClassifierMixin, ClusterMixin, RegressorMixin
    from sklearn.metrics import (
        accuracy_score,
        adjusted_rand_score,
        make_scorer,
        r2_score,
    )

    match model:
        case ClusterMixin():
            return make_scorer(adjusted_rand_score)
        case ClassifierMixin():
            return make_scorer(accuracy_score)
        case RegressorMixin():
            return make_scorer(r2_score)
        case _:
            raise ValueError(
                f"Cannot determine default scorer for model type {type(model).__name__}. "
                "Please specify a scorer explicitly."
            )


@frozen
class Scorer:
    """Normalized representation of a scorer.

    Encapsulates all detection logic: given any input (str, _BaseScorer,
    bare callable, or None), resolves to a consistent set of properties.
    """

    metric_fn: Callable = field(validator=instance_of(Callable))
    sign: int = field(validator=instance_of(int))
    kwargs: tuple = field(
        factory=tuple,
        converter=lambda d: tuple(sorted(d.items())) if isinstance(d, dict) else d,
    )
    response_method: str = field(
        validator=instance_of(str), default=ResponseMethod.PREDICT
    )

    @classmethod
    def from_spec(cls, scorer, model=None):
        """Normalize str | callable | _BaseScorer | None -> Scorer.

        Parameters
        ----------
        scorer : str, callable, _BaseScorer, Scorer, or None
            The scorer specification to normalize.
        model : estimator, optional
            A fitted sklearn estimator, used to determine the default scorer
            when scorer is None.

        Returns
        -------
        Scorer
            A normalized Scorer instance.
        """
        from sklearn.metrics import get_scorer
        from sklearn.metrics._scorer import _BaseScorer

        match scorer:
            case None:
                scorer_obj = _default_scorer_for_model(model)
                return cls._from_scorer_obj(scorer_obj)

            case str():
                scorer_obj = get_scorer(scorer)
                return cls._from_scorer_obj(scorer_obj)

            case _BaseScorer():
                return cls._from_scorer_obj(scorer)

            case Scorer():
                return scorer  # already resolved

            case object(__call__=_):
                known = _build_known_scorer_funcs()
                if scorer in known:
                    # Known scorer function — bare callable = raw metric value
                    # sign=1, kwargs={} (function's own defaults apply;
                    # caller provides overrides via metric_kwargs)
                    return cls(
                        metric_fn=scorer,
                        sign=1,
                        kwargs={},
                        response_method=ResponseMethod.PREDICT,
                    )
                else:
                    raise ValueError(
                        f"Unknown callable {scorer.__name__!r} is not a known sklearn scorer function. "
                        f"Use make_scorer() to wrap custom metrics, or pass a scorer name string."
                    )

            case _:
                raise ValueError(
                    f"scorer must be a string, callable, _BaseScorer, Scorer, or None, "
                    f"got {type(scorer)}"
                )

    @classmethod
    def _from_scorer_obj(cls, scorer_obj):
        """Create a Scorer from an sklearn _BaseScorer object."""
        return cls(
            metric_fn=scorer_obj._score_func,
            sign=scorer_obj._sign,
            kwargs=scorer_obj._kwargs,
            response_method=cls._resolve_response_method(scorer_obj),
        )

    @staticmethod
    def _resolve_response_method(scorer_obj):
        """Extract the response method string from a scorer object."""
        raw = scorer_obj._response_method
        match raw:
            case (method, *_):
                return method
            case str(method):
                return method
            case _:
                raise ValueError(f"Unexpected _response_method: {raw}")


@functools.lru_cache(maxsize=1)
def _build_known_non_scorer_metric_fns():
    """Set of known non-scorer metric functions.

    Computed as all public functions in ``sklearn.metrics`` minus the
    scorer functions from ``_build_known_scorer_funcs()``, excluding
    pairwise/scorer utilities by module and unsupported metrics by name.
    """
    import inspect

    import sklearn.metrics

    _EXCLUDED_NAMES = frozenset(
        {
            "auc",  # helper for curve coords, not (y_true, y_pred)
            "classification_report",  # returns string
            "consensus_score",  # biclustering, different signature
            "multilabel_confusion_matrix",  # returns 3D array (n_classes, 2, 2)
            "precision_recall_fscore_support",  # returns variable-length tuple
        }
    )

    all_metric_fns = frozenset(
        obj
        for name in dir(sklearn.metrics)
        if not name.startswith("_")
        and name not in _EXCLUDED_NAMES
        and callable(obj := getattr(sklearn.metrics, name))
        and inspect.isfunction(obj)
        and not obj.__module__.startswith(
            ("sklearn.metrics.pairwise", "sklearn.metrics._scorer")
        )
    )
    return all_metric_fns - _build_known_scorer_funcs()


@functools.lru_cache(maxsize=1)
def _build_metric_return_types():
    """Registry of non-scalar return types for known sklearn metrics.

    Maps metric functions to their ``dt.DataType`` so that
    ``deferred_sklearn_metric`` can auto-resolve ``return_type``
    when the caller doesn't provide one explicitly.
    """
    from sklearn.metrics import (
        class_likelihood_ratios,
        confusion_matrix,
        det_curve,
        homogeneity_completeness_v_measure,
        pair_confusion_matrix,
        precision_recall_curve,
        roc_curve,
        silhouette_samples,
    )

    return {
        # Tuple of scalars -> Struct
        class_likelihood_ratios: dt.Struct(
            dict(
                positive_likelihood_ratio=dt.float64,
                negative_likelihood_ratio=dt.float64,
            )
        ),
        homogeneity_completeness_v_measure: dt.Struct(
            dict(
                homogeneity=dt.float64,
                completeness=dt.float64,
                v_measure=dt.float64,
            )
        ),
        # Confusion matrices -> Array(Array(int64))
        confusion_matrix: dt.Array(dt.Array(dt.int64)),
        pair_confusion_matrix: dt.Array(dt.Array(dt.int64)),
        # Curves -> Struct of arrays
        roc_curve: dt.Struct(
            dict(
                fpr=dt.Array(dt.float64),
                tpr=dt.Array(dt.float64),
                thresholds=dt.Array(dt.float64),
            )
        ),
        precision_recall_curve: dt.Struct(
            dict(
                precision=dt.Array(dt.float64),
                recall=dt.Array(dt.float64),
                thresholds=dt.Array(dt.float64),
            )
        ),
        det_curve: dt.Struct(
            dict(
                fpr=dt.Array(dt.float64),
                fnr=dt.Array(dt.float64),
                thresholds=dt.Array(dt.float64),
            )
        ),
        # Per-sample metrics -> Array(float64)
        silhouette_samples: dt.Array(dt.float64),
    }


def _validate_str_or_tuple_of_str(instance, attribute, value):
    """Validate value is str or tuple of str."""
    match value:
        case str():
            pass
        case tuple() if all(isinstance(v, str) for v in value):
            pass
        case _:
            raise TypeError(
                f"{attribute.name} must be a str or tuple of str, got {type(value)}"
            )


@frozen
class MetricComputation:
    target = field(validator=_validate_str_or_tuple_of_str)
    pred = field(validator=_validate_str_or_tuple_of_str)
    metric_fn: Callable = field(validator=instance_of(Callable))
    sign: int = field(validator=optional(instance_of(int)), default=None)
    metric_kwargs_tuple: tuple = field(
        default=(),
        converter=compose(tuple, sorted, dict.items, dict),
    )
    return_type = field(validator=instance_of(dt.DataType), default=dt.float64)
    name = field(validator=optional(instance_of(str)), default=None)

    def __attrs_post_init__(self):
        if self.name is None:
            object.__setattr__(
                self,
                "name",
                make_name(
                    prefix=f"metric_{self.metric_fn.__name__}",
                    to_tokenize=self.metric_kwargs_tuple,
                ),
            )

    @property
    def metric_kwargs(self):
        return dict(self.metric_kwargs_tuple)

    @property
    def __name__(self):
        """Return the name of the metric function for UDF registration."""
        return self.metric_fn.__name__

    @property
    def __module__(self):
        """Return the module of the metric function for UDF registration."""
        return self.metric_fn.__module__

    @staticmethod
    def _normalize_columns(value, name):
        """Normalize str or tuple[str] to tuple[str].

        Parameters
        ----------
        value : str or tuple of str
            Column name(s) to normalize.
        name : str
            Parameter name for error messages.

        Returns
        -------
        tuple of str
            Normalized column names as a tuple.
        """
        match value:
            case str():
                return (value,)
            case tuple():
                return value
            case _:
                raise TypeError(
                    f"{name} must be a str or tuple of str, got {type(value)}"
                )

    @property
    def _target_columns(self):
        return self._normalize_columns(self.target, "target")

    @property
    def _pred_columns(self):
        return self._normalize_columns(self.pred, "pred")

    @property
    def _prepare_target(self):
        """Extract the target (first metric argument) from a DataFrame.

        str  -> single column (Series)
        tuple -> multiple columns as 2D ndarray
        """
        match self.target:
            case str():
                return lambda df: df[self.target]
            case tuple():
                return lambda df: df[list(self.target)].values
            case _:
                raise TypeError(
                    f"target must be a str or tuple of str, got {type(self.target)}"
                )

    @property
    def _prepare_predictions(self):
        """Extract predictions (second metric argument) from a DataFrame.

        Dispatch is on (pred type, target type):

        (str, str)   -> single column with positive-class extraction
                        for binary (n, 2) probabilities
        (str, tuple) -> single column, raw (no extraction); handles both
                        scalar columns (clustering) and array columns
                        (multilabel)
        (tuple, any) -> multiple columns as 2D ndarray
        """
        match (self.pred, self.target):
            case (str(), str()):
                return lambda df: self._extract_positive_class_proba(
                    self._coerce_ndarray(df[self.pred])
                )
            case (str(), tuple()):
                return lambda df: self._coerce_ndarray(df[self.pred])
            case (tuple(), _):
                return lambda df: df[list(self.pred)].values
            case _:
                raise TypeError(
                    f"pred must be a str or tuple of str, got {type(self.pred)}"
                )

    @property
    def _apply_sign(self):
        """Dispatch for sign application to a metric result."""
        match self.sign:
            case None:
                return lambda result: result
            case int():
                return lambda result: self.sign * result
            case _:
                raise TypeError(f"sign must be None or int, got {type(self.sign)}")

    @property
    def _convert_result_for_udaf(self):
        """Convert a raw sklearn metric result for the UDAF infrastructure.

        dt.Struct  -> dict keyed by field names (tuple of scalars or arrays)
        dt.Array   -> list (ndarray.tolist() for matrices / per-sample)
        dt.Float64 -> passthrough
        """
        match self.return_type:
            case dt.Struct() as s:
                names = s.names
                return lambda raw: dict(zip(names, raw))
            case dt.Array():
                return lambda raw: raw.tolist()
            case dt.Float64():
                return lambda raw: raw
            case _:
                raise TypeError(f"Unsupported return_type {self.return_type}")

    def __call__(self, df):
        target = self._prepare_target(df)
        predictions = self._prepare_predictions(df)
        result = self.metric_fn(target, predictions, **self.metric_kwargs)
        return self._apply_sign(self._convert_result_for_udaf(result))

    def on_expr(self, expr):
        schema = expr.select((*self._target_columns, *self._pred_columns)).schema()
        metric_udaf = udf.agg.pandas_df(
            fn=self,
            schema=schema,
            return_type=self.return_type,
            name=self.name,
        )
        return metric_udaf.on_expr(expr)

    @staticmethod
    def _coerce_ndarray(predictions):
        """Coerce a column value into a numpy ndarray.

        Handles scalar Series, array-valued Series (np.vstack), and
        raw ndarrays.
        """
        import pandas as pd

        match predictions:
            case pd.Series() as series if len(series) > 0 and isinstance(
                series.iloc[0], (np.ndarray, list, tuple)
            ):
                return np.vstack(series.values)
            case pd.Series() as series:
                return series.values
            case np.ndarray():
                return predictions
            case _:
                return predictions

    @staticmethod
    def _extract_positive_class_proba(y_pred):
        """Extract positive class probabilities for binary classification."""
        match (y_pred.ndim, y_pred.shape):
            case (2, (_, 2)):
                return y_pred[:, 1]
            case (2, (_, 1)):
                return y_pred[:, 0]
            case _:
                return y_pred


def deferred_sklearn_metric(
    expr,
    target,
    pred,
    metric,
    *,
    metric_kwargs=(),
    return_type=dt.float64,
    name=None,
):
    """Compute sklearn metrics on expressions.

    This function expects that predictions have already been added to the
    expression (via fitted_pipeline.predict(), predict_proba(), or
    decision_function()).

    Parameters
    ----------
    expr : ibis.Expr
        Expression containing both target and prediction columns.
    target : str | tuple[str, ...]
        Name of the target column, or a tuple of column names for metrics
        that expect a feature matrix (e.g. clustering metrics like
        calinski_harabasz_score).
    pred : str | tuple[str, ...]
        Name of the prediction column (e.g., "predict", "predict_proba"),
        or a tuple of column names for multi-column predictions.
    metric : str | _BaseScorer | Callable | Scorer
        The metric to compute.  Accepted forms:

        - **str** -- scorer name from ``sklearn.metrics.get_scorer_names()``
          (e.g. ``"accuracy"``, ``"neg_mean_squared_error"``).
        - **_BaseScorer** -- an sklearn scorer object (e.g. from
          ``make_scorer()``).
        - **Scorer** -- an already-resolved ``Scorer`` instance.
        - **callable** -- a known sklearn metric function.  Must be either
          a known scorer function (e.g. ``accuracy_score``) or a known
          non-scorer metric (e.g. ``cohen_kappa_score``,
          ``confusion_matrix``).  Unknown callables are rejected.
    metric_kwargs : dict, optional
        Additional kwargs to pass to the metric function.
    return_type : dt.DataType, optional
        Return type for the metric.  Auto-detected for non-scalar metrics
        in the registry; defaults to ``dt.float64`` for scalar metrics.
    name : str, optional
        Custom name for the UDF.

    Returns
    -------
    deferred_metric : ibis.Expr
        Deferred expression that computes the metric when executed.

    Examples
    --------
    >>> from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
    >>>
    >>> # Scorer name string
    >>> acc = deferred_sklearn_metric(
    ...     expr_with_preds, target="target", pred="predict",
    ...     metric="accuracy",
    ... )
    >>>
    >>> # Known scorer callable
    >>> acc = deferred_sklearn_metric(
    ...     expr_with_preds, target="target", pred="predict",
    ...     metric=accuracy_score,
    ... )
    >>>
    >>> # Non-scorer metric
    >>> kappa = deferred_sklearn_metric(
    ...     expr_with_preds, target="target", pred="predict",
    ...     metric=cohen_kappa_score,
    ... )
    >>>
    >>> # Non-scalar metric (return_type auto-detected)
    >>> cm = deferred_sklearn_metric(
    ...     expr_with_preds, target="target", pred="predict",
    ...     metric=confusion_matrix,
    ... )
    """
    from sklearn.metrics._scorer import _BaseScorer

    # Helper to build MetricComputation from a Scorer
    def _from_scorer(scorer, metric_kwargs):
        merged_kwargs = {**dict(scorer.kwargs), **dict(metric_kwargs)}
        return MetricComputation(
            target=target,
            pred=pred,
            metric_fn=scorer.metric_fn,
            sign=scorer.sign,
            metric_kwargs_tuple=merged_kwargs,
            return_type=return_type,
            name=name,
        ).on_expr(expr)

    # Helper to build MetricComputation from a non-scorer metric function
    def _from_non_scorer_metric_fn(metric_fn, metric_kwargs):
        resolved_return_type = _build_metric_return_types().get(metric_fn, return_type)
        return MetricComputation(
            target=target,
            pred=pred,
            metric_fn=metric_fn,
            sign=None,
            metric_kwargs_tuple=dict(metric_kwargs),
            return_type=resolved_return_type,
            name=name,
        ).on_expr(expr)

    match metric:
        case str() | _BaseScorer():
            return _from_scorer(Scorer.from_spec(metric), metric_kwargs)

        case Scorer():
            return _from_scorer(metric, metric_kwargs)

        case object(__call__=_):
            known_scorers = _build_known_scorer_funcs()
            known_non_scorer_metrics = _build_known_non_scorer_metric_fns()

            match (metric in known_scorers, metric in known_non_scorer_metrics):
                case (True, False):
                    return _from_scorer(Scorer.from_spec(metric), metric_kwargs)
                case (False, True):
                    return _from_non_scorer_metric_fn(metric, metric_kwargs)
                case _:
                    raise ValueError(
                        f"Unknown callable {metric.__name__!r}. "
                        f"Must be a known sklearn scorer function or a known "
                        f"non-scorer metric. Use a scorer name string or "
                        f"make_scorer() for custom metrics."
                    )

        case _:
            raise TypeError(
                f"metric must be a str, _BaseScorer, Scorer, or known callable, "
                f"got {type(metric)}"
            )


# sklearn curve functions return plain tuples — no field-name metadata — so
# the Struct field names in _build_metric_return_types are ours.  The Struct
# field order must match sklearn's positional return order because
# _convert_result_for_udaf zips names with the tuple: dict(zip(names, raw)).
#
# auc(x, y) requires x to be monotonic.  precision_recall_curve returns
# (precision, recall, thresholds) but recall is the monotonic axis, so auc
# needs (recall, precision) — the reverse of the Struct order.  We can't
# reorder the Struct without breaking the zip, so this map exists to resolve
# the correct (x, y) pair for each curve regardless of Struct field order.
_CURVE_FIELD_MAP = {
    frozenset({"fpr", "tpr", "thresholds"}): ("fpr", "tpr"),
    frozenset({"precision", "recall", "thresholds"}): ("recall", "precision"),
    frozenset({"fpr", "fnr", "thresholds"}): ("fpr", "fnr"),
}


def deferred_auc_from_curve(curve_expr):
    """Compute the area under a deferred curve metric.

    Parameters
    ----------
    curve_expr : ibis.Expr
        A deferred curve expression with a Struct return type,
        from ``deferred_sklearn_metric`` with ``roc_curve``,
        ``precision_recall_curve``, or ``det_curve``.

    Returns
    -------
    auc_expr : ibis.Expr
        A deferred float64 scalar expression.

    Examples
    --------
    >>> from sklearn.metrics import roc_curve
    >>> deferred_roc = deferred_sklearn_metric(
    ...     expr=preds, target="target", pred="scores",
    ...     metric=roc_curve,
    ... )
    >>> deferred_roc_auc = deferred_auc_from_curve(deferred_roc)
    """
    import pyarrow as pa
    from sklearn.metrics import auc

    # Validate type and extract field mapping
    curve_type = curve_expr.type()
    match curve_type:
        case dt.Struct():
            field_names = frozenset(curve_type.names)
            if (xy_fields := _CURVE_FIELD_MAP.get(field_names)) is None:
                raise ValueError(
                    f"Unrecognized curve fields {set(field_names)}. "
                    f"Expected fields from roc_curve, precision_recall_curve, "
                    f"or det_curve."
                )
            x_field, y_field = xy_fields
        case _:
            raise TypeError(
                f"Expected a Struct expression from a curve metric, got {curve_type}"
            )

    # Create UDF to compute AUC
    def _auc_fn(struct: curve_type) -> dt.float64:
        x = struct.field(x_field).values.to_pylist()
        y = struct.field(y_field).values.to_pylist()
        return pa.array([auc(x, y)], type=pa.float64())

    _auc_fn.__name__ = f"_auc_{x_field}_{y_field}"
    return udf.scalar.pyarrow(_auc_fn)(curve_expr)
