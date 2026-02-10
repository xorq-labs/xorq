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


def _validate_target(instance, attribute, value):
    """Validate target is str or tuple of str."""
    match value:
        case str():
            pass
        case tuple() if all(isinstance(v, str) for v in value):
            pass
        case _:
            raise TypeError(f"target must be a str or tuple of str, got {type(value)}")


@frozen
class MetricComputation:
    target = field(validator=_validate_target)
    pred_col: str = field(validator=instance_of(str))
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

    @property
    def _target_columns(self):
        match self.target:
            case str():
                return (self.target,)
            case tuple():
                return self.target
            case _:
                raise TypeError(
                    f"target must be a str or tuple of str, got {type(self.target)}"
                )

    @property
    def _first_arg(self):
        """Dispatch for extracting the first metric argument from a DataFrame."""
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
    def _apply_sign(self):
        """Dispatch for sign application to a metric result."""
        match self.sign:
            case None:
                return lambda result: result
            case int():
                return lambda result: self.sign * result
            case _:
                raise TypeError(f"sign must be None or int, got {type(self.sign)}")

    def __call__(self, df):
        first_arg = self._first_arg(df)
        y_pred = self._prepare_predictions(df[self.pred_col])
        result = self.metric_fn(first_arg, y_pred, **self.metric_kwargs)
        return self._apply_sign(result)

    def on_expr(self, expr):
        schema = expr.select((*self._target_columns, self.pred_col)).schema()
        metric_udaf = udf.agg.pandas_df(
            fn=self,
            schema=schema,
            return_type=self.return_type,
            name=self.name,
        )
        return metric_udaf.on_expr(expr)

    @classmethod
    def _prepare_predictions(cls, predictions):
        """Prepare predictions for metric computation using pattern matching."""
        import pandas as pd

        match predictions:
            # Case 1: pandas Series with array-like values (e.g., probabilities)
            case pd.Series() as series if len(series) > 0 and isinstance(
                series.iloc[0], (np.ndarray, list, tuple)
            ):
                return cls._extract_positive_class_proba(np.vstack(series.values))

            # Case 2: pandas Series with scalar values
            case pd.Series() as series:
                return cls._extract_positive_class_proba(series.values)

            # Case 3: already a numpy array
            case np.ndarray() as arr:
                return cls._extract_positive_class_proba(arr)

            # Case 4: anything else, return as-is
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
    pred_col,
    scorer=None,
    metric_fn=None,
    metric_kwargs=(),
    return_type=dt.float64,
    name=None,
):
    """Compute sklearn metrics on expressions.

    This function expects that predictions have already been added to the
    expression (via fitted_pipeline.predict(), predict_proba(), or
    decision_function()).

    Exactly one of ``scorer`` or ``metric_fn`` must be provided.

    Parameters
    ----------
    expr : ibis.Expr
        Expression containing both target and prediction columns
    target : str | tuple[str, ...]
        Name of the target column, or a tuple of column names for metrics
        that expect a feature matrix (e.g. clustering metrics like
        calinski_harabasz_score).
    pred_col : str
        Name of the prediction column (e.g., "predict", "predict_proba")
    scorer : str | _BaseScorer | Callable | Scorer, optional
        Scorer specification. Can be a scorer name string, an sklearn
        _BaseScorer, a known sklearn metric function, or a Scorer instance.
        Mutually exclusive with ``metric_fn``.
    metric_fn : Callable, optional
        A raw metric callable (e.g. cohen_kappa_score, confusion_matrix).
        If the callable is a known scorer function, sign is auto-detected
        from sklearn's registry. Otherwise sign is not applied.
        Mutually exclusive with ``scorer``.
    metric_kwargs : Optional[dict]
        Additional kwargs to pass to metric function
    return_type : dt.DataType, optional
        Return type for the metric (default: dt.float64)
    name: Optional[str]
        Custom name for the UDF

    Returns
    -------
    deferred_metric : ibis.Expr
        Deferred expression that computes the metric when executed

    Examples
    --------
    >>> from sklearn.metrics import accuracy_score, roc_auc_score, cohen_kappa_score
    >>>
    >>> # Using scorer (existing path)
    >>> expr_with_preds = fitted_pipeline.predict(test_data)
    >>> acc = deferred_sklearn_metric(
    ...     expr_with_preds,
    ...     target="target",
    ...     pred_col="predict",
    ...     scorer=accuracy_score
    ... )
    >>>
    >>> # Using metric_fn (non-scorer metrics)
    >>> kappa = deferred_sklearn_metric(
    ...     expr_with_preds,
    ...     target="target",
    ...     pred_col="predict",
    ...     metric_fn=cohen_kappa_score
    ... )
    """
    match (scorer, metric_fn):
        case (None, None):
            raise ValueError("Exactly one of 'scorer' or 'metric_fn' must be provided.")
        case (_, None):
            resolved = (
                Scorer.from_spec(scorer) if not isinstance(scorer, Scorer) else scorer
            )
            merged_kwargs = {**dict(resolved.kwargs), **dict(metric_kwargs)}
            return MetricComputation(
                target=target,
                pred_col=pred_col,
                metric_fn=resolved.metric_fn,
                sign=resolved.sign,
                metric_kwargs_tuple=merged_kwargs,
                return_type=return_type,
                name=name,
            ).on_expr(expr)
        case (None, _):
            known = _build_known_scorer_funcs()
            sign = Scorer.from_spec(metric_fn).sign if metric_fn in known else None
            merged_kwargs = dict(metric_kwargs)
            return MetricComputation(
                target=target,
                pred_col=pred_col,
                metric_fn=metric_fn,
                sign=sign,
                metric_kwargs_tuple=merged_kwargs,
                return_type=return_type,
                name=name,
            ).on_expr(expr)
        case _:
            raise ValueError(
                "Cannot specify both 'scorer' and 'metric_fn'. "
                "Use 'scorer' for sklearn scorers, 'metric_fn' for raw metric callables."
            )
