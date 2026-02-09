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
    response_method: str = field(validator=instance_of(str), default="predict")

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
                        response_method="predict",
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


@frozen
class MetricComputation:
    target: str = field(validator=instance_of(str))
    pred_col: str = field(validator=instance_of(str))
    metric_fn: Callable = field(validator=instance_of(Callable))
    sign: int = field(validator=instance_of(int), default=1)
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

    def __call__(self, df):
        y_true = df[self.target]
        y_pred = self._prepare_predictions(df[self.pred_col])
        result = self.metric_fn(y_true, y_pred, **self.metric_kwargs)
        return self.sign * result

    def on_expr(self, expr):
        schema = expr.select([self.target, self.pred_col]).schema()
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
        if y_pred.ndim == 2:
            if y_pred.shape[1] == 2:
                return y_pred[:, 1]
            elif y_pred.shape[1] == 1:
                return y_pred[:, 0]
        return y_pred


def deferred_sklearn_metric(
    expr,
    target,
    pred_col,
    scorer,
    metric_kwargs=(),
    return_type=dt.float64,
    name=None,
):
    """Compute sklearn metrics on expressions.

    This function expects that predictions have already been added to the
    expression (via fitted_pipeline.predict(), predict_proba(), or
    decision_function()).

    Sign is extracted automatically from the resolved scorer (e.g.
    "neg_mean_squared_error" resolves to sign=-1).

    Parameters
    ----------
    expr : ibis.Expr
        Expression containing both target and prediction columns
    target : str
        Name of the target column
    pred_col : str
        Name of the prediction column (e.g., "predict", "predict_proba")
    scorer : str | _BaseScorer | Callable | Scorer
        Scorer specification. Can be a scorer name string, an sklearn _BaseScorer,
        a known sklearn metric function, or a Scorer instance.
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
    >>> from sklearn.metrics import accuracy_score, roc_auc_score
    >>>
    >>> # For regular predictions
    >>> expr_with_preds = fitted_pipeline.predict(test_data)
    >>> acc = deferred_sklearn_metric(
    ...     expr_with_preds,
    ...     target="target",
    ...     pred_col="predict",
    ...     scorer=accuracy_score
    ... )
    >>>
    >>> # For probabilities
    >>> expr_with_proba = fitted_pipeline.predict_proba(test_data)
    >>> auc = deferred_sklearn_metric(
    ...     expr_with_proba,
    ...     target="target",
    ...     pred_col="predict_proba",
    ...     scorer=roc_auc_score
    ... )
    """
    scorer = Scorer.from_spec(scorer) if not isinstance(scorer, Scorer) else scorer
    merged_kwargs = {**dict(scorer.kwargs), **dict(metric_kwargs)}
    metric = MetricComputation(
        target=target,
        pred_col=pred_col,
        metric_fn=scorer.metric_fn,
        sign=scorer.sign,
        metric_kwargs_tuple=merged_kwargs,
        return_type=return_type,
        name=name,
    )
    return metric.on_expr(expr)
