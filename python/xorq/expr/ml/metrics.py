"""Deferred scikit-learn metrics evaluation for xorq."""

from typing import Any, Callable

import attrs
import numpy as np

import xorq.expr.datatypes as dt
import xorq.expr.udf as udf
from xorq.common.utils.name_utils import make_name


def deferred_sklearn_metric(
    expr,
    target,
    pred_col,
    metric_fn,
    metric_kwargs=None,
    return_type=None,
    name_infix=None,
):
    """Compute sklearn metrics on expressions

    This function expects that predictions have already been added to the
    expression (via fitted_pipeline.predict(), predict_proba(), or
    decision_function()).

    Parameters
    ----------
    expr : ibis.Expr
        Expression containing both target and prediction columns
    target : str
        Name of the target column
    pred_col : str
        Name of the prediction column (e.g., "predicted", "predicted_proba")
    metric_fn : callable
        Scikit-learn metric function (e.g., accuracy_score, roc_auc_score)
    metric_kwargs : Optional[dict]
        Additional kwargs to pass to metric function
    return_type : dt.DataType, optional
        Return type for the metric (default: dt.float64)
    name_infix : Optional[str]
        Custom name infix for the UDF

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
    ...     pred_col="predicted",
    ...     metric_fn=accuracy_score
    ... )
    >>>
    >>> # For probabilities
    >>> expr_with_proba = fitted_pipeline.predict_proba(test_data)
    >>> auc = deferred_sklearn_metric(
    ...     expr_with_proba,
    ...     target="target",
    ...     pred_col="predicted_proba",
    ...     metric_fn=roc_auc_score
    ... )
    """
    metric_kwargs = metric_kwargs or {}
    return_type = return_type or dt.float64
    metric_udaf = _create_metric_aggregation(
        target,
        pred_col,
        metric_fn,
        metric_kwargs,
        return_type,
        name_infix
        or make_name(
            prefix=f"metric_{metric_fn.__name__}",
            to_tokenize=(metric_fn.__name__, str(metric_kwargs)),
        ),
        expr,
    )

    return metric_udaf.on_expr(expr)


def _create_metric_aggregation(
    target,
    predictions_col,
    metric_fn,
    metric_kwargs,
    return_type,
    name_infix,
    expr,
):
    """Create an aggregation UDF for computing the metric."""
    compute_fn = _MetricComputation(
        target=target,
        pred_col=predictions_col,
        metric_fn=metric_fn,
        metric_kwargs=metric_kwargs,
    )
    schema = expr.select([target, predictions_col]).schema()

    return udf.agg.pandas_df(
        fn=compute_fn,
        schema=schema,
        return_type=return_type,
        name=name_infix,
    )


def _extract_positive_class_proba(y_pred):
    """Extract positive class probabilities for binary classification."""
    if y_pred.ndim == 2:
        if y_pred.shape[1] == 2:
            return y_pred[:, 1]
        elif y_pred.shape[1] == 1:
            return y_pred[:, 0]
    return y_pred


@attrs.frozen
class _MetricComputation:
    target: str = attrs.field()
    pred_col: str = attrs.field()
    metric_fn: Callable[..., Any] = attrs.field()
    metric_kwargs: tuple = attrs.field(
        converter=lambda x: tuple(sorted((x or {}).items())),
    )

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
        y_pred = _prepare_predictions(df[self.pred_col])
        return self.metric_fn(y_true, y_pred, **dict(self.metric_kwargs))


def _prepare_predictions(predictions):
    """Prepare predictions for metric computation using pattern matching."""
    import pandas as pd

    match predictions:
        # Case 1: pandas Series with array-like values (e.g., probabilities)
        case pd.Series() as series if len(series) > 0 and isinstance(
            series.iloc[0], (np.ndarray, list, tuple)
        ):
            return _extract_positive_class_proba(np.vstack(series.values))

        # Case 2: pandas Series with scalar values
        case pd.Series() as series:
            return _extract_positive_class_proba(series.values)

        # Case 3: already a numpy array
        case np.ndarray() as arr:
            return _extract_positive_class_proba(arr)

        # Case 4: anything else, return as-is
        case _:
            return predictions
