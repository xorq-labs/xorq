"""Deferred scikit-learn metrics evaluation for xorq."""

from typing import Any, Callable

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


@frozen
class MetricComputation:
    target: str = field(validator=instance_of(str))
    pred_col: str = field(validator=instance_of(str))
    metric_fn: Callable[..., Any] = field(validator=instance_of(Callable))
    metric_kwargs_tuple: tuple = field(
        default=(),
        converter=compose(tuple, sorted, dict.items, dict),
    )
    return_type = field(validator=instance_of(dt.DataType), default=dt.float64)
    name = field(validator=optional(instance_of(str)), default=None)
    sign: int = field(validator=instance_of(int), default=1)

    def __attrs_post_init__(self):
        if self.name is None:
            name = make_name(
                prefix=f"metric_{self.metric_fn.__name__}",
                to_tokenize=self.metric_kwargs_tuple,
            )
            object.__setattr__(self, "name", name)

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
    metric_fn,
    metric_kwargs=(),
    return_type=dt.float64,
    name=None,
    sign=1,
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
    name: Optional[str]
        Custom name for the UDF
    sign : int, optional
        Sign to apply to result (default: 1). Use -1 for neg_* scorers.

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
    metric = MetricComputation(
        target=target,
        pred_col=pred_col,
        metric_fn=metric_fn,
        metric_kwargs_tuple=metric_kwargs,
        return_type=return_type,
        name=name,
        sign=sign,
    )
    return metric.on_expr(expr)
