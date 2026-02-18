"""Cross-validation for xorq ML pipelines."""

import functools
from random import Random

import cloudpickle
import dask
import numpy as np
import pyarrow as pa
from attr import field, frozen
from attr.validators import deep_iterable, instance_of

import xorq.expr.datatypes as dt
from xorq.expr.udf import pyarrow_udwf
from xorq.vendor import ibis
from xorq.vendor.ibis.expr.api import literal
from xorq.vendor.ibis.expr.types import Expr


def make_deterministic_sort_key(expr, random_seed=None):
    """Make a deterministic sort key for an ibis table.

    Concatenates every column as a string (comma-separated), appends a
    deterministic random string derived from *random_seed*, and hashes the
    result.  Two tables with the same data and the same seed will always
    produce the same per-row hash within a given backend.

    Use this to impose a stable row order before any operation that depends
    on positional indexing (e.g. sklearn splitters).  To reproduce the same
    order on a pandas DataFrame::

        key = make_deterministic_sort_key(ibis_table, random_seed=42)
        col = key.get_name()
        df = ibis_table.mutate(key).order_by(col).drop(col).execute()

    .. note::

       The hash function is backend-dependent, so the same data may
       produce different sort orders on different backends.  Within a
       single backend, the ordering is fully deterministic.

    Parameters
    ----------
    expr : ir.Table
        The input table.
    random_seed : int or None
        Seed for reproducibility.  ``None`` uses a fixed default (0).

    Returns
    -------
    ir.IntegerColumn
        Hash column suitable for use in ``order_by``.
    """
    if random_seed is None:
        random_seed = 0
    random_str = str(Random(random_seed).getrandbits(256))
    tmp_name = "_sort_" + dask.base.tokenize(random_str)[:8]
    comb_key = literal(",").join(expr[col].cast("str") for col in expr.columns)
    return comb_key.concat(random_str).hash().name(tmp_name)


def _make_folds_from_int(expr, cv, random_seed):
    """Build a fold expression from k equal-sized hash-based folds.

    Returns the input table with ``fold_0``, ``fold_1``, ..., ``fold_{k-1}``
    columns appended (int8, 0=unused, 1=train, 2=test).

    Parameters
    ----------
    expr : ir.Table
        The input table to split.
    cv : int
        Number of folds.
    random_seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    ir.Table
        The input table with ``fold_0``, ..., ``fold_{k-1}`` columns
        appended (int8, 0=unused, 1=train, 2=test).
    """
    from xorq.expr.ml.split_lib import calc_split_column

    test_sizes = tuple(1.0 / cv for _ in range(cv))
    split_col_name = "__cv_split__"
    split_col = calc_split_column(
        expr,
        unique_key=expr.columns,
        test_sizes=test_sizes,
        num_buckets=10000,
        random_seed=random_seed,
        name=split_col_name,
    )
    base_expr = expr.mutate(split_col)
    # Encode: 2=test (row belongs to fold_i), 1=train (row belongs to another fold)
    fold_columns = (
        ((base_expr[split_col_name] == fold_i).cast("int8") + 1).name(f"fold_{fold_i}")
        for fold_i in range(cv)
    )
    fold_expr = functools.reduce(
        lambda expr, col: expr.mutate(col), fold_columns, base_expr
    )
    return fold_expr.drop(split_col_name)


def _make_fold_udwf(cv, fold_index, features, target, expr):
    """Create a UDWF that marks rows for a given fold.

    The UDWF receives feature and target columns for all rows, runs the sklearn
    splitter to determine fold assignments, and returns a per-row int8 column
    where 0 = unused, 1 = train, 2 = test.

    Parameters
    ----------
    cv : sklearn splitter
        An sklearn cross-validation splitter with a .split() method.
    fold_index : int
        Which fold (0-based) this UDWF assigns test membership for.
    features : tuple of str
        Feature column names.
    target : str
        Target column name.
    expr : ir.Table
        The input table (used to infer column types for the UDWF schema).

    Returns
    -------
    callable
        A UDWF constructor with an .on_expr() method.
    """
    # Build the schema from the actual expression types (features + target)
    # We need to serialize cv because pyarrow_udwf stores config in FrozenDict
    # and sklearn splitters aren't hashable.  We pickle them and unpickle
    # inside the UDWF.
    cv_bytes = cloudpickle.dumps(cv)

    table_schema = expr.schema()
    schema_cols = tuple((col, table_schema[col]) for col in (*features, target))

    @pyarrow_udwf(
        schema=ibis.schema(schema_cols),
        return_type=dt.int8,
        cv_bytes=cv_bytes,
        fold_index=fold_index,
        n_features=len(features),
        name=f"cv_fold_{fold_index}",
    )
    def assign_fold(self, values, num_rows):
        import cloudpickle as cp

        cv_splitter = cp.loads(self.cv_bytes)
        n_feat = self.n_features
        fold_i = self.fold_index

        # values[:n_feat] are feature columns, values[n_feat] is target
        X = np.column_stack([v.to_numpy(zero_copy_only=False) for v in values[:n_feat]])
        y = values[n_feat].to_numpy(zero_copy_only=False)

        # Run the splitter: 0=unused, 1=train, 2=test
        train_idx, test_idx = next(
            (train, test)
            for i, (train, test) in enumerate(cv_splitter.split(X, y))
            if i == fold_i
        )
        result = np.zeros(num_rows, dtype=np.int8)
        result[train_idx] = 1
        result[test_idx] = 2

        return pa.array(result, type=pa.int8())

    return assign_fold


def _make_folds_from_sklearn(
    expr, cv, features, target, random_seed=None, order_by=None
):
    """Build a fold expression from an sklearn splitter using deferred UDWFs.

    Returns the input table with ``fold_0``, ``fold_1``, ..., ``fold_{k-1}``
    columns appended (int8, 0=unused, 1=train, 2=test).

    Row ordering for the UDWF is determined by:

    1. If *order_by* is provided, rows are sorted by those column(s).
    2. Otherwise, rows are sorted by a deterministic seeded hash so
       the UDWF always sees the same order regardless of backend scan
       order.

    Parameters
    ----------
    expr : ir.Table
        The input table.
    cv : sklearn splitter
        An sklearn cross-validation splitter with a .split() method.
    features : tuple of str
        Feature column names (used as X for stratified splitters).
    target : str
        Target column name (used as y for stratified splitters).
    random_seed : int or None
        Seed for the deterministic row-ordering hash.
    order_by : str, tuple of str, or None
        Column(s) to sort by before folding.  Overrides the default
        hash-based sort.  Useful for ``TimeSeriesSplit`` where rows
        must be in temporal order.

    Returns
    -------
    ir.Table
        The input table with ``fold_0``, ..., ``fold_{k-1}`` columns
        appended (int8, 0=unused, 1=train, 2=test).
    """
    n_splits = cv.get_n_splits()
    window = ibis.window()

    match order_by:
        case str() | tuple() | list() as key:
            # User-specified ordering (e.g. a timestamp column).
            base_expr = expr.order_by(key)
            drop_cols = ()
        case None:
            # Default: sort by a deterministic hash so the UDWF sees rows
            # in a stable order regardless of backend scan order.
            sort_key = make_deterministic_sort_key(expr, random_seed)
            sort_col = sort_key.get_name()
            base_expr = expr.mutate(sort_key).order_by(sort_col)
            drop_cols = (sort_col,)
        case _:
            raise TypeError(
                f"order_by must be a str, tuple of str, or None, "
                f"got {type(order_by).__name__}"
            )

    fold_columns = (
        _make_fold_udwf(cv, i, features, target, base_expr)
        .on_expr(base_expr)
        .over(window)
        .name(f"fold_{i}")
        for i in range(n_splits)
    )
    result = functools.reduce(
        lambda expr, col: expr.mutate(col), fold_columns, base_expr
    )
    return result.drop(*drop_cols) if drop_cols else result


def _fold_pairs_from_fold_expr(fold_expr, n_splits):
    """Extract (train, test) pairs from a fold expression.

    Parameters
    ----------
    fold_expr : ir.Table
        Table with ``fold_0``, ..., ``fold_{k-1}`` columns
        (int8, 0=unused, 1=train, 2=test).
    n_splits : int
        Number of folds.

    Returns
    -------
    tuple of (ir.Table, ir.Table)
        One (train, test) pair per fold.  Fold columns are dropped from
        the returned tables.
    """
    return tuple(
        (
            fold_expr.filter(fold_expr[f"fold_{i}"] == 1).drop(f"fold_{i}"),
            fold_expr.filter(fold_expr[f"fold_{i}"] == 2).drop(f"fold_{i}"),
        )
        for i in range(n_splits)
    )


@frozen
class CrossValScore:
    """Deferred cross-validation scores.

    Holds a tuple of per-fold ibis score expressions and a ``fold_expr`` —
    a single ibis table with the original columns plus ``fold_0``,
    ``fold_1``, ..., ``fold_k`` columns (int8, 0 = unused, 1 = train, 2 = test).

    Call ``.execute()`` to materialize the scores, or inspect
    ``fold_expr`` to visualize / debug fold assignments.
    """

    score_exprs = field(
        validator=deep_iterable(instance_of(Expr), instance_of(tuple)),
        converter=tuple,
    )
    fold_expr = field(validator=instance_of(Expr))

    def execute(self):
        """Materialize all per-fold scores into a numpy array.

        Returns
        -------
        numpy.ndarray
            Array of scores, one per fold.
        """
        return np.array(tuple(expr.execute() for expr in self.score_exprs))

    def __len__(self):
        return len(self.score_exprs)


def deferred_cross_val_score(
    pipeline,
    expr,
    features,
    target,
    *,
    cv=5,
    scoring=None,
    random_seed=None,
    order_by=None,
):
    """Evaluate a pipeline using cross-validation, returning deferred per-fold scores.

    This is the xorq equivalent of sklearn's cross_val_score. The pipeline is
    fit and scored on each fold independently. Returns a CrossValScore holding
    deferred ibis expressions — call .execute() to materialize the scores.

    Parameters
    ----------
    pipeline : Pipeline
        An unfitted xorq Pipeline (will be fit on each fold's training data).
    expr : ir.Table
        The input ibis table expression containing features and target.
    features : tuple of str
        Feature column names.
    target : str
        Target column name.
    cv : int or sklearn splitter, optional
        Cross-validation strategy. If int, uses train_test_splits to create
        that many equal-sized folds. If an sklearn splitter object (e.g.
        KFold, StratifiedKFold), uses its .split() method to generate
        train/test indices. Default is 5.
    scoring : str, callable, _BaseScorer, Scorer, or None, optional
        Scorer specification passed to FittedPipeline.score_expr(). If None,
        uses the model's default scorer (accuracy for classifiers, r2 for
        regressors). Default is None.
    random_seed : int or None, optional
        Random seed for reproducibility.  When cv is an int, controls
        hash-based fold partitioning.  When cv is an sklearn splitter,
        controls the deterministic row ordering used to guarantee that
        the UDWF sees rows in a stable order.  To reproduce the same
        fold assignments with standalone sklearn, sort the pandas
        DataFrame by :func:`make_deterministic_sort_key` with the same
        seed before calling ``cross_val_score``.  Default is None.
    order_by : str, tuple of str, or None, optional
        Column(s) to sort by before folding.  Overrides the default
        hash-based sort.  Required for ``TimeSeriesSplit`` to specify
        the temporal ordering column.  Default is None.

    Returns
    -------
    CrossValScore
        Deferred cross-validation result. Call .execute() to get numpy.ndarray.

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline as SklearnPipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.linear_model import LogisticRegression
    >>> import xorq.api as xo
    >>> from xorq.expr.ml.pipeline_lib import Pipeline
    >>> from xorq.expr.ml.cross_validation import deferred_cross_val_score
    >>>
    >>> t = xo.memtable({"x1": range(100), "x2": range(100), "y": [0, 1] * 50})
    >>> pipe = Pipeline.from_instance(SklearnPipeline([
    ...     ("scaler", StandardScaler()),
    ...     ("clf", LogisticRegression()),
    ... ]))
    >>> cv_scores = deferred_cross_val_score(pipe, t, features=("x1", "x2"), target="y", cv=5)
    >>> scores = cv_scores.execute()  # materializes all folds
    """
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.model_selection._split import GroupsConsumerMixin

    from xorq.expr.ml.pipeline_lib import Pipeline

    if not isinstance(pipeline, Pipeline):
        raise TypeError(
            f"pipeline must be a Pipeline instance, got {type(pipeline).__name__}"
        )

    features = tuple(features)

    match cv:
        case int():
            fold_expr = _make_folds_from_int(expr, cv, random_seed)
            n_splits = cv
        case TimeSeriesSplit() if order_by is None:
            raise TypeError(
                "TimeSeriesSplit requires order_by to specify the "
                "temporal ordering column(s)."
            )
        case GroupsConsumerMixin():
            raise TypeError(
                f"Group-based splitters are not supported "
                f"(got {type(cv).__name__}). "
                f"Use a non-group splitter such as KFold, "
                f"StratifiedKFold, ShuffleSplit, etc."
            )
        case object(split=_):
            fold_expr = _make_folds_from_sklearn(
                expr,
                cv,
                features,
                target,
                random_seed=random_seed,
                order_by=order_by,
            )
            n_splits = cv.get_n_splits()
        case _:
            raise TypeError(
                f"cv must be an int or an sklearn splitter with a .split() method, "
                f"got {type(cv).__name__}"
            )

    fold_pairs = _fold_pairs_from_fold_expr(fold_expr, n_splits)

    score_exprs = tuple(
        pipeline.fit(train, features=features, target=target).score_expr(
            test, scorer=scoring
        )
        for train, test in fold_pairs
    )

    return CrossValScore(score_exprs=score_exprs, fold_expr=fold_expr)


__all__ = ["deferred_cross_val_score", "make_deterministic_sort_key"]
