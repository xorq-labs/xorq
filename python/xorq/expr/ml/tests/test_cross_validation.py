"""Tests for deferred_cross_val_score."""

import numpy as np
import pytest
from pytest import param

import xorq.api as xo


sklearn = pytest.importorskip("sklearn")

from sklearn.model_selection import (  # noqa: E402
    KFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
    cross_val_score,
)

from xorq.expr.ml.cross_validation import (  # noqa: E402
    CrossValScore,
    apply_deterministic_sort,
    deferred_cross_val_score,
)


@pytest.fixture(scope="module")
def classification_data():
    """Generate a classification dataset as an ibis table."""
    import pandas as pd
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=200,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)]).assign(
        target=y, t=range(200)
    )
    return xo.memtable(df)


@pytest.fixture(scope="module")
def regression_data():
    """Generate a regression dataset as an ibis table."""
    import pandas as pd
    from sklearn.datasets import make_regression

    X, y = make_regression(
        n_samples=200,
        n_features=4,
        n_informative=3,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)]).assign(
        target=y, t=range(200)
    )
    return xo.memtable(df)


@pytest.fixture(scope="module")
def classifier_pipeline():
    """An unfitted xorq Pipeline for classification."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline as SklearnPipeline
    from sklearn.preprocessing import StandardScaler

    from xorq.expr.ml.pipeline_lib import Pipeline

    return Pipeline.from_instance(
        SklearnPipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        )
    )


@pytest.fixture(scope="module")
def regressor_pipeline():
    """An unfitted xorq Pipeline for regression."""
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline as SklearnPipeline
    from sklearn.preprocessing import StandardScaler

    from xorq.expr.ml.pipeline_lib import Pipeline

    return Pipeline.from_instance(
        SklearnPipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", LinearRegression()),
            ]
        )
    )


FEATURES = tuple(f"f{i}" for i in range(4))
TARGET = "target"
RANDOM_SEED = 42


# --- splitter instances, each with its expected number of folds ---
CLASSIFIER_SPLITTERS = (
    param(
        KFold(n_splits=3, shuffle=True, random_state=42),
        3,
        id="KFold",
    ),
    param(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        5,
        id="StratifiedKFold",
    ),
    param(
        RepeatedKFold(n_splits=3, n_repeats=2, random_state=42),
        6,
        id="RepeatedKFold",
    ),
    param(
        RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=42),
        6,
        id="RepeatedStratifiedKFold",
    ),
    param(
        ShuffleSplit(n_splits=4, test_size=0.25, random_state=42),
        4,
        id="ShuffleSplit",
    ),
    param(
        StratifiedShuffleSplit(n_splits=4, test_size=0.25, random_state=42),
        4,
        id="StratifiedShuffleSplit",
    ),
)


class TestDeferredCrossValScoreIntCV:
    """Tests with integer cv (uses train_test_splits internally)."""

    def test_returns_cross_val_score(self, classification_data, classifier_pipeline):
        result = deferred_cross_val_score(
            classifier_pipeline,
            classification_data,
            features=FEATURES,
            target=TARGET,
            cv=3,
            random_seed=42,
        )
        assert isinstance(result, CrossValScore)
        scores = result.execute()
        assert isinstance(scores, np.ndarray)

    def test_correct_number_of_folds(self, classification_data, classifier_pipeline):
        for k in (3, 5):
            result = deferred_cross_val_score(
                classifier_pipeline,
                classification_data,
                features=FEATURES,
                target=TARGET,
                cv=k,
                random_seed=42,
            )
            assert len(result) == k
            assert len(result.execute()) == k

    def test_scores_are_finite(self, classification_data, classifier_pipeline):
        scores = deferred_cross_val_score(
            classifier_pipeline,
            classification_data,
            features=FEATURES,
            target=TARGET,
            cv=3,
            random_seed=42,
        ).execute()
        assert np.all(np.isfinite(scores))

    def test_classifier_scores_bounded(self, classification_data, classifier_pipeline):
        """Default classifier scorer is accuracy, which is in [0, 1]."""
        scores = deferred_cross_val_score(
            classifier_pipeline,
            classification_data,
            features=FEATURES,
            target=TARGET,
            cv=3,
            random_seed=42,
        ).execute()
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_regressor_default_scorer(self, regression_data, regressor_pipeline):
        scores = deferred_cross_val_score(
            regressor_pipeline,
            regression_data,
            features=FEATURES,
            target=TARGET,
            cv=3,
            random_seed=42,
        ).execute()
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 3
        assert np.all(np.isfinite(scores))

    def test_explicit_scorer(self, classification_data, classifier_pipeline):
        scores = deferred_cross_val_score(
            classifier_pipeline,
            classification_data,
            features=FEATURES,
            target=TARGET,
            cv=3,
            scoring="f1",
            random_seed=42,
        ).execute()
        assert isinstance(scores, np.ndarray)
        assert len(scores) == 3
        assert np.all(np.isfinite(scores))

    def test_mean_close_to_sklearn(self, classification_data, classifier_pipeline):
        """int cv uses hash-based folds (different from sklearn's KFold), so
        per-fold scores differ.  But the mean over k folds should converge to
        the same value.  We allow 5% tolerance on the mean."""
        df = classification_data.execute()
        X = df[list(FEATURES)].values
        y = df[TARGET].values

        for k in (3, 5):
            xorq_scores = deferred_cross_val_score(
                classifier_pipeline,
                classification_data,
                features=FEATURES,
                target=TARGET,
                cv=k,
                random_seed=42,
            ).execute()
            sklearn_scores = cross_val_score(
                classifier_pipeline.instance,
                X,
                y,
                cv=KFold(n_splits=k, shuffle=True, random_state=42),
                scoring="accuracy",
            )
            assert abs(xorq_scores.mean() - sklearn_scores.mean()) < 0.05

    def test_reproducible_with_seed(self, classification_data, classifier_pipeline):
        scores1 = deferred_cross_val_score(
            classifier_pipeline,
            classification_data,
            features=FEATURES,
            target=TARGET,
            cv=3,
            random_seed=42,
        ).execute()
        scores2 = deferred_cross_val_score(
            classifier_pipeline,
            classification_data,
            features=FEATURES,
            target=TARGET,
            cv=3,
            random_seed=42,
        ).execute()
        np.testing.assert_array_equal(scores1, scores2)


class TestDeferredCrossValScoreSklearnSplitter:
    """Tests that each sklearn splitter produces the right shape and finite scores."""

    @pytest.mark.parametrize("cv, expected_folds", CLASSIFIER_SPLITTERS)
    def test_splitter_produces_correct_folds(
        self,
        classification_data,
        classifier_pipeline,
        cv,
        expected_folds,
    ):
        result = deferred_cross_val_score(
            classifier_pipeline,
            classification_data,
            features=FEATURES,
            target=TARGET,
            cv=cv,
        )
        assert len(result) == expected_folds
        scores = result.execute()
        assert isinstance(scores, np.ndarray)
        assert len(scores) == expected_folds
        assert np.all(np.isfinite(scores))


class TestDeferredCrossValScoreMatchesSklearn:
    """Per-fold parity: deferred_cross_val_score must match sklearn exactly.

    When both sides use the same splitter, they see identical train/test
    indices, so scores must be numerically equal.
    """

    # Each entry is (splitter_factory, scoring, id).  Factory is a callable
    # that returns a fresh splitter — splitters are stateful iterators, so
    # xorq and sklearn each need their own instance.
    @pytest.mark.parametrize(
        "make_cv, scoring",
        (
            param(
                lambda: KFold(n_splits=3, shuffle=True, random_state=42),
                "accuracy",
                id="KFold-accuracy",
            ),
            param(
                lambda: StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                "f1",
                id="StratifiedKFold-f1",
            ),
            param(
                lambda: RepeatedKFold(n_splits=3, n_repeats=2, random_state=42),
                "accuracy",
                id="RepeatedKFold-accuracy",
            ),
            param(
                lambda: RepeatedStratifiedKFold(
                    n_splits=3, n_repeats=2, random_state=42
                ),
                "f1",
                id="RepeatedStratifiedKFold-f1",
            ),
            param(
                lambda: ShuffleSplit(n_splits=4, test_size=0.25, random_state=42),
                "accuracy",
                id="ShuffleSplit-accuracy",
            ),
            param(
                lambda: StratifiedShuffleSplit(
                    n_splits=4, test_size=0.25, random_state=42
                ),
                "accuracy",
                id="StratifiedShuffleSplit-accuracy",
            ),
        ),
    )
    def test_classifier_matches_sklearn(
        self,
        classification_data,
        classifier_pipeline,
        make_cv,
        scoring,
    ):
        xorq_scores = deferred_cross_val_score(
            classifier_pipeline,
            classification_data,
            features=FEATURES,
            target=TARGET,
            cv=make_cv(),
            scoring=scoring,
            random_seed=RANDOM_SEED,
        ).execute()

        # Sort by the same deterministic key so sklearn sees identical row order.
        df = apply_deterministic_sort(
            classification_data, random_seed=RANDOM_SEED
        ).execute()
        sklearn_scores = cross_val_score(
            classifier_pipeline.instance,
            df[list(FEATURES)].values,
            df[TARGET].values,
            cv=make_cv(),
            scoring=scoring,
        )

        np.testing.assert_allclose(xorq_scores, sklearn_scores, rtol=1e-9, atol=1e-12)

    def test_timeseries_split_matches_sklearn(
        self, classification_data, classifier_pipeline
    ):
        """TimeSeriesSplit with order_by must match sklearn on sorted data."""
        make_cv = lambda: TimeSeriesSplit(n_splits=4)  # noqa: E731

        xorq_scores = deferred_cross_val_score(
            classifier_pipeline,
            classification_data,
            features=FEATURES,
            target=TARGET,
            cv=make_cv(),
            scoring="accuracy",
            order_by="t",
        ).execute()

        # sklearn sees the same order: sorted by "t".
        df = classification_data.execute().sort_values("t")
        sklearn_scores = cross_val_score(
            classifier_pipeline.instance,
            df[list(FEATURES)].values,
            df[TARGET].values,
            cv=make_cv(),
            scoring="accuracy",
        )

        np.testing.assert_allclose(xorq_scores, sklearn_scores, rtol=1e-9, atol=1e-12)

    def test_regressor_kfold_matches_sklearn(self, regression_data, regressor_pipeline):
        """Per-fold r2 with KFold must match sklearn exactly."""
        make_cv = lambda: KFold(n_splits=3, shuffle=True, random_state=42)  # noqa: E731

        xorq_scores = deferred_cross_val_score(
            regressor_pipeline,
            regression_data,
            features=FEATURES,
            target=TARGET,
            cv=make_cv(),
            random_seed=RANDOM_SEED,
        ).execute()

        df = apply_deterministic_sort(
            regression_data, random_seed=RANDOM_SEED
        ).execute()
        sklearn_scores = cross_val_score(
            regressor_pipeline.instance,
            df[list(FEATURES)].values,
            df[TARGET].values,
            cv=make_cv(),
            scoring="r2",
        )

        np.testing.assert_allclose(xorq_scores, sklearn_scores, rtol=1e-9, atol=1e-12)


class TestDeferredCrossValScoreLazy:
    """Tests that expression construction is fully deferred (no eager execution)."""

    def test_sklearn_splitter_no_execute_during_construction(
        self,
        classification_data,
        classifier_pipeline,
    ):
        """Building a CrossValScore with an sklearn splitter must NOT call .execute().

        The old implementation eagerly materialized the table to get row indices.
        The new UDWF-based implementation should be fully lazy.
        """
        from unittest.mock import patch

        cv = KFold(n_splits=3, shuffle=True, random_state=42)

        with patch.object(
            type(classification_data),
            "execute",
            wraps=classification_data.execute,
        ) as mock_execute:
            result = deferred_cross_val_score(
                classifier_pipeline,
                classification_data,
                features=FEATURES,
                target=TARGET,
                cv=cv,
            )
            # Construction should not trigger any .execute() calls
            mock_execute.assert_not_called()

        # The result should still be a valid CrossValScore
        assert isinstance(result, CrossValScore)
        assert len(result) == 3

    def test_int_cv_no_execute_during_construction(
        self,
        classification_data,
        classifier_pipeline,
    ):
        """Building a CrossValScore with int cv must NOT call .execute()."""
        from unittest.mock import patch

        with patch.object(
            type(classification_data),
            "execute",
            wraps=classification_data.execute,
        ) as mock_execute:
            result = deferred_cross_val_score(
                classifier_pipeline,
                classification_data,
                features=FEATURES,
                target=TARGET,
                cv=3,
                random_seed=42,
            )
            mock_execute.assert_not_called()

        assert isinstance(result, CrossValScore)
        assert len(result) == 3


class TestFoldExpr:
    """Tests for the fold_expr attribute on CrossValScore."""

    def test_fold_expr_has_fold_columns_int_cv(
        self, classification_data, classifier_pipeline
    ):
        result = deferred_cross_val_score(
            classifier_pipeline,
            classification_data,
            features=FEATURES,
            target=TARGET,
            cv=3,
            random_seed=42,
        )
        fold_df = result.fold_expr.execute()
        for i in range(3):
            assert f"fold_{i}" in fold_df.columns
        # Original columns should still be present
        for col in list(FEATURES) + [TARGET]:
            assert col in fold_df.columns

    def test_fold_expr_has_fold_columns_sklearn_cv(
        self, classification_data, classifier_pipeline
    ):
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        result = deferred_cross_val_score(
            classifier_pipeline,
            classification_data,
            features=FEATURES,
            target=TARGET,
            cv=cv,
            random_seed=42,
        )
        fold_df = result.fold_expr.execute()
        for i in range(5):
            assert f"fold_{i}" in fold_df.columns

    def test_fold_expr_values_are_0_1_or_2(
        self, classification_data, classifier_pipeline
    ):
        """Fold columns use ternary encoding: 0=unused, 1=train, 2=test."""
        result = deferred_cross_val_score(
            classifier_pipeline,
            classification_data,
            features=FEATURES,
            target=TARGET,
            cv=3,
            random_seed=42,
        )
        fold_df = result.fold_expr.execute()
        for i in range(3):
            values = set(fold_df[f"fold_{i}"].unique())
            assert values <= {0, 1, 2}

    def test_fold_expr_each_row_is_test_in_exactly_one_fold_int_cv(
        self, classification_data, classifier_pipeline
    ):
        """With int cv (equal-sized folds), each row should be test in exactly one fold."""
        result = deferred_cross_val_score(
            classifier_pipeline,
            classification_data,
            features=FEATURES,
            target=TARGET,
            cv=3,
            random_seed=42,
        )
        fold_df = result.fold_expr.execute()
        # Each row is test (2) in exactly one fold
        test_counts = sum((fold_df[f"fold_{i}"] == 2).astype(int) for i in range(3))
        assert (test_counts == 1).all()

    def test_fold_expr_row_count_matches_original(
        self, classification_data, classifier_pipeline
    ):
        result = deferred_cross_val_score(
            classifier_pipeline,
            classification_data,
            features=FEATURES,
            target=TARGET,
            cv=3,
            random_seed=42,
        )
        assert (
            result.fold_expr.count().execute() == classification_data.count().execute()
        )

    @pytest.mark.parametrize("cv, expected_folds", CLASSIFIER_SPLITTERS)
    def test_fold_expr_sklearn_splitter_columns(
        self, classification_data, classifier_pipeline, cv, expected_folds
    ):
        result = deferred_cross_val_score(
            classifier_pipeline,
            classification_data,
            features=FEATURES,
            target=TARGET,
            cv=cv,
        )
        fold_df = result.fold_expr.execute()
        for i in range(expected_folds):
            assert f"fold_{i}" in fold_df.columns
            values = set(fold_df[f"fold_{i}"].unique())
            assert values <= {0, 1, 2}

    def test_fold_expr_timeseries_split(self, classification_data, classifier_pipeline):
        """TimeSeriesSplit fold_expr uses ternary encoding with unused rows."""
        result = deferred_cross_val_score(
            classifier_pipeline,
            classification_data,
            features=FEATURES,
            target=TARGET,
            cv=TimeSeriesSplit(n_splits=4),
            order_by="t",
        )
        fold_df = result.fold_expr.execute()
        for i in range(4):
            assert f"fold_{i}" in fold_df.columns
            values = set(fold_df[f"fold_{i}"].unique())
            assert values <= {0, 1, 2}
            # TimeSeriesSplit has unused rows in early folds
            if i < 3:
                assert 0 in values, f"fold_{i} should have unused rows"


class TestDeferredCrossValScoreValidation:
    """Tests for input validation."""

    def test_rejects_non_pipeline(self, classification_data):
        with pytest.raises(TypeError, match="pipeline must be a Pipeline instance"):
            deferred_cross_val_score(
                "not_a_pipeline",
                classification_data,
                features=FEATURES,
                target=TARGET,
            )

    @pytest.mark.parametrize(
        "cv",
        (
            param(
                pytest.importorskip("sklearn.model_selection").GroupKFold(n_splits=3),
                id="GroupKFold",
            ),
            param(
                pytest.importorskip("sklearn.model_selection").GroupShuffleSplit(
                    n_splits=3, random_state=42
                ),
                id="GroupShuffleSplit",
            ),
            param(
                pytest.importorskip("sklearn.model_selection").LeaveOneGroupOut(),
                id="LeaveOneGroupOut",
            ),
            param(
                pytest.importorskip("sklearn.model_selection").StratifiedGroupKFold(
                    n_splits=3
                ),
                id="StratifiedGroupKFold",
            ),
        ),
    )
    def test_rejects_group_based_splitters(
        self, classification_data, classifier_pipeline, cv
    ):
        with pytest.raises(TypeError, match="Group-based splitters are not supported"):
            deferred_cross_val_score(
                classifier_pipeline,
                classification_data,
                features=FEATURES,
                target=TARGET,
                cv=cv,
            )

    def test_rejects_timeseries_split_without_order_by(
        self, classification_data, classifier_pipeline
    ):
        with pytest.raises(TypeError, match="TimeSeriesSplit requires order_by"):
            deferred_cross_val_score(
                classifier_pipeline,
                classification_data,
                features=FEATURES,
                target=TARGET,
                cv=TimeSeriesSplit(n_splits=4),
            )
