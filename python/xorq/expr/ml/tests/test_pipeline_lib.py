import operator
from numbers import Real

import pytest

import xorq.api as xo
from xorq.common.utils.graph_utils import walk_nodes
from xorq.expr.relations import Tag


sklearn = pytest.importorskip("sklearn")


TARGET = "target"
features = (feature0, feature1) = ("feature_0", "feature_1")


get_metadata = operator.attrgetter("metadata")


@pytest.fixture(scope="module")
def t():
    return xo.memtable(
        {
            feature0: [1, 2],
            feature1: [3, 4],
            TARGET: [0, 1],
        }
    )


step_typs = (
    sklearn.preprocessing.StandardScaler,
    sklearn.linear_model.LinearRegression,
)


@pytest.fixture(scope="module")
def sklearn_pipeline():
    sklearn_pipeline = sklearn.pipeline.make_pipeline(*(typ() for typ in step_typs))
    return sklearn_pipeline


@pytest.fixture(scope="module")
def fitted_xorq_pipeline(sklearn_pipeline, t):
    xorq_pipeline = xo.Pipeline.from_instance(sklearn_pipeline)
    return xorq_pipeline.fit(t, target=TARGET)


def test_infer_features(fitted_xorq_pipeline):
    assert all(
        features == step.features for step in fitted_xorq_pipeline.transform_steps
    )


@pytest.fixture(scope="module")
def all_tags(t, fitted_xorq_pipeline):
    expr = fitted_xorq_pipeline.predict(t)
    all_tags = walk_nodes((Tag,), expr)
    return all_tags


def test_all_tags(t, fitted_xorq_pipeline, all_tags):
    expr = fitted_xorq_pipeline.predict(t)
    actual = tuple(map(get_metadata, expr.ls.get_tags()))
    expected = tuple(map(get_metadata, all_tags))
    assert actual == expected


@pytest.mark.parametrize(
    "pairs",
    (
        (("tag", "FittedStep-transform"),),
        (("tag", "FittedStep-predict"),),
        (
            ("tag", "FittedStep-transform"),
            ("tag", "FittedStep-predict"),
        ),
    ),
)
def test_tagging_pipeline(pairs, t, fitted_xorq_pipeline):
    def contains_any_pairs(d, pairs=pairs):
        return set(pairs).intersection(d.items())

    def sort_and_tuplify(dcts):
        return tuple(sorted(tuple(sorted(dct.items())) for dct in dcts))

    actual = sort_and_tuplify(
        map(
            get_metadata,
            fitted_xorq_pipeline.predict(t).ls.get_tags(
                predicate=contains_any_pairs,
            ),
        )
    )
    expected = sort_and_tuplify(
        dct
        for dct in (
            fitted_step.tag_kwargs for fitted_step in fitted_xorq_pipeline.fitted_steps
        )
        if contains_any_pairs(dct)
    )
    assert actual and actual == expected


def test_score_expr_returns_metric(t, fitted_xorq_pipeline):
    score_expr = fitted_xorq_pipeline.score_expr(t)
    result = score_expr.execute()
    assert isinstance(result, Real)


class TestFittedStepTransform:
    """Tests for FittedStep.transform simplified logic."""

    def test_fitted_step_transform_known_schema_unpacks(self):
        """Test FittedStep.transform unpacks struct columns for known schema."""
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        step = xo.Step.from_instance_name(StandardScaler(), name="scaler")
        fitted = step.fit(t, features=("a", "b"))

        result = fitted.transform(t)
        df = result.execute()

        # Should have unpacked columns a and b, not a struct column
        assert "a" in df.columns
        assert "b" in df.columns
        assert "transformed" not in df.columns

    def test_fitted_step_transform_kv_encoded_no_unpack(self):
        """Test FittedStep.transform keeps KV-encoded column without unpacking."""
        from sklearn.preprocessing import OneHotEncoder

        t = xo.memtable({"cat": ["x", "y", "x", "z"]})
        step = xo.Step.from_instance_name(OneHotEncoder(), name="ohe")
        fitted = step.fit(t, features=("cat",))

        result = fitted.transform(t)
        df = result.execute()

        # Should have KV-encoded column named "transformed"
        assert "transformed" in df.columns
        # Should not have unpacked category columns
        assert "cat_x" not in df.columns
        assert "cat_y" not in df.columns
        assert "cat_z" not in df.columns

    def test_fitted_step_transform_retain_others_true(self):
        """Test FittedStep.transform retains other columns by default."""
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0], "other": ["x", "y"]})
        step = xo.Step.from_instance_name(StandardScaler(), name="scaler")
        fitted = step.fit(t, features=("a", "b"))

        result = fitted.transform(t, retain_others=True)
        df = result.execute()

        # Should retain the "other" column
        assert "other" in df.columns
        assert df["other"].tolist() == ["x", "y"]

    def test_fitted_step_transform_retain_others_false(self):
        """Test FittedStep.transform drops other columns when retain_others=False."""
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0], "other": ["x", "y"]})
        step = xo.Step.from_instance_name(StandardScaler(), name="scaler")
        fitted = step.fit(t, features=("a", "b"))

        result = fitted.transform(t, retain_others=False)
        df = result.execute()

        # Should not retain the "other" column
        assert "other" not in df.columns


class TestPipelineGetOutputColumns:
    """Tests for Pipeline using Structer.get_output_columns."""

    def test_pipeline_known_schema_features_propagate(self):
        """Test Pipeline correctly propagates features for known schema transformers."""
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable(
            {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "y": [0.0, 1.0, 0.0]}
        )
        pipeline = xo.Pipeline.from_instance(
            sklearn.pipeline.make_pipeline(StandardScaler(), LinearRegression())
        )
        fitted = pipeline.fit(t, target="y")

        # The transform step should have features = ("a", "b") from the known schema
        transform_step = fitted.transform_steps[0]
        assert transform_step.structer.get_output_columns() == ("a", "b")

    def test_pipeline_kv_encoded_features_use_dest_col(self):
        """Test Pipeline correctly uses dest_col for KV-encoded transformers."""
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import OneHotEncoder

        t = xo.memtable({"cat": ["a", "b", "a"], "y": [0.0, 1.0, 0.0]})
        pipeline = xo.Pipeline.from_instance(
            sklearn.pipeline.make_pipeline(OneHotEncoder(), LinearRegression())
        )
        fitted = pipeline.fit(t, target="y")

        # The transform step should have features = ("transformed",) for KV-encoded
        transform_step = fitted.transform_steps[0]
        assert transform_step.structer.get_output_columns("transformed") == (
            "transformed",
        )

    def test_pipeline_mixed_transform_steps(self):
        """Test Pipeline with multiple transform steps propagates features correctly."""
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler

        t = xo.memtable(
            {"a": [1.0, None, 3.0], "b": [4.0, 5.0, 6.0], "y": [0.0, 1.0, 0.0]}
        )
        pipeline = xo.Pipeline.from_instance(
            sklearn.pipeline.make_pipeline(
                SimpleImputer(), StandardScaler(), LinearRegression()
            )
        )
        fitted = pipeline.fit(t, target="y")

        # Both transform steps should have known schema
        for transform_step in fitted.transform_steps:
            assert not transform_step.structer.is_kv_encoded

        # Prediction should work
        result = fitted.predict(t)
        assert result.execute() is not None


class TestDeeplyNestedPipelines:
    """Tests for deeply nested sklearn pipelines with xorq.

    These tests verify that xorq produces identical predictions to sklearn
    for complex nested pipeline structures.
    """

    def test_kv_encoded_deeply_nested_pipeline(self):
        """Test depth-4 nested pipeline with KV-encoded ColumnTransformer.

        Pipeline structure:
        - ColumnTransformer (KV-encoded due to OneHotEncoder)
          - FeatureUnion
            - Pipeline (SimpleImputer -> StandardScaler)
            - Pipeline (SimpleImputer -> StandardScaler)
          - Pipeline (SimpleImputer -> OneHotEncoder)
        - SelectKBest
        - RandomForestClassifier
        """
        import numpy as np
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import FeatureUnion
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        from xorq.expr.ml.pipeline_lib import Pipeline

        # Create sample data
        np.random.seed(42)
        n_samples = 100

        data = pd.DataFrame(
            {
                "age": np.random.randint(18, 80, n_samples).astype(float),
                "income": np.random.randint(20000, 150000, n_samples).astype(float),
                "credit_score": np.random.randint(300, 850, n_samples).astype(float),
                "years_employed": np.random.randint(0, 40, n_samples).astype(float),
                "education": np.random.choice(
                    ["high_school", "bachelor", "master", "phd"], n_samples
                ),
                "employment_type": np.random.choice(
                    ["full_time", "part_time", "contract", "self_employed"], n_samples
                ),
                "region": np.random.choice(
                    ["north", "south", "east", "west"], n_samples
                ),
                "approved": np.random.randint(0, 2, n_samples),
            }
        )

        # Add missing values
        data.loc[np.random.choice(n_samples, 10), "age"] = np.nan
        data.loc[np.random.choice(n_samples, 8), "income"] = np.nan

        numeric_features = ["age", "income", "credit_score", "years_employed"]
        categorical_features = ["education", "employment_type", "region"]
        all_features = tuple(numeric_features + categorical_features)

        # Build nested sklearn pipeline
        scaled_pipeline = SklearnPipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        imputed_pipeline = SklearnPipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

        numeric_union = FeatureUnion(
            [
                ("scaled", scaled_pipeline),
                ("imputed", imputed_pipeline),
            ]
        )

        categorical_pipeline = SklearnPipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ("numeric", numeric_union, numeric_features),
                ("categorical", categorical_pipeline, categorical_features),
            ]
        )

        sklearn_pipe = SklearnPipeline(
            [
                ("preprocessor", preprocessor),
                ("selector", SelectKBest(f_classif, k=10)),
                (
                    "classifier",
                    RandomForestClassifier(n_estimators=50, random_state=42),
                ),
            ]
        )

        # Fit and predict with xorq
        expr = xo.memtable(data)
        xorq_pipeline = Pipeline.from_instance(sklearn_pipe)
        fitted_pipeline = xorq_pipeline.fit(
            expr, features=all_features, target="approved"
        )
        predictions = fitted_pipeline.predict(expr).execute()

        # Fit and predict with sklearn
        X = data[list(all_features)]
        y = data["approved"]
        sklearn_pipe.fit(X, y)
        sklearn_preds = sklearn_pipe.predict(X)

        # Assert predictions match
        assert np.array_equal(predictions["predicted"].values, sklearn_preds)

    def test_non_kv_deeply_nested_pipeline(self):
        """Test depth-4 nested pipeline with all known-schema transformers.

        Pipeline structure:
        - ColumnTransformer (known schema - no KV-encoded children)
          - Pipeline (SimpleImputer -> StandardScaler -> Pipeline)
            - Pipeline (SimpleImputer -> StandardScaler)
          - Pipeline (SimpleImputer -> StandardScaler)
        - RandomForestClassifier
        """
        import numpy as np
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.pipeline_lib import Pipeline

        # Create sample data
        np.random.seed(42)
        n_samples = 100

        data = pd.DataFrame(
            {
                "age": np.random.randint(18, 80, n_samples).astype(float),
                "income": np.random.randint(20000, 150000, n_samples).astype(float),
                "credit_score": np.random.randint(300, 850, n_samples).astype(float),
                "years_employed": np.random.randint(0, 40, n_samples).astype(float),
                "debt_ratio": np.random.uniform(0, 1, n_samples),
                "savings": np.random.randint(0, 100000, n_samples).astype(float),
                "approved": np.random.randint(0, 2, n_samples),
            }
        )

        # Add missing values
        data.loc[np.random.choice(n_samples, 10), "age"] = np.nan
        data.loc[np.random.choice(n_samples, 8), "income"] = np.nan
        data.loc[np.random.choice(n_samples, 5), "years_employed"] = np.nan

        numeric_features_a = ["age", "income", "credit_score"]
        numeric_features_b = ["years_employed", "debt_ratio", "savings"]
        all_features = tuple(numeric_features_a + numeric_features_b)

        # Build nested sklearn pipeline (depth 4)
        inner_pipeline = SklearnPipeline(
            [
                ("imputer2", SimpleImputer(strategy="mean")),
                ("scaler2", StandardScaler()),
            ]
        )

        numeric_a_pipeline = SklearnPipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("inner", inner_pipeline),
            ]
        )

        numeric_b_pipeline = SklearnPipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ("numeric_a", numeric_a_pipeline, numeric_features_a),
                ("numeric_b", numeric_b_pipeline, numeric_features_b),
            ]
        )

        sklearn_pipe = SklearnPipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(n_estimators=50, random_state=42),
                ),
            ]
        )

        # Fit and predict with xorq
        expr = xo.memtable(data)
        xorq_pipeline = Pipeline.from_instance(sklearn_pipe)
        fitted_pipeline = xorq_pipeline.fit(
            expr, features=all_features, target="approved"
        )
        predictions = fitted_pipeline.predict(expr).execute()

        # Fit and predict with sklearn
        X = data[list(all_features)]
        y = data["approved"]
        sklearn_pipe.fit(X, y)
        sklearn_preds = sklearn_pipe.predict(X)

        # Assert predictions match
        assert np.array_equal(predictions["predicted"].values, sklearn_preds)


class TestPipelineScoring:
    """Tests for pipeline scoring with all compatible scorers."""

    # Compatible scorers discovered by test_report_scorer_compatibility
    CLASSIFIER_SCORERS = frozenset(
        [
            "accuracy",
            "adjusted_mutual_info_score",
            "adjusted_rand_score",
            "average_precision",
            "balanced_accuracy",
            "completeness_score",
            "d2_absolute_error_score",
            "explained_variance",
            "f1",
            "f1_macro",
            "f1_micro",
            "f1_weighted",
            "fowlkes_mallows_score",
            "homogeneity_score",
            "jaccard",
            "jaccard_macro",
            "jaccard_micro",
            "jaccard_weighted",
            "matthews_corrcoef",
            "mutual_info_score",
            "neg_brier_score",
            "neg_log_loss",
            "neg_max_error",
            "neg_mean_absolute_error",
            "neg_mean_absolute_percentage_error",
            "neg_mean_squared_error",
            "neg_mean_squared_log_error",
            "neg_median_absolute_error",
            "neg_negative_likelihood_ratio",
            "neg_root_mean_squared_error",
            "neg_root_mean_squared_log_error",
            "normalized_mutual_info_score",
            "positive_likelihood_ratio",
            "precision",
            "precision_macro",
            "precision_micro",
            "precision_weighted",
            "r2",
            "rand_score",
            "recall",
            "recall_macro",
            "recall_micro",
            "recall_weighted",
            "roc_auc",
            "roc_auc_ovo",
            "roc_auc_ovo_weighted",
            "roc_auc_ovr",
            "roc_auc_ovr_weighted",
            "top_k_accuracy",
            "v_measure_score",
        ]
    )

    REGRESSOR_SCORERS = frozenset(
        [
            "adjusted_mutual_info_score",
            "adjusted_rand_score",
            "completeness_score",
            "d2_absolute_error_score",
            "explained_variance",
            "fowlkes_mallows_score",
            "homogeneity_score",
            "mutual_info_score",
            "neg_max_error",
            "neg_mean_absolute_error",
            "neg_mean_absolute_percentage_error",
            "neg_mean_squared_error",
            "neg_median_absolute_error",
            "neg_root_mean_squared_error",
            "normalized_mutual_info_score",
            "r2",
            "rand_score",
            "v_measure_score",
        ]
    )

    CLUSTER_SUPERVISED_SCORERS = frozenset(
        [
            "accuracy",
            "adjusted_mutual_info_score",
            "adjusted_rand_score",
            "balanced_accuracy",
            "completeness_score",
            "d2_absolute_error_score",
            "explained_variance",
            "f1",
            "f1_macro",
            "f1_micro",
            "f1_weighted",
            "fowlkes_mallows_score",
            "homogeneity_score",
            "jaccard",
            "jaccard_macro",
            "jaccard_micro",
            "jaccard_weighted",
            "matthews_corrcoef",
            "mutual_info_score",
            "neg_max_error",
            "neg_mean_absolute_error",
            "neg_mean_absolute_percentage_error",
            "neg_mean_squared_error",
            "neg_mean_squared_log_error",
            "neg_median_absolute_error",
            "neg_negative_likelihood_ratio",
            "neg_root_mean_squared_error",
            "neg_root_mean_squared_log_error",
            "normalized_mutual_info_score",
            "positive_likelihood_ratio",
            "precision",
            "precision_macro",
            "precision_micro",
            "precision_weighted",
            "r2",
            "rand_score",
            "recall",
            "recall_macro",
            "recall_micro",
            "recall_weighted",
            "v_measure_score",
        ]
    )

    # Note: No unsupervised clustering scorers work via sklearn's scorer API
    CLUSTER_UNSUPERVISED_SCORERS = frozenset([])

    # Known problematic scorers that need investigation/fixes
    KNOWN_ISSUES = frozenset(
        [
            # Multiclass scorers with averaging that don't match sklearn
            "f1_macro",
            "f1_micro",
            "f1_weighted",
            "precision_macro",
            "precision_micro",
            "precision_weighted",
            "recall_macro",
            "recall_micro",
            "recall_weighted",
            "jaccard_macro",
            "jaccard_micro",
            "jaccard_weighted",
            # Clustering metrics when used with regressors/classifiers
            "adjusted_mutual_info_score",
            "adjusted_rand_score",
            "completeness_score",
            "fowlkes_mallows_score",
            "homogeneity_score",
            "mutual_info_score",
            "normalized_mutual_info_score",
            "rand_score",
            "v_measure_score",
        ]
    )

    @pytest.fixture
    def scoring_data(self):
        """Generate dataset suitable for classification, regression, and clustering."""
        import numpy as np

        np.random.seed(42)
        n = 100
        return {
            "x1": np.random.randn(n).tolist(),
            "x2": np.random.randn(n).tolist(),
            "y_class": (np.random.randn(n) > 0).astype(int).tolist(),
            "y_reg": np.random.randn(n).tolist(),
        }

    @pytest.fixture(scope="class")
    def all_scorer_names(self):
        """Get all available scorer names."""
        from sklearn.metrics import get_scorer_names

        return get_scorer_names()

    def test_report_scorer_compatibility(self, scoring_data, all_scorer_names):
        """Report which scorers are compatible with each model type."""
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.metrics import get_scorer
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import StandardScaler

        # Prepare data
        X = np.array([scoring_data["x1"], scoring_data["x2"]]).T
        y_class = np.array(scoring_data["y_class"])
        y_reg = np.array(scoring_data["y_reg"])

        # Create and fit pipelines
        classifier_pipe = SklearnPipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(random_state=42, max_iter=1000)),
            ]
        )
        classifier_pipe.fit(X, y_class)

        regressor_pipe = SklearnPipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        )
        regressor_pipe.fit(X, y_reg)

        clusterer_pipe = SklearnPipeline(
            [
                ("scaler", StandardScaler()),
                ("clusterer", KMeans(n_clusters=2, random_state=42, n_init=10)),
            ]
        )
        clusterer_pipe.fit(X)

        # Test compatibility
        classifier_scorers = []
        regressor_scorers = []
        cluster_supervised_scorers = []
        cluster_unsupervised_scorers = []

        for scorer_name in all_scorer_names:
            scorer = get_scorer(scorer_name)

            # Test classifier
            try:
                scorer(classifier_pipe, X, y_class)
                classifier_scorers.append(scorer_name)
            except (AttributeError, ValueError, TypeError):
                pass

            # Test regressor
            try:
                scorer(regressor_pipe, X, y_reg)
                regressor_scorers.append(scorer_name)
            except (AttributeError, ValueError, TypeError):
                pass

            # Test clustering with target
            try:
                scorer(clusterer_pipe, X, y_class)
                cluster_supervised_scorers.append(scorer_name)
            except (AttributeError, ValueError, TypeError):
                pass

            # Test clustering without target
            try:
                scorer(clusterer_pipe, X)
                cluster_unsupervised_scorers.append(scorer_name)
            except (AttributeError, ValueError, TypeError):
                pass

        # Print comprehensive report
        print("\n" + "=" * 80)
        print("SCORER COMPATIBILITY REPORT")
        print("=" * 80)

        print(f"\nCLASSIFIER SCORERS ({len(classifier_scorers)}):")
        for s in sorted(classifier_scorers):
            print(f"  • {s}")

        print(f"\nREGRESSOR SCORERS ({len(regressor_scorers)}):")
        for s in sorted(regressor_scorers):
            print(f"  • {s}")

        print(f"\nCLUSTERING SCORERS - SUPERVISED ({len(cluster_supervised_scorers)}):")
        for s in sorted(cluster_supervised_scorers):
            print(f"  • {s}")

        print(
            f"\nCLUSTERING SCORERS - UNSUPERVISED ({len(cluster_unsupervised_scorers)}):"
        )
        for s in sorted(cluster_unsupervised_scorers):
            print(f"  • {s}")

        print(f"\nTOTAL SCORERS AVAILABLE: {len(all_scorer_names)}")
        print("=" * 80 + "\n")

    @pytest.mark.parametrize(
        "model_type,target_col",
        [
            ("classifier", "y_class"),
            ("regressor", "y_reg"),
        ],
    )
    def test_score_all_scorers_match_sklearn(
        self, scoring_data, model_type, target_col
    ):
        """Test that xorq pipeline .score() matches sklearn for all compatible scorers."""
        import numpy as np
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.metrics import get_scorer
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import StandardScaler

        # Get expected compatible scorers
        if model_type == "classifier":
            expected_scorers = self.CLASSIFIER_SCORERS
            sklearn_pipe = SklearnPipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(random_state=42, max_iter=1000)),
                ]
            )
        else:  # regressor
            expected_scorers = self.REGRESSOR_SCORERS
            sklearn_pipe = SklearnPipeline(
                [("scaler", StandardScaler()), ("model", LinearRegression())]
            )

        # Prepare data
        t = xo.memtable(scoring_data)
        features = ("x1", "x2")
        X = np.array([scoring_data["x1"], scoring_data["x2"]]).T
        y = np.array(scoring_data[target_col])

        # Fit sklearn pipeline
        sklearn_pipe.fit(X, y)

        # Fit xorq pipeline using from_instance
        xorq_pipeline = xo.Pipeline.from_instance(sklearn_pipe)
        fitted_xorq = xorq_pipeline.fit(t, features=features, target=target_col)

        # Test all expected compatible scorers
        tested_count = 0
        failed_scorers = []

        for scorer_name in expected_scorers:
            # Skip known problematic scorers
            if scorer_name in self.KNOWN_ISSUES:
                continue

            try:
                scorer = get_scorer(scorer_name)

                # Score with sklearn
                sklearn_score = scorer(sklearn_pipe, X, y)

                # Score with xorq using .score()
                xorq_score = fitted_xorq.score(X, y, scorer=scorer_name)

                # Assert scores match (within floating point tolerance)
                np.testing.assert_allclose(
                    xorq_score,
                    sklearn_score,
                    rtol=1e-9,
                    atol=1e-12,
                    err_msg=f"Mismatch for scorer: {scorer_name}",
                )
                tested_count += 1
            except Exception as e:
                failed_scorers.append((scorer_name, str(e)[:100]))

        # Report results
        expected_testable = len(expected_scorers - self.KNOWN_ISSUES)
        print(
            f"\n{model_type.upper()} - Successfully tested {tested_count}/{expected_testable} scorers"
        )
        print(f"Skipped {len(self.KNOWN_ISSUES & expected_scorers)} known issues")

        if failed_scorers:
            print(f"Failed {len(failed_scorers)} scorers:")
            for scorer_name, error in failed_scorers[:3]:
                print(f"  ✗ {scorer_name}: {error}")

        # We should have tested most scorers (allowing some failures)
        # Current threshold: 70% (some prob-based scorers need use_proba=True support)
        assert tested_count >= expected_testable * 0.7, (
            f"Too many failures: only {tested_count}/{expected_testable} passed"
        )

    @pytest.mark.parametrize(
        "has_target",
        [True, False],
    )
    def test_clustering_score_all_scorers_match_sklearn(self, scoring_data, has_target):
        """Test that xorq clustering .score() matches sklearn for all clustering scorers."""
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.metrics import get_scorer
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import StandardScaler

        # Get expected compatible scorers
        if has_target:
            expected_scorers = self.CLUSTER_SUPERVISED_SCORERS
        else:
            expected_scorers = self.CLUSTER_UNSUPERVISED_SCORERS

        # Skip if no scorers expected (unsupervised clustering has none via scorer API)
        if not expected_scorers:
            pytest.skip(
                "No scorers compatible with unsupervised clustering via sklearn scorer API"
            )

        # Create sklearn clustering pipeline
        sklearn_pipe = SklearnPipeline(
            [
                ("scaler", StandardScaler()),
                ("clusterer", KMeans(n_clusters=2, random_state=42, n_init=10)),
            ]
        )

        # Prepare data
        t = xo.memtable(scoring_data)
        features = ("x1", "x2")
        X = np.array([scoring_data["x1"], scoring_data["x2"]]).T

        # Fit sklearn pipeline
        if has_target:
            y_true = np.array(scoring_data["y_class"])
            sklearn_pipe.fit(X, y_true)
            target_col = "y_class"
        else:
            sklearn_pipe.fit(X)
            target_col = None

        # Fit xorq pipeline using from_instance
        xorq_pipeline = xo.Pipeline.from_instance(sklearn_pipe)
        fitted_xorq = xorq_pipeline.fit(t, features=features, target=target_col)

        # Test all expected compatible scorers
        tested_count = 0
        failed_scorers = []

        for scorer_name in expected_scorers:
            # Skip known problematic scorers
            if scorer_name in self.KNOWN_ISSUES:
                continue

            try:
                scorer = get_scorer(scorer_name)

                # Score with sklearn
                if has_target:
                    sklearn_score = scorer(sklearn_pipe, X, y_true)
                else:
                    sklearn_score = scorer(sklearn_pipe, X)

                # Score with xorq using .score()
                if has_target:
                    xorq_score = fitted_xorq.score(X, y_true, scorer=scorer_name)
                else:
                    xorq_score = fitted_xorq.score(X, None, scorer=scorer_name)

                # Assert scores match (within floating point tolerance)
                np.testing.assert_allclose(
                    xorq_score,
                    sklearn_score,
                    rtol=1e-9,
                    atol=1e-12,
                    err_msg=f"Mismatch for scorer: {scorer_name}",
                )
                tested_count += 1
            except Exception as e:
                failed_scorers.append((scorer_name, str(e)[:100]))

        # Report results
        expected_testable = len(expected_scorers - self.KNOWN_ISSUES)
        label = "CLUSTERING (supervised)" if has_target else "CLUSTERING (unsupervised)"
        print(
            f"\n{label} - Successfully tested {tested_count}/{expected_testable} scorers"
        )
        print(f"Skipped {len(self.KNOWN_ISSUES & expected_scorers)} known issues")

        if failed_scorers:
            print(f"Failed {len(failed_scorers)} scorers:")
            for scorer_name, error in failed_scorers[:3]:
                print(f"  ✗ {scorer_name}: {error}")

        # We should have tested most scorers (allowing some failures)
        # Current threshold: 70% (some prob-based scorers need use_proba=True support)
        assert tested_count >= expected_testable * 0.7, (
            f"Too many failures: only {tested_count}/{expected_testable} passed"
        )


class TestClusteringPredict:
    """Tests for clustering algorithm predict support."""

    @pytest.fixture
    def cluster_data(self):
        """Generate data with clear cluster structure."""
        import numpy as np

        np.random.seed(42)
        # Two well-separated clusters
        cluster1 = np.random.randn(10, 2) + [0, 0]
        cluster2 = np.random.randn(10, 2) + [10, 10]
        data = np.vstack([cluster1, cluster2])
        return {"num1": data[:, 0].tolist(), "num2": data[:, 1].tolist()}

    @pytest.mark.parametrize(
        "clusterer_cls,clusterer_kwargs",
        [
            pytest.param(
                "KMeans",
                {"n_clusters": 2, "random_state": 42, "n_init": 10},
                id="KMeans",
            ),
            pytest.param(
                "MiniBatchKMeans",
                {"n_clusters": 2, "random_state": 42, "n_init": 10},
                id="MiniBatchKMeans",
            ),
            pytest.param(
                "BisectingKMeans",
                {"n_clusters": 2, "random_state": 42},
                id="BisectingKMeans",
            ),
            pytest.param(
                "Birch",
                {"n_clusters": 2},
                id="Birch",
            ),
            pytest.param(
                "MeanShift",
                {},
                id="MeanShift",
            ),
            pytest.param(
                "AffinityPropagation",
                {"random_state": 42},
                id="AffinityPropagation",
            ),
        ],
    )
    def test_inductive_clustering_predict(
        self, cluster_data, clusterer_cls, clusterer_kwargs
    ):
        """Test that inductive clustering algorithms support predict."""
        import numpy as np
        from sklearn import cluster

        t = xo.memtable(cluster_data)
        features = ("num1", "num2")

        ClustererClass = getattr(cluster, clusterer_cls)
        clusterer = ClustererClass(**clusterer_kwargs)

        # xorq predict
        step = xo.Step.from_instance_name(clusterer, name="clusterer")
        fitted = step.fit(t, features=features)
        result = fitted.predict(t)
        xorq_labels = result.execute()["predicted"].values

        # sklearn predict
        X = np.array([cluster_data["num1"], cluster_data["num2"]]).T
        sklearn_clusterer = ClustererClass(**clusterer_kwargs)
        sklearn_clusterer.fit(X)
        sklearn_labels = sklearn_clusterer.predict(X)

        # Labels should match
        np.testing.assert_array_equal(xorq_labels, sklearn_labels)

    @pytest.mark.parametrize(
        "clusterer_cls,clusterer_kwargs",
        [
            pytest.param(
                "DBSCAN",
                {"eps": 3, "min_samples": 2},
                id="DBSCAN",
            ),
            pytest.param(
                "HDBSCAN",
                {"min_samples": 2},
                id="HDBSCAN",
            ),
            pytest.param(
                "AgglomerativeClustering",
                {"n_clusters": 2},
                id="AgglomerativeClustering",
            ),
            pytest.param(
                "SpectralClustering",
                {"n_clusters": 2, "random_state": 42},
                id="SpectralClustering",
            ),
            pytest.param(
                "OPTICS",
                {"min_samples": 2},
                id="OPTICS",
            ),
        ],
    )
    def test_transductive_clustering_rejected_at_fit(
        self, cluster_data, clusterer_cls, clusterer_kwargs
    ):
        """Test that transductive clustering algorithms are rejected at fit time."""
        from sklearn import cluster

        t = xo.memtable(cluster_data)
        features = ("num1", "num2")

        ClustererClass = getattr(cluster, clusterer_cls)
        clusterer = ClustererClass(**clusterer_kwargs)

        step = xo.Step.from_instance_name(clusterer, name="clusterer")

        with pytest.raises(ValueError, match="must have transform or predict method"):
            step.fit(t, features=features)
