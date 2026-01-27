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

    def test_non_kv_deeply_nested_pipeline(self):  # noqa: C901
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


class TestIsContainerTransformer:
    """Tests for _is_container_transformer helper function."""

    def test_column_transformer_is_container(self):
        """Test ColumnTransformer is recognized as a container."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.pipeline_lib import _is_container_transformer

        ct = ColumnTransformer([("scaler", StandardScaler(), ["a"])])
        assert _is_container_transformer(ct) is True

    def test_feature_union_is_container(self):
        """Test FeatureUnion is recognized as a container."""
        from sklearn.pipeline import FeatureUnion
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.pipeline_lib import _is_container_transformer

        fu = FeatureUnion([("scaler", StandardScaler())])
        assert _is_container_transformer(fu) is True

    def test_sklearn_pipeline_is_container(self):
        """Test sklearn Pipeline is recognized as a container."""
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.pipeline_lib import _is_container_transformer

        pipe = SklearnPipeline([("scaler", StandardScaler())])
        assert _is_container_transformer(pipe) is True

    def test_simple_transformer_is_not_container(self):
        """Test simple transformers are not recognized as containers."""
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        from xorq.expr.ml.pipeline_lib import _is_container_transformer

        assert _is_container_transformer(StandardScaler()) is False
        assert _is_container_transformer(OneHotEncoder()) is False

    def test_estimator_is_not_container(self):
        """Test estimators are not recognized as containers."""
        from sklearn.linear_model import LinearRegression

        from xorq.expr.ml.pipeline_lib import _is_container_transformer

        assert _is_container_transformer(LinearRegression()) is False


class TestAnalyzeChildStructer:
    """Tests for _analyze_child_structer helper function."""

    def test_drop_returns_none(self):
        """Test 'drop' transformer returns None."""
        from xorq.expr.ml.pipeline_lib import _analyze_child_structer

        result = _analyze_child_structer("drop", None, None)
        assert result is None

    def test_passthrough_returns_none(self):
        """Test 'passthrough' transformer returns None."""
        from xorq.expr.ml.pipeline_lib import _analyze_child_structer

        result = _analyze_child_structer("passthrough", None, None)
        assert result is None

    def test_known_schema_transformer(self):
        """Test known-schema transformer is correctly identified."""
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.pipeline_lib import _analyze_child_structer

        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = _analyze_child_structer(StandardScaler(), t, ("a", "b"))

        assert result is not None
        assert result["is_kv_encoded"] is False
        assert "structer" in result

    def test_kv_encoded_transformer(self):
        """Test KV-encoded transformer is correctly identified."""
        from sklearn.preprocessing import OneHotEncoder

        from xorq.expr.ml.pipeline_lib import _analyze_child_structer

        t = xo.memtable({"cat": ["x", "y", "z"]})
        result = _analyze_child_structer(OneHotEncoder(), t, ("cat",))

        assert result is not None
        assert result["is_kv_encoded"] is True
        assert "structer" in result

    def test_nested_container_with_kv_child(self):
        """Test nested container with KV-encoded child is correctly identified."""
        from sklearn.pipeline import FeatureUnion
        from sklearn.preprocessing import OneHotEncoder

        from xorq.expr.ml.pipeline_lib import _analyze_child_structer

        t = xo.memtable({"cat": ["x", "y", "z"]})
        fu = FeatureUnion([("encoder", OneHotEncoder())])
        result = _analyze_child_structer(fu, t, ("cat",))

        assert result is not None
        assert result["is_kv_encoded"] is True
        assert "child_info" in result

    def test_nested_container_with_known_children(self):
        """Test nested container with known-schema children."""
        from sklearn.pipeline import FeatureUnion
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.pipeline_lib import _analyze_child_structer

        t = xo.memtable({"a": [1.0, 2.0]})
        fu = FeatureUnion([("scaler", StandardScaler())])
        result = _analyze_child_structer(fu, t, ("a",))

        assert result is not None
        assert result["is_kv_encoded"] is False
        assert "child_info" in result


class TestAnalyzeContainer:
    """Tests for _analyze_container helper function."""

    def test_column_transformer_children(self):
        """Test ColumnTransformer child analysis."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        from xorq.expr.ml.pipeline_lib import _analyze_container

        t = xo.memtable({"num": [1.0, 2.0], "cat": ["x", "y"]})
        ct = ColumnTransformer(
            [
                ("scaler", StandardScaler(), ["num"]),
                ("encoder", OneHotEncoder(), ["cat"]),
            ]
        )
        children = _analyze_container(ct, t)

        assert len(children) == 2
        # Find by name
        scaler_child = next(c for c in children if c["name"] == "scaler")
        encoder_child = next(c for c in children if c["name"] == "encoder")

        assert scaler_child["is_kv_encoded"] is False
        assert scaler_child["columns"] == ("num",)
        assert encoder_child["is_kv_encoded"] is True
        assert encoder_child["columns"] == ("cat",)

    def test_column_transformer_drop_ignored(self):
        """Test ColumnTransformer 'drop' transformers are ignored."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.pipeline_lib import _analyze_container

        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        ct = ColumnTransformer(
            [
                ("scaler", StandardScaler(), ["a"]),
                ("dropped", "drop", ["b"]),
            ]
        )
        children = _analyze_container(ct, t)

        assert len(children) == 1
        assert children[0]["name"] == "scaler"

    def test_feature_union_children(self):
        """Test FeatureUnion child analysis."""
        from sklearn.pipeline import FeatureUnion
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        from xorq.expr.ml.pipeline_lib import _analyze_container

        t = xo.memtable({"a": [1.0, 2.0]})
        fu = FeatureUnion(
            [
                ("scaler", StandardScaler()),
                ("encoder", OneHotEncoder()),
            ]
        )
        children = _analyze_container(fu, t, features=("a",))

        assert len(children) == 2
        scaler_child = next(c for c in children if c["name"] == "scaler")
        encoder_child = next(c for c in children if c["name"] == "encoder")

        assert scaler_child["is_kv_encoded"] is False
        assert encoder_child["is_kv_encoded"] is True

    def test_sklearn_pipeline_children(self):
        """Test sklearn Pipeline child analysis."""
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        from xorq.expr.ml.pipeline_lib import _analyze_container

        t = xo.memtable({"a": [1.0, 2.0]})
        pipe = SklearnPipeline(
            [
                ("scaler", StandardScaler()),
                ("encoder", OneHotEncoder()),
            ]
        )
        children = _analyze_container(pipe, t, features=("a",))

        assert len(children) == 2
        assert children[0]["name"] == "scaler"
        assert children[0]["is_kv_encoded"] is False
        assert children[1]["name"] == "encoder"
        assert children[1]["is_kv_encoded"] is True

    def test_pipeline_passthrough_step(self):
        """Test Pipeline with passthrough step."""
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.pipeline_lib import _analyze_container

        t = xo.memtable({"a": [1.0, 2.0]})
        pipe = SklearnPipeline(
            [
                ("pass", "passthrough"),
                ("scaler", StandardScaler()),
            ]
        )
        children = _analyze_container(pipe, t, features=("a",))

        assert len(children) == 2
        assert children[0]["name"] == "pass"
        assert children[0]["is_kv_encoded"] is False
        assert children[1]["name"] == "scaler"


class TestComplexStep:
    """Tests for ComplexStep class."""

    def test_from_column_transformer(self):
        """Test ComplexStep creation from ColumnTransformer."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        from xorq.expr.ml.pipeline_lib import ComplexStep

        t = xo.memtable({"num": [1.0, 2.0], "cat": ["x", "y"]})
        ct = ColumnTransformer(
            [
                ("scaler", StandardScaler(), ["num"]),
                ("encoder", OneHotEncoder(), ["cat"]),
            ]
        )
        step = ComplexStep.from_instance_name(ct, name="my_ct", expr=t)

        assert step.typ == ColumnTransformer
        assert step.name == "my_ct"
        assert len(step.child_info) == 2
        assert step.kv_child_names == ("encoder",)
        assert step.known_child_names == ("scaler",)
        assert step.is_hybrid is True

    def test_from_feature_union(self):
        """Test ComplexStep creation from FeatureUnion."""
        from sklearn.pipeline import FeatureUnion
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.pipeline_lib import ComplexStep

        t = xo.memtable({"a": [1.0, 2.0]})
        fu = FeatureUnion([("scaler", StandardScaler())])
        step = ComplexStep.from_instance_name(fu, name="my_fu", expr=t, features=("a",))

        assert step.typ == FeatureUnion
        assert step.name == "my_fu"
        assert step.is_all_known is True
        assert step.is_hybrid is False

    def test_from_simple_transformer_raises(self):
        """Test ComplexStep rejects simple transformers."""
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.pipeline_lib import ComplexStep

        with pytest.raises(ValueError, match="container transformer"):
            ComplexStep.from_instance_name(StandardScaler())

    def test_is_all_kv(self):
        """Test is_all_kv property."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder

        from xorq.expr.ml.pipeline_lib import ComplexStep

        t = xo.memtable({"cat1": ["x", "y"], "cat2": ["a", "b"]})
        ct = ColumnTransformer(
            [
                ("enc1", OneHotEncoder(), ["cat1"]),
                ("enc2", OneHotEncoder(), ["cat2"]),
            ]
        )
        step = ComplexStep.from_instance_name(ct, expr=t)

        assert step.is_all_kv is True
        assert step.is_hybrid is False
        assert step.is_all_known is False

    def test_is_all_known(self):
        """Test is_all_known property."""
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.pipeline_lib import ComplexStep

        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        ct = ColumnTransformer(
            [
                ("scaler", StandardScaler(), ["a"]),
                ("imputer", SimpleImputer(), ["b"]),
            ]
        )
        step = ComplexStep.from_instance_name(ct, expr=t)

        assert step.is_all_known is True
        assert step.is_hybrid is False
        assert step.is_all_kv is False

    def test_deferred_child_analysis(self):
        """Test ComplexStep with deferred child analysis (no expr)."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.pipeline_lib import ComplexStep

        ct = ColumnTransformer([("scaler", StandardScaler(), ["a"])])
        step = ComplexStep.from_instance_name(ct, name="deferred")

        # No child_info yet since no expr was provided
        assert step.child_info == ()
        assert step.kv_child_names == ()
        assert step.known_child_names == ()

    def test_auto_generated_name(self):
        """Test ComplexStep auto-generates name if not provided."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.pipeline_lib import ComplexStep

        ct = ColumnTransformer([("scaler", StandardScaler(), ["a"])])
        step = ComplexStep.from_instance_name(ct)

        assert step.name.startswith("columntransformer_")


class TestComplexFittedStep:
    """Tests for ComplexFittedStep class."""

    def test_fitted_step_from_complex_step(self):
        """Test ComplexFittedStep creation via ComplexStep.fit()."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        from xorq.expr.ml.pipeline_lib import ComplexStep

        t = xo.memtable({"num": [1.0, 2.0, 3.0], "cat": ["x", "y", "x"]})
        ct = ColumnTransformer(
            [
                ("scaler", StandardScaler(), ["num"]),
                ("encoder", OneHotEncoder(), ["cat"]),
            ]
        )
        step = ComplexStep.from_instance_name(ct, expr=t)
        fitted = step.fit(t)

        assert fitted.features == ("num", "cat")
        assert fitted.is_hybrid is True
        assert fitted.kv_child_names == ("encoder",)

    def test_fitted_step_deferred_analysis(self):
        """Test ComplexFittedStep performs deferred analysis if needed."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.pipeline_lib import ComplexStep

        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        ct = ColumnTransformer([("scaler", StandardScaler(), ["a", "b"])])

        # Create step without expr (deferred analysis)
        step = ComplexStep.from_instance_name(ct)
        assert step.child_info == ()

        # Fit should trigger analysis
        fitted = step.fit(t)
        assert len(fitted.step.child_info) == 1
        assert fitted.step.is_all_known is True

    def test_fitted_step_structer(self):
        """Test ComplexFittedStep structer property."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        from xorq.expr.ml.pipeline_lib import ComplexStep

        t = xo.memtable({"num": [1.0, 2.0], "cat": ["x", "y"]})
        ct = ColumnTransformer(
            [
                ("scaler", StandardScaler(), ["num"]),
                ("encoder", OneHotEncoder(), ["cat"]),
            ]
        )
        step = ComplexStep.from_instance_name(ct, expr=t)
        fitted = step.fit(t)

        structer = fitted.structer
        assert structer.is_hybrid is True
        assert "encoder" in structer.kv_child_names

    def test_fitted_step_is_transform(self):
        """Test ComplexFittedStep is_transform property."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.pipeline_lib import ComplexStep

        t = xo.memtable({"a": [1.0, 2.0]})
        ct = ColumnTransformer([("scaler", StandardScaler(), ["a"])])
        step = ComplexStep.from_instance_name(ct, expr=t)
        fitted = step.fit(t)

        assert fitted.is_transform is True

    def test_fitted_step_with_explicit_features(self):
        """Test ComplexFittedStep with explicit features."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        from xorq.expr.ml.pipeline_lib import ComplexStep

        t = xo.memtable({"a": [1.0, 2.0], "b": [3.0, 4.0], "y": [0, 1]})
        ct = ColumnTransformer([("scaler", StandardScaler(), ["a", "b"])])
        step = ComplexStep.from_instance_name(ct)
        # Explicitly pass features excluding target
        fitted = step.fit(t, features=("a", "b"), target="y")

        assert fitted.features == ("a", "b")
        assert "y" not in fitted.features
        assert fitted.target == "y"
