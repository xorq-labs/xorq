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
