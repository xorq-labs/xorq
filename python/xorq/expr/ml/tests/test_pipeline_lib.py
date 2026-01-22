import operator

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


class TestStepContainerTypes:
    """Tests for Step wrapping container types (ColumnTransformer, etc)."""

    @pytest.fixture
    def column_transformer(self):
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), ["age", "income"]),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ["gender", "region"],
                ),
            ],
            remainder="drop",
        )

    @pytest.fixture
    def sklearn_pipeline(self):
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import StandardScaler

        return SklearnPipeline([("scaler", StandardScaler())])

    def test_step_from_instance_column_transformer(self, column_transformer):
        from xorq.expr.ml.pipeline_lib import Step

        step = Step.from_instance(column_transformer, name="preprocessor")
        assert step.name == "preprocessor"
        assert step.typ.__name__ == "ColumnTransformer"

    def test_step_from_instance_standard_scaler(self):
        from xorq.expr.ml.pipeline_lib import Step

        step = Step.from_instance(sklearn.preprocessing.StandardScaler(), name="scaler")
        assert step.name == "scaler"
        assert step.typ.__name__ == "StandardScaler"

    def test_step_from_instance_sklearn_pipeline(self, sklearn_pipeline):
        from xorq.expr.ml.pipeline_lib import Step

        step = Step.from_instance(sklearn_pipeline, name="pipeline")
        assert step.name == "pipeline"
        assert step.typ.__name__ == "Pipeline"


class TestPipelineWithColumnTransformer:
    """Tests for Pipeline with ColumnTransformer as a Step."""

    @pytest.fixture
    def mixed_data_expr(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "age": [25.0, 30.0, 35.0, 40.0, 45.0],
                "income": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
                "gender": ["M", "F", "M", "F", "M"],
                "target": [0, 1, 0, 1, 0],
            }
        )
        return xo.memtable(df)

    @pytest.fixture
    def sklearn_ct_pipeline(self):
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline as SklearnPipeline
        from sklearn.preprocessing import StandardScaler

        return SklearnPipeline(
            [
                (
                    "preprocessor",
                    ColumnTransformer([("num", StandardScaler(), ["age", "income"])]),
                ),
            ]
        )

    def test_from_instance_wraps_column_transformer(self, sklearn_ct_pipeline):
        pipeline = xo.Pipeline.from_instance(sklearn_ct_pipeline)

        assert len(pipeline.steps) == 1
        assert pipeline.steps[0].typ.__name__ == "ColumnTransformer"


class TestAllowUnregistered:
    """Tests for allow_unregistered flag on Step and Pipeline from_instance."""

    @pytest.fixture
    def unregistered_transformer(self):
        from sklearn.base import BaseEstimator, TransformerMixin

        class CustomTransformer(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self

            def transform(self, X):
                return X

        return CustomTransformer()

    def test_step_from_instance_raises_for_unregistered_by_default(
        self, unregistered_transformer
    ):
        from xorq.expr.ml.pipeline_lib import Step

        with pytest.raises(ValueError, match="No Structer registration"):
            Step.from_instance(unregistered_transformer, name="custom")

    def test_step_from_instance_allows_unregistered_with_flag(
        self, unregistered_transformer
    ):
        from xorq.expr.ml.pipeline_lib import Step

        step = Step.from_instance(
            unregistered_transformer, name="custom", allow_unregistered=True
        )
        assert step.name == "custom"
        assert step.typ.__name__ == "CustomTransformer"

    def test_step_from_instance_works_for_registered_types(self):
        from xorq.expr.ml.pipeline_lib import Step

        step = Step.from_instance(sklearn.preprocessing.StandardScaler(), name="scaler")
        assert step.name == "scaler"
        assert step.typ.__name__ == "StandardScaler"

    def test_pipeline_from_instance_raises_for_unregistered_by_default(
        self, unregistered_transformer
    ):
        pipe = sklearn.pipeline.Pipeline(
            [
                ("scaler", sklearn.preprocessing.StandardScaler()),
                ("custom", unregistered_transformer),
            ]
        )

        with pytest.raises(ValueError, match="No Structer registration"):
            xo.Pipeline.from_instance(pipe)

    def test_pipeline_from_instance_allows_unregistered_with_flag(
        self, unregistered_transformer
    ):
        pipe = sklearn.pipeline.Pipeline(
            [
                ("scaler", sklearn.preprocessing.StandardScaler()),
                ("custom", unregistered_transformer),
            ]
        )

        xorq_pipe = xo.Pipeline.from_instance(pipe, allow_unregistered=True)
        assert len(xorq_pipe.steps) == 2
        assert xorq_pipe.steps[0].typ.__name__ == "StandardScaler"
        assert xorq_pipe.steps[1].typ.__name__ == "CustomTransformer"

    def test_nested_container_allows_unregistered_child_with_flag(
        self, unregistered_transformer
    ):
        from sklearn.compose import ColumnTransformer

        from xorq.expr.ml.pipeline_lib import Step

        ct = ColumnTransformer([("custom", unregistered_transformer, ["a", "b"])])

        step = Step.from_instance(ct, name="preprocessor", allow_unregistered=True)
        assert step.name == "preprocessor"
        assert step.typ.__name__ == "ColumnTransformer"
