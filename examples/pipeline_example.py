"""Full ML pipeline with StandardScaler and KNeighborsClassifier, comparing a manual
deferred approach against the Pipeline API.

Traditional approach: You would build an sklearn Pipeline with fit/transform/predict
calls, operating entirely in-memory on pandas DataFrames. There is no caching of
intermediate results, so rerunning the pipeline repeats all computation. Moving data
between backends (e.g., from a database to local memory) requires manual extraction.

With xorq: The Pipeline API wraps sklearn estimators with deferred execution, so
transformations and predictions are built as expressions before being materialized.
Intermediate results are automatically cached, and data loading is backend-agnostic,
meaning the same pipeline code works whether data comes from DuckDB, Postgres, or
pandas.
"""

import operator

import sklearn
import toolz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import xorq.api as xo
from xorq.expr.ml import (
    deferred_fit_predict_sklearn,
    deferred_fit_transform_sklearn_struct,
    train_test_splits,
)
from xorq.expr.ml.enums import ResponseMethod
from xorq.expr.ml.pipeline_lib import (
    Pipeline,
)


STRUCTED = "structed"
ORIGINAL_ROW = "original_row"
PREDICTED = str(ResponseMethod.PREDICT)


@toolz.curry
def as_struct(expr, name=None):
    struct = xo.struct({c: expr[c] for c in expr.columns})
    if name:
        struct = struct.name(name)
    return struct


def make_manual_expr(train, test, features, target, scaler_step, kneighbor_step):
    (_, scaler_instance), (_, kneighbor_instance) = (scaler_step, kneighbor_step)
    deferred_transform = deferred_fit_transform_sklearn_struct(
        train,
        features=features,
        cls=scaler_instance.__class__,
        params=tuple(scaler_instance.get_params().items()),
    ).deferred_other
    transformed = train.select(
        deferred_transform.on_expr(train).name(STRUCTED),
        target,
    ).unpack(STRUCTED)
    deferred_predict = deferred_fit_predict_sklearn(
        transformed,
        target=target,
        features=features,
        cls=kneighbor_instance.__class__,
        return_type=iris[target].type(),
        params=tuple(kneighbor_instance.get_params().items()),
    ).deferred_other
    test_predicted = (
        test.mutate(as_struct(name=ORIGINAL_ROW))
        .select(
            deferred_transform.on_expr(test).name(STRUCTED),
            ORIGINAL_ROW,
        )
        .unpack(STRUCTED)
        .mutate(**{PREDICTED: deferred_predict.on_expr})
        .select(PREDICTED, ORIGINAL_ROW)
        .unpack(ORIGINAL_ROW)
    )
    return test_predicted


def make_pipeline_expr(train, test, features, target, sklearn_pipeline):
    xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)
    fitted_pipeline = xorq_pipeline.fit(train, features=features, target=target)
    test_predicted = (
        test.mutate(as_struct(name=ORIGINAL_ROW))
        # we have into backends sprinkled in here: unclear if we need that for predict
        .pipe(fitted_pipeline.predict)
        .drop(target)
        .unpack(ORIGINAL_ROW)
    )
    return test_predicted


def train_predict_sklearn(sklearn_pipeline, train, test, features, target):
    train_df = train.execute()
    sklearn_pipeline.fit(X=train_df[list(features)], y=train_df[target])
    df = test.execute().assign(
        **{
            PREDICTED: toolz.compose(
                sklearn_pipeline.predict,
                operator.itemgetter(list(features)),
            ),
        }
    )
    return df


(scaler_step, kneighbor_step) = steps = (
    ("scaler", StandardScaler()),
    ("k-neighbors", KNeighborsClassifier(n_neighbors=11)),
)
sklearn_pipeline = sklearn.pipeline.Pipeline(steps)


target = "species"
iris = xo.examples.iris.fetch()
features = tuple(iris.drop(target).schema())
(train, test) = train_test_splits(iris, 0.2)


test_predicted_manual = make_manual_expr(
    train,
    test,
    features,
    target,
    scaler_step,
    kneighbor_step,
)
test_predicted_pipeline = make_pipeline_expr(
    train, test, features, target, sklearn_pipeline
)


if __name__ in ("__pytest_main__", "__main__"):
    sklearn_df = train_predict_sklearn(sklearn_pipeline, train, test, features, target)
    manual_df = test_predicted_manual.execute().reindex_like(sklearn_df)
    pipeline_df = test_predicted_pipeline.execute().reindex_like(sklearn_df)
    assert manual_df.equals(sklearn_df)
    assert pipeline_df.equals(sklearn_df)
    pytest_examples_passed = True
