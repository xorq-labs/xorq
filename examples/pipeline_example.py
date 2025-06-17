import operator

import sklearn
import toolz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import xorq as xo
from xorq.expr.ml import (
    deferred_fit_predict_sklearn,
    deferred_fit_transform_sklearn_struct,
    train_test_splits,
)
from xorq.expr.ml.pipeline_lib import (
    Pipeline,
)


UNIQUE_KEY = "unique_key"
STRUCTED = "structed"
ORIGINAL_ROW = "original_row"
PREDICTED = "predicted"


@toolz.curry
def as_struct(expr, name=None):
    struct = xo.struct({c: expr[c] for c in expr.columns})
    if name:
        struct = struct.name(name)
    return struct


def make_manual_expr(train, test, features, target, scaler_step, kneighbor_step):
    (_, scaler_instance), (_, kneighbor_instance) = (scaler_step, kneighbor_step)
    *_, deferred_transform = deferred_fit_transform_sklearn_struct(
        train,
        features=features,
        cls=scaler_instance.__class__,
        params=tuple(scaler_instance.get_params().items()),
    )
    transformed = train.select(
        deferred_transform.on_expr(train).name(STRUCTED),
        target,
    ).unpack(STRUCTED)
    *_, deferred_predict = deferred_fit_predict_sklearn(
        transformed,
        target=target,
        features=features,
        cls=kneighbor_instance.__class__,
        return_type=iris[target].type(),
        params=tuple(kneighbor_instance.get_params().items()),
    )
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
(train, test) = (
    expr.drop(UNIQUE_KEY)
    for expr in train_test_splits(
        iris.mutate(**{UNIQUE_KEY: xo.row_number()}), UNIQUE_KEY, 0.2
    )
)


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


if __name__ == "__pytest_main__":
    sklearn_df = train_predict_sklearn(sklearn_pipeline, train, test, features, target)
    manual_df = test_predicted_manual.execute().reindex_like(sklearn_df)
    pipeline_df = test_predicted_pipeline.execute().reindex_like(sklearn_df)
    assert manual_df.equals(sklearn_df)
    assert pipeline_df.equals(sklearn_df)
    pytest_examples_passed = True
