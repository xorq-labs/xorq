"""Skill scaffold: penguins_demo
Minimal multi-engine example (penguins template) for demonstrations.
"""

import xorq.api as xo
# https://inria.github.io/scikit-learn-mooc/python_scripts/trees_classification.html
import sklearn
from sklearn.linear_model import LogisticRegression

import xorq.api as xo
from xorq.caching import ParquetCache
from xorq.expr.ml.pipeline_lib import (
    Pipeline,
)


features = ("bill_length_mm", "bill_depth_mm")
target = "species"
data_url = "https://storage.googleapis.com/letsql-pins/penguins/20250703T145709Z-c3cde/penguins.parquet"


def gen_splits(expr, test_size=.2, random_seed=42, **split_kwargs):
    # inject and drop row number
    assert "test_sizes" not in split_kwargs
    assert isinstance(test_size, float)
    yield from (
        expr
        for expr in xo.train_test_splits(
            expr,
            test_sizes=test_size,
            random_seed=random_seed,
            **split_kwargs,
        )
    )


def get_penguins_splits(storage=None, **split_kwargs):
    t = (
        xo.deferred_read_parquet(
            con=xo.duckdb.connect(),
            path=data_url,
            table_name="t",
        )
        .select(features+(target,))
        .drop_null()
    )
    (train, test) = (
        expr
        .cache(storage or ParquetCache.from_kwargs())
        for expr in gen_splits(t, **split_kwargs)
    )
    return (train, test)


def make_pipeline(params=()):
    clf = (
        sklearn.pipeline.Pipeline(
            steps=[
                ("logistic", LogisticRegression()),
            ]
        )
        .set_params(**dict(params))
    )
    return clf


def fit_and_score_sklearn_pipeline(pipeline, train, test):
    (
        (X_train, y_train),
        (X_test, y_test),
    ) = (
        expr.execute().pipe(lambda t: (
            t.filter(regex=f"^(?!{target})"),
            t.filter(regex=f"^{target}"),
        ))
        for expr in (train, test)
    )
    clf = pipeline.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return clf, score


params = {
    "logistic__C": 1E-4,
    "logistic__penalty": "l2",
}
(train, test) = get_penguins_splits()
sklearn_pipeline = make_pipeline(params=params)
xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)
# still no work done: deferred fit expression
fitted_pipeline = xorq_pipeline.fit(train, features=features, target=target)
#
train_predicted = fitted_pipeline.fitted_steps[-1].predicted
expr = test_predicted = fitted_pipeline.predict(test[features])


if __name__ == "__pytest_main__":
    clf, score_sklearn = fit_and_score_sklearn_pipeline(sklearn_pipeline, train, test)
    score_xorq = fitted_pipeline.score_expr(test)
    assert score_xorq == score_sklearn
