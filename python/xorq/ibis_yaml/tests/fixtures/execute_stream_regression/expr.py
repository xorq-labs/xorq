# https://scikit-learn.org/stable/auto_examples/compose/plot_digits_pipe.html#sphx-glr-auto-examples-compose-plot-digits-pipe-py
import sklearn
from sklearn.datasets._base import resources
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import xorq.api as xo
from xorq.caching import ParquetCache
from xorq.expr.ml.pipeline_lib import (
    Pipeline,
)
from xorq.expr.ml.structer import (
    Structer,
    structer_from_instance,
)


features = tuple(f"feature_{i}" for i in range(64))
target = "target"


def gen_splits(expr, test_size=0.2, random_seed=42, **split_kwargs):
    # inject and drop row number
    assert "test_sizes" not in split_kwargs
    assert isinstance(test_size, float)
    row_number = "row_number"
    yield from (
        expr.drop(row_number)
        for expr in xo.train_test_splits(
            expr.mutate(**{row_number: xo.row_number()}),
            unique_key=row_number,
            test_sizes=test_size,
            random_seed=random_seed,
            **split_kwargs,
        )
    )


def get_digits_splits(cache=None, **split_kwargs):
    t = xo.deferred_read_csv(
        con=xo.duckdb.connect(),
        path=str(resources.files("sklearn.datasets.data") / "digits.csv.gz"),
        schema=xo.schema(dict.fromkeys((*features, target), float)),
        header=False,
        table_name="t",
    )
    (train, test) = (
        expr.cache(cache or ParquetCache.from_kwargs())
        for expr in gen_splits(t, **split_kwargs)
    )
    return (train, test)


def make_pipeline(params=()):
    clf = sklearn.pipeline.Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=5)),
            ("logistic", LogisticRegression(max_iter=10000, tol=0.1, C=1)),
        ]
    ).set_params(**dict(params))
    return clf


def fit_and_score_sklearn_pipeline(pipeline, train, test):
    (
        (X_train, y_train),
        (X_test, y_test),
    ) = (
        expr.execute().pipe(
            lambda t: (
                t.filter(regex=f"^(?!{target})"),
                t.filter(regex=f"^{target}"),
            )
        )
        for expr in (train, test)
    )
    clf = pipeline.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return clf, score


@structer_from_instance.register(PCA)
def _(instance, expr, features=None):
    # complex return types must have their schema pre-defined
    structer = Structer.from_n_typ_prefix(
        n=instance.n_components or len(expr.schema()),
    )
    return structer


params = {
    "pca__n_components": 5,
    "logistic__C": 1e-4,
    "logistic__penalty": "l2",
}
(train, test) = get_digits_splits()
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
