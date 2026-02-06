"""Demonstrates setting pipeline parameters dynamically and computing classification
scores with different configurations.

Traditional approach: You would use sklearn's set_params to change hyperparameters,
refit the pipeline, and call score on the test set, all operating in-memory on numpy
arrays. Repeated runs with different parameter values redo all computation from scratch,
including data loading.

With xorq: The same set_params API is available on xorq Pipelines, but execution is
deferred. Repeated runs with different parameters reuse cached data loading and
preprocessing steps, and scores are computed as expressions that can be composed or
stored.
"""

# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import xorq.api as xo
from xorq.expr.ml.pipeline_lib import (
    Pipeline,
)


def make_pipeline():
    clf = sklearn.pipeline.Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=11)),
        ]
    )
    return clf


def make_data():
    iris = load_iris(as_frame=True)
    X = iris.data[["sepal length (cm)", "sepal width (cm)"]].rename(
        columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
        }
    )
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=0
    )
    return X_train, X_test, y_train, y_test


def make_exprs(X_train, X_test, y_train, y_test):
    con = xo.connect()
    features = tuple(X_train.columns)
    target = y_train.name
    train = con.register(X_train.assign(**{target: y_train}), "train")
    test = con.register(X_test.assign(**{target: y_test}), "test")
    return train, test, features, target


def calc_scores_sklearn(X_train, X_test, y_train, y_test):
    dct = {
        weights: make_pipeline().set_params(knn__weights=weights).fit(X_train, y_train)
        for weights in ("uniform", "distance")
    }
    scores = {weights: clf.score(X_test, y_test) for weights, clf in dct.items()}
    return scores


def calc_scores_xorq(train, test, features, target):
    xorq_pipeline = Pipeline.from_instance(make_pipeline())
    dct = {
        weights: xorq_pipeline.set_params(knn__weights=weights).fit(
            train, features=features, target=target
        )
        for weights in ("uniform", "distance")
    }
    scores = {weights: clf.score_expr(test).execute() for weights, clf in dct.items()}
    return scores


X_train, X_test, y_train, y_test = make_data()
train, test, features, target = make_exprs(X_train, X_test, y_train, y_test)


if __name__ in ("__pytest_main__", "__main__"):
    scores_sklearn = calc_scores_sklearn(X_train, X_test, y_train, y_test)
    scores_xorq = calc_scores_xorq(train, test, features, target)
    assert scores_xorq == scores_sklearn
    pytest_examples_passed = True
