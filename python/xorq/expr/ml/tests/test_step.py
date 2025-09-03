import pytest

import xorq.api as xo
from xorq.caching import (
    ParquetStorage,
    SourceStorage,
)
from xorq.expr.ml.pipeline_lib import (
    Pipeline,
)


sklearn = pytest.importorskip("sklearn")
load_iris = sklearn.datasets.load_iris
train_test_split = sklearn.model_selection.train_test_split
KNeighborsClassifier = sklearn.neighbors.KNeighborsClassifier
StandardScaler = sklearn.preprocessing.StandardScaler


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


@pytest.mark.parametrize("storage_cls", (None, ParquetStorage, SourceStorage))
def test_fittedstep_model(storage_cls):
    storage = storage_cls() if storage_cls else storage_cls
    X_train, X_test, y_train, y_test = make_data()
    train, test, features, target = make_exprs(X_train, X_test, y_train, y_test)
    xorq_pipeline = Pipeline.from_instance(make_pipeline())
    fitted_pipeline = xorq_pipeline.fit(
        train, features=features, target=target, storage=storage
    )
    for fitted_step in (*fitted_pipeline.transform_steps, fitted_pipeline.predict_step):
        fitted_step.model
