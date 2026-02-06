# https://scikit-learn.org/stable/_downloads/5a7e586367163444711012a4c5214817/plot_feature_selection_pipeline.py
import pandas as pd
import toolz
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

import xorq.api as xo
import xorq.expr.ml.pipeline_lib


def make_data():
    X, y = make_classification(
        n_features=20,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        n_clusters_per_class=2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return X_train, X_test, y_train, y_test


def make_exprs(X_train, X_test, y_train, y_test):
    rename_f = toolz.compose("feature_".__add__, str)
    features = tuple(map(rename_f, range(X_test.shape[1])))
    target = "target"
    train = pd.DataFrame(X_train).rename(columns=rename_f).assign(**{target: y_train})
    test = pd.DataFrame(X_test).rename(columns=rename_f).assign(**{target: y_test})
    con = xo.connect()
    train = con.register(train, "train")
    test = con.register(test, "test")
    return (train, test, features, target)


def make_sklearn_pipeline():
    anova_filter = SelectKBest(f_classif, k=3)
    clf = LinearSVC()
    anova_svm = make_pipeline(anova_filter, clf)
    return anova_svm


X_train, X_test, y_train, y_test = make_data()
(train, test, features, target) = make_exprs(X_train, X_test, y_train, y_test)
anova_svm = make_sklearn_pipeline()
xorq_pipeline = xorq.expr.ml.pipeline_lib.Pipeline.from_instance(anova_svm)
fitted_pipline = xorq_pipeline.fit(train, features=features, target=target)
predicted = fitted_pipline.predict(test)


if __name__ in ("__pytest_main__", "__main__"):
    y_pred = anova_svm.fit(X_train, y_train).predict(X_test)
    df = predicted.execute().assign(from_sklearn=y_pred)
    assert df["predicted"].equals(df["from_sklearn"])
    pytest_examples_passed = True
