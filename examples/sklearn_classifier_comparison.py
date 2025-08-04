# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pandas as pd
from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import xorq as xo
from xorq.expr.ml import Pipeline
from xorq.expr.ml.pipeline_lib import (
    get_predict_return_type,
    get_target_type,
    step_typ_to_f,
)


target = "target"
feature_prefix = "feature_"


def make_linearly_separable():
    X, y = make_classification(
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=1,
        n_clusters_per_class=1,
    )
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    return (X, y)


def make_exprs(X_train, y_train, X_test, y_test):
    con = xo.connect()
    (train, test) = (
        con.register(
            pd.DataFrame(X)
            .rename(columns=(feature_prefix + "{}").format)
            .assign(**{target: y}),
            name,
        )
        for (X, y, name) in (
            (X_train, y_train, "train"),
            (X_test, y_test, "test"),
        )
    )
    features = train.select(xo.selectors.startswith(feature_prefix)).columns
    return (train, test, features)


classifiers = {
    "Nearest Neighbors": KNeighborsClassifier(3),
    "Linear SVM": SVC(kernel="linear", C=0.025, random_state=42),
    "RBF SVM": SVC(gamma=2, C=1, random_state=42),
    # # issue with hashability of RBF
    # "Gaussian Process": GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=42
    ),
    "Neural Net": MLPClassifier(alpha=1, max_iter=1000, random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "QDA": QuadraticDiscriminantAnalysis(),
}


for typ in set(clf.__class__ for clf in classifiers.values()).difference(step_typ_to_f):
    get_predict_return_type.register(typ, get_target_type)


datasets = {
    "moons": make_moons(noise=0.3, random_state=0),
    "circles": make_circles(noise=0.2, factor=0.5, random_state=1),
    "linearly separable": make_linearly_separable(),
}


dct = {}
for ds_name, (X, y) in datasets.items():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    (train, test, features) = make_exprs(X_train, y_train, X_test, y_test)
    for clf_name, clf in classifiers.items():
        sklearn_pipeline = make_pipeline(StandardScaler(), clf).fit(X_train, y_train)
        sklearn_score = sklearn_pipeline.score(X_test, y_test)
        xorq_pipeline = Pipeline.from_instance(sklearn_pipeline).fit(
            train, features=features, target=target
        )
        xorq_score = xorq_pipeline.score_expr(test)
        dct[(ds_name, clf_name)] = {
            "sklearn_pipeline": sklearn_pipeline,
            "sklearn_score": sklearn_score,
            "xorq_pipeline": xorq_pipeline,
            "xorq_score": xorq_score,
        }
