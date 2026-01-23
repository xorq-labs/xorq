from pathlib import Path

import sklearn.pipeline
import xgboost as xgb
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder

import xorq.api as xo
import xorq.expr.selectors as s
import xorq.vendor.ibis.expr.datatypes as dt
from xorq.caching import ParquetCache
from xorq.common.utils.defer_utils import deferred_read_csv
from xorq.expr.ml import train_test_splits
from xorq.expr.ml.pipeline_lib import Pipeline


class XGBoostClassifier(BaseEstimator):
    def __init__(self, num_boost_round=10, max_depth=4, eta=1, seed=0):
        self.num_boost_round = num_boost_round
        self.max_depth = max_depth
        self.eta = eta
        self.seed = seed
        self.model = None

    return_type = dt.float64

    def fit(self, X, y):
        params = {
            "max_depth": self.max_depth,
            "eta": self.eta,
            "objective": "binary:logistic",
            "seed": self.seed,
        }
        dtrain = xgb.DMatrix(X, y)
        self.model = xgb.train(params, dtrain, num_boost_round=self.num_boost_round)
        return self

    def predict(self, X):
        dmatrix = xgb.DMatrix(X)
        return self.model.predict(dmatrix)


sklearn_pipeline = sklearn.pipeline.make_pipeline(
    OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
    XGBoostClassifier(),
)


def make_pipeline(dataset_name, target_column, predicted_col, con=None, cache=None):
    con = con or xo.connect()
    expr = deferred_read_csv(
        path=xo.options.pins.get_path(dataset_name),
        con=con,
    ).mutate(
        **{
            target_column: (xo._[target_column] == "yes").cast("int"),
        }
    )
    train_table, test_table = expr.pipe(
        train_test_splits,
        test_sizes=[0.5, 0.5],
        num_buckets=2,
        random_seed=42,
    )
    pattern = f"^(?!{target_column}$)"
    features = tuple(
        train_table.select(
            s.any_of(s.numeric(), s.of_type(str)) & s.matches(pattern)
        ).columns
    )
    xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)
    fitted_pipeline = xorq_pipeline.fit(
        train_table,
        features=features,
        target=target_column,
        cache=cache,
    )
    return (train_table, test_table, fitted_pipeline)


con = xo.connect()
cache = ParquetCache.from_kwargs(
    source=con,
    relative_path="./tmp-cache",
    base_path=Path(".").absolute(),
)
(dataset_name, target_column, predicted_col) = (
    "bank-marketing",
    "deposit",
    "predicted",
)
train_table, test_table, fitted_pipeline = make_pipeline(
    dataset_name, target_column, predicted_col, con=con, cache=cache
)
encoded_test = fitted_pipeline.transform(test_table)
predicted_test = fitted_pipeline.predict(test_table)


if __name__ == "__pytest_main__":
    predictions_df = predicted_test.execute()
    binary_predictions = (predictions_df[predicted_col] >= 0.5).astype(int)

    cm = confusion_matrix(
        predictions_df[target_column],
        binary_predictions,
    )
    print("\nConfusion Matrix:")
    print(f"TN: {cm[0, 0]}, FP: {cm[0, 1]}")
    print(f"FN: {cm[1, 0]}, TP: {cm[1, 1]}")

    auc = roc_auc_score(predictions_df[target_column], predictions_df[predicted_col])
    print(f"\nAUC Score: {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(predictions_df[target_column], binary_predictions))
    pytest_examples_passed = True
