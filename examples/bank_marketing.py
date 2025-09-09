import functools
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.base import (
    BaseEstimator,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import (
    OneHotEncoder,
)

import xorq.api as xo
import xorq.expr.selectors as s
import xorq.vendor.ibis.expr.datatypes as dt
from xorq.caching import (
    ParquetStorage,
)
from xorq.common.utils.defer_utils import deferred_read_csv
from xorq.expr.ml import (
    train_test_splits,
)
from xorq.expr.ml.fit_lib import (
    transform_sklearn_feature_names_out,
)
from xorq.expr.ml.pipeline_lib import (
    FittedPipeline,
    Step,
)
from xorq.expr.ml.structer import (
    ENCODED,
)


class OneHotStep(OneHotEncoder):
    @functools.wraps(OneHotEncoder.transform)
    def transform(self, *args, **kwargs):
        transformed = transform_sklearn_feature_names_out(super(), *args, **kwargs)
        return transformed

    @classmethod
    def get_step_f_kwargs(cls, kwargs):
        from xorq.expr.ml.fit_lib import deferred_fit_transform_sklearn

        f = deferred_fit_transform_sklearn
        kwargs = kwargs | {
            "return_type": dt.Array(dt.Struct({"key": str, "value": float})),
            "target": None,
        }
        return (f, kwargs)


class XGBoostModelExplodeEncoded(BaseEstimator):
    def __init__(self, num_boost_round=10, params=None, encoded_col=ENCODED):
        self.encoded_col = encoded_col
        self.num_boost_round = num_boost_round
        self.params = params or {
            "max_depth": 4,
            "eta": 1,
            "objective": "binary:logistic",
            "seed": 0,
        }
        self.model = None

    return_type = dt.float64

    def do_explode_encoded(self, X):
        def explode_series(series):
            (keys, values) = (
                [tuple(dct[which] for dct in lst) for lst in series]
                for which in ("key", "value")
            )
            (columns, *rest) = keys
            assert all(el == columns for el in rest)
            df = pd.DataFrame(
                values,
                index=series.index,
                columns=columns,
            )
            return df

        X = X.drop(columns=self.encoded_col).join(explode_series(X[self.encoded_col]))
        return X

    def make_dmatrix(self, X, y=None):
        X = self.do_explode_encoded(X)
        dmatrix = xgb.DMatrix(X, y)
        return dmatrix

    def fit(self, X, y):
        dtrain = self.make_dmatrix(X, y)
        self.model = xgb.train(
            self.params, dtrain, num_boost_round=self.num_boost_round
        )
        print("fitting")
        return self

    def predict(self, X):
        dmatrix = self.make_dmatrix(X)
        return self.model.predict(dmatrix)


one_hot_step = Step(
    OneHotStep,
    "one_hot_step",
    params_tuple=(("handle_unknown", "ignore"), ("drop", "first")),
)
xgbee_step = Step(
    XGBoostModelExplodeEncoded,
    name="xgbee_step",
    params_tuple=(("encoded_col", ENCODED),),
)


def make_pipeline(dataset_name, target_column, predicted_col, con=None, storage=None):
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
        # FIXME: default unique_key to s.all()
        unique_key=expr.columns,
        test_sizes=[0.5, 0.5],
        num_buckets=2,
        random_seed=42,
    )
    pattern = f"^(?!{target_column}$)"
    numeric_features = tuple(
        train_table.select(s.all_of(s.numeric(), s.matches(pattern))).columns
    )
    fitted_one_hot_step = one_hot_step.fit(
        train_table,
        features=train_table.select(s.of_type(str)).columns,
        dest_col=ENCODED,
        storage=storage,
    )
    fitted_xgbee_step = xgbee_step.fit(
        expr=train_table.mutate(fitted_one_hot_step.mutate),
        features=numeric_features + (ENCODED,),
        target=target_column,
        dest_col=predicted_col,
        storage=storage,
    )
    fitted_pipeline = FittedPipeline(
        (fitted_one_hot_step, fitted_xgbee_step),
        train_table,
    )
    return (train_table, test_table, fitted_pipeline)


con = xo.connect()
storage = ParquetStorage(
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
    dataset_name, target_column, predicted_col, con=con, storage=storage
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
