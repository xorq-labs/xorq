import functools

import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

import xorq as xo
import xorq.selectors as s
import xorq.vendor.ibis.expr.datatypes as dt
from xorq.common.utils.defer_utils import deferred_read_csv
from xorq.common.utils.toolz_utils import curry
from xorq.expr.ml import (
    deferred_fit_predict_sklearn,
    deferred_fit_transform,
    train_test_splits,
)


def fit(
    df,
    cls=functools.partial(OneHotEncoder, handle_unknown="ignore", drop="first"),
    features=slice(None),
):
    model = cls().fit(df[features])
    return model


@curry
def transform(model, df, features=slice(None)):
    names = model.get_feature_names_out()
    return pd.Series(
        (
            tuple({"key": key, "value": float(value)} for key, value in zip(names, row))
            for row in model.transform(df[features]).toarray()
        )
    )


return_type = dt.Array(dt.Struct({"key": str, "value": float}))
deferred_one_hot = deferred_fit_transform(
    fit=fit,
    transform=transform,
    return_type=return_type,
)


class XGBoostModelExplodeEncoded:
    def __init__(self, encoded_col, num_boost_round=10, params=None):
        self.encoded_col = encoded_col
        self.num_boost_round = num_boost_round
        self.params = params or {
            "max_depth": 4,
            "eta": 1,
            "objective": "binary:logistic",
            "seed": 0,
        }
        self.model = None

    def do_explode_encoded(self, X):
        X = X.drop(columns=self.encoded_col).join(
            X[self.encoded_col].apply(
                lambda lst: pd.Series({dct["key"]: dct["value"] for dct in lst})
            )
        )
        return X

    def fit(self, X, y):
        X = self.do_explode_encoded(X)
        dtrain = xgb.DMatrix(X, y)
        self.model = xgb.train(
            self.params, dtrain, num_boost_round=self.num_boost_round
        )
        return self

    def predict(self, X):
        X = self.do_explode_encoded(X)
        dmatrix = xgb.DMatrix(X)
        return self.model.predict(dmatrix)


def make_pipeline_exprs(dataset_name, target_column, predicted_col):
    ROW_NUMBER = "row_number"
    ENCODED = "encoded"

    con = xo.connect()
    train_table, test_table = (
        expr.drop(ROW_NUMBER)
        for expr in (
            deferred_read_csv(
                path=xo.options.pins.get_path(dataset_name),
                con=con,
            )
            .mutate(
                **{
                    target_column: (xo._[target_column] == "yes").cast("int"),
                    ROW_NUMBER: xo.row_number(),
                }
            )
            .pipe(
                train_test_splits,
                unique_key=ROW_NUMBER,
                test_sizes=[0.5, 0.5],
                num_buckets=2,
                random_seed=42,
            )
        )
    )

    deferred_encoder, model_udaf, deferred_encode = deferred_one_hot(
        train_table,
        features=train_table.select(s.of_type(str)).columns,
    )
    (encoded_train, encoded_test) = (
        expr.mutate(**{ENCODED: deferred_encode.on_expr})
        for expr in (train_table, test_table)
    )

    numeric_features = [
        col
        for col in encoded_train.select(s.numeric()).columns
        if col != target_column and col != target_column + "_yes"
    ]
    deferred_model, model_udaf, deferred_predict = deferred_fit_predict_sklearn(
        expr=encoded_train,
        target=target_column,
        features=numeric_features + [ENCODED],
        cls=functools.partial(XGBoostModelExplodeEncoded, encoded_col=ENCODED),
        return_type=dt.float64,
        name_infix="xgb_prediction",
    )
    predictions = encoded_test.mutate(
        **{predicted_col: deferred_predict.on_expr(encoded_test)}
    ).drop(ENCODED)

    return {
        "encoded_train": encoded_train,
        "encoded_test": encoded_test,
        "predictions": predictions,
        "encoder": deferred_encoder,
        "model": deferred_model,
    }


(dataset_name, target_column, predicted_col) = (
    "bank-marketing",
    "deposit",
    "predicted",
)
results = make_pipeline_exprs(dataset_name, target_column, predicted_col)
encoded_test = results["encoded_test"]


if __name__ == "__pytest_main__":
    predictions_df = results["predictions"].execute()
    binary_predictions = (predictions_df[predicted_col] >= 0.5).astype(int)

    cm = confusion_matrix(predictions_df[target_column], binary_predictions)
    print("\nConfusion Matrix:")
    print(f"TN: {cm[0, 0]}, FP: {cm[0, 1]}")
    print(f"FN: {cm[1, 0]}, TP: {cm[1, 1]}")

    auc = roc_auc_score(predictions_df[target_column], predictions_df[predicted_col])
    print(f"\nAUC Score: {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(predictions_df[target_column], binary_predictions))
    pytest_examples_passed = True
