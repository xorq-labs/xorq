import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import xorq as xo
import xorq.selectors as s
import xorq.vendor.ibis.expr.datatypes as dt
from xorq.common.utils.defer_utils import deferred_read_csv
from xorq.expr.ml import deferred_fit_predict, train_test_splits


class DeferredOneHotEncoder:
    def __init__(
        self,
        features=None,
        drop=None,
        prefix_separator="_",
    ):
        self.features = features
        self.drop = drop
        self.prefix_separator = prefix_separator
        self.categories_exprs = {}
        self._fitted_categories = {}

    def fit(self, expr):
        if self.features is None:
            non_numeric_columns = set(expr.columns) - set(
                expr.select(s.numeric()).columns
            )
            self.features = list(non_numeric_columns)

        self.categories_exprs = {}
        for col in self.features:
            categories_expr = expr.select(col).distinct().cache()
            self.categories_exprs[col] = categories_expr

        return self

    def transform(self, expr):
        result_expr = expr

        categories_dict = self._fitted_categories or {}
        if not categories_dict:
            for col, categories_expr in self.categories_exprs.items():
                categories = categories_expr.execute()[col].tolist()
                categories.sort()

                if self.drop == "first" and len(categories) > 0:
                    categories_dict[col] = categories[1:]
                else:
                    categories_dict[col] = categories
        mutations = {}
        for col in self.features:
            categories = categories_dict[col]
            for category in categories:
                new_col = f"{col}{self.prefix_separator}{category}".replace("-", "_")
                mutations[new_col] = (expr[col] == category).cast("int")

        if mutations:
            result_expr = expr.mutate(**mutations)

        return result_expr


class XGBoostModel:
    def __init__(self, num_boost_round=10, params=None):
        self.num_boost_round = num_boost_round
        self.params = params or {
            "max_depth": 4,
            "eta": 1,
            "objective": "binary:logistic",
            "seed": 0,
        }
        self.model = None

    def fit(self, X, y):
        dtrain = xgb.DMatrix(X, y)
        self.model = xgb.train(
            self.params, dtrain, num_boost_round=self.num_boost_round
        )
        return self

    def predict(self, X):
        dmatrix = xgb.DMatrix(X)
        return self.model.predict(dmatrix)


def run_full_pipeline(dataset_name, target_column):
    csv_path = xo.options.pins.get_path(dataset_name)
    con = xo.connect()

    t = deferred_read_csv(path=csv_path, con=con).mutate(row_number=xo.row_number())

    t = t.mutate(**{target_column: (t[target_column] == "yes").cast("int")})

    train_table, test_table = train_test_splits(
        t, unique_key="row_number", test_sizes=[0.5, 0.5], num_buckets=2, random_seed=42
    )

    train_table = train_table.drop("row_number")
    test_table = test_table.drop("row_number")

    encoder = DeferredOneHotEncoder(drop="first")
    encoder.fit(train_table)

    encoded_train = encoder.transform(train_table).cache()
    encoded_test = encoder.transform(test_table).cache()

    numeric_features = encoded_train.select(s.numeric()).columns
    numeric_features = [
        col
        for col in numeric_features
        if col != target_column and col != target_column + "_yes"
    ]

    deferred_model, model_udaf, deferred_predict = deferred_fit_predict(
        expr=encoded_train,
        target=target_column,
        features=numeric_features,
        cls=XGBoostModel,
        return_type=dt.float64,
        name="xgb_prediction",
    )

    predictions = encoded_test.mutate(prediction=deferred_predict.on_expr(encoded_test))

    return {
        "encoded_train": encoded_train,
        "encoded_test": encoded_test,
        "predictions": predictions,
        "encoder": encoder,
        "model": deferred_model,
    }


if __name__ == "__main__":
    results = run_full_pipeline("bank-marketing", "deposit")

    predictions_expr = results["predictions"]
    encoded_test = results["encoded_test"]
    encoded_train = results["encoded_train"]

    predictions_df = predictions_expr.execute()

    binary_predictions = (predictions_df["prediction"] >= 0.5).astype(int)

    cm = confusion_matrix(predictions_df["deposit"], binary_predictions)

    print("\nConfusion Matrix:")
    print(f"TN: {cm[0, 0]}, FP: {cm[0, 1]}")
    print(f"FN: {cm[1, 0]}, TP: {cm[1, 1]}")

    auc = roc_auc_score(predictions_df["deposit"], predictions_df["prediction"])
    print(f"\nAUC Score: {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(predictions_df["deposit"], binary_predictions))
