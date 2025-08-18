import functools

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

import xorq as xo
import xorq.expr.selectors as s
import xorq.vendor.ibis.expr.datatypes as dt
from xorq.caching import ParquetStorage
from xorq.common.utils.defer_utils import deferred_read_csv

# ===== EVALUATION UDF SETUP =====
from xorq.expr import udf
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


def compute_evaluation_metrics(df):
    """Comprehensive evaluation function that outputs all metrics."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    actual = df["actual"].values
    predicted_prob = df["predicted"].values
    predicted_binary = (predicted_prob >= 0.5).astype(int)

    # Confusion matrix
    cm = confusion_matrix(actual, predicted_binary)
    tn, fp, fn, tp = cm.ravel()

    # Metrics
    accuracy = accuracy_score(actual, predicted_binary)
    precision = precision_score(actual, predicted_binary, zero_division=0)
    recall = recall_score(actual, predicted_binary, zero_division=0)
    f1 = f1_score(actual, predicted_binary, zero_division=0)
    auc = roc_auc_score(actual, predicted_prob)

    return {
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc),
        },
        "n_samples": len(actual),
    }


# Create the aggregation UDF
evaluation_udaf = udf.agg.pandas_df(
    fn=compute_evaluation_metrics,
    schema=xo.schema({"actual": int, "predicted": float}),
    return_type=dt.Struct(
        {
            "confusion_matrix": dt.Struct(
                {"tn": dt.int64, "fp": dt.int64, "fn": dt.int64, "tp": dt.int64}
            ),
            "metrics": dt.Struct(
                {
                    "accuracy": dt.float64,
                    "precision": dt.float64,
                    "recall": dt.float64,
                    "f1": dt.float64,
                    "auc": dt.float64,
                }
            ),
            "n_samples": dt.int64,
        }
    ),
    name="evaluate_classification",
)


# ===== ORIGINAL PIPELINE CLASSES (UNCHANGED) =====


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
        X = X.drop(columns=self.encoded_col).join(
            X[self.encoded_col].apply(
                lambda lst: pd.Series({dct["key"]: dct["value"] for dct in lst})
            )
        )
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


def make_pipeline(dataset_name, target_column, predicted_col, make_storage=None):
    con = xo.connect()

    if make_storage is not None:
        storage = make_storage(con)
    else:
        storage = None

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


def create_cached_evaluation(
    test_predictions, target_column, predicted_col, storage=None
):
    # Prepare evaluation input
    eval_input = test_predictions.select(
        actual=xo._[target_column], predicted=xo._[predicted_col]
    )
    evaluation_expr = eval_input.agg(
        evaluation_udaf.on_expr(eval_input).name("evaluation_result")
    )
    if storage:
        evaluation_expr = evaluation_expr.cache(storage=storage)

    return evaluation_expr


def print_cached_evaluation(evaluation_expr):
    """Print formatted evaluation results from cached expression."""
    result = evaluation_expr.execute()
    eval_data = result.iloc[0]["evaluation_result"]

    # Extract data
    cm = eval_data["confusion_matrix"]
    metrics = eval_data["metrics"]
    n_samples = eval_data["n_samples"]

    print("Cached Evaluation Results")
    print("=" * 30)
    print(f"Dataset size: {n_samples:,} samples")
    print()

    print("Confusion Matrix:")
    print(f"TN: {cm['tn']:6,} | FP: {cm['fp']:6,}")
    print(f"FN: {cm['fn']:6,} | TP: {cm['tp']:6,}")
    print()

    print("Performance Metrics:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"AUC:       {metrics['auc']:.4f}")

    return eval_data


(dataset_name, target_column, predicted_col) = (
    "bank-marketing",
    "deposit",
    "predicted",
)

train_table, test_table, fitted_pipeline = make_pipeline(
    dataset_name, target_column, predicted_col, make_storage=ParquetStorage
)

con = train_table._find_backend()
storage = ParquetStorage(source=con)

train_predicted = fitted_pipeline.fitted_steps[-1].predicted
deferred_model = fitted_pipeline.fitted_steps[-1].deferred_model
encoded_test = fitted_pipeline.transform(test_table)
test_predicted = fitted_pipeline.predict(test_table)

test_evaluation = create_cached_evaluation(
    test_predicted, target_column, predicted_col, storage=storage
)
expr = test_predicted

if __name__ == "__pytest_main__":
    print("=== ORIGINAL SKLEARN EVALUATION ===")
    predictions_df = test_predicted.execute()
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

    print("\n" + "=" * 50)

    # NEW: Cached evaluation
    print("=== CACHED EVALUATION (NEW) ===")
    eval_data = print_cached_evaluation(cached_evaluation)

    # Demonstrate caching
    print("\n=== CACHE DEMONSTRATION ===")
    print("Second call (should use cache)...")
    cached_result = cached_evaluation.execute()
    print("✓ Retrieved from cache successfully")

    # Verify cache key exists
    print(f"Cache key: {cached_evaluation.ls.get_key()}")
    print(f"Cache exists: {cached_evaluation.ls.op.parent.to_expr().ls.exists()}")

    pytest_examples_passed = True
