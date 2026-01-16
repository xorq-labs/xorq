"""
Diamonds price prediction example expression.

Builds feature-engineered training data from the Snowflake ``DIAMONDS`` table,
fits a deferred sklearn pipeline, and exposes expressions for predictions and
simple error diagnostics.
"""

import toolz
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler

import xorq.api as xo
from xorq.api import _
from xorq.expr.ml import train_test_splits
from xorq.expr.ml.pipeline_lib import Pipeline
from xorq.vendor import ibis


TARGET = "PRICE"
ORIGINAL_ROW = "original_row"
PREDICTED_COL = "predicted_price"


@toolz.curry
def as_struct(expr, name=None):
    struct = xo.struct({column: expr[column] for column in expr.columns})
    if name:
        struct = struct.name(name)
    return struct


def safe_ratio(numerator, denominator):
    """Return numerator / denominator and guard against divide-by-zero/null."""
    return (
        ibis.case()
        .when(denominator.isnull() | (denominator == 0), None)
        .else_(numerator / denominator)
        .end()
    )


def encode_quality_columns(table):
    """Map CUT, COLOR, and CLARITY categoricals to ordered numeric scores."""
    return table.mutate(
        cut_score=(
            _.CUT.case()
            .when("Fair", 0)
            .when("Good", 1)
            .when("Very Good", 2)
            .when("Premium", 3)
            .when("Ideal", 4)
            .else_(2)
            .end()
        ),
        color_score=(
            _.COLOR.case()
            .when("J", 0)
            .when("I", 1)
            .when("H", 2)
            .when("G", 3)
            .when("F", 4)
            .when("E", 5)
            .when("D", 6)
            .else_(3)
            .end()
        ),
        clarity_score=(
            _.CLARITY.case()
            .when("I1", 0)
            .when("SI2", 1)
            .when("SI1", 2)
            .when("VS2", 3)
            .when("VS1", 4)
            .when("VVS2", 5)
            .when("VVS1", 6)
            .when("IF", 7)
            .else_(3)
            .end()
        ),
    )


def build_feature_view(table):
    """Create a clean, feature-rich view ready for ML."""
    filtered = table.filter(
        [
            _.CARAT.notnull(),
            _.PRICE.notnull(),
            _.DEPTH.notnull(),
            _.TABLE.notnull(),
            _.X.notnull(),
            _.Y.notnull(),
            _.Z.notnull(),
            _.CARAT > 0,
            _.DEPTH > 0,
            _.TABLE > 0,
            _.X > 0,
            _.Y > 0,
            _.Z > 0,
        ]
    )

    encoded = encode_quality_columns(filtered)
    engineered = encoded.mutate(
        log_carat=(_.CARAT + 1e-6).ln(),
        carat_squared=_.CARAT**2,
        depth_table_ratio=safe_ratio(_.DEPTH, _.TABLE),
        slenderness_ratio=safe_ratio(_.X, _.Y),
        volume=_.X * _.Y * _.Z,
        surface_area=_.X * _.Y,
    )

    feature_columns = (
        "CARAT",
        "DEPTH",
        "TABLE",
        "X",
        "Y",
        "Z",
        "cut_score",
        "color_score",
        "clarity_score",
        "log_carat",
        "carat_squared",
        "depth_table_ratio",
        "slenderness_ratio",
        "volume",
        "surface_area",
    )

    feature_view = engineered.select(*feature_columns, TARGET)
    return feature_view, feature_columns


def make_price_prediction_pipeline():
    """Return a xorq Pipeline wrapping a scaler + linear regressor."""
    sklearn_pipeline = SkPipeline(
        steps=[
            ("scale", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    return Pipeline.from_instance(sklearn_pipeline)


def build_diamonds_price_prediction(con=None):
    """
    Build expressions for fitting and evaluating a price regression model.

    Returns:
        dict[str, xo.Expr]: view of the raw data, train/test splits, predictions,
            and aggregate error metrics.
    """

    con = con or xo.connect()
    diamonds = xo.examples.diamonds.fetch(backend=con, table_name="DIAMONDS")
    diamonds = diamonds.rename("ALL_CAPS")
    print(diamonds.schema())  # Inspect schema when running this module manually.

    feature_view, feature_columns = build_feature_view(diamonds)
    train, test = train_test_splits(feature_view, 0.2)

    xorq_pipeline = make_price_prediction_pipeline()
    fitted_pipeline = xorq_pipeline.fit(
        train,
        features=feature_columns,
        target=TARGET,
    )

    test_with_predictions = (
        test.mutate(as_struct(name=ORIGINAL_ROW))
        .pipe(fitted_pipeline.predict)
        .drop(TARGET)
        .unpack(ORIGINAL_ROW)
        .mutate(
            **{
                PREDICTED_COL: _.predicted,
                "abs_error": (_.predicted - _[TARGET]).abs(),
                "pct_error": safe_ratio(_.predicted - _[TARGET], _[TARGET]),
            }
        )
        .drop("predicted")
    )

    prediction_metrics = test_with_predictions.aggregate(
        [
            _.abs_error.mean().name("mae"),
            _.abs_error.median().name("median_abs_error"),
            _.pct_error.abs().mean().name("mean_abs_pct_error"),
            (_.pct_error.abs() > 0.1).mean().name("share_over_10pct_error"),
        ]
    )

    return {
        "diamonds": diamonds,
        "feature_view": feature_view,
        "train": train,
        "test": test,
        "predictions": test_with_predictions,
        "prediction_metrics": prediction_metrics,
    }


artifacts = build_diamonds_price_prediction()
diamonds_price_predictions = artifacts["predictions"]
diamonds_price_metrics = artifacts["prediction_metrics"]
