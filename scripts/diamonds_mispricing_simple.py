"""
Simple diamonds mispricing detector - WORKING VERSION.

This version trains on numeric features and provides mispricing scores.
For categorical analysis, post-process the results by joining with the source data.

Build with: xorq build scripts/diamonds_mispricing_simple.py -e mispricing_scores
Run with: xorq run diamonds-mispricing-simple -o mispricing.parquet
"""

import toolz
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler

import xorq.api as xo
from xorq.api import _
from xorq.expr.ml.pipeline_lib import Pipeline
from xorq.vendor import ibis


TARGET = "price"
ORIGINAL_ROW = "original_row"


@toolz.curry
def as_struct(expr, name=None):
    """Pack all columns into a struct."""
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


def encode_and_engineer_features(table):
    """Create numeric features including encoded categorical scores."""
    # Filter nulls and invalid values
    filtered = table.filter(
        [
            _.carat.notnull(),
            _.price.notnull(),
            _.depth.notnull(),
            _.table.notnull(),
            _.x.notnull(),
            _.y.notnull(),
            _.z.notnull(),
            _.carat > 0,
            _.depth > 0,
            _.table > 0,
            _.x > 0,
            _.y > 0,
            _.z > 0,
        ]
    )

    # Encode categoricals as numeric scores
    encoded = filtered.mutate(
        cut_score=(
            _.cut.case()
            .when("Fair", 0.0)
            .when("Good", 1.0)
            .when("Very Good", 2.0)
            .when("Premium", 3.0)
            .when("Ideal", 4.0)
            .else_(2.0)
            .end()
        ),
        color_score=(
            _.color.case()
            .when("J", 0.0)
            .when("I", 1.0)
            .when("H", 2.0)
            .when("G", 3.0)
            .when("F", 4.0)
            .when("E", 5.0)
            .when("D", 6.0)
            .else_(3.0)
            .end()
        ),
        clarity_score=(
            _.clarity.case()
            .when("I1", 0.0)
            .when("SI2", 1.0)
            .when("SI1", 2.0)
            .when("VS2", 3.0)
            .when("VS1", 4.0)
            .when("VVS2", 5.0)
            .when("VVS1", 6.0)
            .when("IF", 7.0)
            .else_(3.0)
            .end()
        ),
    )

    # Engineer features
    engineered = encoded.mutate(
        log_carat=(_.carat + 1e-6).ln(),
        carat_squared=_.carat**2,
        depth_table_ratio=safe_ratio(_.depth, _.table),
        slenderness_ratio=safe_ratio(_.x, _.y),
        volume=_.x * _.y * _.z,
        surface_area=_.x * _.y,
    )

    return engineered


# Load diamonds
con = xo.connect()
diamonds = xo.examples.diamonds.fetch(backend=con).tag(
    "source", type="dataset", name="diamonds"
)

# Build features
feature_table = encode_and_engineer_features(diamonds)

# Define feature columns for training (numeric only)
feature_columns = [
    "carat",
    "depth",
    "table",
    "x",
    "y",
    "z",
    "cut_score",
    "color_score",
    "clarity_score",
    "log_carat",
    "carat_squared",
    "depth_table_ratio",
    "slenderness_ratio",
    "volume",
    "surface_area",
]

# Select only features and target for modeling
model_data = feature_table.select(*feature_columns, TARGET)

# Train price prediction model on full dataset
pipeline = Pipeline.from_instance(
    SkPipeline(
        [
            ("scale", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
)

fitted = pipeline.fit(model_data, features=feature_columns, target=TARGET)

# Generate predictions and mispricing scores
mispricing_scores = (
    model_data.mutate(as_struct(name=ORIGINAL_ROW))
    .pipe(fitted.predict)
    .drop(TARGET)
    .unpack(ORIGINAL_ROW)
    .mutate(
        predicted_price=_.predicted,
        abs_error=(_.predicted - _.price).abs(),
        pct_error=safe_ratio(_.predicted - _.price, _.price),
        # Deal score: positive = underpriced (good deal), negative = overpriced
        deal_score=safe_ratio(_.predicted - _.price, _.predicted),
        abs_pct_deviation=safe_ratio((_.predicted - _.price).abs(), _.price),
        # Mispricing flags
        is_mispriced=(safe_ratio((_.predicted - _.price).abs(), _.price) > 0.20),
        price_category=(
            ibis.case()
            .when(safe_ratio(_.predicted - _.price, _.predicted) > 0.20, "underpriced")
            .when(safe_ratio(_.predicted - _.price, _.predicted) < -0.20, "overpriced")
            .else_("fair_priced")
            .end()
        ),
        # Arbitrage score (higher = better deal for underpriced items)
        arbitrage_score=(
            ibis.case()
            .when(
                safe_ratio(_.predicted - _.price, _.predicted) > 0,
                safe_ratio(_.predicted - _.price, _.predicted) * 100,
            )
            .else_(0)
            .end()
        ),
    )
    .drop("predicted")
    .tag("mispricing", type="analysis", name="diamond_mispricing_scores")
)

# Main output
expr = mispricing_scores

# Top arbitrage opportunities
top_deals = (
    mispricing_scores.filter(_.price_category == "underpriced")
    .order_by(ibis.desc("deal_score"))
    .tag("deals", type="top_arbitrage", name="best_deals")
)

# Summary by price category
category_stats = (
    mispricing_scores.group_by(_.price_category)
    .aggregate(
        [
            _.count().name("count"),
            _.price.mean().name("avg_price"),
            _.predicted_price.mean().name("avg_predicted"),
            _.deal_score.mean().name("avg_deal_score"),
            _.carat.mean().name("avg_carat"),
            _.cut_score.mean().name("avg_cut_score"),
            _.color_score.mean().name("avg_color_score"),
            _.clarity_score.mean().name("avg_clarity_score"),
        ]
    )
    .order_by(ibis.desc("count"))
    .tag("summary", type="stats", name="category_summary")
)
