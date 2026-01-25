"""
Baseball breakout predictor using xorq.

Breakout definition: Season where OPS jumps >20% vs career average
Model: Predict next-season breakouts using prior trends, age, playing time

This example demonstrates:
- Complex window functions (lag, rank, cumulative sums, std)
- Career statistics with time-ordered aggregations
- Feature engineering from historical trends
- Train/test temporal split
- RandomForestClassifier with class imbalance handling
- as_struct pattern for column preservation through predict

Critical patterns showcased:
1. Chained mutates - split when referencing newly created columns
2. rank() returns Decimal128 - MUST cast to float for sklearn
3. Target labels - use float literals (1.0, 0.0) not integers
4. as_struct workflow - pack → predict → drop → unpack
"""

import xorq.api as xo
import toolz
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from xorq.expr.ml.pipeline_lib import Pipeline

# Get batting source from catalog
batting_source = xo.examples.batting.fetch()

# Check schema
print("Batting schema:")
print(batting_source.schema())

# Calculate OPS and other key metrics per player-season
batting_with_ops = batting_source.mutate(
    # On-base percentage: (H + BB + HBP) / (AB + BB + HBP + SF)
    obp=xo.ifelse(
        (xo._.AB + xo._.BB + xo._.HBP + xo._.SF) > 0,
        (xo._.H + xo._.BB + xo._.HBP) / (xo._.AB + xo._.BB + xo._.HBP + xo._.SF),
        0.0
    ),
    # Slugging percentage: (1B + 2*2B + 3*3B + 4*HR) / AB
    slg=xo.ifelse(
        xo._.AB > 0,
        ((xo._.H - xo._.X2B - xo._.X3B - xo._.HR) + 2 * xo._.X2B + 3 * xo._.X3B + 4 * xo._.HR) / xo._.AB,
        0.0
    ),
    # Plate appearances (proxy for playing time)
    pa=xo._.AB + xo._.BB + xo._.HBP + xo._.SF + xo._.SH
).mutate(
    ops=xo._.obp + xo._.slg
)

# Filter to meaningful seasons (>200 PA)
qualified_seasons = batting_with_ops.filter(xo._.pa >= 200)

# Calculate career average OPS for each player (up to that year)
career_stats = qualified_seasons.mutate(
    # Calculate cumulative stats
    career_pa=xo._.pa.sum().over(
        group_by="playerID",
        order_by="yearID",
        rows=(None, 0)  # All rows up to current
    ),
    career_ops_sum=xo._.ops.sum().over(
        group_by="playerID",
        order_by="yearID",
        rows=(None, 0)
    )
).mutate(
    career_avg_ops=xo.ifelse(
        xo._.career_pa > 0,
        xo._.career_ops_sum / xo._.career_pa,
        xo._.ops
    )
)

# Identify breakout seasons: OPS jump >20% vs career average
breakout_labels = career_stats.mutate(
    # Calculate OPS jump
    ops_vs_career=(xo._.ops - xo._.career_avg_ops) / xo._.career_avg_ops
).mutate(
    # Label next season as breakout candidate
    # CRITICAL: Use float literals (1.0, 0.0) not integers for sklearn
    next_year_breakout=xo.ifelse(
        xo._.ops_vs_career > 0.20,
        1.0,
        0.0
    )
).mutate(
    # Shift breakout label to prior year for prediction
    breakout_label=xo._.next_year_breakout.lag(1).over(
        group_by="playerID",
        order_by="yearID"
    )
)

# Engineer features from prior seasons
features_engineered = breakout_labels.mutate(
    # Prior year stats
    prior_ops=xo._.ops.lag(1).over(group_by="playerID", order_by="yearID"),
    prior_pa=xo._.pa.lag(1).over(group_by="playerID", order_by="yearID"),
    prior_hr=xo._.HR.lag(1).over(group_by="playerID", order_by="yearID"),
    prior_sb=xo._.SB.lag(1).over(group_by="playerID", order_by="yearID")
).mutate(
    # Two years ago stats
    prior2_ops=xo._.ops.lag(2).over(group_by="playerID", order_by="yearID"),
    prior2_pa=xo._.pa.lag(2).over(group_by="playerID", order_by="yearID")
).mutate(
    # Trend: OPS change from 2 years ago to last year
    ops_trend=xo._.prior_ops - xo._.prior2_ops,
    # Playing time trend
    pa_trend=xo._.prior_pa - xo._.prior2_pa
).mutate(
    # Career length (proxy for age/experience)
    # CRITICAL: rank() returns Decimal128, MUST cast to float for sklearn
    career_years=xo._.yearID.rank().over(
        group_by="playerID",
        order_by="yearID"
    ).cast("float"),  # Cast to float (workaround for Decimal128 issue)
    # Recent performance vs career
    recent_vs_career=xo._.prior_ops - xo._.career_avg_ops
).mutate(
    # Power trend
    hr_rate=xo.ifelse(xo._.prior_pa > 0, xo._.prior_hr / xo._.prior_pa, 0.0),
    # Speed component
    sb_rate=xo.ifelse(xo._.prior_pa > 0, xo._.prior_sb / xo._.prior_pa, 0.0)
).mutate(
    # Volatility: Did they have big swings in performance?
    ops_volatility=xo._.ops.std().over(
        group_by="playerID",
        order_by="yearID",
        rows=(None, -1)  # Up to prior year
    )
)

# Filter to training set: Must have 3+ prior seasons and valid label
training_data = features_engineered.filter([
    xo._.career_years >= 3,
    xo._.breakout_label.notnull(),
    xo._.prior_ops.notnull(),
    xo._.prior2_ops.notnull()
])

# Export labeled dataset for analysis
breakout_dataset = training_data

# Define feature columns
FEATURES = [
    "prior_ops",
    "prior_pa",
    "prior_hr",
    "prior_sb",
    "prior2_ops",
    "prior2_pa",
    "ops_trend",
    "pa_trend",
    "career_years",
    "recent_vs_career",
    "hr_rate",
    "sb_rate",
    "ops_volatility",
    "career_avg_ops"
]

# Train/test split by year (train on <2020, test on 2020+)
train = training_data.filter(xo._.yearID < 2020)
test = training_data.filter(xo._.yearID >= 2020)

# Create as_struct helper
@toolz.curry
def as_struct(expr, name=None):
    """Pack all columns into a struct for preservation through predict."""
    struct = xo.struct({c: expr[c] for c in expr.columns})
    return struct.name(name) if name else struct

# Build sklearn pipeline
sklearn_pipeline = SkPipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        random_state=42,
        class_weight="balanced"  # Handle class imbalance
    ))
])

xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)

# Fit on training data
fitted_pipeline = xorq_pipeline.fit(
    train,
    features=FEATURES,
    target="breakout_label"
)

# Make predictions using as_struct pattern
# CRITICAL: as_struct workflow for column preservation:
# 1. Pack all columns into struct
# 2. Predict (adds 'predicted' column)
# 3. Drop all original columns (they're duplicated)
# 4. Unpack struct to restore original columns
# 5. Use predicted column
with_struct = test.mutate(as_struct(name="original_row"))
predicted = fitted_pipeline.predict(with_struct)

# After predict, we have: original batting columns + original_row struct + predicted
# Drop all the duplicated columns except the struct
predictions = (
    predicted
    .drop("playerID", "yearID", "stint", "teamID", "lgID", "G", "AB", "R", "H",
          "X2B", "X3B", "HR", "RBI", "SB", "CS", "BB", "SO", "IBB", "HBP",
          "SH", "SF", "GIDP", "obp", "slg", "pa", "ops", "career_pa",
          "career_ops_sum", "ops_vs_career", "next_year_breakout", "breakout_label")
    .unpack("original_row")
    .mutate(
        breakout_predicted=xo._.predicted
    )
    .select(
        "playerID",
        "yearID",
        "prior_ops",
        "ops",
        "breakout_predicted",
        "breakout_label",
        "ops_trend",
        "pa_trend",
        "career_years",
        "recent_vs_career"
    )
)

# Export final expressions
expr = predictions
dataset_expr = breakout_dataset
model_expr = fitted_pipeline

if __name__ == "__main__":
    # Example usage
    print("\nBreakout predictions (first 10):")
    result = expr.execute()
    print(result.head(10))

    print("\nModel fitted successfully!")
    print(f"Features used: {', '.join(FEATURES)}")
