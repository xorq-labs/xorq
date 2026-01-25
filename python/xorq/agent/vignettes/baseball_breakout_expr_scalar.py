import xorq.api as xo
from xorq.expr.udf import agg, make_pandas_expr_udf
import xorq.expr.datatypes as dt
import toolz
import pickle

batting = xo.examples.batting.fetch()

batting_with_ops = batting.mutate(
    obp=xo.ifelse(
        (xo._.AB + xo._.BB + xo._.HBP + xo._.SF) > 0,
        (xo._.H + xo._.BB + xo._.HBP) / (xo._.AB + xo._.BB + xo._.HBP + xo._.SF),
        0.0
    ),
    slg=xo.ifelse(
        xo._.AB > 0,
        ((xo._.H - xo._.X2B - xo._.X3B - xo._.HR) + 2 * xo._.X2B + 3 * xo._.X3B + 4 * xo._.HR) / xo._.AB,
        0.0
    ),
    pa=xo._.AB + xo._.BB + xo._.HBP + xo._.SF + xo._.SH
).mutate(
    ops=xo._.obp + xo._.slg
)

qualified_seasons = batting_with_ops.filter(xo._.pa >= 200)

lag_window = xo.window(
    group_by="playerID",
    order_by="yearID",
    rows=(None, -1)
)

career_stats = (qualified_seasons
    .mutate(
        career_pa_prior=xo._.pa.sum().over(lag_window),
        career_ops_weighted_prior=(xo._.ops * xo._.pa).sum().over(lag_window)
    )
    .mutate(
        career_avg_ops_prior=xo.ifelse(
            xo._.career_pa_prior > 0,
            xo._.career_ops_weighted_prior / xo._.career_pa_prior,
            None
        )
    )
)

breakout_labels = career_stats.mutate(
    ops_improvement=xo.ifelse(
        xo._.career_avg_ops_prior.notnull(),
        (xo._.ops - xo._.career_avg_ops_prior) / xo._.career_avg_ops_prior,
        None
    )
).mutate(
    is_breakout=xo.ifelse(
        xo._.ops_improvement.notnull(),
        xo.ifelse(xo._.ops_improvement > 0.15, 1.0, 0.0),
        None
    )
)

features_engineered = breakout_labels.mutate(
    prior_ops=xo._.ops.lag(1).over(group_by="playerID", order_by="yearID"),
    prior_pa=xo._.pa.lag(1).over(group_by="playerID", order_by="yearID"),
    prior_hr=xo._.HR.lag(1).over(group_by="playerID", order_by="yearID"),
    prior_sb=xo._.SB.lag(1).over(group_by="playerID", order_by="yearID"),
    prior_breakout=xo._.is_breakout.lag(1).over(group_by="playerID", order_by="yearID"),

    prior2_ops=xo._.ops.lag(2).over(group_by="playerID", order_by="yearID"),
    prior2_pa=xo._.pa.lag(2).over(group_by="playerID", order_by="yearID")
).mutate(
    ops_trend=xo._.prior_ops - xo._.prior2_ops,
    pa_trend=xo._.prior_pa - xo._.prior2_pa,

    career_years=xo._.yearID.rank().over(
        group_by="playerID",
        order_by="yearID"
    ).cast("float"),

    ops_vs_career=xo.ifelse(
        xo._.career_avg_ops_prior.notnull(),
        (xo._.prior_ops - xo._.career_avg_ops_prior) / xo._.career_avg_ops_prior,
        0.0
    )
).mutate(
    hr_rate=xo.ifelse(xo._.prior_pa > 0, xo._.prior_hr / xo._.prior_pa, 0.0),
    sb_rate=xo.ifelse(xo._.prior_pa > 0, xo._.prior_sb / xo._.prior_pa, 0.0),

    ops_volatility=xo._.ops.std().over(
        group_by="playerID",
        order_by="yearID",
        rows=(None, -1)  # Up to prior year
    )
).mutate(
    breakout_label=xo._.is_breakout
)

training_data = features_engineered.filter([
    xo._.career_years >= 3,
    xo._.breakout_label.notnull(),
    xo._.prior_ops.notnull(),
    xo._.prior2_ops.notnull(),
    xo._.career_avg_ops_prior.notnull()
])

FEATURES = [
    "prior_ops",
    "prior_pa",
    "prior_hr",
    "prior_sb",
    "prior_breakout",
    "prior2_ops",
    "prior2_pa",
    "ops_trend",
    "pa_trend",
    "career_years",
    "ops_vs_career",
    "hr_rate",
    "sb_rate",
    "ops_volatility",
    "career_avg_ops_prior"
]

@toolz.curry
def train_model(df, features=FEATURES, target="breakout_label"):
    from sklearn.ensemble import RandomForestClassifier
    import pickle

    X = df[list(features)].fillna(0)
    y = df[target]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        max_depth=10
    )
    model.fit(X, y)

    return pickle.dumps(model)

@toolz.curry
def predict_with_model(model_bytes, df, features=FEATURES):
    model = model_bytes # ExprScalarUDF automatically unpickles
    X = df[list(features)].fillna(0)
    probs = model.predict_proba(X)
    return probs[:, 1]

training_subset = training_data.select(FEATURES + ["breakout_label"])

model_udaf = agg.pandas_df(
    fn=train_model,
    schema=training_subset.schema(),
    return_type=dt.binary,
    name="model"
)

train = training_subset.limit(10000).cache()
test = training_subset.tail(500)

trained_model_expr = model_udaf.on_expr(train)

predict_expr_udf = make_pandas_expr_udf(
    computed_kwargs_expr=trained_model_expr,
    fn=predict_with_model,
    schema=test[FEATURES].schema(),
    return_type=dt.float64,
    name="predicted"
)

predictions = test.mutate(
    predicted=predict_expr_udf.on_expr(test).name("predicted")
)

result_df = predictions.select("breakout_label", "predicted").limit(20).execute()

model_bytes = trained_model_expr.execute()
model = pickle.loads(model_bytes)
feature_importance = sorted(zip(FEATURES, model.feature_importances_),
                          key=lambda x: x[1], reverse=True)


for feat, imp in feature_importance[:5]:
    if imp > 0.001:
        print(f"  {feat}: {imp:.4f}")
