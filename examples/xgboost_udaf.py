import pandas as pd
import xgboost as xgb

import xorq as xo
import xorq.expr.datatypes as dt
from xorq.common.utils.toolz_utils import curry
from xorq.expr import udf


ROWNUM = "rownum"


def train_xgboost_model(df, features, target, seed=0):
    if ROWNUM in df:
        df = df.sort_values(ROWNUM, ignore_index=True)
    param = {"max_depth": 4, "eta": 1, "objective": "binary:logistic", "seed": seed}
    num_round = 10
    X = df[list(features)]
    y = df[target]
    dtrain = xgb.DMatrix(X, y)
    bst = xgb.train(param, dtrain, num_boost_round=num_round)
    return bst


def calc_best_features(df, candidates, target, n):
    return (
        pd.Series(train_xgboost_model(df, candidates, target).get_score())
        .tail(n)
        .pipe(lambda s: tuple({"feature": k, "score": v} for k, v in s.items()))
    )


candidates = (
    "emp_length",
    "dti",
    "annual_inc",
    "loan_amnt",
    "fico_range_high",
    "cr_age_days",
)
by = "issue_y"
target = "event_occurred"
cols = list(candidates) + [by, target, ROWNUM]
curried_calc_best_features = curry(
    calc_best_features, candidates=candidates, target=target, n=2
)
ibis_output_type = dt.infer(({"feature": "feature", "score": 0.0},))


t = xo.deferred_read_parquet(
    xo.connect(),
    xo.options.pins.get_path("lending-club"),
)
agg_udf = udf.agg.pandas_df(
    curried_calc_best_features,
    t[cols].schema(),
    ibis_output_type,
    name="calc_best_features",
)
expr = t.group_by(by).agg(agg_udf.on_expr(t).name("best_features")).order_by(by)


if __name__ == "__pytest_main__":
    result = xo.execute(expr)
    pytest_examples_passed = True
