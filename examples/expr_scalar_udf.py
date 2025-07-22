import pickle

import pandas as pd
import toolz
import xgboost as xgb

import xorq as xo
import xorq.expr.datatypes as dt
import xorq.expr.udf as udf
from xorq.common.utils.toolz_utils import curry
from xorq.expr.udf import (
    make_pandas_expr_udf,
)


ROWNUM = "rownum"
features = (
    "emp_length",
    "dti",
    "annual_inc",
    "loan_amnt",
    "fico_range_high",
    "cr_age_days",
)
target = "event_occurred"
model_key = "model"
prediction_key = "predicted"
prediction_typ = "float32"


@curry
def train_xgboost_model(df, features=features, target=target, seed=0):
    param = {"max_depth": 4, "eta": 1, "objective": "binary:logistic", "seed": seed}
    num_round = 10
    if ROWNUM in df:
        # enforce order for reproducibility
        df = df.sort_values(ROWNUM, ignore_index=True)
    X = df[list(features)]
    y = df[target]
    dtrain = xgb.DMatrix(X, y)
    bst = xgb.train(param, dtrain, num_boost_round=num_round)
    return bst


@curry
def predict_xgboost_model(model, df, features=features):
    return model.predict(xgb.DMatrix(df[list(features)]))


def run_pd(train, test):
    train_df = train.execute()
    test_df = test.execute()
    model = train_xgboost_model(train_df)
    from_pd = test_df.assign(
        **{prediction_key: predict_xgboost_model(model)},
    )
    return from_pd


t = xo.deferred_read_parquet(
    xo.config.options.pins.get_path("lending-club"),
    xo.connect(),
)

(train, test) = xo.train_test_splits(
    t,
    unique_key=ROWNUM,
    test_sizes=0.7,
    random_seed=42,
)
model_udaf = udf.agg.pandas_df(
    fn=toolz.compose(pickle.dumps, train_xgboost_model),
    schema=t[features + (target,)].schema(),
    return_type=dt.binary,
    name=model_key,
)
predict_expr_udf = make_pandas_expr_udf(
    computed_kwargs_expr=model_udaf.on_expr(train),
    fn=predict_xgboost_model,
    schema=t[features].schema(),
    return_type=dt.dtype(prediction_typ),
    name=prediction_key,
)
expr = test.mutate(predict_expr_udf.on_expr(test).name(prediction_key))


if __name__ == "__pytest_main__":
    from_pd = run_pd(train, test)
    from_xo = expr.execute()
    pd._testing.assert_frame_equal(from_xo, from_pd)
    pytest_examples_passed = True
