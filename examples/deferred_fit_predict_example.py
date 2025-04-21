import pandas as pd
from sklearn.linear_model import LinearRegression

import xorq as xo
import xorq.vendor.ibis.expr.datatypes as dt
from xorq.caching import ParquetStorage
from xorq.ml import deferred_fit_predict_sklearn


def make_data():
    import numpy as np

    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    # y = 1 * x_0 + 2 * x_1 + 3
    y = np.dot(X, np.array([1, 2])) + 3
    df = pd.DataFrame(np.hstack((X, y[:, np.newaxis]))).rename(
        columns=lambda x: chr(x + ord("a"))
    )
    (*features, target) = df.columns
    return (df, features, target)


deferred_linear_regression = deferred_fit_predict_sklearn(
    cls=LinearRegression, return_type=dt.float64
)


con = xo.connect()
storage = ParquetStorage(source=con)
(df, features, target) = make_data()
t = con.register(df, "t")


# uncached run
(deferred_model, model_udaf, predict) = deferred_linear_regression(t, target, features)
predicted = t.mutate(predict.on_expr(t))


# cached run
(cached_deferred_model, cached_model_udaf, cached_predict) = deferred_linear_regression(
    t, target, features, storage=storage
)
cached_predicted = t.mutate(cached_predict.on_expr(t))


if __name__ == "__pytest_main__":
    # model = deferred_model.execute()
    # ((cached_model,),) = cached_deferred_model.execute().values
    predicted_df = predicted.execute()
    cached_predicted_df = cached_predicted.execute()
    assert predicted_df.equals(cached_predicted_df)
    pytest_examples_passed = True
