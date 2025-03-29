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
(df, features, target) = make_data()
t = con.register(df, "t")


# uncached run
(computed_kwargs_expr, model_udaf, predict_expr_udf) = deferred_linear_regression(
    t, target, features
)
model = computed_kwargs_expr.execute()
predicted = t.mutate(predict_expr_udf.on_expr(t)).execute()


# cached run
storage = ParquetStorage(source=con)
(computed_kwargs_expr, model_udaf, predict_expr_udf) = deferred_linear_regression(
    t, target, features, storage=storage
)
((cached_model,),) = computed_kwargs_expr.execute().values
cached_predicted = t.mutate(predict_expr_udf.on_expr(t)).execute()


assert predicted.equals(cached_predicted)
pytest_examples_passed = True
