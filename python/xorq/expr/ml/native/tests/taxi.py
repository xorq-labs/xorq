from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC = ("trip_distance", "passenger_count", "hour", "dow")
CATEGORICAL = ("payment_type", "RatecodeID")
FEATURES = NUMERIC + CATEGORICAL
TARGET = "fare_amount"


def make_pipeline(model):
    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), list(NUMERIC)),
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL)),
        ]
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])
