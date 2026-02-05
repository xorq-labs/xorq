"""Home Run Prediction with Deferred sklearn Pipeline

This example demonstrates xorq's deferred execution with a complex sklearn pipeline.
The pipeline is defined declaratively and only executes when .execute() is called,
enabling lazy evaluation, caching, and distributed execution.

Pipeline structure:
1. Load batting dataset with missing values (deferred)
2. ColumnTransformer: KNNImputer -> FeatureUnion(PCA, ICA, NMF) on numeric features,
   passthrough other numeric, OneHotEncoder on categoricals
3. FeatureUnion: 2 feature selectors (SelectFromModel with Lasso, VarianceThreshold)
4. RandomForestRegressor to predict home runs (HR)

All transformations are deferred until predictions.execute() is called.
"""

# ruff: noqa: E402
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import NMF, PCA, FastICA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.impute import KNNImputer
from sklearn.linear_model import Lasso
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

import xorq.api as xo
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.expr.ml import train_test_splits
from xorq.expr.ml.pipeline_lib import Pipeline


con = xo.connect()
batting = deferred_read_parquet(
    path=xo.options.pins.get_path("batting"),
    con=con,
)

target = "HR"

# Numeric features with missing values - KNN impute then decomposition (PCA, ICA, NMF)
numeric_to_impute = [
    "AB",
    "R",
    "H",
    "X2B",
    "X3B",
    "RBI",
    "SB",
    "CS",
    "BB",
    "SO",
    "IBB",
    "HBP",
    "SH",
    "SF",
    "GIDP",
]

# Numeric features without missing - passthrough
numeric_passthrough = ["yearID", "stint", "G"]

# Categorical features - one-hot encode
categorical_features = ["teamID", "lgID"]

all_features = numeric_to_impute + numeric_passthrough + categorical_features

# Filter null target
batting_filtered = batting.filter(batting[target].notnull())
# Make example smaller ~2k
batting_clean = batting_filtered.filter((batting_filtered.playerID.hash() % 50) == 0)

train, test = train_test_splits(batting_clean, test_sizes=0.2, random_seed=42)

n_pca_components = 5
n_ica_components = 5
n_nmf_components = 5
n_decomposition_cols = n_pca_components + n_ica_components + n_nmf_components

# FeatureUnion: PCA, ICA, and NMF
decomposition_union = FeatureUnion(
    [
        ("pca", PCA(n_components=n_pca_components, svd_solver="full")),
        ("ica", FastICA(n_components=n_ica_components, random_state=42, max_iter=500)),
        (
            "nmf",
            SklearnPipeline(
                [
                    ("minmax", MinMaxScaler()),  # NMF requires non-negative values
                    (
                        "nmf",
                        NMF(
                            n_components=n_nmf_components,
                            random_state=42,
                            max_iter=1000,
                        ),
                    ),
                ]
            ),
        ),
    ]
)

# ColumnTransformer: KNNImputer -> FeatureUnion(PCA, ICA, NMF) on numeric,
# passthrough other numeric, OneHotEncoder on categorical
preprocessor = ColumnTransformer(
    [
        (
            "decomp",
            SklearnPipeline(
                [
                    ("knn_impute", KNNImputer(n_neighbors=5)),
                    ("union", decomposition_union),
                ]
            ),
            numeric_to_impute,
        ),
        ("pass_numeric", "passthrough", numeric_passthrough),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_features,
        ),
    ]
)

# FeatureUnion: 2 feature selectors (using Lasso for SelectFromModel)
feature_selection_union = FeatureUnion(
    [
        (
            "from_model",
            SklearnPipeline(
                [
                    ("scaler", StandardScaler()),  # Lasso needs scaled features
                    ("selector", SelectFromModel(Lasso(alpha=0.1, random_state=42))),
                ]
            ),
        ),
        ("variance", VarianceThreshold(threshold=0.01)),
    ]
)

sklearn_pipeline = SklearnPipeline(
    [
        ("preprocessor", preprocessor),
        ("feature_selection", feature_selection_union),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
)

xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)
deferred_fitted_pipeline = xorq_pipeline.fit(
    train, features=all_features, target=target
)
predictions = deferred_fitted_pipeline.predict(
    test
)  # predictions is just a deferred expression , you can .execute() later


if __name__ in ("__main__", "__pytest_main__"):
    results = predictions.execute()
    print(f"Predictions shape: {results.shape}")
    print(f"\nColumns: {list(results.columns)}")
    print("\nSample predictions:")
    print(results.head(10))

    # Calculate simple metrics
    mae = (results[target] - results["predicted"]).abs().mean()
    print(f"\nMean Absolute Error: {mae:.2f}")

    pytest_examples_passed = True
