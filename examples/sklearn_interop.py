"""
sklearn ↔ xorq Interoperability Example

This example demonstrates the full round-trip between sklearn and xorq:
1. from_instance(): Wrap any sklearn pipeline for deferred execution
2. to_sklearn(): Get the fitted sklearn pipeline back
3. predict_proba() and decision_function() support
4. Packed format for transformers with dynamic schemas

Key features demonstrated:
- Pipeline.from_instance() accepts any sklearn Pipeline
- fit(), transform(), predict() are deferred (lazy)
- Fitted models are cached
- to_sklearn() returns fitted sklearn objects for direct use
- All execution happens at .execute() time
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import xorq.api as xo
from xorq.expr.ml.pipeline_lib import Pipeline, Step


def create_sample_data(include_missing=False):
    """Create a sample dataset with numeric and categorical features."""
    np.random.seed(42)
    n_samples = 1000

    # Generate numeric features
    X_numeric, y = make_classification(
        n_samples=n_samples,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        random_state=42,
    )

    # Create DataFrame with numeric features
    df = pd.DataFrame(
        X_numeric,
        columns=["age", "income", "score", "balance"],
    )

    # Add categorical features
    df["gender"] = np.random.choice(["M", "F"], n_samples)
    df["region"] = np.random.choice(["North", "South", "East", "West"], n_samples)
    df["category"] = np.random.choice(["A", "B", "C"], n_samples)

    # Optionally add some missing values (for imputer examples)
    if include_missing:
        df.loc[np.random.choice(n_samples, 50, replace=False), "income"] = np.nan
        df.loc[np.random.choice(n_samples, 30, replace=False), "balance"] = np.nan

    # Add target
    df["target"] = y

    return df


# =============================================================================
# Example 1: Basic Pipeline Round-Trip
# =============================================================================


def example_basic_roundtrip():
    """
    Demonstrate basic from_instance() and to_sklearn() round-trip.
    """
    print("\n" + "=" * 60)
    print("Example 1: Basic Pipeline Round-Trip")
    print("=" * 60)

    # Create sample data
    df = create_sample_data()
    numeric_features = ["age", "income", "score", "balance"]

    # Create xorq expression
    expr = xo.memtable(df)

    # Create sklearn pipeline
    sklearn_pipe = SklearnPipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )

    # Wrap with xorq for deferred execution
    xorq_pipe = Pipeline.from_instance(sklearn_pipe)
    print(f"Created xorq Pipeline from sklearn: {type(xorq_pipe)}")

    # Get unfitted sklearn pipeline back
    sklearn_unfitted = xorq_pipe.to_sklearn()
    print(f"Round-trip unfitted: {type(sklearn_unfitted)}")
    print(f"Steps match: {[s[0] for s in sklearn_unfitted.steps]}")

    # Fit the xorq pipeline (deferred)
    fitted_xorq = xorq_pipe.fit(
        expr,
        features=tuple(numeric_features),
        target="target",
    )
    print(f"Fitted xorq pipeline: {type(fitted_xorq)}")

    # Get fitted sklearn pipeline back
    sklearn_fitted = fitted_xorq.to_sklearn()
    print(f"Round-trip fitted: {type(sklearn_fitted)}")
    print(f"Scaler fitted: {hasattr(sklearn_fitted.named_steps['scaler'], 'mean_')}")
    print(f"Classifier fitted: {hasattr(sklearn_fitted.named_steps['clf'], 'coef_')}")

    # Use sklearn directly on pandas
    X_test = df[numeric_features].head(10).fillna(0)
    sklearn_predictions = sklearn_fitted.predict(X_test)
    print(f"Direct sklearn predictions: {sklearn_predictions[:5]}")

    # Use xorq for deferred execution
    xorq_predictions = fitted_xorq.predict(expr)
    result = xorq_predictions.execute()
    print(f"xorq predictions shape: {result.shape}")

    return fitted_xorq


# =============================================================================
# Example 2: predict_proba() and decision_function()
# =============================================================================


def example_probabilistic_outputs():
    """
    Demonstrate predict_proba() and decision_function() support.
    """
    print("\n" + "=" * 60)
    print("Example 2: Probabilistic Outputs")
    print("=" * 60)

    df = create_sample_data()
    numeric_features = ["age", "income", "score", "balance"]
    expr = xo.memtable(df)

    # Example with LogisticRegression (has predict_proba)
    print("\n--- LogisticRegression with predict_proba ---")
    sklearn_pipe = SklearnPipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )

    xorq_pipe = Pipeline.from_instance(sklearn_pipe)
    fitted = xorq_pipe.fit(expr, features=tuple(numeric_features), target="target")

    # Get class probabilities (packed format)
    proba_result = fitted.predict_proba(expr)
    proba_df = proba_result.execute()
    print(f"predict_proba result columns: {proba_df.columns.tolist()}")
    print(f"First row proba (packed): {proba_df['proba'].iloc[0]}")

    # Example with SVC (has decision_function)
    print("\n--- SVC with decision_function ---")
    sklearn_pipe_svc = SklearnPipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="linear")),
        ]
    )

    xorq_pipe_svc = Pipeline.from_instance(sklearn_pipe_svc)
    fitted_svc = xorq_pipe_svc.fit(
        expr, features=tuple(numeric_features), target="target"
    )

    # Get decision function values
    decision_result = fitted_svc.decision_function(expr)
    decision_df = decision_result.execute()
    print(f"decision_function result columns: {decision_df.columns.tolist()}")
    print(f"First row decision (packed): {decision_df['decision'].iloc[0]}")

    return fitted


# =============================================================================
# Example 3: Unregistered Transformers (Packed Format Fallback)
# =============================================================================


def example_unregistered_transformers():
    """
    Demonstrate that unregistered transformers automatically use packed format.
    """
    print("\n" + "=" * 60)
    print("Example 3: Unregistered Transformers (Auto Packed Format)")
    print("=" * 60)

    df = create_sample_data()
    numeric_features = ["age", "income", "score", "balance"]
    expr = xo.memtable(df)

    # MinMaxScaler is not explicitly registered - uses packed format fallback
    from sklearn.preprocessing import MinMaxScaler, RobustScaler

    print("\n--- MinMaxScaler (unregistered, uses packed format) ---")
    step = Step.from_instance_name(MinMaxScaler(), name="minmax")
    fitted = step.fit(expr, features=tuple(numeric_features))

    # Transform returns packed format
    result = fitted.transform_raw(expr)
    result_df = result.as_table().execute()
    print("Transform result type: packed Array[Struct{key, value}]")
    print(f"First row (packed): {result_df['transformed'].iloc[0][:2]}...")

    print("\n--- RobustScaler (unregistered, uses packed format) ---")
    step2 = Step.from_instance_name(RobustScaler(), name="robust")
    fitted2 = step2.fit(expr, features=tuple(numeric_features))
    result2 = fitted2.transform_raw(expr)
    result2_df = result2.as_table().execute()
    print(f"First row (packed): {result2_df['transformed'].iloc[0][:2]}...")

    return fitted


# =============================================================================
# Example 4: Regressor with Mixin Catch-All
# =============================================================================


def example_regressor_mixin():
    """
    Demonstrate RegressorMixin catch-all for regressors.
    """
    print("\n" + "=" * 60)
    print("Example 4: Regressor Mixin Catch-All")
    print("=" * 60)

    df = create_sample_data()
    numeric_features = ["age", "income", "score", "balance"]
    expr = xo.memtable(df)

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge

    # Ridge regression - uses RegressorMixin catch-all
    print("\n--- Ridge Regression ---")
    sklearn_pipe = SklearnPipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0)),
        ]
    )

    xorq_pipe = Pipeline.from_instance(sklearn_pipe)
    fitted = xorq_pipe.fit(expr, features=tuple(numeric_features), target="target")

    predictions = fitted.predict(expr)
    result = predictions.execute()
    print(f"Ridge predictions shape: {result.shape}")
    print(f"First 5 predictions: {result['predicted'].head().tolist()}")

    # GradientBoostingRegressor - also uses RegressorMixin
    print("\n--- GradientBoostingRegressor ---")
    step = Step.from_instance_name(
        GradientBoostingRegressor(n_estimators=10, max_depth=3),
        name="gbr",
    )
    fitted_gbr = step.fit(expr, features=tuple(numeric_features), target="target")

    # Get sklearn model back
    sklearn_gbr = fitted_gbr.to_sklearn()
    print(f"Fitted GBR n_estimators: {sklearn_gbr.n_estimators_}")

    return fitted


# =============================================================================
# Example 5: Complex Pipeline with Multiple Steps
# =============================================================================


def example_complex_pipeline():
    """
    Demonstrate a complex multi-step pipeline.
    """
    print("\n" + "=" * 60)
    print("Example 5: Complex Multi-Step Pipeline")
    print("=" * 60)

    df = create_sample_data()
    expr = xo.memtable(df)

    # Create individual steps
    from sklearn.feature_selection import SelectKBest, f_classif

    numeric_features = ["age", "income", "score", "balance"]

    # Multi-step sklearn pipeline
    sklearn_pipe = SklearnPipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(f_classif, k=3)),
            ("clf", RandomForestClassifier(n_estimators=10, random_state=42)),
        ]
    )

    # Wrap and fit
    xorq_pipe = Pipeline.from_instance(sklearn_pipe)
    fitted = xorq_pipe.fit(expr, features=tuple(numeric_features), target="target")

    # Get all fitted sklearn models
    sklearn_fitted = fitted.to_sklearn()
    print(f"Pipeline steps: {[name for name, _ in sklearn_fitted.steps]}")
    print(
        f"Imputer statistics: {sklearn_fitted.named_steps['imputer'].statistics_[:2]}..."
    )
    print(f"Scaler mean: {sklearn_fitted.named_steps['scaler'].mean_[:2]}...")
    print(
        f"Selected features mask: {sklearn_fitted.named_steps['selector'].get_support()}"
    )
    print(f"RF n_estimators: {sklearn_fitted.named_steps['clf'].n_estimators}")

    # Execute predictions
    result = fitted.predict(expr).execute()
    print(f"Predictions shape: {result.shape}")

    return fitted


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("sklearn ↔ xorq Interoperability Examples")
    print("=" * 60)

    # Run all examples
    example_basic_roundtrip()
    example_probabilistic_outputs()
    example_unregistered_transformers()
    example_regressor_mixin()
    example_complex_pipeline()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
