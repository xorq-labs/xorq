"""
sklearn ColumnTransformer with xorq Example

This example demonstrates using sklearn's ColumnTransformer with xorq
for complex preprocessing pipelines that handle different feature types.

Key features demonstrated:
- ColumnTransformer with mixed numeric/categorical preprocessing
- OneHotEncoder for categorical features (dynamic schema)
- StandardScaler for numeric features
- Packed format handles dynamic output schemas automatically
- Full round-trip with to_sklearn()

The key insight: sklearn handles all the complexity internally.
xorq just wraps it for deferred execution.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import xorq.api as xo
from xorq.expr.ml.pipeline_lib import Step


def create_mixed_data(include_missing=False):
    """Create dataset with numeric and categorical features."""
    np.random.seed(42)
    n_samples = 500

    df = pd.DataFrame(
        {
            # Numeric features
            "age": np.random.randint(18, 80, n_samples).astype(float),
            "income": np.random.exponential(50000, n_samples),
            "credit_score": np.random.randint(300, 850, n_samples).astype(float),
            "account_balance": np.random.normal(10000, 5000, n_samples),
            # Categorical features
            "gender": np.random.choice(["M", "F", "Other"], n_samples),
            "education": np.random.choice(
                ["High School", "Bachelor", "Master", "PhD"], n_samples
            ),
            "employment": np.random.choice(
                ["Employed", "Self-Employed", "Unemployed", "Retired"], n_samples
            ),
            "region": np.random.choice(
                ["North", "South", "East", "West", "Central"], n_samples
            ),
            # Target
            "approved": np.random.randint(0, 2, n_samples),
        }
    )

    # Optionally add some missing values
    if include_missing:
        df.loc[np.random.choice(n_samples, 30, replace=False), "income"] = np.nan
        df.loc[np.random.choice(n_samples, 20, replace=False), "credit_score"] = np.nan

    return df


# =============================================================================
# Example 1: ColumnTransformer as a Single Step
# =============================================================================


def example_column_transformer_step():
    """
    Use ColumnTransformer as a single deferred Step.

    The ColumnTransformer handles:
    - Routing features to appropriate sub-transformers
    - Fitting all sub-transformers
    - Concatenating outputs
    - Providing get_feature_names_out()

    xorq treats it as a single opaque unit.
    """
    print("\n" + "=" * 60)
    print("Example 1: ColumnTransformer as Single Step")
    print("=" * 60)

    df = create_mixed_data()
    expr = xo.memtable(df)

    numeric_features = ["age", "income", "credit_score", "account_balance"]
    categorical_features = ["gender", "education", "employment", "region"]

    # Create ColumnTransformer with nested sklearn Pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                SklearnPipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    # Wrap as xorq Step - uses packed format for dynamic output
    step = Step.from_instance_name(preprocessor, name="preprocessor")
    print("Created Step from ColumnTransformer")

    # Fit (deferred)
    all_features = tuple(numeric_features + categorical_features)
    fitted = step.fit(expr, features=all_features)
    print(f"Fitted step type: {type(fitted)}")

    # Get sklearn ColumnTransformer back
    sklearn_ct = fitted.to_sklearn()
    print("\nFitted sklearn ColumnTransformer:")
    print(f"  Transformers: {[name for name, _, _ in sklearn_ct.transformers_]}")

    # Check output feature names
    output_names = sklearn_ct.get_feature_names_out()
    print(f"  Output features count: {len(output_names)}")
    print(f"  First 5 output names: {output_names[:5].tolist()}")
    print(
        f"  Categorical output names: {[n for n in output_names if n.startswith('cat__')][:5]}"
    )

    # Transform produces packed format
    result = fitted.transform_raw(expr)
    result_df = result.as_table().execute()
    print("\nTransform result (packed format):")
    print(f"  Columns: {result_df.columns.tolist()}")
    first_row = result_df["transformed"].iloc[0]
    print(f"  First row has {len(first_row)} features")
    print(f"  Sample features: {[item['key'] for item in first_row[:3]]}")

    return fitted


# =============================================================================
# Example 2: Full Pipeline with ColumnTransformer + Classifier
# =============================================================================


def example_full_pipeline_column_transformer():
    """
    Complete ML pipeline with ColumnTransformer preprocessing and classifier.

    NOTE: For complex sklearn pipelines with ColumnTransformer, wrap the entire
    sklearn Pipeline as a single Step (not decomposed). This is because xorq's
    Pipeline.from_instance() decomposes into steps, and intermediate packed format
    output from ColumnTransformer can't be consumed by subsequent steps.
    """
    print("\n" + "=" * 60)
    print("Example 2: Full Pipeline (ColumnTransformer + Classifier)")
    print("=" * 60)

    df = create_mixed_data()
    expr = xo.memtable(df)

    numeric_features = ["age", "income", "credit_score", "account_balance"]
    categorical_features = ["gender", "education", "employment", "region"]

    # Create full sklearn pipeline with nested ColumnTransformer
    sklearn_pipeline = SklearnPipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        (
                            "num",
                            SklearnPipeline(
                                [
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            numeric_features,
                        ),
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            categorical_features,
                        ),
                    ],
                    remainder="drop",
                ),
            ),
            ("classifier", LogisticRegression(max_iter=500)),
        ]
    )

    # Wrap the ENTIRE sklearn pipeline as a single Step
    # This keeps all sklearn processing together, avoiding packed format issues
    step = Step.from_instance_name(sklearn_pipeline, name="full_pipeline")
    print("Created xorq Step wrapping entire sklearn Pipeline")

    # Fit (deferred)
    all_features = tuple(numeric_features + categorical_features)
    fitted = step.fit(expr, features=all_features, target="approved")
    print("Fitted step")

    # Get fitted sklearn pipeline back
    sklearn_fitted = fitted.to_sklearn()
    print("\nFitted sklearn Pipeline:")
    print(f"  Steps: {[name for name, _ in sklearn_fitted.steps]}")

    # Check preprocessor
    preprocessor = sklearn_fitted.named_steps["preprocessor"]
    print(
        f"  Preprocessor output features: {len(preprocessor.get_feature_names_out())}"
    )

    # Check classifier
    classifier = sklearn_fitted.named_steps["classifier"]
    print(f"  Classifier coef shape: {classifier.coef_.shape}")
    print(f"  Classifier classes: {classifier.classes_}")

    # Execute predictions with xorq (deferred)
    predictions = fitted.predict(expr)
    result = predictions.execute()
    print("\nPredictions:")
    print(f"  Shape: {result.shape}")
    print(f"  Columns: {result.columns.tolist()}")
    print(f"  First 5 predictions: {result['predicted'].head().tolist()}")

    # Get probabilities
    proba = fitted.predict_proba(expr)
    proba_result = proba.execute()
    print("\nProbabilities (packed format):")
    print(f"  First row: {proba_result['proba'].iloc[0]}")

    # Compare with direct sklearn prediction
    X_test = df[numeric_features + categorical_features].head(5)
    sklearn_pred = sklearn_fitted.predict(X_test)
    print(f"\nDirect sklearn predictions on same data: {sklearn_pred.tolist()}")

    return fitted


# =============================================================================
# Example 3: ColumnTransformer with remainder="passthrough"
# =============================================================================


def example_passthrough_columns():
    """
    ColumnTransformer with remainder="passthrough" to keep unprocessed columns.
    """
    print("\n" + "=" * 60)
    print("Example 3: ColumnTransformer with Passthrough")
    print("=" * 60)

    df = create_mixed_data()
    # Add an ID column that should pass through
    df["customer_id"] = range(len(df))
    expr = xo.memtable(df)

    numeric_features = ["age", "income"]
    categorical_features = ["gender"]

    # ColumnTransformer with passthrough for remaining columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(sparse_output=False), categorical_features),
        ],
        remainder="passthrough",  # Keep other columns
    )

    step = Step.from_instance_name(preprocessor, name="preprocessor_passthrough")
    all_features = tuple(
        numeric_features + categorical_features + ["credit_score", "customer_id"]
    )
    fitted = step.fit(expr, features=all_features)

    sklearn_ct = fitted.to_sklearn()
    output_names = sklearn_ct.get_feature_names_out()
    print("Output feature names with passthrough:")
    print(f"  {output_names.tolist()}")
    print("  Note: 'remainder__' prefix for passthrough columns")

    return fitted


# =============================================================================
# Example 4: Nested Pipelines in ColumnTransformer
# =============================================================================


def example_nested_pipelines():
    """
    ColumnTransformer with nested Pipeline objects for each transformer.
    """
    print("\n" + "=" * 60)
    print("Example 4: Nested Pipelines in ColumnTransformer")
    print("=" * 60)

    df = create_mixed_data()
    expr = xo.memtable(df)

    numeric_features = ["age", "income", "credit_score", "account_balance"]
    categorical_features = ["gender", "education", "employment", "region"]

    # Complex nested structure
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric_pipeline",
                SklearnPipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical_pipeline",
                SklearnPipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="missing"),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
    )

    step = Step.from_instance_name(preprocessor, name="nested_preprocessor")
    all_features = tuple(numeric_features + categorical_features)
    fitted = step.fit(expr, features=all_features)

    sklearn_ct = fitted.to_sklearn()
    print("Nested pipeline structure:")
    for name, transformer, cols in sklearn_ct.transformers_:
        print(f"  {name}:")
        print(f"    Columns: {cols}")
        if hasattr(transformer, "steps"):
            print(f"    Sub-steps: {[s[0] for s in transformer.steps]}")

    output_names = sklearn_ct.get_feature_names_out()
    print(f"\nOutput features: {len(output_names)} total")
    print(f"  Numeric: {[n for n in output_names if n.startswith('numeric')]}")
    print(
        f"  Categorical (sample): {[n for n in output_names if n.startswith('categorical')][:5]}..."
    )

    return fitted


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("sklearn ColumnTransformer with xorq Examples")
    print("=" * 60)

    # Run all examples
    example_column_transformer_step()
    example_full_pipeline_column_transformer()
    example_passthrough_columns()
    example_nested_pipelines()

    print("\n" + "=" * 60)
    print("All ColumnTransformer examples completed!")
    print("=" * 60)
