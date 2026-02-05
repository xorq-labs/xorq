#!/usr/bin/env python
"""
Production Preprocessing with ColumnTransformer
================================================

This intermediate vignette demonstrates real-world ML preprocessing patterns using
scikit-learn's ColumnTransformer to handle heterogeneous features (numeric and
categorical) with different preprocessing strategies.

What you'll learn:
1. **ColumnTransformer**: Applying different transformations to different feature types
2. **Imputation Strategies**: Handling missing values with median (numeric) and constant (categorical)
3. **Feature Scaling**: StandardScaler for numeric normalization
4. **Categorical Encoding**: OneHotEncoder for categorical features
5. **Nested Pipelines**: Composing multi-step transformations for each feature type
6. **Production Patterns**: Building realistic preprocessing pipelines for deferred execution

The Pipeline Architecture:
-------------------------
                        ColumnTransformer
                              |
                    +---------+---------+
                    |                   |
            Numeric Pipeline    Categorical Pipeline
                    |                   |
            SimpleImputer        SimpleImputer
            (strategy=median)    (strategy='missing')
                    |                   |
            StandardScaler       OneHotEncoder
                    |                   |
                    +--------+----------+
                             |
                    GradientBoostingClassifier

Key Pattern: ColumnTransformer allows you to apply different preprocessing pipelines
to different subsets of columns, which is essential for real-world datasets with
mixed feature types.

Dataset: Bank Marketing
-----------------------
Predict whether a customer will subscribe to a term deposit based on:
- Numeric features: age, balance, duration, campaign metrics
- Categorical features: job, education, marital status, contact method
Target: deposit (yes/no)
"""

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# =============================================================================
# CRITICAL IMPORTS: xorq Module Structure
# =============================================================================
import xorq.api as xo              # Main API
from xorq.caching import ParquetCache
from xorq.common.utils.defer_utils import deferred_read_csv
from xorq.expr.ml import train_test_splits
from xorq.expr.ml.pipeline_lib import Pipeline

# =============================================================================
# CONFIGURATION: Feature Definitions
# =============================================================================
# Organizing features by type is crucial for ColumnTransformer
TARGET_COLUMN = "deposit"

# Numeric features: continuous or discrete numeric values
NUMERIC_FEATURES = [
    "age",          # Customer age
    "balance",      # Account balance
    "day",          # Last contact day of month
    "duration",     # Last contact duration in seconds
    "campaign",     # Number of contacts during campaign
    "pdays",        # Days since last contact from previous campaign
    "previous",     # Number of contacts before this campaign
]

# Categorical features: discrete categories or labels
CATEGORICAL_FEATURES = [
    "job",          # Type of job
    "marital",      # Marital status
    "education",    # Education level
    "default",      # Has credit in default?
    "housing",      # Has housing loan?
    "loan",         # Has personal loan?
    "contact",      # Contact communication type
    "month",        # Last contact month
    "poutcome",     # Outcome of previous marketing campaign
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# =============================================================================
# STEP 1: Data Loading and Preparation
# =============================================================================
def get_bank_data_expr():
    """
    Load and prepare the bank marketing dataset.

    This function demonstrates:
    - Loading CSV data with deferred execution
    - Binary encoding of target variable
    - Working with xorq's built-in pins for data access

    The target is converted from "yes"/"no" strings to 1/0 integers,
    which is required by scikit-learn classifiers.
    """
    # Get connection
    con = xo.connect()

    # Load data with deferred_read_csv
    # This creates an expression without loading data into memory!
    expr = deferred_read_csv(
        path=xo.options.pins.get_path("bank-marketing"),
        con=con,
    )

    # Convert target to binary integer (yes=1, no=0)
    # Note: We use xo._ for clean column references
    expr = expr.mutate(
        **{TARGET_COLUMN: (xo._[TARGET_COLUMN] == "yes").cast("int")}
    )

    return expr


# =============================================================================
# STEP 2: Train/Test Split with Caching
# =============================================================================
def create_train_test_split(data_expr, test_size=0.5, random_seed=42):
    """
    Split data into train and test sets with caching.

    This demonstrates:
    - Using train_test_splits for reproducible splitting
    - Configuring cache for intermediate results
    - Returning deferred expressions

    Why 50% test size?
    This is just for demonstration - in production you'd typically use 20-30%.
    The larger test set helps visualize evaluation metrics.
    """
    # Set up cache backend
    # We use a temporary cache directory for this vignette
    cache = ParquetCache.from_kwargs(
        source=xo.connect(),
        relative_path="./tmp-cache",
        base_path=Path(".").absolute(),
    )

    # Perform train/test split
    train_table, test_table = data_expr.pipe(
        train_test_splits,
        test_sizes=[test_size, test_size],  # [train%, test%]
        num_buckets=2,                      # Number of splits
        random_seed=random_seed,            # For reproducibility
    )

    return train_table, test_table, cache


# =============================================================================
# STEP 3: Building the ColumnTransformer Preprocessing Pipeline
# =============================================================================
def create_preprocessing_pipeline():
    """
    Create a ColumnTransformer for heterogeneous feature preprocessing.

    This is the core pattern for production ML pipelines!

    ColumnTransformer allows you to:
    - Apply different transformations to different columns
    - Keep transformations organized and maintainable
    - Ensure consistent preprocessing between train and test

    Structure:
    ---------
    ColumnTransformer([
        (name, transformer, columns),  # For numeric features
        (name, transformer, columns),  # For categorical features
    ])

    Why nested pipelines?
    Each feature type needs multiple sequential transformations:
    1. Imputation (fill missing values)
    2. Scaling/Encoding (normalize or encode)
    """

    # Define numeric preprocessing pipeline
    numeric_pipeline = SklearnPipeline([
        # Step 1: Impute missing values with median
        # Median is robust to outliers, good for numeric features
        ("imputer", SimpleImputer(strategy="median")),

        # Step 2: Standardize features (zero mean, unit variance)
        # Essential for models sensitive to feature scale (like GBM)
        ("scaler", StandardScaler()),
    ])

    # Define categorical preprocessing pipeline
    categorical_pipeline = SklearnPipeline([
        # Step 1: Impute missing values with constant "missing"
        # Creates an explicit "missing" category rather than dropping rows
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),

        # Step 2: One-hot encode categorical variables
        # handle_unknown="ignore" prevents errors on unseen categories
        # sparse_output=False returns dense arrays (required by some models)
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Combine into ColumnTransformer
    # This routes each feature set to its appropriate pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            # (name, transformer, columns)
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


# =============================================================================
# STEP 4: Complete ML Pipeline with Model
# =============================================================================
def create_sklearn_pipeline():
    """
    Create the complete sklearn pipeline: preprocessing + model.

    This demonstrates:
    - Chaining preprocessing with modeling
    - Using GradientBoostingClassifier (more sophisticated than LogisticRegression)
    - Configuring model hyperparameters

    Pipeline structure:
    -------------------
    preprocessing (ColumnTransformer)
        → gradient boosting classifier

    Why GradientBoostingClassifier?
    - Handles non-linear relationships
    - Robust to feature scale (but we still normalize for best practices)
    - Good baseline for binary classification
    """
    # Get preprocessor
    preprocessor = create_preprocessing_pipeline()

    # Create full pipeline
    sklearn_pipeline = SklearnPipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(
            n_estimators=50,      # Number of boosting stages
            random_state=42       # For reproducibility
        )),
    ])

    return sklearn_pipeline


# =============================================================================
# STEP 5: Training in xorq's Deferred Framework
# =============================================================================
def fit_pipeline_deferred(train_table, cache):
    """
    Fit the sklearn pipeline in xorq's deferred framework.

    This demonstrates:
    - Wrapping sklearn pipelines for deferred execution
    - Using Pipeline.from_instance()
    - Fitting with cache support

    Key Pattern: xorq's Pipeline wrapper makes sklearn pipelines work
    with deferred expressions. The fit() operation itself becomes part
    of the deferred computation graph!
    """
    # Create sklearn pipeline
    sklearn_pipeline = create_sklearn_pipeline()

    # Wrap in xorq Pipeline for deferred execution
    xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)

    # Fit pipeline on training data
    # This creates a fitted pipeline expression (still deferred!)
    fitted_pipeline = xorq_pipeline.fit(
        train_table,
        features=tuple(ALL_FEATURES),  # All numeric + categorical features
        target=TARGET_COLUMN,
        cache=cache,                   # Enable caching for fitted artifacts
    )

    return fitted_pipeline


# =============================================================================
# STEP 6: Prediction and Evaluation
# =============================================================================
def make_predictions(fitted_pipeline, test_table):
    """
    Generate predictions on test data.

    This demonstrates:
    - Making predictions with fitted pipeline
    - Working with deferred prediction expressions

    The predictions are still deferred at this point!
    They only execute when you call .execute()
    """
    # Generate predictions
    predicted_test = fitted_pipeline.predict(test_table)

    return predicted_test


def evaluate_predictions(predictions_df):
    """
    Evaluate model performance with sklearn metrics.

    This demonstrates:
    - Computing standard classification metrics
    - Interpreting confusion matrix
    - Displaying classification report

    Note: This function operates on executed (materialized) predictions,
    not deferred expressions.
    """
    # Extract predictions and ground truth
    y_true = predictions_df[TARGET_COLUMN]
    y_pred = predictions_df["predicted"]

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(f"TN (True Negative):  {cm[0, 0]:5d}  |  FP (False Positive): {cm[0, 1]:5d}")
    print(f"FN (False Negative): {cm[1, 0]:5d}  |  TP (True Positive):  {cm[1, 1]:5d}")

    # Compute AUC score
    try:
        auc = roc_auc_score(y_true, y_pred)
        print(f"\nAUC Score: {auc:.4f}")
    except Exception as e:
        print(f"\nAUC Score: Could not compute ({str(e)})")

    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Deposit', 'Deposit']))


# =============================================================================
# STEP 7: Pipeline Orchestration
# =============================================================================
def build_complete_pipeline():
    """
    Orchestrate the complete ML pipeline from data to predictions.

    This ties together:
    1. Data loading
    2. Train/test split
    3. Preprocessing pipeline creation
    4. Model training
    5. Predictions

    Everything remains deferred until execute() is called!
    """
    print("\n" + "="*70)
    print("BUILDING DEFERRED PIPELINE (Production Preprocessing with ColumnTransformer)")
    print("="*70)

    # Step 1: Load data
    print("\n[1/5] Loading bank marketing data...")
    data_expr = get_bank_data_expr()
    print(f"✓ Data expression created")
    print(f"  Features: {len(ALL_FEATURES)} ({len(NUMERIC_FEATURES)} numeric, {len(CATEGORICAL_FEATURES)} categorical)")
    print(f"  Target: {TARGET_COLUMN}")

    # Step 2: Train/test split
    print("\n[2/5] Creating train/test split...")
    train_table, test_table, cache = create_train_test_split(data_expr)
    print("✓ Train/test split configured (50/50 split)")
    print("  Cache: ParquetCache enabled")

    # Step 3: Build preprocessing pipeline
    print("\n[3/5] Building ColumnTransformer preprocessing pipeline...")
    preprocessor = create_preprocessing_pipeline()
    print("✓ Preprocessor created")
    print("  Numeric pipeline: SimpleImputer(median) → StandardScaler")
    print("  Categorical pipeline: SimpleImputer('missing') → OneHotEncoder")

    # Step 4: Fit pipeline
    print("\n[4/5] Fitting deferred pipeline...")
    fitted_pipeline = fit_pipeline_deferred(train_table, cache)
    print("✓ Pipeline fitted (deferred)")
    print("  Model: GradientBoostingClassifier(n_estimators=50)")

    # Step 5: Make predictions
    print("\n[5/5] Creating prediction expressions...")
    predicted_test = make_predictions(fitted_pipeline, test_table)
    print("✓ Predictions configured (deferred)")

    return {
        'fitted_pipeline': fitted_pipeline,
        'predictions': predicted_test,
        'train_table': train_table,
        'test_table': test_table,
    }


# =============================================================================
# STEP 8: Execution and Display
# =============================================================================
def execute_and_evaluate(pipeline_dict):
    """
    Execute the deferred pipeline and evaluate results.

    THIS IS WHERE COMPUTATION HAPPENS!
    All the deferred expressions we built finally execute here.
    """
    print("\n" + "="*70)
    print("EXECUTING DEFERRED PIPELINE")
    print("="*70)

    import time

    # Execute predictions
    print("\nExecuting predictions on test set...")
    start = time.time()
    predictions_df = pipeline_dict['predictions'].execute()
    elapsed = time.time() - start

    print(f"✓ Predictions executed in {elapsed:.2f}s")
    print(f"  Test samples: {len(predictions_df)}")

    # Evaluate performance
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)

    evaluate_predictions(predictions_df)

    return predictions_df


# =============================================================================
# MAIN: Complete Workflow
# =============================================================================
def main():
    """
    Main execution demonstrating the complete workflow.

    Key Learning Points:
    1. ColumnTransformer enables heterogeneous preprocessing
    2. Nested pipelines organize multi-step transformations
    3. Different imputation strategies for different feature types
    4. Everything stays deferred until explicit execution
    5. Production-ready preprocessing patterns
    """
    print("\n" + "="*70)
    print("XORQ VIGNETTE: Bank Marketing with ColumnTransformer")
    print("="*70)
    print("\nThis vignette demonstrates production preprocessing patterns:")
    print("  • ColumnTransformer for heterogeneous features")
    print("  • Separate pipelines for numeric and categorical features")
    print("  • Imputation strategies (median vs constant)")
    print("  • StandardScaler and OneHotEncoder")
    print("  • GradientBoostingClassifier")
    print("  • Full deferred execution with xorq")

    # Build pipeline
    pipeline_dict = build_complete_pipeline()

    # Execute and evaluate
    predictions_df = execute_and_evaluate(pipeline_dict)

    print("\n" + "="*70)
    print("VIGNETTE COMPLETE!")
    print("="*70)
    print("\nWhat you learned:")
    print("  ✓ Using ColumnTransformer for mixed feature types")
    print("  ✓ Building nested sklearn pipelines")
    print("  ✓ Applying different imputation strategies")
    print("  ✓ Scaling numeric and encoding categorical features")
    print("  ✓ Training complex models in deferred execution")
    print("\nNext steps:")
    print("  • Experiment with different imputation strategies")
    print("  • Try different scalers (RobustScaler, MinMaxScaler)")
    print("  • Add feature engineering steps to the pipelines")
    print("  • Tune GradientBoostingClassifier hyperparameters")
    print("  • Add cross-validation for model selection")

    return pipeline_dict, predictions_df


if __name__ == "__main__":
    # Set a flag for pytest to detect successful execution
    pipeline, predictions = main()
    pytest_examples_passed = True
