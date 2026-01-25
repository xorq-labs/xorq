#!/usr/bin/env python
"""
Introduction to xorq: Penguin Species Classification
====================================================

This beginner-friendly vignette introduces core xorq concepts through a complete
machine learning pipeline that classifies penguin species. Perfect as your first
xorq project!

What you'll learn:
1. **Deferred Execution**: How xorq builds computation graphs without executing
2. **Expression Composition**: Creating reusable, composable data pipelines
3. **ML Integration**: Using scikit-learn models in deferred expressions
4. **Caching Strategy**: Efficient intermediate result storage with ParquetCache
5. **Metrics & Visualization**: Computing ML metrics and ROC curves in deferred mode

The Pipeline Flow:
-----------------
    data_expr → train_expr → fitted_pipeline → predictions_with_probs → roc_expr
         ↓           ↓              ↓                    ↓                  ↓
    (deferred)  (deferred)     (deferred)         (deferred)         (deferred)
                                                                          ↓
                                                                     execute()

Key Insight: Everything stays deferred until you explicitly call execute().
This enables optimization, caching, and lazy evaluation across your entire pipeline.

Dataset: Palmer Penguins
-----------------------
We'll classify three penguin species (Adelie, Chinstrap, Gentoo) using just two
features: bill length and depth. This simplicity helps focus on xorq patterns
rather than complex ML.
"""

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
import toolz
import pickle
import base64

# =============================================================================
# CRITICAL IMPORTS: Understanding xorq's Module Structure
# =============================================================================
import xorq.api as xo              # Main API for catalog, connections, and utilities
from xorq.api import _             # The underscore for elegant column references
from xorq.vendor import ibis       # xorq's enhanced ibis (ALWAYS use this!)

# Why xorq.vendor.ibis instead of regular ibis?
# xorq extends ibis with custom operators essential for deferred execution:
# - .cache() for intermediate result storage
# - .into_backend() for multi-backend support
# - ExprScalarUDF for passing expression results to UDFs
# These extensions are what make xorq's deferred model possible!

from xorq.caching import ParquetCache               # Efficient cache backend
from xorq.expr.ml.pipeline_lib import Pipeline      # ML pipeline wrapper
from xorq.expr.ml.metrics import deferred_sklearn_metric  # Deferred metrics
from xorq.expr.udf import scalar, agg               # User-defined functions
import xorq.expr.datatypes as dt                    # Data type definitions

# Import metrics for evaluation
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =============================================================================
# CONFIGURATION: Keep It Simple for Learning
# =============================================================================
# We use only two features to keep the model simple and interpretable
FEATURES = ["bill_length_mm", "bill_depth_mm"]
TARGET = "species"  # Three classes: Adelie, Chinstrap, Gentoo


# =============================================================================
# STEP 1: Data Loading and Cleaning
# =============================================================================
def get_data_expr():
    """
    Load and clean the penguins dataset.

    This function demonstrates:
    - Loading built-in example data
    - Filtering with the underscore (_) pattern for clean column references
    - Returning a deferred expression (no data loaded yet!)

    Key Pattern: The underscore (_) represents "current expression's columns"
    This allows writing _.column instead of expr['column'], making code cleaner.
    """
    # Load the example dataset (returns an expression, not actual data!)
    penguins = xo.examples.penguins.fetch()

    # Filter out rows with missing values
    # Note: We use _ for column references - this is idiomatic xorq
    clean_data = penguins.filter(
        _.bill_length_mm.notnull()      # Remove nulls from bill length
        & _.bill_depth_mm.notnull()      # Remove nulls from bill depth
        & _.flipper_length_mm.notnull()  # Remove nulls from flipper length
        & _.body_mass_g.notnull()        # Remove nulls from body mass
        & _.species.notnull()            # Remove nulls from target
    )

    # Return the expression - still no execution!
    return clean_data


# =============================================================================
# STEP 2: Train/Test Split with Caching
# =============================================================================
def create_train_test_expr(data_expr, test_size=0.2, random_seed=42):
    """
    Split data into train/test sets and cache them.

    This function demonstrates:
    - Using xorq's built-in train_test_splits for reproducible splitting
    - Caching intermediate results with ParquetCache
    - Maintaining deferred execution through the split

    Why cache here?
    The train/test sets are used multiple times (training, prediction, evaluation).
    Caching prevents recomputation of the split, improving performance.
    """
    # Use xorq's train_test_splits for deterministic splitting
    # num_buckets controls the granularity of the split
    train_expr, test_expr = xo.train_test_splits(
        data_expr,
        test_sizes=test_size,     # 20% for testing
        num_buckets=1000,          # Higher = more precise split
        random_seed=random_seed,   # For reproducibility
    )

    # Create a cache backend - ParquetCache is efficient for tabular data
    cache = ParquetCache.from_kwargs()

    # Cache both expressions and return
    # The .cache() method stores results after first execution
    return train_expr.cache(cache), test_expr.cache(cache)


# =============================================================================
# STEP 3: Model Training with xorq Pipeline
# =============================================================================
def create_fitted_pipeline_expr(train_expr, model_params=None):
    """
    Create and fit a scikit-learn pipeline within xorq's deferred framework.

    This function demonstrates:
    - Wrapping sklearn pipelines for deferred execution
    - Configuring model hyperparameters
    - Fitting models on expression data (still deferred!)

    The magic: Even model fitting is deferred! The actual training only
    happens when you execute expressions that depend on this fitted pipeline.
    """
    # Default hyperparameters for LogisticRegression
    params = model_params or {
        "classifier__C": 0.1,                    # Regularization strength
        "classifier__penalty": "l2",             # L2 regularization
        "classifier__max_iter": 2000,            # Max iterations for convergence
        "classifier__random_state": 42           # Reproducibility
    }

    # Create a standard scikit-learn pipeline
    # This is your normal sklearn code!
    sklearn_pipeline = SkPipeline([
        ("scaler", StandardScaler()),           # Normalize features
        ("classifier", LogisticRegression())    # Logistic regression classifier
    ]).set_params(**params)

    # Wrap the sklearn pipeline in xorq's Pipeline
    # This enables deferred execution of fit/predict operations
    xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)

    # Fit the pipeline on training data
    # IMPORTANT: This returns a fitted pipeline expression, not actual fitted model!
    fitted_pipeline = xorq_pipeline.fit(
        train_expr,
        features=FEATURES,  # Which columns to use as features
        target=TARGET       # Which column to predict
    )

    return fitted_pipeline


# =============================================================================
# STEP 4: Utility Function - as_struct Pattern
# =============================================================================
@toolz.curry
def as_struct(expr, name=None):
    """
    Bundle all columns into a single struct column.

    This utility pattern is useful when you want to:
    - Preserve all original columns through transformations
    - Pass multiple columns as a single unit
    - Avoid losing columns during operations

    The @toolz.curry decorator makes this function partially applicable,
    allowing clean usage in pipelines: expr.mutate(as_struct(name="original"))
    """
    # Create a struct containing all columns
    struct = xo.struct({c: expr[c] for c in expr.columns})
    # Optionally name the struct column
    return struct.name(name) if name else struct


# =============================================================================
# STEP 5: Predictions with Probability Extraction
# =============================================================================
def create_predictions_with_probs_expr(fitted_pipeline, test_expr):
    """
    Generate predictions and extract individual class probabilities.

    This function demonstrates:
    - Making predictions with predict_proba
    - Extracting array elements from predicted_proba column
    - Using xo.cases() for conditional logic (from xorq's vendored ibis!)
    - Creating interpretable prediction columns

    Pattern: Convert probability arrays into named columns for each class.
    This makes downstream analysis and visualization much easier.
    """
    # Get probability predictions (returns array of probabilities per class)
    predictions_proba = fitted_pipeline.predict_proba(test_expr)

    # Transform the probability array into individual columns
    predictions_with_probs = predictions_proba.mutate(
        # Extract probability for each species
        # predicted_proba is an array, we access elements by index
        prob_Adelie=predictions_proba['predicted_proba'][0],      # First class
        prob_Chinstrap=predictions_proba['predicted_proba'][1],   # Second class
        prob_Gentoo=predictions_proba['predicted_proba'][2],      # Third class

        # Determine final prediction based on highest probability
        # xo.cases() is like SQL CASE WHEN, but for expressions!
        predicted=xo.cases(
            # If Adelie has highest probability
            (
                (predictions_proba['predicted_proba'][0] >= predictions_proba['predicted_proba'][1]) &
                (predictions_proba['predicted_proba'][0] >= predictions_proba['predicted_proba'][2]),
                'Adelie'
            ),
            # If Chinstrap has highest probability
            (
                (predictions_proba['predicted_proba'][1] >= predictions_proba['predicted_proba'][0]) &
                (predictions_proba['predicted_proba'][1] >= predictions_proba['predicted_proba'][2]),
                'Chinstrap'
            ),
            # Otherwise, it's Gentoo
            else_='Gentoo'
        )
    )

    return predictions_with_probs, predictions_proba


# =============================================================================
# STEP 6: ROC Visualization with UDAF (Advanced Pattern)
# =============================================================================
def create_roc_analysis_udf():
    """
    Create a User-Defined Aggregate Function for ROC curve generation.

    This advanced pattern demonstrates:
    - Creating complex visualizations within deferred execution
    - Using UDAFs to aggregate all data for analysis
    - Returning binary data (images) from expressions

    How it works:
    1. The UDAF receives all prediction data as a pandas DataFrame
    2. It computes ROC curves and AUC scores
    3. It generates a matplotlib plot
    4. It returns everything as pickled binary data

    This keeps visualization generation deferred until execution!
    """
    def compute_roc(df):
        """
        The actual UDAF function that computes ROC analysis.

        This runs on the executor when the expression is evaluated.
        It receives a pandas DataFrame with all the prediction data.
        """
        try:
            from sklearn.metrics import roc_auc_score, roc_curve, auc
            from sklearn.preprocessing import label_binarize
            import matplotlib.pyplot as plt
            import numpy as np
            import io

            # Extract true labels and predictions
            y_true = df[TARGET].values
            y_pred = df['predicted'].values
            classes = sorted(df[TARGET].unique())
            n_classes = len(classes)

            # Get probability scores for ROC computation
            prob_cols = ['prob_' + cls for cls in classes]
            if all(col in df.columns for col in prob_cols):
                # Use actual probabilities if available
                y_score = df[prob_cols].values
            else:
                # Fallback: create synthetic probabilities
                # This handles cases where probabilities weren't computed
                y_score = np.zeros((len(df), n_classes))
                for i, pred in enumerate(y_pred):
                    y_score[i, classes.index(pred)] = 0.9
                    y_score[i, :] = y_score[i, :] / y_score[i, :].sum()

            # Binarize labels for multi-class ROC
            y_binarized = label_binarize(y_true, classes=classes)
            roc_scores = {}

            # Compute ROC scores based on number of classes
            if n_classes == 2:
                # Binary classification
                y_binarized = y_binarized.ravel()
                roc_scores['binary'] = roc_auc_score(y_binarized, y_score[:, 1])
            else:
                # Multi-class classification (our case with 3 penguin species)
                per_class = {}
                for i, cls in enumerate(classes):
                    # Compute AUC for each class vs rest
                    per_class[cls] = roc_auc_score(y_binarized[:, i], y_score[:, i])
                roc_scores['per_class'] = per_class

                # Compute aggregate metrics
                roc_scores['macro'] = roc_auc_score(
                    y_binarized, y_score, multi_class='ovr', average='macro'
                )
                roc_scores['weighted'] = roc_auc_score(
                    y_binarized, y_score, multi_class='ovr', average='weighted'
                )

            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = plt.colormaps['tab10'](np.linspace(0, 1, n_classes))

            # Plot ROC curve for each class
            for i, (cls, color) in enumerate(zip(classes, colors)):
                if n_classes == 2:
                    fpr, tpr, _ = roc_curve(y_binarized, y_score[:, 1] if i == 1 else 1 - y_score[:, 1])
                else:
                    fpr, tpr, _ = roc_curve(y_binarized[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=color, lw=2, label=f'{cls} (AUC = {roc_auc:.3f})')

            # Add reference line (random classifier)
            ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves - Penguin Species Classification')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)

            # Convert plot to base64 string for storage
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            plt.close()  # Important: prevent memory leaks
            buffer.seek(0)
            plot_b64 = base64.b64encode(buffer.read()).decode('utf-8')

            # Return results as pickled dictionary
            return pickle.dumps({
                'status': 'success',
                'roc_scores': roc_scores,
                'plot_base64': plot_b64,
                'n_samples': len(df)
            })

        except Exception as e:
            import traceback
            # Return error information for debugging
            return pickle.dumps({
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            })

    return compute_roc


def create_roc_expr(predictions_with_probs):
    """
    Create a deferred ROC analysis expression.

    This wraps the ROC UDAF in an aggregation expression that:
    - Remains deferred until execution
    - Computes ROC curves for all prediction data
    - Returns results as binary data
    """
    # Get the UDAF function
    roc_udf_fn = create_roc_analysis_udf()

    # Get schema from the predictions expression
    schema = predictions_with_probs.schema()

    # Create the aggregation UDF
    roc_udf = agg.pandas_df(
        fn=roc_udf_fn,
        schema=schema,
        return_type=dt.binary,  # Returns pickled binary data
        name="compute_roc_analysis"
    )

    # Create aggregation expression
    # This aggregates all rows into a single ROC analysis result
    roc_expr = predictions_with_probs.aggregate([
        roc_udf.on_expr(predictions_with_probs).name('roc_analysis')
    ])

    return roc_expr


# =============================================================================
# STEP 7: Deferred Metrics Computation
# =============================================================================
def create_metric_expressions(predictions, predictions_proba):
    """
    Create deferred metric expressions for model evaluation.

    This demonstrates:
    - Computing multiple metrics in deferred mode
    - Using deferred_sklearn_metric for standard ML metrics
    - Handling both predictions and probabilities

    Each metric is an expression that computes when executed.
    This allows selective computation - only calculate metrics you need!
    """
    metrics = {
        # Classification accuracy
        'accuracy': deferred_sklearn_metric(
            expr=predictions,
            target=TARGET,
            pred_col="predicted",
            metric_fn=accuracy_score,
        ),

        # Precision (weighted average across classes)
        'precision': deferred_sklearn_metric(
            expr=predictions,
            target=TARGET,
            pred_col="predicted",
            metric_fn=precision_score,
            metric_kwargs={"average": "weighted", "zero_division": 0},
        ),

        # Recall (weighted average across classes)
        'recall': deferred_sklearn_metric(
            expr=predictions,
            target=TARGET,
            pred_col="predicted",
            metric_fn=recall_score,
            metric_kwargs={"average": "weighted", "zero_division": 0},
        ),

        # F1 score (harmonic mean of precision and recall)
        'f1': deferred_sklearn_metric(
            expr=predictions,
            target=TARGET,
            pred_col="predicted",
            metric_fn=f1_score,
            metric_kwargs={"average": "weighted", "zero_division": 0},
        ),

        # ROC-AUC (requires probability predictions)
        'roc_auc': deferred_sklearn_metric(
            expr=predictions_proba,
            target=TARGET,
            pred_col="predicted_proba",
            metric_fn=roc_auc_score,
            metric_kwargs={"multi_class": "ovr", "average": "weighted"},
        )
    }

    return metrics


# =============================================================================
# STEP 8: Pipeline Orchestration
# =============================================================================
def build_complete_pipeline():
    """
    Orchestrate the complete ML pipeline.

    This function ties everything together:
    1. Load and clean data
    2. Split into train/test
    3. Train the model
    4. Make predictions
    5. Compute metrics
    6. Generate ROC visualization

    Important: This entire function executes instantly because everything
    is deferred! No actual computation happens until you call execute().
    """
    # Step 1: Get clean data expression
    data_expr = get_data_expr()

    # Step 2: Create train/test split with caching
    train_expr, test_expr = create_train_test_expr(data_expr)

    # Step 3: Fit the ML pipeline
    fitted_pipeline = create_fitted_pipeline_expr(train_expr)

    # Step 4: Generate predictions with probabilities
    predictions_with_probs, predictions_proba = create_predictions_with_probs_expr(
        fitted_pipeline, test_expr
    )

    # Step 5: Get regular predictions for metrics
    predictions = fitted_pipeline.predict(test_expr)

    # Step 6: Create ROC analysis expression
    roc_expr = create_roc_expr(predictions_with_probs)

    # Step 7: Create metric expressions
    metrics = create_metric_expressions(predictions, predictions_proba)

    # Return everything as a dictionary for selective execution
    return {
        'roc_expr': roc_expr,
        'metrics': metrics,
        'predictions': predictions,
        'predictions_proba': predictions_proba,
        'fitted_pipeline': fitted_pipeline
    }


# =============================================================================
# STEP 9: Execution and Display
# =============================================================================
def execute_and_display(pipeline_dict):
    """
    Execute the deferred pipeline and display results.

    THIS IS WHERE THE MAGIC HAPPENS!
    All the deferred expressions we built finally execute here.

    Notice:
    - Each metric_expr.execute() triggers computation
    - We time each execution to show performance
    - ROC visualization is computed and saved to disk
    """
    import time

    print("\n" + "="*60)
    print("EXECUTING DEFERRED PIPELINE")
    print("="*60)
    print("\nMetrics:")

    # Execute each metric and display results
    for name, metric_expr in pipeline_dict['metrics'].items():
        start = time.time()
        value = metric_expr.execute()  # <-- EXECUTION happens here!
        elapsed = time.time() - start
        print(f"  {name:10s}: {value:.4f} (executed in {elapsed:.3f}s)")

    print("\nROC Analysis:")

    # Execute ROC analysis
    start = time.time()
    roc_result = pipeline_dict['roc_expr'].execute()  # <-- EXECUTION happens here!
    elapsed = time.time() - start

    # Unpickle the ROC analysis results
    roc_data = pickle.loads(roc_result['roc_analysis'].iloc[0])

    if roc_data['status'] == 'success':
        roc_scores = roc_data['roc_scores']

        # Display per-class AUC scores
        if 'per_class' in roc_scores:
            print("  Per-class AUC:")
            for cls, score in roc_scores['per_class'].items():
                print(f"    {cls}: {score:.4f}")
            print(f"  Macro-average: {roc_scores['macro']:.4f}")
            print(f"  Weighted-average: {roc_scores['weighted']:.4f}")

        # Save the ROC plot to disk
        if 'plot_base64' in roc_data:
            plot_data = base64.b64decode(roc_data['plot_base64'])
            with open('penguin_roc_curves.png', 'wb') as f:
                f.write(plot_data)
            print(f"\n✓ ROC plot saved to 'penguin_roc_curves.png'")
            print(f"  (executed in {elapsed:.3f}s)")

        print(f"\nTotal samples analyzed: {roc_data['n_samples']}")

    return roc_data


# =============================================================================
# MAIN: Putting It All Together
# =============================================================================
def main():
    """
    Main execution demonstrating the complete workflow.

    Key Learning Points:
    1. Building is instant (deferred)
    2. Execution only happens when needed
    3. Caching prevents recomputation
    4. Everything uses xorq.vendor.ibis
    """
    print("\n" + "="*60)
    print("XORQ BEGINNER VIGNETTE: Penguin Classification")
    print("="*60)

    print("\nBuilding pipeline (deferred - instant!)...")
    pipeline_dict = build_complete_pipeline()
    print("✓ Pipeline built")
    print("  - Data loading configured")
    print("  - Train/test split prepared")
    print("  - Model training ready")
    print("  - Predictions defined")
    print("  - Metrics and visualization prepared")

    # Execute and display results
    roc_data = execute_and_display(pipeline_dict)

    # Demonstrate caching benefit
    print("\n" + "="*60)
    print("DEMONSTRATING CACHE EFFICIENCY")
    print("="*60)

    # Cache the ROC expression
    cached_roc = pipeline_dict['roc_expr'].cache(ParquetCache.from_kwargs())

    print("\nRe-executing cached ROC analysis...")
    import time
    start = time.time()
    result2 = cached_roc.execute()  # Should be much faster!
    elapsed = time.time() - start
    print(f"✓ Cached execution completed in {elapsed:.3f}s (much faster!)")

    return pipeline_dict


if __name__ == "__main__":
    # Run the complete pipeline
    pipeline = main()

    print("\n" + "="*60)
    print("VIGNETTE COMPLETE!")
    print("="*60)
    print("\nWhat you learned:")
    print("  ✓ Deferred execution with xorq expressions")
    print("  ✓ Using xorq.vendor.ibis for enhanced operators")
    print("  ✓ Building ML pipelines that stay deferred")
    print("  ✓ Computing metrics and visualizations lazily")
    print("  ✓ Caching strategies for efficiency")
    print("\nNext steps:")
    print("  • Try modifying the features or model")
    print("  • Add more metrics or visualizations")
    print("  • Explore other vignettes for advanced patterns")
    print("  • Build your own deferred pipelines!")