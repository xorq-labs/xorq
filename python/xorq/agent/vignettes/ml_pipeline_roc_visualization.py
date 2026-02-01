#!/usr/bin/env python
"""
Complete ML Pipeline with ROC Visualization using Functional Pattern
=====================================================================

This vignette demonstrates a production-ready ML pipeline in xorq that showcases:

1. **Deferred ML Pipeline**: Building a complete classifier pipeline that remains
   deferred until final execution - no eager computation!

2. **Functional Composition**: Using .pipe() for clean, composable transformations
   that chain together naturally.

3. **Advanced Metrics**: Computing standard ML metrics (accuracy, precision, recall,
   F1, ROC-AUC) in a deferred manner.

4. **ROC Visualization in UDAF**: Generating ROC curves and storing them as binary
   data using a User-Defined Aggregate Function (UDAF).

5. **xorq.vendor.ibis**: Leveraging xorq's enhanced ibis for caching, multi-backend
   support, and custom operators.

Key Architectural Pattern:
--------------------------
    data_expr.pipe(clean).pipe(split).pipe(train).pipe(predict).pipe(analyze)
         ↓          ↓         ↓         ↓         ↓          ↓
    (deferred) (deferred) (deferred) (deferred) (deferred) (deferred)
                                                              ↓
                                                          execute()

The entire pipeline remains deferred until you call execute(), enabling:
- Optimization across the entire computation graph
- Lazy evaluation for efficient resource usage
- Easy composition and modification without recomputation

This vignette is ideal for:
- Building production ML pipelines with xorq
- Understanding advanced UDAF patterns for visualization
- Learning functional composition patterns with .pipe()
- Seeing how xorq's vendored ibis enables complex workflows
"""

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
import pickle
import base64
from typing import Tuple, Dict, Any, Optional

# CRITICAL: Import xorq's vendored ibis (not standalone ibis!)
# This gives us custom operators like .cache(), .into_backend(), and more
import xorq.api as xo
from xorq.api import _  # The underscore for column references
from xorq.vendor import ibis  # xorq's enhanced ibis with custom operators
from xorq.caching import ParquetCache
from xorq.expr.ml.pipeline_lib import Pipeline
from xorq.expr.ml.metrics import deferred_sklearn_metric
from xorq.expr.udf import scalar, agg
import xorq.expr.datatypes as dt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# Configuration: Features and target for our classification task
FEATURES = ["bill_length_mm", "bill_depth_mm"]
TARGET = "species"


# =============================================================================
# SECTION 1: Functional Pipeline Components for Expression Chaining
# =============================================================================
# These functions are designed to be used with .pipe() for clean composition.
# Each function takes an expression and returns a transformed expression,
# maintaining the deferred execution model throughout.

def clean_missing_values(expr):
    """
    Remove rows with null values in key columns.

    This is a pure transformation function designed for .pipe() chaining.
    It filters out incomplete records that would cause issues in ML training.

    Pattern: expr.pipe(clean_missing_values)
    """
    return expr.filter(
        _.bill_length_mm.notnull()
        & _.bill_depth_mm.notnull()
        & _.flipper_length_mm.notnull()
        & _.body_mass_g.notnull()
        & _.species.notnull()
    )


def add_row_id(expr):
    """
    Add a row ID for deterministic splitting.

    Row IDs enable reproducible train/test splits without random sampling,
    which is important for deferred execution where we can't use random.split().
    """
    return expr.mutate(row_id=xo.row_number())


def split_train_test(test_size=0.2, random_seed=42):
    """
    Split data into train/test using deterministic row sampling.

    Returns a function that can be used with .pipe().
    This closure pattern allows parameterized transformations.

    Example:
        splits = data.pipe(split_train_test(test_size=0.3))
    """
    def _split(expr):
        # Deterministic split using modulo - ensures reproducibility
        train_mask = (expr.row_id % 100) >= (test_size * 100)
        train_expr = expr.filter(train_mask)
        test_expr = expr.filter(~train_mask)

        # Cache both splits to avoid recomputation
        # ParquetCache is efficient for intermediate results
        cache = ParquetCache.from_kwargs()
        return {
            'train': train_expr.cache(cache),
            'test': test_expr.cache(cache),
        }
    return _split


def prepare_for_training(expr):
    """
    Prepare data for ML training by extracting features and target.

    Reduces the expression to only the columns needed for training,
    improving efficiency and reducing memory usage.
    """
    return expr.select(FEATURES + [TARGET])


# =============================================================================
# SECTION 2: ROC Visualization using UDAF
# =============================================================================
# This section demonstrates how to create complex visualizations within
# the deferred execution framework using User-Defined Aggregate Functions.

def create_roc_visualization_expr(expr):
    """
    Create ROC visualization using a pandas UDAF.

    This is an advanced pattern that shows how to:
    1. Aggregate all prediction data into a single computation
    2. Generate visualizations within the deferred framework
    3. Return binary data (plots) as part of the expression result

    The UDAF runs once on all data, generating a comprehensive ROC analysis
    including multi-class curves, AUC scores, and a matplotlib figure.

    Key insight: By using a UDAF, the visualization becomes part of the
    deferred computation graph and only executes when needed.
    """
    def compute_roc(df):
        """
        The actual UDAF function that computes ROC curves and metrics.

        This function receives a pandas DataFrame with all prediction data
        and returns a pickled dictionary containing:
        - ROC scores (per-class, macro, weighted)
        - Base64-encoded plot image
        - Sample count and status information
        """
        try:
            from sklearn.metrics import roc_auc_score, roc_curve, auc
            from sklearn.preprocessing import label_binarize
            import matplotlib.pyplot as plt
            import numpy as np
            import io

            # Extract predictions and true labels from the DataFrame
            y_true = df[TARGET].values
            y_pred = df['predicted'].values
            classes = sorted(df[TARGET].unique())
            n_classes = len(classes)

            # Get probability scores for each class
            # This demonstrates defensive programming - handle missing prob columns
            prob_cols = ['prob_' + cls for cls in classes]
            if all(col in df.columns for col in prob_cols):
                y_score = df[prob_cols].values
            else:
                # Fallback: create synthetic probabilities if not available
                y_score = np.zeros((len(df), n_classes))
                for i, pred in enumerate(y_pred):
                    y_score[i, classes.index(pred)] = 0.9
                    y_score[i, :] = y_score[i, :] / y_score[i, :].sum()

            # Binarize labels for multi-class ROC computation
            y_binarized = label_binarize(y_true, classes=classes)
            roc_scores = {}

            # Compute ROC scores based on classification type
            if n_classes == 2:
                # Binary classification: simpler computation
                y_binarized = y_binarized.ravel()
                roc_scores['binary'] = roc_auc_score(y_binarized, y_score[:, 1])
            else:
                # Multi-class: compute per-class and aggregate metrics
                per_class = {}
                for i, cls in enumerate(classes):
                    per_class[cls] = roc_auc_score(y_binarized[:, i], y_score[:, i])
                roc_scores['per_class'] = per_class

                # Macro-average: unweighted mean of per-class AUCs
                roc_scores['macro'] = roc_auc_score(
                    y_binarized, y_score, multi_class='ovr', average='macro'
                )
                # Weighted-average: weighted by class support
                roc_scores['weighted'] = roc_auc_score(
                    y_binarized, y_score, multi_class='ovr', average='weighted'
                )

            # Create publication-quality ROC visualization
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = plt.colormaps['tab10'](np.linspace(0, 1, n_classes))

            # Plot ROC curve for each class with distinct colors
            for i, (cls, color) in enumerate(zip(classes, colors)):
                if n_classes == 2:
                    # Binary: need to handle both classes
                    fpr, tpr, _ = roc_curve(
                        y_binarized,
                        y_score[:, 1] if i == 1 else 1 - y_score[:, 1]
                    )
                else:
                    # Multi-class: one-vs-rest for each class
                    fpr, tpr, _ = roc_curve(y_binarized[:, i], y_score[:, i])

                roc_auc = auc(fpr, tpr)
                ax.plot(
                    fpr, tpr, color=color, lw=2,
                    label=f'{cls} (AUC = {roc_auc:.3f})'
                )

            # Add diagonal reference line (random classifier)
            ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random')

            # Formatting for professional appearance
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves - Multi-class Classification')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)

            # Convert plot to base64-encoded bytes for storage
            # This allows the plot to be stored as part of the expression result
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            plt.close()  # Important: free memory
            buffer.seek(0)
            plot_b64 = base64.b64encode(buffer.read()).decode('utf-8')

            # Return comprehensive results as pickled dictionary
            # Pickling allows complex Python objects in the deferred result
            return pickle.dumps({
                'status': 'success',
                'roc_scores': roc_scores,
                'plot_base64': plot_b64,
                'n_samples': len(df)
            })

        except Exception as e:
            import traceback
            # Robust error handling - return error info for debugging
            return pickle.dumps({
                'status': 'error',
                'error': str(e),
                'traceback': traceback.format_exc()
            })

    # Create the UDAF with proper schema and return type
    schema = expr.schema()
    roc_udf = agg.pandas_df(
        fn=compute_roc,
        schema=schema,
        return_type=dt.binary,  # Binary for pickled data
        name="compute_roc_analysis"
    )

    # Return an aggregation expression containing the ROC analysis
    # This remains deferred until execute() is called
    return expr.aggregate([
        roc_udf.on_expr(expr).name('roc_analysis')
    ])


# =============================================================================
# SECTION 3: Metrics Computation
# =============================================================================

def create_metrics_dict(predictions_expr, predictions_proba_expr=None):
    """
    Create a dictionary of deferred metric expressions.

    Each metric is computed using deferred_sklearn_metric, which ensures
    the computation only happens when execute() is called.

    This pattern allows you to:
    - Define all metrics upfront
    - Execute them selectively
    - Cache intermediate results automatically
    """
    metrics = {
        'accuracy': deferred_sklearn_metric(
            expr=predictions_expr,
            target=TARGET,
            pred_col="predicted",
            metric_fn=accuracy_score,
        ),
        'precision': deferred_sklearn_metric(
            expr=predictions_expr,
            target=TARGET,
            pred_col="predicted",
            metric_fn=precision_score,
            metric_kwargs={"average": "weighted", "zero_division": 0},
        ),
        'recall': deferred_sklearn_metric(
            expr=predictions_expr,
            target=TARGET,
            pred_col="predicted",
            metric_fn=recall_score,
            metric_kwargs={"average": "weighted", "zero_division": 0},
        ),
        'f1': deferred_sklearn_metric(
            expr=predictions_expr,
            target=TARGET,
            pred_col="predicted",
            metric_fn=f1_score,
            metric_kwargs={"average": "weighted", "zero_division": 0},
        ),
    }

    # ROC-AUC requires probability scores
    if predictions_proba_expr is not None:
        metrics['roc_auc'] = deferred_sklearn_metric(
            expr=predictions_proba_expr,
            target=TARGET,
            pred_col="predicted_proba",
            metric_fn=roc_auc_score,
            metric_kwargs={"multi_class": "ovr", "average": "weighted"},
        )

    return metrics


# =============================================================================
# SECTION 4: Main Pipeline Builder using Functional Composition
# =============================================================================

def add_probability_columns(expr):
    """
    Transform predicted probabilities into individual columns.

    This helper function demonstrates how to:
    1. Extract array elements from predicted_proba
    2. Create the final prediction based on max probability
    3. Use ibis.cases() for conditional logic (xorq's vendored version)
    """
    return expr.mutate(
        # Extract individual class probabilities
        prob_Adelie=expr['predicted_proba'][0],
        prob_Chinstrap=expr['predicted_proba'][1],
        prob_Gentoo=expr['predicted_proba'][2],

        # Determine prediction from probabilities using xo.cases
        # Each (condition, result) pair is a single tuple argument
        predicted=xo.cases(
            ((expr['predicted_proba'][0] >= expr['predicted_proba'][1]) &
             (expr['predicted_proba'][0] >= expr['predicted_proba'][2]), 'Adelie'),
            ((expr['predicted_proba'][1] >= expr['predicted_proba'][0]) &
             (expr['predicted_proba'][1] >= expr['predicted_proba'][2]), 'Chinstrap'),
            else_='Gentoo'
        )
    )


def build_ml_pipeline():
    """
    Build the complete ML pipeline using functional composition.

    This is the main orchestrator that demonstrates:
    1. How to chain transformations using .pipe()
    2. How to use xorq's caching for efficiency
    3. How to integrate sklearn models into deferred expressions
    4. How everything remains deferred until execute()

    The pipeline follows this flow:
    Data → Clean → Split → Train → Predict → Analyze

    All steps are deferred, creating a computation graph that's
    optimized and executed only when results are needed.
    """

    # Step 1: Data preparation with functional chaining
    # The .pipe() pattern makes the transformation sequence clear
    clean_data = (
        xo.examples.penguins.fetch()  # Load example data
        .pipe(clean_missing_values)   # Remove nulls
        .pipe(prepare_for_training)   # Select features + target
    )

    # Step 2: Train/test split using xorq's built-in method
    # This is more efficient than custom splitting for large datasets
    train_expr, test_expr = xo.train_test_splits(
        clean_data,
        test_sizes=0.2,        # 20% test set
        num_buckets=1000,      # Granularity for splitting
        random_seed=42,        # Reproducibility
    )

    # Step 3: Cache the splits for efficiency
    # ParquetCache stores intermediate results to avoid recomputation
    cache = ParquetCache.from_kwargs()
    train_cached = train_expr.cache(cache)
    test_cached = test_expr.cache(cache)

    # Step 4: Create and fit the ML pipeline
    # First create a standard sklearn pipeline
    sklearn_pipeline = SkPipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression())
    ]).set_params(**{
        "classifier__C": 0.1,
        "classifier__penalty": "l2",
        "classifier__max_iter": 2000,
        "classifier__random_state": 42
    })

    # Wrap it in xorq's Pipeline for deferred execution
    xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)

    # Fit the model (still deferred!)
    fitted_pipeline = xorq_pipeline.fit(
        train_cached,
        features=FEATURES,
        target=TARGET
    )

    # Step 5: Generate predictions
    predictions = fitted_pipeline.predict(test_cached)
    predictions_proba = fitted_pipeline.predict_proba(test_cached)

    # Step 6: Enhanced predictions with probability columns
    # Using .pipe() with lambda for inline transformations
    predictions_with_probs = (
        predictions_proba
        .pipe(add_probability_columns)
    )

    # Step 7: Create ROC visualization expression
    roc_expr = predictions_with_probs.pipe(create_roc_visualization_expr)

    # Step 8: Create metric expressions
    metrics = create_metrics_dict(predictions, predictions_proba)

    # Return all components for selective execution
    return {
        'train': train_cached,
        'test': test_cached,
        'fitted_pipeline': fitted_pipeline,
        'predictions': predictions,
        'predictions_proba': predictions_proba,
        'predictions_with_probs': predictions_with_probs,
        'roc_expr': roc_expr,
        'metrics': metrics
    }


# =============================================================================
# SECTION 5: Execution and Results Display
# =============================================================================

def execute_pipeline(pipeline_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute the deferred pipeline and collect results.

    This function demonstrates selective execution - you only
    execute the parts of the pipeline you need. Each execute()
    call triggers computation for that specific branch.
    """
    import time
    results = {}

    # Execute metrics one by one, timing each
    if 'metrics' in pipeline_dict:
        results['metrics'] = {}
        for name, metric_expr in pipeline_dict['metrics'].items():
            start = time.time()
            value = metric_expr.execute()  # Trigger computation here!
            elapsed = time.time() - start
            results['metrics'][name] = {
                'value': value,
                'time': elapsed
            }

    # Execute ROC analysis (includes visualization)
    if 'roc_expr' in pipeline_dict:
        start = time.time()
        roc_result = pipeline_dict['roc_expr'].execute()  # Trigger computation!
        elapsed = time.time() - start

        # Unpickle the UDAF results
        roc_data = pickle.loads(roc_result['roc_analysis'].iloc[0])
        results['roc'] = {
            'data': roc_data,
            'time': elapsed
        }

    return results


def display_results(results: Dict[str, Any]) -> None:
    """
    Display the execution results in a formatted manner.

    This includes:
    - Standard ML metrics with execution times
    - ROC analysis with per-class scores
    - Saving the ROC plot to a file
    """
    print("ML Pipeline Execution Results\n")
    print("=" * 50)

    # Display standard metrics
    if 'metrics' in results:
        print("\nStandard Metrics:")
        for name, info in results['metrics'].items():
            print(f"  {name:10s}: {info['value']:.4f} ({info['time']:.3f}s)")

    # Display ROC analysis
    if 'roc' in results:
        roc_data = results['roc']['data']
        print("\nROC Analysis:")

        if roc_data['status'] == 'success':
            roc_scores = roc_data['roc_scores']
            if 'per_class' in roc_scores:
                print("  Per-class AUC scores:")
                for cls, score in roc_scores['per_class'].items():
                    print(f"    {cls}: {score:.4f}")
                print(f"  Macro-average AUC: {roc_scores['macro']:.4f}")
                print(f"  Weighted-average AUC: {roc_scores['weighted']:.4f}")

            # Save the plot to a file
            if 'plot_base64' in roc_data:
                plot_data = base64.b64decode(roc_data['plot_base64'])
                with open('roc_visualization.png', 'wb') as f:
                    f.write(plot_data)
                print(f"\n✓ ROC plot saved to 'roc_visualization.png'")
                print(f"  (executed in {results['roc']['time']:.3f}s)")

            print(f"\nTotal samples analyzed: {roc_data['n_samples']}")
        else:
            print(f"  Error: {roc_data['error']}")


def main():
    """
    Main execution demonstrating the complete workflow.

    Key points:
    1. Building the pipeline is instant (deferred)
    2. Execution only happens when we call execute()
    3. Results can be selectively computed
    4. The entire workflow uses xorq's vendored ibis
    """
    print("Building ML Pipeline with Functional Pattern...")
    print("Using .pipe() for composable transformations")
    print("Leveraging xorq.vendor.ibis for enhanced operations\n")

    # Build the pipeline (instant - nothing executed)
    pipeline_dict = build_ml_pipeline()
    print("✓ Pipeline built (fully deferred)")
    print("  - Train/test splits created")
    print("  - Model fitting configured")
    print("  - Predictions prepared")
    print("  - ROC visualization ready")
    print("  - Metrics defined")

    # Now execute and display results
    print("\nExecuting pipeline (this triggers computation)...")
    results = execute_pipeline(pipeline_dict)

    print("\n" + "=" * 50)
    display_results(results)

    return pipeline_dict, results


if __name__ == "__main__":
    # Run the complete pipeline
    pipeline, results = main()
    print("\n✓ Pipeline execution complete!")
    print("\nThis vignette demonstrated:")
    print("  • Fully deferred ML pipeline with xorq")
    print("  • Functional composition using .pipe()")
    print("  • ROC visualization via UDAF")
    print("  • xorq.vendor.ibis custom operators")
    print("  • Selective execution of pipeline components")