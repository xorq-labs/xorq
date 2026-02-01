#!/usr/bin/env python
"""
ML Pipeline with ROC Visualization using Functional Pattern

This example demonstrates:
1. Building a complete deferred ML pipeline using functional .pipe() pattern
2. Training a classifier and making predictions through composable transformations
3. Computing standard ML metrics (accuracy, precision, recall, F1, ROC-AUC)
4. Generating ROC curves and storing them as binary data in a UDAF
5. Maintaining pure deferred execution until the final execute() calls

Key Pattern:
    data_expr.pipe(clean).pipe(split).pipe(train).pipe(predict).pipe(analyze)
         ↓          ↓         ↓         ↓         ↓          ↓
    (deferred) (deferred) (deferred) (deferred) (deferred) (deferred)
                                                              ↓
                                                          execute()

The functional pattern makes the pipeline more composable and easier to extend.
"""

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
import toolz
import pickle
import base64
from typing import Tuple, Dict, Any, Optional

import xorq.api as xo
from xorq.api import _
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


FEATURES = ["bill_length_mm", "bill_depth_mm"]
TARGET = "species"


# =============================================================================
# Functional Pipeline Components for Expression Chaining
# =============================================================================

def clean_missing_values(expr):
    """Remove rows with null values in key columns."""
    return expr.filter(
        _.bill_length_mm.notnull()
        & _.bill_depth_mm.notnull()
        & _.flipper_length_mm.notnull()
        & _.body_mass_g.notnull()
        & _.species.notnull()
    )


def add_row_id(expr):
    """Add a row ID for splitting."""
    return expr.mutate(row_id=xo.row_number())


def split_train_test(test_size=0.2, random_seed=42):
    """Split data into train/test using row sampling."""
    def _split(expr):
        # Use modulo for deterministic split based on row_id
        train_mask = (expr.row_id % 100) >= (test_size * 100)
        train_expr = expr.filter(train_mask)
        test_expr = expr.filter(~train_mask)

        # Cache both splits
        cache = ParquetCache.from_kwargs()
        return {
            'train': train_expr.cache(cache),
            'test': test_expr.cache(cache),
        }
    return _split


def prepare_for_training(expr):
    """Prepare data for ML training by extracting features and target."""
    return expr.select(FEATURES + [TARGET])


def fit_classifier(features=None, target=None, model_params=None):
    """Fit a classifier on the training data."""
    features = features or FEATURES
    target = target or TARGET
    params = model_params or {
        "classifier__C": 0.1,
        "classifier__penalty": "l2",
        "classifier__max_iter": 2000,
        "classifier__random_state": 42
    }

    def _fit(expr):
        sklearn_pipeline = SkPipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression())
        ]).set_params(**params)

        xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)

        # Return the fitted pipeline as an attribute on the expression
        fitted = xorq_pipeline.fit(expr, features=features, target=target)

        # Store fitted pipeline in expression metadata (conceptually)
        expr._fitted_pipeline = fitted
        return expr

    return _fit


def add_predictions_column(fitted_pipeline, test_expr):
    """Add predictions as a new column to the test expression."""
    predictions = fitted_pipeline.predict(test_expr)
    # Join predictions back to test data
    return predictions


def add_probability_columns(fitted_pipeline, test_expr):
    """Add probability columns for each class."""
    predictions_proba = fitted_pipeline.predict_proba(test_expr)

    # Add individual probability columns
    enhanced = predictions_proba.mutate(
        prob_Adelie=predictions_proba['predicted_proba'][0],
        prob_Chinstrap=predictions_proba['predicted_proba'][1],
        prob_Gentoo=predictions_proba['predicted_proba'][2],
    )

    return enhanced


def create_roc_visualization_expr(expr):
    """Create ROC visualization using UDAF."""
    def compute_roc(df):
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

            # Get probability scores for each class
            prob_cols = ['prob_' + cls for cls in classes]
            if all(col in df.columns for col in prob_cols):
                y_score = df[prob_cols].values
            else:
                # Fallback if probabilities not available
                y_score = np.zeros((len(df), n_classes))
                for i, pred in enumerate(y_pred):
                    y_score[i, classes.index(pred)] = 0.9
                    y_score[i, :] = y_score[i, :] / y_score[i, :].sum()

            # Binarize labels for multi-class ROC
            y_binarized = label_binarize(y_true, classes=classes)
            roc_scores = {}

            # Compute ROC scores
            if n_classes == 2:
                # Binary classification
                y_binarized = y_binarized.ravel()
                roc_scores['binary'] = roc_auc_score(y_binarized, y_score[:, 1])
            else:
                # Multi-class classification
                per_class = {}
                for i, cls in enumerate(classes):
                    per_class[cls] = roc_auc_score(y_binarized[:, i], y_score[:, i])
                roc_scores['per_class'] = per_class
                roc_scores['macro'] = roc_auc_score(
                    y_binarized, y_score, multi_class='ovr', average='macro'
                )
                roc_scores['weighted'] = roc_auc_score(
                    y_binarized, y_score, multi_class='ovr', average='weighted'
                )

            # Create ROC visualization
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

            # Add diagonal reference line
            ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves - Multi-class Classification')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)

            # Save plot to bytes buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            plt.close()  # Important: close to free memory
            buffer.seek(0)
            plot_b64 = base64.b64encode(buffer.read()).decode('utf-8')

            # Return all results as pickled dictionary
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

    schema = expr.schema()
    roc_udf = agg.pandas_df(
        fn=compute_roc,
        schema=schema,
        return_type=dt.binary,
        name="compute_roc_analysis"
    )

    return expr.aggregate([
        roc_udf.on_expr(expr).name('roc_analysis')
    ])


def create_metrics_dict(predictions_expr, predictions_proba_expr=None):
    """Create a dictionary of metric expressions."""
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
# Main Pipeline Builder using Functional Composition
# =============================================================================

def build_ml_pipeline():
    """
    Build the complete ML pipeline using functional composition with .pipe().

    This demonstrates how to chain transformations on xorq expressions
    to create a clean, readable pipeline.
    """

    # Step 1: Data preparation pipeline
    clean_data = (
        xo.examples.penguins.fetch()
        .pipe(clean_missing_values)
        .pipe(prepare_for_training)
    )

    # Step 2: Train/test split (using xorq's built-in method)
    train_expr, test_expr = xo.train_test_splits(
        clean_data,
        test_sizes=0.2,
        num_buckets=1000,
        random_seed=42,
    )

    # Cache the splits
    cache = ParquetCache.from_kwargs()
    train_cached = train_expr.cache(cache)
    test_cached = test_expr.cache(cache)

    # Step 3: Fit the model
    sklearn_pipeline = SkPipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression())
    ]).set_params(**{
        "classifier__C": 0.1,
        "classifier__penalty": "l2",
        "classifier__max_iter": 2000,
        "classifier__random_state": 42
    })

    xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)
    fitted_pipeline = xorq_pipeline.fit(
        train_cached,
        features=FEATURES,
        target=TARGET
    )

    # Step 4: Generate predictions using functional chaining
    predictions = fitted_pipeline.predict(test_cached)
    predictions_proba = fitted_pipeline.predict_proba(test_cached)

    # Step 5: Enhanced predictions with probabilities - using pipe for transformations
    predictions_with_probs = (
        predictions_proba
        .pipe(lambda expr: expr.mutate(
            prob_Adelie=expr['predicted_proba'][0],
            prob_Chinstrap=expr['predicted_proba'][1],
            prob_Gentoo=expr['predicted_proba'][2],
            predicted=xo.cases(
                ((expr['predicted_proba'][0] >= expr['predicted_proba'][1]) &
                 (expr['predicted_proba'][0] >= expr['predicted_proba'][2]), 'Adelie'),
                ((expr['predicted_proba'][1] >= expr['predicted_proba'][0]) &
                 (expr['predicted_proba'][1] >= expr['predicted_proba'][2]), 'Chinstrap'),
                else_='Gentoo'
            )
        ))
    )

    # Step 6: Create ROC analysis using pipe
    roc_expr = predictions_with_probs.pipe(create_roc_visualization_expr)

    # Step 7: Create metrics
    metrics = create_metrics_dict(predictions, predictions_proba)

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


def add_probability_columns(expr):
    return expr.mutate(
        prob_Adelie=expr['predicted_proba'][0],
        prob_Chinstrap=expr['predicted_proba'][1],
        prob_Gentoo=expr['predicted_proba'][2],
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
    Alternative implementation with more aggressive functional chaining.

    This shows how far we can push the .pipe() pattern.
    """

    # Prepare clean data with single chain
    clean_data = (
        xo.examples.penguins.fetch()
        .pipe(clean_missing_values)
        .pipe(lambda expr: expr.select(FEATURES + [TARGET]))
        .pipe(lambda expr: expr.cache(ParquetCache.from_kwargs()))
    )

    # Split data
    train_expr, test_expr = xo.train_test_splits(
        clean_data,
        test_sizes=0.2,
        num_buckets=1000,
        random_seed=42,
    )

    # Cache splits
    cache = ParquetCache.from_kwargs()
    train_cached = train_expr.pipe(lambda e: e.cache(cache))
    test_cached = test_expr.pipe(lambda e: e.cache(cache))

    # Fit pipeline
    fitted_pipeline = Pipeline.from_instance(
        SkPipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=0.1, penalty="l2", max_iter=2000, random_state=42))
        ])
    ).fit(train_cached, features=FEATURES, target=TARGET)

    # Generate predictions with chaining
    predictions_with_probs = (
        fitted_pipeline.predict_proba(test_cached)
        .pipe(add_probability_columns)
    )

    # Create ROC visualization
    roc_expr = predictions_with_probs.pipe(create_roc_visualization_expr)

    return {
        'predictions_with_probs': predictions_with_probs,
        'roc_expr': roc_expr,
        'fitted_pipeline': fitted_pipeline,
    }


def execute_pipeline(pipeline_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the deferred pipeline and collect results."""
    import time

    results = {}

    # Execute metrics
    if 'metrics' in pipeline_dict:
        results['metrics'] = {}
        for name, metric_expr in pipeline_dict['metrics'].items():
            start = time.time()
            value = metric_expr.execute()
            elapsed = time.time() - start
            results['metrics'][name] = {
                'value': value,
                'time': elapsed
            }

    # Execute ROC analysis
    if 'roc_expr' in pipeline_dict:
        start = time.time()
        roc_result = pipeline_dict['roc_expr'].execute()
        elapsed = time.time() - start

        # Unpickle the results
        roc_data = pickle.loads(roc_result['roc_analysis'].iloc[0])
        results['roc'] = {
            'data': roc_data,
            'time': elapsed
        }

    return results


def display_results(results: Dict[str, Any]) -> None:
    """Display the execution results."""
    print("ML Pipeline Execution Results\n")
    print("=" * 50)

    # Display metrics
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

            # Save plot to file
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
    """Main execution function demonstrating the functional pipeline."""
    print("Building ML Pipeline with Functional Pattern...")
    print("Using .pipe() for composable transformations\n")

    # Build the pipeline (nothing executed yet)
    pipeline_dict = build_ml_pipeline()
    print("✓ Pipeline built (deferred execution)")

    # Execute and display results
    print("\nExecuting pipeline...")
    results = execute_pipeline(pipeline_dict)

    print("\n" + "=" * 50)
    display_results(results)

    return pipeline_dict, results


if __name__ == "__main__":
    pipeline, results = main()
    print("\n✓ Pipeline execution complete!")
