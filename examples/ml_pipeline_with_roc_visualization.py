#!/usr/bin/env python
"""
ML Pipeline with ROC Visualization using UDAFs

This example demonstrates:
1. Building a complete deferred ML pipeline
2. Training a classifier and making predictions
3. Computing standard ML metrics (accuracy, precision, recall, F1, ROC-AUC)
4. Generating ROC curves and storing them as binary data in a UDAF
5. Maintaining pure deferred execution until the final execute() calls

Key Pattern:
    data_expr → train_expr → fitted_pipeline → predictions_with_probs → roc_expr
         ↓           ↓              ↓                    ↓                  ↓
    (deferred)  (deferred)     (deferred)         (deferred)         (deferred)
                                                                          ↓
                                                                     execute()

The ROC visualization is generated inside a UDAF and stored as base64-encoded binary data,
demonstrating how to integrate plotting into xorq's deferred expression pipeline.
"""

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
import toolz
import pickle
import base64

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


# Configuration
FEATURES = ["bill_length_mm", "bill_depth_mm"]  # Only bill measurements
TARGET = "species"


def get_data_expr():
    """Load and clean the penguins dataset as a deferred expression."""
    penguins = xo.examples.penguins.fetch()

    clean_data = penguins.filter(
        _.bill_length_mm.notnull()
        & _.bill_depth_mm.notnull()
        & _.flipper_length_mm.notnull()
        & _.body_mass_g.notnull()
        & _.species.notnull()
    )

    return clean_data


def create_train_test_expr(data_expr, test_size=0.2, random_seed=42):
    """Split data into train/test sets and cache them."""
    train_expr, test_expr = xo.train_test_splits(
        data_expr,
        test_sizes=test_size,
        num_buckets=1000,
        random_seed=random_seed,
    )
    cache = ParquetCache.from_kwargs()
    return train_expr.cache(cache), test_expr.cache(cache)


def create_fitted_pipeline_expr(train_expr, model_params=None):
    """Create and fit a sklearn pipeline in a deferred manner."""
    params = model_params or {
        "classifier__C": 0.1,
        "classifier__penalty": "l2",
        "classifier__max_iter": 2000,
        "classifier__random_state": 42
    }

    sklearn_pipeline = SkPipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression())
    ]).set_params(**params)

    xorq_pipeline = Pipeline.from_instance(sklearn_pipeline)

    fitted_pipeline = xorq_pipeline.fit(
        train_expr,
        features=FEATURES,
        target=TARGET
    )

    return fitted_pipeline


def create_predictions_with_probs_expr(fitted_pipeline, test_expr):
    """Generate predictions with probability scores for each class."""
    predictions_proba = fitted_pipeline.predict_proba(test_expr)

    # Extract individual class probabilities
    predictions_with_probs = predictions_proba.mutate(
        prob_Adelie=predictions_proba['predicted_proba'][0],
        prob_Chinstrap=predictions_proba['predicted_proba'][1],
        prob_Gentoo=predictions_proba['predicted_proba'][2],
        # Determine predicted class based on highest probability
        predicted=xo.cases(
            (
                (predictions_proba['predicted_proba'][0] >= predictions_proba['predicted_proba'][1]) &
                (predictions_proba['predicted_proba'][0] >= predictions_proba['predicted_proba'][2]),
                'Adelie'
            ),
            (
                (predictions_proba['predicted_proba'][1] >= predictions_proba['predicted_proba'][0]) &
                (predictions_proba['predicted_proba'][1] >= predictions_proba['predicted_proba'][2]),
                'Chinstrap'
            ),
            else_='Gentoo'
        )
    )

    return predictions_with_probs, predictions_proba


def create_roc_analysis_udf():
    """
    Create a UDAF that computes ROC analysis and generates visualization.

    This function demonstrates how to:
    1. Compute ROC scores for multi-class classification
    2. Generate ROC curves using matplotlib
    3. Store the plot as base64-encoded binary data
    4. Return everything as a pickled dictionary with dt.binary type
    """
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

    return compute_roc


def create_roc_expr(predictions_with_probs):
    """Create a deferred expression for ROC analysis using UDAF."""
    roc_udf_fn = create_roc_analysis_udf()
    schema = predictions_with_probs.schema()

    # Create UDAF with binary return type for pickled data
    roc_udf = agg.pandas_df(
        fn=roc_udf_fn,
        schema=schema,
        return_type=dt.binary,
        name="compute_roc_analysis"
    )

    # Apply UDAF to aggregate all predictions
    roc_expr = predictions_with_probs.aggregate([
        roc_udf.on_expr(predictions_with_probs).name('roc_analysis')
    ])

    return roc_expr


def create_metric_expressions(predictions, predictions_proba):
    """Create deferred expressions for standard ML metrics."""
    metrics = {
        'accuracy': deferred_sklearn_metric(
            expr=predictions,
            target=TARGET,
            pred_col="predicted",
            metric_fn=accuracy_score,
        ),
        'precision': deferred_sklearn_metric(
            expr=predictions,
            target=TARGET,
            pred_col="predicted",
            metric_fn=precision_score,
            metric_kwargs={"average": "weighted", "zero_division": 0},
        ),
        'recall': deferred_sklearn_metric(
            expr=predictions,
            target=TARGET,
            pred_col="predicted",
            metric_fn=recall_score,
            metric_kwargs={"average": "weighted", "zero_division": 0},
        ),
        'f1': deferred_sklearn_metric(
            expr=predictions,
            target=TARGET,
            pred_col="predicted",
            metric_fn=f1_score,
            metric_kwargs={"average": "weighted", "zero_division": 0},
        ),
        'roc_auc': deferred_sklearn_metric(
            expr=predictions_proba,
            target=TARGET,
            pred_col="predicted_proba",
            metric_fn=roc_auc_score,
            metric_kwargs={"multi_class": "ovr", "average": "weighted"},
        )
    }

    return metrics


def build_complete_pipeline():
    """
    Build the complete deferred ML pipeline.

    Returns a dictionary with all deferred expressions.
    Nothing is executed until explicitly calling .execute()
    """
    # Build deferred pipeline components
    data_expr = get_data_expr()
    train_expr, test_expr = create_train_test_expr(data_expr)
    fitted_pipeline = create_fitted_pipeline_expr(train_expr)

    # Generate predictions
    predictions_with_probs, predictions_proba = create_predictions_with_probs_expr(
        fitted_pipeline, test_expr
    )
    predictions = fitted_pipeline.predict(test_expr)

    # Create analysis expressions
    roc_expr = create_roc_expr(predictions_with_probs)
    metrics = create_metric_expressions(predictions, predictions_proba)

    return {
        'roc_expr': roc_expr,
        'metrics': metrics,
        'predictions': predictions,
        'predictions_proba': predictions_proba,
        'fitted_pipeline': fitted_pipeline
    }


def execute_and_display(pipeline_dict):
    """Execute the deferred pipeline and display results."""
    import time

    print("Executing ML Pipeline with ROC Visualization\n")
    print("Standard Metrics:")

    # Execute metric expressions
    for name, metric_expr in pipeline_dict['metrics'].items():
        start = time.time()
        value = metric_expr.execute()  # <-- EXECUTION happens here
        elapsed = time.time() - start
        print(f"  {name:10s}: {value:.4f} (executed in {elapsed:.3f}s)")

    print("\nROC Analysis:")
    # Execute ROC analysis
    start = time.time()
    roc_result = pipeline_dict['roc_expr'].execute()  # <-- EXECUTION happens here
    elapsed = time.time() - start

    # Unpickle the results
    roc_data = pickle.loads(roc_result['roc_analysis'].iloc[0])

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
            print(f"  (executed in {elapsed:.3f}s)")

        print(f"\nTotal samples analyzed: {roc_data['n_samples']}")
    else:
        print(f"  Error: {roc_data['error']}")
        if 'traceback' in roc_data:
            print(f"  Traceback:\n{roc_data['traceback']}")

    return roc_data


def main():
    """Main execution function."""
    print("Building deferred ML pipeline...")
    pipeline_dict = build_complete_pipeline()
    print("Pipeline built (nothing executed yet)\n")

    # Execute the entire pipeline
    roc_data = execute_and_display(pipeline_dict)

    # Optionally cache the ROC analysis for reuse
    print("\nCaching ROC analysis for reuse...")
    cached_roc = pipeline_dict['roc_expr'].cache(ParquetCache.from_kwargs())
    result2 = cached_roc.execute()
    print("✓ ROC analysis cached successfully")

    return pipeline_dict


if __name__ == "__main__":
    pipeline = main()
    print("\nPipeline execution complete!")