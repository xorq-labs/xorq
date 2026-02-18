"""Visualizing Cross-Validation Fold Assignments: sklearn vs xorq

Side-by-side comparison of sklearn's native cross-validation fold
assignments with xorq's deferred_cross_val_score fold_expr.

The left column shows sklearn's fold assignments (running the splitter
directly on the same DataFrame that fold_expr.execute() returns).
The right column shows xorq's fold_expr — they should match exactly
because both see rows in the same order.

Based on sklearn's plot_cv_indices example:
https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html
"""

import pathlib

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
)
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler

import xorq.api as xo
from xorq.expr.ml.cross_validation import deferred_cross_val_score
from xorq.expr.ml.pipeline_lib import Pipeline


mpl.use("Agg")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
N_SPLITS = 4
N_SAMPLES = 100

TRAIN_COLOR = "#3575D5"
TEST_COLOR = "#E8432A"
UNUSED_COLOR = "#E0E0E0"
cmap_data = plt.cm.Paired


def make_data(n_samples=N_SAMPLES, random_state=RANDOM_STATE):
    """Generate synthetic classification data matching the sklearn plot_cv_indices example.

    Returns (df, feature_names).
    """
    rng = np.random.RandomState(random_state)

    n_features = 10
    feature_names = tuple(f"f{i}" for i in range(n_features))
    X = rng.randn(n_samples, n_features)

    percentiles_classes = [0.1, 0.3, 0.6]
    y = np.hstack(
        [[ii] * int(n_samples * perc) for ii, perc in enumerate(percentiles_classes)]
    )

    group_prior = rng.dirichlet([2] * 10)
    groups = np.repeat(np.arange(10), rng.multinomial(n_samples, group_prior))

    df = pd.DataFrame(X, columns=list(feature_names)).assign(
        target=y, group=groups, t=range(n_samples)
    )
    return df, feature_names


# ---------------------------------------------------------------------------
# Data — same structure as the sklearn plot_cv_indices example
# ---------------------------------------------------------------------------
con = xo.connect()
df, feature_names = make_data()
data = con.register(df, "cv_data")

# ---------------------------------------------------------------------------
# Pipeline (unfitted — deferred_cross_val_score fits per fold)
# ---------------------------------------------------------------------------
sk_pipeline = SklearnPipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)),
    ]
)
pipeline = Pipeline.from_instance(sk_pipeline)


# ---------------------------------------------------------------------------
# Splitters — factories so each side gets a fresh instance
# ---------------------------------------------------------------------------
# Each entry: (name, splitter_factory, order_by)
splitters = [
    (
        "KFold",
        lambda: KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        None,
    ),
    (
        "StratifiedKFold",
        lambda: StratifiedKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE
        ),
        None,
    ),
    (
        "ShuffleSplit",
        lambda: ShuffleSplit(
            n_splits=N_SPLITS, test_size=0.25, random_state=RANDOM_STATE
        ),
        None,
    ),
    (
        "StratifiedShuffleSplit",
        lambda: StratifiedShuffleSplit(
            n_splits=N_SPLITS, test_size=0.25, random_state=RANDOM_STATE
        ),
        None,
    ),
    ("TimeSeriesSplit", lambda: TimeSeriesSplit(n_splits=N_SPLITS), "t"),
]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _build_sklearn_fold_df(cv, X_arr, y_arr):
    """Build a fold DataFrame from sklearn splitter (same format as fold_expr).

    Encoding: 0=unused, 1=train, 2=test.
    """

    def _make_col(train_idx, test_idx):
        col = np.zeros(len(y_arr), dtype=np.int8)
        col[train_idx] = 1
        col[test_idx] = 2
        return col

    return pd.DataFrame(
        {
            f"fold_{fold_i}": _make_col(train_idx, test_idx)
            for fold_i, (train_idx, test_idx) in enumerate(cv.split(X_arr, y_arr))
        }
    )


def plot_fold_bars(ax, fold_values, n_splits, y_values, group_values, title):
    """Plot fold assignments as continuous horizontal bars.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    fold_values : dict[str, np.ndarray]
        Mapping of fold_i column name to 0/1/2 array (0=unused, 1=train, 2=test).
    n_splits : int
    y_values : array-like
        Target class labels (for the class-label bar).
    group_values : array-like
        Group labels (for the group bar).
    title : str
    """
    n_samples = len(y_values)

    for fold_i in range(n_splits):
        indices = fold_values[f"fold_{fold_i}"]
        color_map = {0: UNUSED_COLOR, 1: TRAIN_COLOR, 2: TEST_COLOR}
        colors = [color_map[v] for v in indices]
        ax.barh(
            [fold_i] * n_samples,
            width=1,
            left=range(n_samples),
            height=0.8,
            color=colors,
            edgecolor="none",
        )

    # Class labels bar
    n_classes = max(len(set(y_values)), 1)
    class_colors = [cmap_data(c / max(n_classes - 1, 1)) for c in y_values]
    ax.barh(
        [n_splits] * n_samples,
        width=1,
        left=range(n_samples),
        height=0.8,
        color=class_colors,
        edgecolor="none",
    )

    # Group labels bar
    n_groups = max(len(set(group_values)), 1)
    group_colors = [cmap_data(g / max(n_groups - 1, 1)) for g in group_values]
    ax.barh(
        [n_splits + 1] * n_samples,
        width=1,
        left=range(n_samples),
        height=0.8,
        color=group_colors,
        edgecolor="none",
    )

    ax.set(
        yticks=range(n_splits + 2),
        yticklabels=[f"Fold {i}" for i in range(n_splits)] + ["Class", "Group"],
        xlabel="Sample index",
        xlim=(0, n_samples),
        ylim=(n_splits + 1.8, -0.5),
    )
    ax.set_title(title, fontsize=12, fontweight="bold")


def plot_splitter_row(
    axes_row,
    name,
    make_cv,
    order_by,
    pipeline,
    data,
    feature_names,
    n_splits,
    random_state,
):
    """Plot one row of the comparison figure for a single splitter."""
    result = deferred_cross_val_score(
        pipeline,
        data,
        features=feature_names,
        target="target",
        cv=make_cv(),
        random_seed=random_state,
        order_by=order_by,
    )
    fold_df = result.fold_expr.execute()

    # The UDWF saw rows in this order, so sklearn must too for parity.
    sklearn_fold_df = _build_sklearn_fold_df(
        make_cv(),
        fold_df[list(feature_names)].values,
        fold_df["target"].values,
    )

    # Sort both by class label for visual clarity
    sort_idx = fold_df["target"].argsort(kind="stable")
    y_display = fold_df["target"].values[sort_idx]
    group_display = fold_df["group"].values[sort_idx]

    sklearn_sorted = {
        f"fold_{i}": sklearn_fold_df[f"fold_{i}"].values[sort_idx]
        for i in range(n_splits)
    }
    xorq_sorted = {
        f"fold_{i}": fold_df[f"fold_{i}"].values[sort_idx] for i in range(n_splits)
    }

    plot_fold_bars(
        axes_row[0],
        sklearn_sorted,
        n_splits,
        y_display,
        group_display,
        f"{name} (sklearn)",
    )
    plot_fold_bars(
        axes_row[1], xorq_sorted, n_splits, y_display, group_display, f"{name} (xorq)"
    )


if __name__ == "__pytest_main__":
    n_rows = len(splitters)
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(16, 2.5 * n_rows), constrained_layout=True
    )

    for row, (name, make_cv, order_by) in enumerate(splitters):
        plot_splitter_row(
            axes[row],
            name,
            make_cv,
            order_by,
            pipeline,
            data,
            feature_names,
            N_SPLITS,
            RANDOM_STATE,
        )

    # Legend
    legend_patches = [
        mpatches.Patch(color=TRAIN_COLOR, label="Train"),
        mpatches.Patch(color=TEST_COLOR, label="Test"),
    ]
    fig.legend(handles=legend_patches, loc="upper right", fontsize=11)
    fig.suptitle(
        "CV fold assignments: sklearn vs xorq fold_expr",
        fontsize=15,
        fontweight="bold",
    )
    plot_path = pathlib.Path("plot_cv_indices.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot to {plot_path}")
    assert plot_path.exists(), f"Expected plot file {plot_path} was not created"
    plot_path.unlink()

    pytest_examples_passed = True
