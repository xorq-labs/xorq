"""Demonstrates FittedPipelineBuilder as a catalog builder: fit, catalog, recover, rebind inference.

With xorq: FittedPipelineBuilder wraps a fitted Pipeline as a catalog builder entry. The
builder is a selector — pick a method (predict, transform, predict_proba) and provide
inference data to yield an expression. Training data is fixed in the fit subgraph;
only the inference input changes between dev and prd.
"""

import tempfile
from pathlib import Path

import sklearn.pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import xorq.api as xo
from xorq.catalog.catalog import Catalog
from xorq.expr.builders.fitted_pipeline import FittedPipelineBuilder
from xorq.expr.ml.pipeline_lib import Pipeline

# ---------------------------------------------------------------------------
# 1. Set up connection and training data
# ---------------------------------------------------------------------------

con = xo.connect()

train_data = con.create_table(
    "iris_train",
    {
        "sepal_length": [5.1, 4.9, 7.0, 6.5, 6.3, 5.8, 5.0, 6.7, 5.9, 6.0],
        "sepal_width": [3.5, 3.0, 3.2, 2.8, 3.3, 2.7, 3.4, 3.1, 3.0, 2.2],
        "petal_length": [1.4, 1.4, 4.7, 4.6, 6.0, 5.1, 1.5, 4.7, 5.1, 5.0],
        "petal_width": [0.2, 0.2, 1.4, 1.5, 2.5, 1.9, 0.2, 1.5, 1.8, 1.5],
        "species": [0, 0, 1, 1, 2, 2, 0, 1, 2, 2],
    },
)

features = ("sepal_length", "sepal_width", "petal_length", "petal_width")
target = "species"

# ---------------------------------------------------------------------------
# 2. Create and fit an ML pipeline
# ---------------------------------------------------------------------------

sklearn_pipe = sklearn.pipeline.Pipeline([
    ("scaler", StandardScaler()),
    ("clf", KNeighborsClassifier(n_neighbors=3)),
])

xorq_pipeline = Pipeline.from_instance(sklearn_pipe)
fitted = xorq_pipeline.fit(train_data, features=features, target=target)
fitted_builder = FittedPipelineBuilder(fitted_pipeline=fitted)

print("Pipeline steps:", fitted_builder.steps)
print("Is predict pipeline:", fitted_builder.is_predict)

# ---------------------------------------------------------------------------
# 3. Add the fitted pipeline builder to a catalog
# ---------------------------------------------------------------------------

catalog_dir = Path(tempfile.mkdtemp()) / "pipeline-catalog"
catalog = Catalog.from_repo_path(catalog_dir, init=True)
print(f"\nCatalog directory: {catalog_dir}")

catalog.add_builder(fitted_builder, __file__, aliases=("iris-classifier",), sync=False)
print("Catalog entries:", catalog.list())
print("Catalog aliases:", catalog.list_aliases())

# ---------------------------------------------------------------------------
# 4. Recover the builder from the catalog and rebind training data
# ---------------------------------------------------------------------------

recovered_builder = catalog.get_builder("iris-classifier")
recovered_builder = recovered_builder.rebind(train_data)
print("\nRecovered builder type:", type(recovered_builder).__name__)
print("Recovered steps:", recovered_builder.steps)
print("Recovered is_predict:", recovered_builder.is_predict)

# ---------------------------------------------------------------------------
# 5. Build prediction on dev inference data
# ---------------------------------------------------------------------------

dev_inference = con.create_table(
    "inference_dev",
    {
        "sepal_length": [5.0, 6.2, 6.9],
        "sepal_width": [3.3, 2.9, 3.1],
        "petal_length": [1.4, 4.3, 5.4],
        "petal_width": [0.2, 1.3, 2.1],
    },
)

dev_predictions = recovered_builder.build_expr(data=dev_inference, method="predict")
print("\nDev predictions:")
print(dev_predictions.execute())

# ---------------------------------------------------------------------------
# 6. Build prediction on prd inference data — same model, different data
# ---------------------------------------------------------------------------

prd_inference = con.create_table(
    "inference_prd",
    {
        "sepal_length": [4.8, 6.0, 7.2, 5.7],
        "sepal_width": [3.1, 2.5, 3.0, 2.8],
        "petal_length": [1.6, 3.9, 5.8, 4.1],
        "petal_width": [0.2, 1.1, 1.6, 1.3],
    },
)

prd_predictions = recovered_builder.build_expr(data=prd_inference, method="predict")
print("\nProd predictions:")
print(prd_predictions.execute())

# ---------------------------------------------------------------------------
# 7. Catalog the prediction expressions
# ---------------------------------------------------------------------------

catalog.add(dev_predictions, aliases=("iris-predictions-dev",), sync=False)
catalog.add(prd_predictions, aliases=("iris-predictions-prd",), sync=False)

# ---------------------------------------------------------------------------
# 8. Final catalog state — one builder, multiple prediction expressions
# ---------------------------------------------------------------------------

print("\nFinal catalog entries:", catalog.list())
print("Final catalog aliases:", catalog.list_aliases())


if __name__ == "__pytest_main__":
    pytest_examples_passed = True
