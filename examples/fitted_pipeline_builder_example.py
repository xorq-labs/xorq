"""Demonstrates fitted ML pipeline as a catalog ExprBuilder entry.

With xorq: A fitted Pipeline produces prediction expressions directly. The prediction
expression carries ML pipeline metadata in its tags. When cataloged, it becomes an
ExprBuilder entry. The sidecar records pipeline steps and builder type. No separate
builder wrapper is needed — the pipeline is the builder.
"""

import tempfile
from pathlib import Path

import sklearn.pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

import xorq.api as xo
from xorq.caching import ParquetCache
from xorq.catalog.catalog import Catalog
from xorq.expr.ml.pipeline_lib import Pipeline


# ---------------------------------------------------------------------------
# 1. Set up connection and training data
# ---------------------------------------------------------------------------

con = xo.connect()

train_data = xo.memtable(
    {
        "sepal_length": [5.1, 4.9, 7.0, 6.5, 6.3, 5.8, 5.0, 6.7, 5.9, 6.0],
        "sepal_width": [3.5, 3.0, 3.2, 2.8, 3.3, 2.7, 3.4, 3.1, 3.0, 2.2],
        "petal_length": [1.4, 1.4, 4.7, 4.6, 6.0, 5.1, 1.5, 4.7, 5.1, 5.0],
        "petal_width": [0.2, 0.2, 1.4, 1.5, 2.5, 1.9, 0.2, 1.5, 1.8, 1.5],
        "species": [0, 0, 1, 1, 2, 2, 0, 1, 2, 2],
    },
    name="iris_train",
)

features = ("sepal_length", "sepal_width", "petal_length", "petal_width")
target = "species"

# ---------------------------------------------------------------------------
# 2. Create and fit an ML pipeline
# ---------------------------------------------------------------------------

sklearn_pipe = sklearn.pipeline.Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=3)),
    ]
)

xorq_pipeline = Pipeline.from_instance(sklearn_pipe)
cache = ParquetCache.from_kwargs(source=con)
fitted = xorq_pipeline.fit(train_data, features=features, target=target, cache=cache)

print("Pipeline fitted.")

# ---------------------------------------------------------------------------
# 3. Build predictions — the expression carries ML tags automatically
# ---------------------------------------------------------------------------

dev_inference = xo.memtable(
    {
        "sepal_length": [5.0, 6.2, 6.9],
        "sepal_width": [3.3, 2.9, 3.1],
        "petal_length": [1.4, 4.3, 5.4],
        "petal_width": [0.2, 1.3, 2.1],
    },
    name="inference_dev",
)

predictions = fitted.predict(dev_inference)
print("\nDev predictions expression built.")

# ---------------------------------------------------------------------------
# 4. Catalog the prediction expression (it becomes an ExprBuilder entry)
# ---------------------------------------------------------------------------

catalog_dir = Path(tempfile.mkdtemp()) / "pipeline-catalog"
catalog = Catalog.from_repo_path(catalog_dir, init=True)
print(f"\nCatalog directory: {catalog_dir}")

catalog.add(predictions, aliases=("iris-predictions-dev",), sync=False)
print("Catalog entries:", catalog.list())
print("Catalog aliases:", catalog.list_aliases())

# Check the entry kind and builder metadata
entry = catalog.get_catalog_entry("iris-predictions-dev", maybe_alias=True)
print(f"\nEntry kind: {entry.kind}")
print(f"Builder metadata: {entry.metadata.builders}")

# ---------------------------------------------------------------------------
# 5. Recover the FittedPipeline from the catalog entry via from_tag_node
# ---------------------------------------------------------------------------

recovered_fitted = entry.expr.ls.builder
print(f"\nRecovered type: {type(recovered_fitted).__name__}")
print(f"Recovered pipeline: {recovered_fitted.pipeline}")

# ---------------------------------------------------------------------------
# 6. Use the recovered FittedPipeline on production data
# ---------------------------------------------------------------------------

prd_inference = xo.memtable(
    {
        "sepal_length": [4.8, 6.0, 7.2, 5.7],
        "sepal_width": [3.1, 2.5, 3.0, 2.8],
        "petal_length": [1.6, 3.9, 5.8, 4.1],
        "petal_width": [0.2, 1.1, 1.6, 1.3],
    },
    name="inference_prd",
)

prd_transform = recovered_fitted.transform(prd_inference)
print("\nProd transform:")
print(prd_transform.execute())

prd_predictions = recovered_fitted.predict(prd_inference)
print("\nProd predictions:")
print(prd_predictions.execute())


if __name__ == "__pytest_main__":
    pytest_examples_passed = True
