"""
Demonstrates a hybrid ColumnTransformer pipeline with numeric, text (TF-IDF), and categorical (OneHotEncoder) features.

Traditional approach: With plain sklearn, you build a ColumnTransformer with manual
column selection and fit it on in-memory pandas DataFrames. Switching data sources
(e.g., from CSV to a database) means rewriting your loading code. There is no
separation between data definition and execution.

With xorq: You use the same sklearn transformers, but data loading is backend-agnostic
via Ibis expressions -- swap a memtable for a database table with no pipeline changes.
The ColumnTransformer definition is composable with deferred execution, and the
intermediate IR schemas are inspectable before any computation runs.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import xorq.api as xo
from xorq.expr.ml.pipeline_lib import Pipeline


# Create sample data
np.random.seed(42)
n_samples = 50

data = pd.DataFrame(
    {
        "age": np.random.randint(18, 80, n_samples).astype(float),
        "income": np.random.randint(20000, 150000, n_samples).astype(float),
        "text": [f"word{i % 5} text{i % 3} doc" for i in range(n_samples)],
        "category": np.random.choice(["A", "B", "C"], n_samples),
        "target": np.random.randint(0, 2, n_samples),
    }
)

numeric_features = ["age", "income"]
text_features = ["text"]
categorical_features = ["category"]

# Build pipeline - mixed children: known-schema + KV-encoded = HYBRID
preprocessor = ColumnTransformer(
    [
        ("numeric", StandardScaler(), numeric_features),  # known-schema
        ("tfidf", TfidfVectorizer(), "text"),  # KV-encoded
        ("cat", OneHotEncoder(sparse_output=False), categorical_features),  # KV-encoded
    ],
)

sklearn_pipe = SklearnPipeline(
    [
        ("preprocessor", preprocessor),  # hybrid output
        ("selector", SelectKBest(f_classif, k=5)),  # selects k features -> known schema
        ("classifier", RandomForestClassifier(n_estimators=10, random_state=42)),
    ]
)

# xorq pipeline
expr = xo.memtable(data)
xorq_pipeline = Pipeline.from_instance(sklearn_pipe)

all_features = tuple(numeric_features + text_features + categorical_features)
fitted_pipeline = xorq_pipeline.fit(expr, features=all_features, target="target")

# Show the ColumnTransformer's hybrid output schema
ct_step = fitted_pipeline.transform_steps[0]
print("ColumnTransformer output schema (hybrid):")
for field_name, field_type in ct_step.structer.struct.fields.items():
    print(f"  {field_name}: {field_type}")

# Show intermediate IR after ColumnTransformer
ct_transform = ct_step.transform(expr)
print("\nIntermediate IR schema after ColumnTransformer:")
print(ct_transform.schema())

# Compare with sklearn's feature names
X = data[list(all_features)]
y = data["target"]
preprocessor.fit(X)
print("\nsklearn feature_names_out:")
print(preprocessor.get_feature_names_out())

if __name__ in ("__main__", "__pytest_main__"):
    # Execute and verify predictions match
    sklearn_pipe.fit(X, y)
    sklearn_preds = sklearn_pipe.predict(X)
    predictions = fitted_pipeline.predict(expr).execute()

    match = np.array_equal(predictions["predicted"].values, sklearn_preds)
    print(f"\nxorq matches sklearn: {match}")
    pytest_examples_passed = True
