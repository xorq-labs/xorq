# sklearn ↔ xorq Interoperability Design

## Philosophy & Goals

### Core Principle: Wrap, Don't Reimplement

The fundamental design philosophy is to **wrap sklearn for deferred execution**, not to reimplement sklearn functionality within xorq. This means:

- sklearn handles all ML complexity internally (fitting, transforming, feature engineering)
- xorq provides the deferred execution layer and caching
- We treat sklearn estimators as opaque units where possible
- We avoid creating custom `Structer` registrations for every new estimator type

### Why This Approach?

1. **Maintainability**: sklearn has hundreds of estimators; registering each is unsustainable
2. **Correctness**: sklearn's internal logic is well-tested; reimplementing risks bugs
3. **Flexibility**: Users can use any sklearn-compatible estimator without xorq changes
4. **Simplicity**: One generic solution beats many special cases

### The Structer Problem

Previously, each sklearn transformer needed a `Structer` registration to define its output schema:

```python
# OLD APPROACH - doesn't scale
@Structer.register(StandardScaler)
def standard_scaler_structer(...):
    return StructType(...)

@Structer.register(OneHotEncoder)
def one_hot_encoder_structer(...):
    # Complex logic to determine output columns
    ...

@Structer.register(ColumnTransformer)
def column_transformer_structer(...):
    # Even more complex...
    ...
```

**Problems:**
- OneHotEncoder output columns depend on categories found in training data
- ColumnTransformer combines multiple transformers with different schemas
- PCA output dimensions are configurable
- New estimators require new registrations

**Solution**: Packed format as universal fallback (see below).

---

## Key APIs

### `Step.from_instance_name(instance, name)`

Wraps any sklearn estimator instance as a xorq Step.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Simple transformer
step = Step.from_instance_name(StandardScaler(), name="scaler")

# Complex transformer with nested structure
ct = ColumnTransformer([
    ("num", StandardScaler(), ["age", "income"]),
    ("cat", OneHotEncoder(), ["gender", "region"])
])
step = Step.from_instance_name(ct, name="preprocessor")
```

**Implementation details:**
- Extracts `params_tuple` from `instance.get_params(deep=False)`
- Stores estimator class type for reconstruction
- Parameters are pickled before passing to deferred functions (handles unhashable types)

### `Pipeline.from_instance(sklearn_pipeline)`

Decomposes an sklearn Pipeline into individual xorq Steps.

```python
from sklearn.pipeline import Pipeline as SklearnPipeline

sklearn_pipe = SklearnPipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression())
])
xorq_pipe = Pipeline.from_instance(sklearn_pipe)
```

**Important limitation:**
When a pipeline step outputs packed format (ColumnTransformer, OneHotEncoder), subsequent steps cannot consume it directly. The packed format is `Array[Struct{key, value}]`, not individual columns.

**Workaround:** For complex pipelines with ColumnTransformer, wrap the entire sklearn Pipeline as a single Step:

```python
# RECOMMENDED for ColumnTransformer pipelines
step = Step.from_instance_name(sklearn_pipeline, name="full_pipeline")
```

### `to_sklearn()`

Returns the sklearn estimator (fitted or unfitted) for direct use outside xorq.

```python
# Unfitted Step -> unfitted sklearn estimator
sklearn_scaler = step.to_sklearn()

# Fitted Step -> triggers execution, returns fitted sklearn estimator
fitted_step = step.fit(expr, features=("a", "b"))
sklearn_fitted = fitted_step.to_sklearn()
sklearn_fitted.transform(new_df)  # Use sklearn directly on pandas

# Fitted Pipeline -> reconstructs sklearn Pipeline with all fitted steps
fitted_pipeline = xorq_pipe.fit(expr, features=features, target="y")
sklearn_pipeline = fitted_pipeline.to_sklearn()
```

**Design decisions:**
- `Step.to_sklearn()`: Returns `self.instance` (unfitted clone)
- `FittedStep.to_sklearn()`: Executes deferred model, unpickles bytes, returns fitted estimator
- `FittedPipeline.to_sklearn()`: Reconstructs `sklearn.pipeline.Pipeline` with all fitted steps in order

---

## Packed Format: The Universal Fallback

### The Problem

Transformers have varying output schemas:
- **StandardScaler**: Same columns as input
- **OneHotEncoder**: Depends on categories found in training data
- **PCA**: Configurable number of components
- **ColumnTransformer**: Combines outputs from multiple sub-transformers

At graph construction time, we don't know what the output schema will be. We only know after fitting on actual data.

### The Solution: Packed Format

A universal output type that works for any transformer:

```
Array[Struct{key: string, value: float64}]
```

Each row contains an array of key-value pairs representing the transformed features:

```python
# Example packed output
[
    [{"key": "pca_0", "value": 1.23}, {"key": "pca_1", "value": -0.45}],
    [{"key": "pca_0", "value": 0.89}, {"key": "pca_1", "value": 0.12}],
    ...
]
```

**Advantages:**
- Fixed return type regardless of actual output schema
- Feature names resolved at execution time via `get_feature_names_out()`
- Works for ALL transformers without registration
- Enables fully deferred execution

**Trade-off:**
- Less convenient than individual columns for downstream SQL operations
- Can be unpacked if needed via `unpack_packed_column()`

### When Packed Format Is Used

1. **Unregistered transformers**: Any transformer without a Structer registration
2. **Dynamic schema transformers**: OneHotEncoder, CountVectorizer, ColumnTransformer
3. **Dimensionality reduction**: PCA, TruncatedSVD, TSNE
4. **Probabilistic outputs**: `predict_proba()`, `decision_function()`

### Fallback Logic

```python
def _pieces(self):
    if has_structer_registration(self.step.instance, self.expr, self.features):
        # Use Structer-based approach for registered types
        f = deferred_fit_transform_sklearn_struct
    else:
        # Fallback to packed format for unregistered types
        f = deferred_fit_transform_sklearn_packed
```

---

## Predictor Support

### Return Type Registry

A dispatch-based registry determines predict output types:

| Estimator Type | Return Type | Rationale |
|---------------|-------------|-----------|
| `LogisticRegression` | Target column type | Classification returns class labels |
| `LinearRegression` | `float64` | Regression returns continuous values |
| `ClassifierMixin` | Target column type | Generic classifier catch-all |
| `RegressorMixin` | `float64` | Generic regressor catch-all |
| `ClusterMixin` | `int64` | Cluster labels are integers |
| `sklearn.Pipeline` | Inferred from final step | Pipeline output = final step output |

**Registration hierarchy:**
1. Specific class registrations (LogisticRegression, KNeighborsClassifier)
2. Mixin catch-alls (ClassifierMixin, RegressorMixin, ClusterMixin)
3. Default raises `ValueError` for unregistered types

```python
@registry.register_lazy("sklearn")
def lazy_register_sklearn():
    registry.register(LogisticRegression, get_target_type)
    registry.register(LinearRegression, return_constant(dt.float64))
    registry.register(ClassifierMixin, get_target_type)  # Catch-all
    registry.register(RegressorMixin, return_constant(dt.float64))  # Catch-all
    registry.register(SklearnPipeline, get_pipeline_return_type)
```

### Probabilistic Outputs

Beyond `predict()`, classifiers often provide probability estimates:

```python
# Class probabilities (LogisticRegression, RandomForest, etc.)
proba_result = fitted.predict_proba(expr)
# Returns packed format: [{"key": "class_0", "value": 0.3}, {"key": "class_1", "value": 0.7}]

# Decision function values (SVC, LinearSVC, etc.)
decision_result = fitted.decision_function(expr)
# Returns packed format: [{"key": "decision", "value": 1.23}]
```

**Design decisions:**
- Both methods use packed format for consistency
- Raises `AttributeError` if estimator doesn't support the method
- Available on both `FittedStep` and `FittedPipeline`

---

## Transform vs Predict Disambiguation

sklearn estimators can have `transform()`, `predict()`, or both:

| Estimator | transform() | predict() | Classification |
|-----------|-------------|-----------|----------------|
| StandardScaler | ✓ | ✗ | Transformer |
| LogisticRegression | ✗ | ✓ | Predictor |
| KMeans | ✓ | ✓ | **Predictor** (prioritized) |
| sklearn.Pipeline | ✓ | ✓ | **Predictor** (prioritized) |

**Rule:** If an estimator has both methods, treat it as a predictor.

```python
@property
def is_transform(self):
    has_transform = hasattr(self.step.typ, "transform")
    has_predict = hasattr(self.step.typ, "predict")
    if has_transform and has_predict:
        return False  # Prioritize predict
    return has_transform

@property
def is_predict(self):
    return hasattr(self.step.typ, "predict")
```

**Rationale:**
- Pipelines ending with classifiers should predict, not transform
- KMeans primary use case is getting cluster assignments (predict), not distances (transform)
- This matches sklearn's typical usage patterns

---

## Transductive Estimators

Some estimators can only produce results for training data and cannot generalize to new data.

### Fit-Transform-Only (Embeddings)

Estimators like TSNE, MDS, Isomap only have `fit_transform()`:

```python
TRANSDUCTIVE_FIT_TRANSFORM_ONLY = {
    "TSNE", "MDS", "Isomap", "LocallyLinearEmbedding",
    "SpectralEmbedding", "UMAP"  # if installed
}

def is_fit_transform_only(cls):
    if cls.__name__ in TRANSDUCTIVE_FIT_TRANSFORM_ONLY:
        return True
    # Check if transform raises NotImplementedError
    ...
```

### Fit-Predict-Only (Transductive Clusterers)

Estimators like DBSCAN, OPTICS only have `fit_predict()`:

```python
TRANSDUCTIVE_FIT_PREDICT_ONLY = {
    "DBSCAN", "OPTICS", "HDBSCAN", "AgglomerativeClustering",
    "SpectralClustering", "Birch"
}

def is_fit_predict_only(cls):
    return cls.__name__ in TRANSDUCTIVE_FIT_PREDICT_ONLY
```

### Error Handling

When users try to call `transform()` or `predict()` on transductive estimators:

```python
def transform(self, expr):
    if self.is_fit_transform_only:
        raise TypeError(
            f"{self.step.typ.__name__} is a transductive estimator that only supports "
            f"fit_transform(). It cannot transform new data. "
            f"Use the training data results from fit() or re-fit on the new data."
        )
    ...

def predict_raw(self, expr):
    if self.is_fit_predict_only:
        raise TypeError(
            f"{self.step.typ.__name__} is a transductive clusterer that only supports "
            f"fit_predict(). It cannot predict on new data. "
            f"Use the training data results from fit() or re-fit on the new data."
        )
    ...
```

### Properties for Introspection

```python
fitted_step.is_fit_transform_only  # True for TSNE, etc.
fitted_step.is_fit_predict_only    # True for DBSCAN, etc.
fitted_step.is_transductive        # True if either above is True
```

---

## Hashability for Deferred Execution

### The Problem

xorq uses `FrozenDict` and `toolz.curry` for deferred execution, which require hashable values. sklearn objects with list parameters break this:

```python
# ColumnTransformer has a list of transformers
ct = ColumnTransformer([
    ("num", StandardScaler(), ["age", "income"]),  # This list breaks hashing
    ("cat", OneHotEncoder(), ["gender"])
])
```

When passed through curry → FrozenDict → hash(), we get:
```
TypeError: unhashable type: 'list'
```

### Solution 1: Pickle Parameters

Before passing params to curry functions, pickle them to bytes (which are hashable):

```python
@toolz.curry
def fit_sklearn(df, target=None, *, cls, params_pickled):
    # Unpickle params to handle unhashable types
    params = cloudpickle.loads(params_pickled)
    obj = cls(**dict(params))
    obj.fit(df, target)
    return obj

# At call site
params_pickled = cloudpickle.dumps(params)
fit=fit_sklearn(cls=cls, params_pickled=params_pickled)
```

Applied to all deferred functions:
- `deferred_fit_transform_sklearn`
- `deferred_fit_predict_sklearn`
- `deferred_fit_transform_sklearn_packed`
- `deferred_fit_transform_sklearn_struct`
- `deferred_fit_transform_only_sklearn_packed`
- `deferred_fit_predict_only_sklearn`
- `deferred_fit_transform_series_sklearn`

### Solution 2: Dask Tokenization

For `dask.base.tokenize()` which is used for caching keys:

```python
def lazy_register_sklearn():
    from sklearn.base import BaseEstimator
    from sklearn.pipeline import Pipeline as SklearnPipeline

    @dask.base.normalize_token.register(SklearnPipeline)
    def normalize_sklearn_pipeline(pipeline):
        steps_data = tuple(
            (name, dask.base.tokenize(step))
            for name, step in pipeline.steps
        )
        return normalize_seq_with_caller("SklearnPipeline", steps_data, pipeline.memory)

    @dask.base.normalize_token.register(BaseEstimator)
    def normalize_sklearn_estimator(estimator):
        params = estimator.get_params(deep=False)
        params_hashable = make_hashable(params)  # Convert lists to tuples recursively
        return normalize_seq_with_caller(
            type(estimator).__name__,
            type(estimator).__module__,
            params_hashable
        )
```

### Solution 3: Tag Kwargs

For expression tagging/metadata:

```python
@property
def tag_kwargs(self):
    def make_hashable(v):
        if isinstance(v, list):
            return tuple(make_hashable(x) for x in v)
        elif isinstance(v, dict):
            return tuple(sorted((k, make_hashable(val)) for k, val in v.items()))
        elif isinstance(v, tuple):
            return tuple(make_hashable(x) for x in v)
        return v

    params_hashable = make_hashable(self.params_tuple)
    return {"typ": self.typ, "name": self.name, "params_tuple": params_hashable}
```

---

## FittedStep Caching Considerations

`FittedStep` is a frozen attrs class (`@frozen`), which means attributes cannot be set after initialization. This affects property caching:

```python
@frozen
class FittedStep:
    # Cannot use @functools.cache because it tries to hash self
    # Cannot use object.__setattr__ because class is frozen

    @property
    def _pieces(self):
        # Computed on each access (no caching)
        # This is acceptable because it's lightweight graph construction
        ...

    @property
    def model(self):
        # Triggers deferred execution on each access
        # Users should call to_sklearn() once and store the result
        ...
```

**Recommendation:** Call `to_sklearn()` once and store the result rather than accessing repeatedly.

---

## Usage Patterns

### Pattern 1: Simple Transformer

```python
from sklearn.preprocessing import StandardScaler

step = Step.from_instance_name(StandardScaler(), name="scaler")
fitted = step.fit(expr, features=("age", "income"))

# Deferred execution
result = fitted.transform(expr).execute()

# Get sklearn for direct use
sklearn_scaler = fitted.to_sklearn()
sklearn_scaler.transform(new_pandas_df)
```

### Pattern 2: Simple Predictor

```python
from sklearn.linear_model import LogisticRegression

step = Step.from_instance_name(LogisticRegression(max_iter=200), name="clf")
fitted = step.fit(expr, features=("age", "income"), target="approved")

# Predictions
predictions = fitted.predict(expr).execute()

# Probabilities
proba = fitted.predict_proba(expr).execute()

# Get sklearn for direct use
sklearn_clf = fitted.to_sklearn()
sklearn_clf.predict_proba(new_pandas_df)
```

### Pattern 3: Unregistered Transformer (Packed Format)

```python
from sklearn.preprocessing import MinMaxScaler  # Not in Structer registry

step = Step.from_instance_name(MinMaxScaler(), name="minmax")
fitted = step.fit(expr, features=("age", "income"))

# Returns packed format
result = fitted.transform_raw(expr).as_table().execute()
# result["transformed"] contains Array[Struct{key, value}]

# Unpack if needed
from xorq.expr.ml.fit_lib import unpack_packed_column
unpacked = unpack_packed_column(result, "transformed")
```

### Pattern 4: ColumnTransformer (Recommended Approach)

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SklearnPipeline

# Create complex sklearn pipeline
sklearn_pipe = SklearnPipeline([
    ("preprocessor", ColumnTransformer([
        ("num", SklearnPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])),
    ("classifier", LogisticRegression(max_iter=500))
])

# Wrap ENTIRE pipeline as single Step (recommended)
step = Step.from_instance_name(sklearn_pipe, name="full_pipeline")
fitted = step.fit(expr, features=all_features, target="approved")

# Predictions
predictions = fitted.predict(expr).execute()

# Get full fitted sklearn pipeline
sklearn_fitted = fitted.to_sklearn()
sklearn_fitted.predict(new_pandas_df)
```

### Pattern 5: Pipeline Decomposition (Simple Transformers Only)

```python
# Only use this when all steps have known output schemas
sklearn_pipe = SklearnPipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression())
])

xorq_pipe = Pipeline.from_instance(sklearn_pipe)
fitted = xorq_pipe.fit(expr, features=features, target="y")

# Reconstruct sklearn Pipeline
sklearn_fitted = fitted.to_sklearn()
```

---

## Assumptions & Limitations

### Assumptions

1. **sklearn API stability**: We assume sklearn's `get_params()`, `fit()`, `transform()`, `predict()` interfaces remain stable
2. **Picklability**: All sklearn estimators can be pickled via cloudpickle
3. **get_feature_names_out()**: Transformers provide this method for packed format feature names (falls back to generic names if not available)
4. **BaseEstimator inheritance**: All sklearn estimators inherit from `sklearn.base.BaseEstimator`

### Limitations

1. **Intermediate packed format**: When `Pipeline.from_instance()` decomposes a pipeline with ColumnTransformer, subsequent steps can't consume packed format. **Workaround**: Wrap entire pipeline as single Step.

2. **No caching in FittedStep**: Properties like `model` execute on each access due to frozen class constraints. **Workaround**: Call `to_sklearn()` once and store.

3. **Transductive estimators**: TSNE, DBSCAN, etc. can only produce results for training data. Calling `transform()`/`predict()` on new data raises `TypeError`.

4. **Estimators with both transform and predict**: Treated as predictors only. If you need transform behavior, use the estimator directly via `to_sklearn()`.

5. **Custom estimators**: Must follow sklearn conventions (inherit BaseEstimator, implement get_params, etc.) to work with `from_instance_name()`.

---

## Future Considerations

### Not Implemented (Out of Scope)

1. **Feature unions**: `sklearn.pipeline.FeatureUnion` is not specially handled
2. **Grid search**: `GridSearchCV`, `RandomizedSearchCV` not wrapped
3. **Model selection**: `cross_val_score`, train/test split utilities
4. **Incremental learning**: `partial_fit()` not supported

### Potential Enhancements

1. **Automatic unpacking**: Option to automatically unpack packed format to columns
2. **Schema inference**: Pre-fit schema inference for registered transformers
3. **Caching improvements**: External cache for FittedStep.model
4. **Better Pipeline decomposition**: Handle intermediate packed format by auto-unpacking

---

## Summary

| Feature | Approach |
|---------|----------|
| Wrapping sklearn | `Step.from_instance_name()`, `Pipeline.from_instance()` |
| Round-trip conversion | `to_sklearn()` on Step, FittedStep, Pipeline, FittedPipeline |
| Unknown output schemas | Packed format `Array[Struct{key, value}]` |
| Predictor return types | Mixin-based registry with catch-alls |
| Probabilistic outputs | `predict_proba()`, `decision_function()` in packed format |
| Transductive estimators | Detection + helpful error messages |
| Unhashable params | Pickle before curry, normalize for dask tokenization |
| Complex pipelines | Wrap entire sklearn Pipeline as single Step |
