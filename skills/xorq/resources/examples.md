# Xorq Examples Reference

Complete, runnable examples demonstrating xorq patterns. All examples are in `examples/` directory and can be run with `python examples/<filename>.py`.

---

## Quick Start Examples

### simple_example.py
**Pattern:** Minimal xorq pipeline
**Demonstrates:** Basic filtering and aggregation

```python
import xorq.api as xo

expr = (
    xo.examples.iris.fetch(backend=xo.connect())
    .filter([xo._.sepal_length > 5])
    .group_by("species")
    .agg(xo._.sepal_width.sum())
)

result = expr.execute()
```

**Use when:** Learning xorq basics, testing setup

---

### pandas_example.py
**Pattern:** Pandas integration
**Demonstrates:** Register pandas DataFrame, execute query

**Use when:** Working with existing pandas DataFrames

---

### iris_example.py
**Pattern:** Local caching with ParquetCache
**Demonstrates:** Cache filtered data to local parquet files

**Use when:** Caching intermediate results locally

---

## ML & sklearn Integration

### penguins_classification_quickstart.py ⭐
**Pattern:** Complete ML workflow with metrics
**Demonstrates:**
- Load example dataset
- Train/test split
- sklearn pipeline (StandardScaler + RandomForestClassifier)
- Predictions and predict_proba
- Deferred sklearn metrics (accuracy, precision, recall, F1, ROC-AUC)
- Feature importances extraction

**Key code:**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xorq.expr.ml.metrics import deferred_sklearn_metric
from xorq.expr.ml.pipeline_lib import Pipeline

# Train
fitted_pipeline = Pipeline.from_instance(sklearn_pipeline).fit(
    train_data, features=feature_cols, target="species"
)

# Predict
predictions = fitted_pipeline.predict(test_data)

# Evaluate (deferred)
accuracy = deferred_sklearn_metric(
    expr=predictions,
    target="species",
    pred_col="predicted",
    metric_fn=accuracy_score,
).execute()
```

**Use when:** Building classification pipelines with sklearn

---

### pipeline_example.py
**Pattern:** Manual pipeline vs Pipeline wrapper comparison
**Demonstrates:**
- Building manual deferred pipeline
- Using Pipeline.from_instance()
- Struct preservation pattern with `as_struct()`

**Use when:** Understanding pipeline internals

---

### pipeline_example_SelectKBest.py
**Pattern:** Feature selection in pipeline
**Demonstrates:** SelectKBest + LinearSVC pipeline

**Use when:** Using feature selection with sklearn

---

### pipeline_example_set_params.py
**Pattern:** Hyperparameter tuning
**Demonstrates:** set_params() for grid search patterns

**Use when:** Tuning model hyperparameters

---

### sklearn_classifier_comparison.py
**Pattern:** Multi-model comparison
**Demonstrates:**
- Test multiple classifiers (KNN, SVM, Decision Tree, Random Forest, etc.)
- Compare sklearn vs xorq scores
- Synthetic datasets (moons, circles, linearly separable)

**Use when:** Comparing multiple sklearn models

---

### sklearn_metrics_comparison.py
**Pattern:** Comprehensive metrics validation
**Demonstrates:**
- Test all major sklearn metrics
- predict_proba() for ROC-AUC
- decision_function() for LinearSVC
- feature_importances() for tree models
- Validation against sklearn reference

**Use when:** Validating metric implementations

---

### train_test_splits.py
**Pattern:** Data splitting strategies
**Demonstrates:**
- Single test_size float (train/test)
- Multiple test_sizes list (hold_out/test/validation/training)
- calc_split_column() for stratification

**Use when:** Creating train/test/validation splits

---

### bank_marketing.py
**Pattern:** Full ML pipeline with custom transformers
**Demonstrates:**
- OneHotEncoder with custom transform
- XGBoost with array-encoded features
- Classification report, confusion matrix, AUC
- ParquetCache for expensive operations

**Use when:** Building production ML pipelines

---

## Deferred ML Patterns

### deferred_fit_predict_example.py
**Pattern:** Deferred fit + predict
**Demonstrates:** LinearRegression with deferred execution and caching

**Use when:** Custom fit/predict patterns

---

### deferred_fit_transform_example.py
**Pattern:** Deferred fit + transform
**Demonstrates:** TfidfVectorizer with train/test

**Use when:** Feature transformations (TF-IDF, PCA, etc.)

---

### deferred_fit_transform_predict_example.py ⭐
**Pattern:** Complete deferred ML pipeline
**Demonstrates:**
- TfidfVectorizer transform step
- XGBoost predict step
- FittedPipeline composition
- Caching strategy

**Key code:**
```python
transform_step = Step(TfidfVectorizer)
predict_step = Step.from_fit_predict(
    fit=fit_xgboost_model,
    predict=predict_xgboost_model,
    return_type=dt.float64,
)
fitted_pipeline = FittedPipeline((fitted_transform, fitted_predict), train_expr)
```

**Use when:** Multi-step ML pipelines with custom models

---

## UDFs & UDXFs

### expr_scalar_udf.py
**Pattern:** ExprScalarUDF for model inference
**Demonstrates:**
- Train XGBoost with UDAF
- Predict with ExprScalarUDF (computed kwargs from trained model)
- Pattern for unsupported sklearn models

**Key code:**
```python
model_udaf = udf.agg.pandas_df(
    fn=toolz.compose(pickle.dumps, train_xgboost_model),
    schema=t[features + (target,)].schema(),
    return_type=dt.binary,
    name=model_key,
)
predict_expr_udf = make_pandas_expr_udf(
    computed_kwargs_expr=model_udaf.on_expr(train),
    fn=predict_xgboost_model,
    schema=t[features].schema(),
    return_type=dt.dtype(prediction_typ),
    name=prediction_key,
)
```

**Use when:** sklearn models not supported by Pipeline, custom ML models

---

### xgboost_udaf.py
**Pattern:** UDAF for feature selection
**Demonstrates:** XGBoost get_score() for best features via UDAF

**Use when:** Custom aggregations with ML models

---

### quickgrove_udf.py
**Pattern:** Compiled XGBoost models
**Demonstrates:** Load XGBoost JSON, compile to quickgrove, rewrite expression

**Use when:** Production XGBoost serving with performance optimization

---

### python_udwf.py
**Pattern:** Window functions (UDWF)
**Demonstrates:**
- Exponential smoothing
- Bounded execution
- Rank-based smoothing
- Window frames
- Multi-column windows

**Use when:** Custom window functions (rolling aggregations, smoothing, etc.)

---

## Flight Servers & UDXFs

### quickstart.py ⭐
**Pattern:** Complete ML serving example
**Demonstrates:**
- Load pre-trained TF-IDF + XGBoost
- Create prediction UDF
- Serve as Flight UDXF
- Client/server pattern

**Use when:** Serving ML models over network

---

### flight_udtf_example.py
**Pattern:** UDTF for data fetching
**Demonstrates:** HackerNews API fetcher with caching

**Use when:** Fetching external data in pipeline

---

### flight_udtf_llm_example.py
**Pattern:** UDTF with LLM API
**Demonstrates:** OpenAI sentiment analysis via Flight

**Use when:** Integrating LLM APIs into pipelines

---

### flight_serve_model.py
**Pattern:** Serve fitted model
**Demonstrates:** flight_serve() for TF-IDF transformer

**Use when:** Serving transformation models

---

### flight_dummy_exchanger.py
**Pattern:** Minimal exchanger
**Demonstrates:** Basic UDXF with schemas

**Use when:** Learning Flight exchanger pattern

---

### flight_exchange_example.py
**Pattern:** Streaming exchange
**Demonstrates:** Iterative split training with streaming_split_exchange

**Use when:** Custom streaming data processing

---

### duckdb_flight_example.py
**Pattern:** Concurrent DuckDB access
**Demonstrates:** Flight server with concurrent readers/writers

**Use when:** Multi-client database access

---

### mcp_flight_server.py
**Pattern:** MCP (Model Context Protocol) integration
**Demonstrates:** Wrap Flight UDXF as MCP tool for Claude Desktop

**Use when:** Integrating xorq with Claude Desktop

---

### weather_flight.py
**Pattern:** Feature store with Flight
**Demonstrates:**
- Offline (batch) and online (flight) feature sources
- FeatureStore, FeatureView, Entity patterns
- Materialize online features
- Historical feature retrieval
- Weather API UDXF

**Use when:** Building feature stores

---

## Data Loading & Caching

### deferred_read_csv.py
**Pattern:** Deferred CSV reading
**Demonstrates:** Read CSV into pandas and postgres backends

**Use when:** Loading CSV data lazily

---

### local_cache.py
**Pattern:** Local parquet caching
**Demonstrates:** ParquetCache with relative path

**Use when:** Caching to local filesystem

---

### postgres_caching.py
**Pattern:** PostgreSQL caching
**Demonstrates:** ParquetCache with postgres source

**Use when:** Caching with database backend

---

### remote_caching.py
**Pattern:** Multi-backend caching
**Demonstrates:** SourceCache with postgres

**Use when:** Caching across different backends

---

### gcstorage_example.py
**Pattern:** Google Cloud Storage caching
**Demonstrates:** GCCache with GCS bucket

**Use when:** Caching to cloud storage

---

### sqlite_example.py
**Pattern:** SQLite integration
**Demonstrates:**
- Read PyArrow batches into SQLite
- into_backend() + SourceCache

**Use when:** Using SQLite as backend

---

### pyiceberg_backend_simple.py
**Pattern:** PyIceberg backend
**Demonstrates:** Create Iceberg table, query

**Use when:** Working with Apache Iceberg

---

## Multi-Engine & Composition

### multi_engine.py
**Pattern:** Cross-backend joins
**Demonstrates:** Join postgres table with duckdb table

**Use when:** Combining data from different backends

---

### into_backend_example.py
**Pattern:** Backend transition
**Demonstrates:** Move from postgres to duckdb with into_backend()

**Use when:** Moving data between backends

---

### yaml_roundtrip.py
**Pattern:** Build serialization
**Demonstrates:** build_expr() + load_expr() roundtrip

**Use when:** Serializing/deserializing expressions

---

## Complex Workflows

### complex_cached_expr.py
**Pattern:** Multi-stage ML serving
**Demonstrates:**
- HackerNews fetcher UDXF
- Sentiment analysis (OpenAI) UDXF
- TF-IDF transform
- XGBoost predict
- Serve/predict architecture
- Command validation

**Key architecture:**
```python
# Build stages
train_expr → do_fit() → (trained_model, transform, predict)
test_expr → do_transform_predict() → predictions
live_expr → do_serve() → (transform_server, predict_server)

# Client
transform_client.do_exchange(data) → predict_client.do_exchange() → results
```

**Use when:** Production ML pipelines with multiple services

---

### profiles.py
**Pattern:** Connection profiles
**Demonstrates:**
- Save connection configs with env var references
- Load profiles
- Clone profiles with modifications
- Security (never stores actual secrets)

**Use when:** Managing database connections across environments

---

### simple_lineage.py
**Pattern:** Column-level lineage
**Demonstrates:**
- build_column_trees()
- print_tree() for visualization
- Track data flow through UDFs and aggregations

**Use when:** Understanding data lineage, auditing transformations

---

## Summary by Use Case

### Learning Xorq
1. **simple_example.py** - Start here
2. **pandas_example.py** - Pandas integration
3. **iris_example.py** - Local caching

### ML Pipelines
1. **penguins_classification_quickstart.py** - Classification with metrics
2. **pipeline_example.py** - Pipeline patterns
3. **bank_marketing.py** - Production pipeline
4. **deferred_fit_transform_predict_example.py** - Custom models

### Unsupported sklearn Models
1. **expr_scalar_udf.py** - ExprScalarUDF pattern (train with UDAF, predict with UDF)
2. **xgboost_udaf.py** - Custom aggregations

### Model Serving
1. **quickstart.py** - Basic serving
2. **flight_serve_model.py** - Serve transformers
3. **complex_cached_expr.py** - Multi-stage serving
4. **mcp_flight_server.py** - Claude Desktop integration

### Data Engineering
1. **deferred_read_csv.py** - Data loading
2. **multi_engine.py** - Cross-backend joins
3. **train_test_splits.py** - Data splitting

### Custom Functions
1. **python_udwf.py** - Window functions
2. **flight_udtf_example.py** - Table functions
3. **quickgrove_udf.py** - Compiled models

### Feature Engineering
1. **weather_flight.py** - Feature stores
2. **simple_lineage.py** - Track transformations

---

## Running Examples

All examples use `__name__ == "__pytest_main__"` pattern:

```bash
# Run example
python examples/simple_example.py

# Run via pytest
pytest examples/simple_example.py -v

# Run all examples
pytest examples/ -v
```

---

## Testing Examples

Examples demonstrate best practices:
- Deferred execution
- Proper caching
- Schema checking
- Error handling
- Resource cleanup

Each example sets `pytest_examples_passed = True` when successful.

---

## Finding Examples

```bash
# List all examples
ls examples/

# Find examples by pattern
ls examples/*flight*.py     # Flight server examples
ls examples/*pipeline*.py   # ML pipeline examples
ls examples/*cache*.py      # Caching examples

# Search by content
grep -l "TfidfVectorizer" examples/*.py
grep -l "RandomForest" examples/*.py
```

---

## Key Patterns Reference

| Pattern | Examples | When to Use |
|---------|----------|-------------|
| **sklearn Pipeline** | penguins_classification_quickstart.py, pipeline_example.py | Supported sklearn models |
| **ExprScalarUDF + UDAF** | expr_scalar_udf.py, xgboost_udaf.py | Unsupported sklearn models, custom ML |
| **FittedPipeline** | deferred_fit_transform_predict_example.py | Multi-step ML pipelines |
| **Flight UDXF** | quickstart.py, flight_udtf_example.py | Model serving, data fetching |
| **Flight MCP** | mcp_flight_server.py | Claude Desktop integration |
| **Multi-backend** | multi_engine.py, into_backend_example.py | Cross-database queries |
| **Caching** | local_cache.py, postgres_caching.py | Performance optimization |
| **Feature Store** | weather_flight.py | Online/offline features |
| **Profiles** | profiles.py | Connection management |
| **Lineage** | simple_lineage.py | Data auditing |

---

For detailed patterns, see:
- [ml-pipelines.md](ml-pipelines.md) - ML patterns with UDAF examples
- [udf-udxf.md](udf-udxf.md) - Custom function patterns
- [caching.md](caching.md) - Caching strategies
- [WORKFLOWS.md](WORKFLOWS.md) - Step-by-step guides
