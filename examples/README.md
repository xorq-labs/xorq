How to run the examples
=======================

## Setup

From the repo root:

```bash
just download-data
just up postgres
#On Apple Silicon Macs, you may see a platform mismatch warning for the Postgres image (linux/amd64 vs linux/arm64); this is not an error — Docker runs it under emulation and it works normally.

# macOS only
brew install cmake libomp

source .venv/bin/activate # Activate the venv
# Install dependencies (from repo root)
uv sync --extra examples --extra postgres

# run the examples
cd examples
xorq build simple_example.py
xorq run simple_example.py

python xorq_build_and_run.py iris_example.py # build and run in a single step
```


## Expressions vs. Scripts

Most examples define a top-level `expr` variable and should be run with `python xorq_build_and_run.py <file>` (or `xorq build` + `xorq run` separately), which compiles the expression graph into a versioned YAML manifest and executes it. A few examples involve runtime behavior like starting servers or CLI subcommands — these are marked as **script** below and should be run directly with `python <file>`.

## Examples

### Getting Started

| File | Description |
|------|-------------|
| `simple_example.py` | Minimal example: loads iris data, filters, groups by species, and aggregates |
| `pandas_example.py` | Registers a pandas DataFrame as a table and queries it |
| `quickstart.py` | End-to-end HackerNews sentiment analysis using pre-trained TF-IDF and XGBoost models, with Flight server integration. Build with `xorq build quickstart.py -e pipeline` |

### Data Loading & Backends

| File | Description |
|------|-------------|
| `deferred_read_csv.py` | Deferred CSV reading with multiple backends (pandas, postgres) and different table creation modes. **Script**: `python deferred_read_csv.py` (supports `--cleanup`) |
| `into_backend_example.py` | Transferring data between backends (postgres to duckdb) with joins and caching |
| `iris_example.py` | Loading the iris dataset with ParquetCache for caching |
| `multi_engine.py` | Working with multiple backends (postgres, duckdb) and joining across them |
| `sqlite_example.py` | Using the SQLite backend with RecordBatch input and chaining to other backends |
| `pyiceberg_backend_simple.py` | Creating and querying PyIceberg tables |

### ML Pipelines & Scikit-Learn

| File | Description |
|------|-------------|
| `pipeline_example.py` | Full ML pipeline with StandardScaler and KNeighborsClassifier; compares manual deferred approach vs the Pipeline API |
| `pipeline_example_SelectKBest.py` | Feature selection pipeline using SelectKBest and LinearSVC |
| `pipeline_example_set_params.py` | Setting pipeline parameters dynamically and computing scores |
| `bank_marketing.py` | Classification pipeline on bank marketing data with preprocessing, caching, confusion matrix, and ROC-AUC |
| `penguins_classification_quickstart.py` | Classification on penguins dataset with StandardScaler + RandomForest, reporting accuracy, precision, recall, F1, ROC-AUC, and feature importances |
| `sklearn_classifier_comparison.py` | Compares 8 classifiers (KNN, SVM, Decision Tree, Random Forest, Neural Net, AdaBoost, Naive Bayes, QDA) across 3 datasets |
| `sklearn_metrics_comparison.py` | Deferred metrics computation for multiple model configurations with predict_proba, decision_function, and feature_importances |
| `train_test_splits.py` | Train/test splitting with scalar and list-based partition sizes |

### Feature Engineering & Transformation

| File | Description |
|------|-------------|
| `demonstrate_ir_two_categorical.py` | Hybrid ColumnTransformer with numeric, text (TF-IDF), and categorical (OneHotEncoder) features |
| `deferred_fit_transform_example.py` | TF-IDF transformation with and without caching on HackerNews data |
| `deferred_fit_predict_example.py` | Linear regression fit/predict with deferred execution and caching |
| `deferred_fit_transform_predict_example.py` | Chained TF-IDF transform + XGBoost prediction pipeline with Flight server serving |

### User-Defined Functions (UDFs, UDAFs, UDWFs)

| File | Description |
|------|-------------|
| `expr_scalar_udf.py` | XGBoost pandas scalar UDF with pickle serialization for model storage |
| `python_udwf.py` | User-defined window functions with exponential smoothing (bounded, rank-based, and frame-based variants) |
| `xgboost_udaf.py` | XGBoost aggregate UDF for computing feature importance grouped by category |
| `quickgrove_udf.py` | XGBoost model loaded from JSON with expression rewriting for optimization |
| `simple_lineage.py` | Column lineage tracking through expressions with custom UDFs |

### Caching

| File | Description |
|------|-------------|
| `local_cache.py` | ParquetCache for local filesystem caching from postgres |
| `postgres_caching.py` | ParquetCache for caching window function results with postgres |
| `remote_caching.py` | Remote caching using DuckDB as source and postgres as cache backend |
| `complex_cached_expr.py` | Multi-level caching with DuckDB and SourceCache. **Script**: `python complex_cached_expr.py` (subcommands: `run`, `serve`, `predict`) |
| `gcstorage_example.py` | Google Cloud Storage caching with GCCache |

### Flight Server & Distribution

[Arrow Flight](https://arrow.apache.org/docs/format/Flight.html) lets you move columnar data between processes over gRPC with zero-copy semantics -- no CSV/JSON serialization overhead. This is useful when a computation must run in a separate process (e.g. a Python model that can't be pushed into the database engine), or when you want to expose a transformation as a network service that multiple clients can call. The `do_exchange` RPC is especially powerful: the client streams input batches and receives output batches in a single bidirectional call, so large datasets never need to be fully materialized on either side.

| File | Description |
|------|-------------|
| `flight_dummy_exchanger.py` | Simple Flight server with a dummy UDXF that returns fixed data |
| `flight_exchange_example.py` | Iterative split-train exchanger using Flight with streaming exchange |
| `flight_serve_model.py` | Flight server serving TF-IDF model transformations |
| `flight_udtf_example.py` | Flight UDTF for fetching HackerNews data via live API calls |
| `flight_udtf_llm_example.py` | Flight UDTF using OpenAI API for sentiment analysis on HackerNews comments *requires openAI key*  |
| `duckdb_flight_example.py` | Concurrent Flight server with DuckDB supporting concurrent read/write. **Script**: `python duckdb_flight_example.py serve\|read\|write` |
| `mcp_flight_server.py` | MCP (Model Context Protocol) server wrapping a Flight sentiment analyzer. **Script**: `python mcp_flight_server.py` |


### Configuration & Serialization

| File | Description |
|------|-------------|
| `profiles.py` | Profile management for database connections with environment variable references |
| `yaml_roundtrip.py` | YAML serialization and deserialization of expressions across multiple backends |

### Helper Libraries (`libs/`)

| File | Description |
|------|-------------|
| `libs/hackernews_lib.py` | HackerNews data fetching via Firebase API with disk caching |
| `libs/openai_lib.py` | OpenAI ChatGPT integration for sentiment analysis with disk caching |
| `libs/mcp_lib.py` | FastMCP server wrapper that exposes Flight UDXFs as MCP tools |

### Data Files (`data/`)

| File | Description |
|------|-------------|
| `data/iris.csv` | Classic Iris classification dataset |
| `data/data.rownum.parquet` | Lending Club dataset with row numbering for ML examples |


### Examples that require API keys

| Script | Environment variable |
|--------|---------------------|
| `weather_flight.py` | `OPENWEATHER_API_KEY` |
| `flight_udtf_llm_example.py` | `OPENAI_API_KEY` |

### Feature Store

| File | Description |
|------|-------------|
| `weather_flight.py` | End-to-end feature store with offline/online features, materialization, inference, and historical retrieval for weather data. **Script**: `python weather_flight.py` |
