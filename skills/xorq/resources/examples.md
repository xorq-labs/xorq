# Complete Examples

## Overview

Complete, runnable examples demonstrating xorq patterns for common use cases.

---

## Example 1: ETL Pipeline

### Pattern: End-to-End Data Transformation

**Use case**: Load, transform, aggregate, and cache data

```python
import xorq.api as xo
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.caching import ParquetStorage
from pathlib import Path

# Setup
con = xo.connect()
storage = ParquetStorage(
    source=con,
    relative_path=Path("./cache")
)

# Build ETL pipeline
pipeline = (
    # Extract: Load data deferred
    deferred_read_parquet(
        path="/data/raw/events.parquet",
        connection=con,
        name="raw_events"
    )

    # Transform: Clean and filter
    .filter([
        xo._.status.isin(["active", "pending"]),
        xo._.timestamp >= "2024-01-01"
    ])
    .select("id", "timestamp", "value", "category", "user_id")

    # Transform: Add derived columns
    .mutate(
        date=xo._.timestamp.date(),
        value_normalized=xo._.value / xo._.value.max(),
        is_high_value=xo._.value > 1000
    )

    # Cache expensive transformation
    .cache(storage=storage, name="cleaned_events")

    # Load: Aggregate for reporting
    .group_by(["category", "date"])
    .agg(
        event_count=xo._.id.count(),
        unique_users=xo._.user_id.nunique(),
        total_value=xo._.value_normalized.sum(),
        avg_value=xo._.value_normalized.mean(),
        high_value_count=xo._.is_high_value.sum()
    )

    # Final sort
    .order_by([xo._.date.desc(), xo._.event_count.desc()])
)

# Execute pipeline
result = pipeline.execute()

# Save results
result.to_csv("daily_category_summary.csv", index=False)
print(f"Processed {len(result)} category-date combinations")
```

**Key patterns used:**
- Deferred reading for large files
- Early filtering to reduce data
- Derived columns with `.mutate()`
- Caching after expensive transformations
- Group-by aggregations
- Multiple aggregation functions

---

## Example 2: ML Training Pipeline

### Pattern: Complete ML Workflow with Sklearn

**Use case**: Train, evaluate, and deploy ML model

```python
import xorq.api as xo
from xorq.expr.ml import train_test_splits
from xorq.expr.ml.pipeline_lib import Pipeline
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import toolz

# Load data
data = xo.examples.iris.fetch()
print(f"Loaded {len(data.execute())} samples")

# Define features and target
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
target = "species"

# Split data (80/20)
train, test = train_test_splits(data, test_size=0.2)
print(f"Train: {len(train.execute())}, Test: {len(test.execute())}")

# Create sklearn pipeline
sk_pipeline = SkPipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=11))
])

# Convert to xorq pipeline
xorq_pipeline = Pipeline.from_instance(sk_pipeline)

# Fit pipeline on training data
fitted_pipeline = xorq_pipeline.fit(
    train,
    features=features,
    target=target
)
print("Model fitted")

# Helper to preserve original data
@toolz.curry
def as_struct(expr, name=None):
    """Pack all columns into struct"""
    struct = xo.struct({c: expr[c] for c in expr.columns})
    if name:
        struct = struct.name(name)
    return struct

# Predict on test data with original preserved
predictions = (
    test
    .mutate(original=as_struct(name="original_row"))  # Preserve
    .pipe(fitted_pipeline.predict)                     # Predict
    .unpack("original_row")                            # Restore
)

# Execute and evaluate
result = predictions.execute()

# Calculate metrics
accuracy = accuracy_score(result[target], result["prediction"])
print(f"\nAccuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(result[target], result["prediction"]))

# Show sample predictions
print("\nSample Predictions:")
print(result[[target, "prediction"] + features].head(10))

# Save model
import pickle
with open("iris_model.pkl", "wb") as f:
    pickle.dump(fitted_pipeline, f)
print("\nModel saved to iris_model.pkl")
```

**Key patterns used:**
- Train/test splitting
- Sklearn pipeline conversion
- Struct preservation pattern
- Pipeline composition with `.pipe()`
- Model persistence

---

## Example 3: Flight Server Data Service

### Pattern: Distributed Data Processing Service

**Use case**: Serve ML model predictions over network

#### Server Code

```python
# server.py
import pandas as pd
import pickle
from xorq.flight import FlightServer
from xorq.flight.exchanger import make_udxf
import xorq.api as xo
import xorq.expr.datatypes as dt
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load trained model
class IrisPredictor:
    """Stateful predictor with loaded model"""

    def __init__(self, model_path: str):
        logger.info(f"Loading model from {model_path}")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        logger.info("Model loaded successfully")

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict on input data"""
        logger.info(f"Processing {len(df)} rows")

        features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

        # Validate input
        missing = set(features) - set(df.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")

        # Make predictions
        predictions = self.model.predict(df[features])

        # Return with predictions
        result = df.copy()
        result["prediction"] = predictions

        logger.info(f"Predictions complete: {len(result)} rows")
        return result

# Create predictor
predictor = IrisPredictor("iris_model.pkl")

# Define schemas
schema_in = xo.schema({
    "sepal_length": dt.float64,
    "sepal_width": dt.float64,
    "petal_length": dt.float64,
    "petal_width": dt.float64
})

schema_out = xo.schema({
    "sepal_length": dt.float64,
    "sepal_width": dt.float64,
    "petal_length": dt.float64,
    "petal_width": dt.float64,
    "prediction": dt.string
})

# Create exchanger
iris_exchanger = make_udxf(
    predictor,
    schema_in,
    schema_out,
    name="iris_predictor"
)

# Create and start server
server = FlightServer(
    exchangers=[iris_exchanger],
    host="0.0.0.0",
    port=8815
)

logger.info("Starting Flight server on port 8815")
logger.info("Press Ctrl+C to stop")
server.serve()
```

#### Client Code

```python
# client.py
from xorq.flight import FlightClient
import pyarrow as pa
import pandas as pd

# Connect to server
client = FlightClient(host="localhost", port=8815)
print("Connected to Flight server")

# Prepare test data
test_data = pd.DataFrame({
    "sepal_length": [5.1, 6.3, 4.9],
    "sepal_width": [3.5, 2.9, 3.0],
    "petal_length": [1.4, 5.6, 1.4],
    "petal_width": [0.2, 1.8, 0.2]
})

# Convert to PyArrow Table
input_table = pa.Table.from_pandas(test_data)

# Call prediction service
result_table = client.do_exchange(
    iris_exchanger.command,
    input_table
)

# Convert back to pandas
result = result_table.to_pandas()

# Display results
print("\nPrediction Results:")
print(result)
```

**Key patterns used:**
- Stateful exchanger with loaded model
- Schema validation
- Logging for debugging
- Error handling
- Flight client/server architecture

---

## Example 4: Feature Engineering Pipeline

### Pattern: Complex Feature Engineering

**Use case**: Engineer features for ML model

```python
import xorq.api as xo
from datetime import datetime
import numpy as np

# Load data
con = xo.connect()
data = con.table("customer_transactions")

# Feature engineering pipeline
engineered = (
    data
    # Time-based features
    .mutate(
        transaction_date=xo._.timestamp.date(),
        transaction_hour=xo._.timestamp.hour(),
        transaction_day_of_week=xo._.timestamp.day_of_week(),
        is_weekend=xo._.timestamp.day_of_week().isin([5, 6])
    )

    # Monetary features
    .mutate(
        log_amount=xo._.amount.log(),
        amount_squared=xo._.amount ** 2,
        amount_per_item=xo._.amount / xo._.item_count
    )

    # Categorical encoding
    .mutate(
        category_risk=xo.case()
            .when(xo._.category == "high_risk", 3)
            .when(xo._.category == "medium_risk", 2)
            .when(xo._.category == "low_risk", 1)
            .else_(0)
            .end()
    )

    # Interaction features
    .mutate(
        amount_hour_interaction=xo._.amount * xo._.transaction_hour,
        risk_amount_interaction=xo._.category_risk * xo._.log_amount
    )

    # User aggregates (window functions)
    .mutate(
        user_avg_amount=xo._.amount.mean().over(
            group_by="user_id"
        ),
        user_total_transactions=xo._.transaction_id.count().over(
            group_by="user_id"
        ),
        user_max_amount=xo._.amount.max().over(
            group_by="user_id"
        )
    )

    # Relative features
    .mutate(
        amount_vs_user_avg=xo._.amount / xo._.user_avg_amount,
        is_max_amount=xo._.amount == xo._.user_max_amount
    )

    # Select final features
    .select(
        "transaction_id",
        "user_id",
        "timestamp",
        # Time features
        "transaction_hour",
        "transaction_day_of_week",
        "is_weekend",
        # Monetary features
        "amount",
        "log_amount",
        "amount_squared",
        "amount_per_item",
        # Categorical features
        "category_risk",
        # Interaction features
        "amount_hour_interaction",
        "risk_amount_interaction",
        # User features
        "user_avg_amount",
        "user_total_transactions",
        "amount_vs_user_avg",
        "is_max_amount"
    )
)

# Execute
features = engineered.execute()
print(f"Engineered {len(features.columns)} features for {len(features)} transactions")
```

**Key patterns used:**
- Time-based feature extraction
- Mathematical transformations
- Conditional features with `case()`
- Window functions for user aggregates
- Interaction features
- Relative features (ratios, comparisons)

---

## Example 5: Real-Time Data Processing

### Pattern: Streaming-Style Processing

**Use case**: Process data in batches

```python
import xorq.api as xo
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.caching import ParquetStorage
from pathlib import Path
import glob

# Setup
con = xo.connect()
storage = ParquetStorage(
    source=con,
    relative_path=Path("./processed")
)

def process_batch(input_path: str, output_path: str):
    """Process a single batch of data"""

    # Build processing pipeline
    pipeline = (
        deferred_read_parquet(input_path, con, "batch")

        # Validate data
        .filter([
            xo._.timestamp.notnull(),
            xo._.value >= 0
        ])

        # Transform
        .mutate(
            processed_at=xo.literal(datetime.now()),
            value_log=xo._.value.log1p(),  # log(1 + x)
            is_anomaly=xo._.value > (xo._.value.mean() + 3 * xo._.value.std())
        )

        # Aggregate by time window
        .mutate(
            time_window=xo._.timestamp.truncate("5 minutes")
        )
        .group_by("time_window")
        .agg(
            count=xo._.id.count(),
            avg_value=xo._.value.mean(),
            max_value=xo._.value.max(),
            anomaly_count=xo._.is_anomaly.sum()
        )
    )

    # Execute and save
    result = pipeline.execute()
    result.to_parquet(output_path)

    return len(result)

# Process all batches
input_pattern = "/data/input/batch_*.parquet"
input_files = sorted(glob.glob(input_pattern))

for i, input_file in enumerate(input_files):
    output_file = f"./processed/batch_{i:04d}.parquet"
    count = process_batch(input_file, output_file)
    print(f"Processed {input_file}: {count} windows")

print(f"Processed {len(input_files)} batches")
```

**Key patterns used:**
- Batch processing pattern
- Data validation with filtering
- Time window aggregations
- Anomaly detection
- File-based processing

---

## Example 6: Multi-Source Data Integration

### Pattern: Combine Data from Multiple Sources

**Use case**: Join data from different backends

```python
import xorq.api as xo
from xorq.common.utils.defer_utils import deferred_read_parquet

# Connect to multiple backends
duckdb = xo.connect()  # Local DuckDB
snowflake = xo.connect("snowflake://account/db/schema")
postgres = xo.connect("postgresql://user:pass@host/db")

# Load from different sources
transactions = deferred_read_parquet(
    "/data/transactions.parquet",
    duckdb,
    "transactions"
)

customers = snowflake.table("customers")
products = postgres.table("products")

# Integrate data
integrated = (
    transactions
    # Join with customer data
    .join(
        customers,
        transactions.customer_id == customers.id,
        how="left"
    )
    .select(
        "transaction_id",
        "customer_id",
        "product_id",
        "amount",
        customer_name=customers.name,
        customer_tier=customers.tier
    )

    # Join with product data
    .join(
        products,
        transactions.product_id == products.id,
        how="left"
    )
    .select(
        "transaction_id",
        "customer_id",
        "customer_name",
        "customer_tier",
        "product_id",
        product_name=products.name,
        product_category=products.category,
        "amount"
    )

    # Add derived fields
    .mutate(
        is_premium_customer=xo._.customer_tier == "premium",
        is_high_value=xo._.amount > 1000
    )

    # Filter and aggregate
    .filter(xo._.is_premium_customer == True)
    .group_by(["customer_name", "product_category"])
    .agg(
        total_transactions=xo._.transaction_id.count(),
        total_amount=xo._.amount.sum(),
        avg_amount=xo._.amount.mean()
    )
    .order_by(xo._.total_amount.desc())
)

# Execute (pulls data from all sources)
result = integrated.execute()
print(result)
```

**Key patterns used:**
- Multi-backend connections
- Cross-backend joins
- Expression-based integration
- Filtering after joins
- Aggregation across sources

---

## Summary

**Example patterns covered:**
1. **ETL Pipeline** - Deferred loading, transformation, caching, aggregation
2. **ML Training** - Split, fit, predict, evaluate, persist
3. **Flight Server** - Client/server architecture, model serving
4. **Feature Engineering** - Complex transformations, window functions
5. **Batch Processing** - File-based processing, time windows
6. **Multi-Source Integration** - Cross-backend joins and aggregations

**Key takeaways:**
- Build expressions declaratively
- Use deferred execution for large data
- Cache expensive operations
- Preserve data with structs
- Test incrementally
- Document complex pipelines
