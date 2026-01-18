# UDFs, UDXFs, and Flight Servers

## Overview

Xorq provides User Defined Functions (UDFs) and User Defined Exchangers (UDXFs) for custom data processing, along with Arrow Flight server support for distributed data exchange.

---

## User Defined Functions (UDFs)

### Pattern: Pandas UDFs

**Use case**: Apply custom Python functions to expressions

```python
import xorq.api as xo
import xorq.expr.datatypes as dt

# Define pandas UDF
@xo.udf.make_pandas_udf(
    schema=xo.schema({"title": str, "description": str}),
    return_type=dt.float64,
    name="sentiment_score"
)
def score_sentiment(df):
    """Process pandas DataFrame and return scores"""
    import some_model
    return some_model.predict(df["title"] + " " + df["description"])

# Apply UDF in expression
scored = data.mutate(
    sentiment=score_sentiment.on_expr
)

# Execute
result = scored.execute()
```

**Key components:**
- `schema`: Input schema definition
- `return_type`: Output type (scalar or struct)
- `name`: UDF name for debugging
- `.on_expr`: Applies UDF to expression

---

### Pattern: Scalar UDFs

**Use case**: Simple scalar transformations

```python
@xo.udf.make_scalar_udf(
    return_type=dt.string,
    name="normalize_text"
)
def normalize(text: str) -> str:
    """Normalize text to lowercase and strip whitespace"""
    return text.lower().strip()

# Apply to column
normalized = data.mutate(
    normalized_name=normalize(xo._.name)
)
```

---

### Pattern: UDFs with Complex Return Types

**Return structs from UDFs:**

```python
# Define struct return type
return_schema = dt.struct({
    "prediction": dt.float64,
    "confidence": dt.float64,
    "label": dt.string
})

@xo.udf.make_pandas_udf(
    schema=xo.schema({"features": dt.array(dt.float64)}),
    return_type=return_schema,
    name="predict_with_confidence"
)
def predict(df):
    """Return structured prediction results"""
    predictions = model.predict(df["features"])
    confidences = model.predict_proba(df["features"]).max(axis=1)
    labels = label_encoder.inverse_transform(predictions)

    return pd.DataFrame({
        "prediction": predictions,
        "confidence": confidences,
        "label": labels
    })

# Apply and unpack
results = (
    data
    .mutate(results=predict.on_expr)
    .unpack("results")  # Expands struct to columns
)
```

---

### Pattern: UDFs with External Dependencies

**Use case**: UDFs that need models or external resources

```python
class SentimentUDF:
    """UDF class with stateful dependencies"""

    def __init__(self, model_path: str):
        # Load model once
        import torch
        self.model = torch.load(model_path)
        self.model.eval()

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process batch of data"""
        texts = df["text"].tolist()
        scores = self.model.predict(texts)
        return pd.DataFrame({"sentiment": scores})

# Create UDF instance
sentiment_udf = SentimentUDF("model.pth")

# Wrap with make_pandas_udf
@xo.udf.make_pandas_udf(
    schema=xo.schema({"text": str}),
    return_type=dt.float64,
    name="sentiment"
)
def sentiment_scorer(df):
    return sentiment_udf(df)

# Use in expression
scored = data.mutate(sentiment=sentiment_scorer.on_expr)
```

---

## User Defined Exchangers (UDXFs)

### Pattern: Basic Exchanger

**Use case**: Transform data in a Flight server

```python
import pandas as pd
from xorq.flight.exchanger import make_udxf
import xorq.expr.datatypes as dt
import xorq.api as xo

# Define schemas
schema_in = xo.schema({
    "text": dt.string,
    "metadata": dt.string
})

schema_out = xo.schema({
    "text": dt.string,
    "result": dt.float64,
    "processed_at": dt.timestamp
})

# Define transformation function
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Transform input data"""
    import datetime

    results = model.predict(df["text"])

    return pd.DataFrame({
        "text": df["text"],
        "result": results,
        "processed_at": [datetime.datetime.now()] * len(df)
    })

# Create exchanger
exchanger = make_udxf(
    process_data,
    schema_in,
    schema_out,
    name="text_processor"
)
```

**Key components:**
- `schema_in`: Expected input schema
- `schema_out`: Output schema
- `name`: Exchanger identifier
- Function must accept and return pandas DataFrame

---

### Pattern: Stateful Exchangers

**Use case**: Exchangers with loaded models or state

```python
class ModelExchanger:
    """Stateful exchanger with loaded model"""

    def __init__(self, model_path: str):
        import torch
        self.model = torch.load(model_path)
        self.model.eval()

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process batch"""
        features = df[["feature1", "feature2"]].values
        predictions = self.model.predict(features)

        return pd.DataFrame({
            "prediction": predictions,
            "model_version": ["v1.0"] * len(df)
        })

# Create exchanger with state
model_exchanger = ModelExchanger("model.pth")

exchanger = make_udxf(
    model_exchanger,
    schema_in=xo.schema({
        "feature1": dt.float64,
        "feature2": dt.float64
    }),
    schema_out=xo.schema({
        "prediction": dt.float64,
        "model_version": dt.string
    }),
    name="model_predictor"
)
```

---

## Arrow Flight Servers

### Pattern: Basic Flight Server

**Use case**: Serve exchangers over network

```python
from xorq.flight import FlightServer

# Create exchangers
exchanger1 = make_udxf(
    process_func1,
    schema_in1,
    schema_out1,
    name="processor1"
)

exchanger2 = make_udxf(
    process_func2,
    schema_in2,
    schema_out2,
    name="processor2"
)

# Create and start server
server = FlightServer(
    exchangers=[exchanger1, exchanger2],
    host="0.0.0.0",
    port=8815
)

# Serve (blocking)
server.serve()
```

**Server features:**
- Multiple exchangers per server
- Arrow Flight protocol for efficient data transfer
- Automatic serialization/deserialization
- gRPC-based communication

---

### Pattern: Flight Client

**Use case**: Call exchangers from client

```python
from xorq.flight import FlightClient
import pyarrow as pa

# Connect to server
client = FlightClient(host="localhost", port=8815)

# Prepare input data
input_table = pa.table({
    "text": ["hello world", "test message"],
    "metadata": ["meta1", "meta2"]
})

# Call exchanger
result = client.do_exchange(
    exchanger.command,  # Command from exchanger
    input_table
)

# Result is PyArrow Table
print(result.to_pandas())
```

---

### Pattern: Exchanger in Expression Pipeline

**Use case**: Use exchangers in xorq expressions

```python
# Create exchanger
exchanger = make_udxf(
    transform_func,
    schema_in,
    schema_out,
    name="transformer"
)

# Serve in background or different process
server = FlightServer(exchangers=[exchanger])
# server.serve() in background thread

# Use in expression via client
client = server.client

def apply_exchanger(expr):
    """Apply exchanger to expression"""
    # Execute expression to get data
    data = expr.execute()

    # Convert to PyArrow
    table = pa.Table.from_pandas(data)

    # Call exchanger
    result_table = client.do_exchange(exchanger.command, table)

    # Convert back to expression
    result_df = result_table.to_pandas()
    return con.create_table("result", result_df)

# Use in pipeline
transformed = expr.pipe(apply_exchanger)
```

---

## Complete Flight Server Example

### Pattern: Production Flight Server

```python
import pandas as pd
import pyarrow as pa
from xorq.flight import FlightServer
from xorq.flight.exchanger import make_udxf
import xorq.api as xo
import xorq.expr.datatypes as dt
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
class TextProcessor:
    def __init__(self, model_path: str):
        logger.info(f"Loading model from {model_path}")
        self.model = load_model(model_path)

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Processing {len(df)} rows")

        texts = df["text"].tolist()
        scores = self.model.predict(texts)

        result = pd.DataFrame({
            "text": texts,
            "score": scores,
            "confidence": self.model.predict_proba(texts).max(axis=1)
        })

        logger.info(f"Processed {len(result)} rows")
        return result

# Create processor
processor = TextProcessor("models/sentiment.pkl")

# Create exchanger
sentiment_exchanger = make_udxf(
    processor,
    schema_in=xo.schema({"text": dt.string}),
    schema_out=xo.schema({
        "text": dt.string,
        "score": dt.float64,
        "confidence": dt.float64
    }),
    name="sentiment_processor"
)

# Create server with multiple exchangers
server = FlightServer(
    exchangers=[sentiment_exchanger],
    host="0.0.0.0",
    port=8815
)

logger.info("Starting Flight server on port 8815")
server.serve()
```

**Client usage:**

```python
from xorq.flight import FlightClient
import pyarrow as pa

# Connect
client = FlightClient(host="localhost", port=8815)

# Prepare data
data = pa.table({
    "text": ["This is great!", "This is terrible", "Neutral statement"]
})

# Call service
result = client.do_exchange(
    sentiment_exchanger.command,
    data
)

# Display results
print(result.to_pandas())
```

---

## Testing UDFs and Exchangers

### Pattern: Test UDFs Locally

```python
# Create test data
test_data = xo.memtable(
    {"text": ["test1", "test2", "test3"]},
    schema=xo.schema({"text": str})
)

# Apply UDF
@xo.udf.make_pandas_udf(
    schema=xo.schema({"text": str}),
    return_type=dt.int64,
    name="text_length"
)
def text_len(df):
    return df["text"].str.len()

# Test
result = test_data.mutate(length=text_len.on_expr).execute()
assert result["length"].tolist() == [5, 5, 5]
```

### Pattern: Test Exchangers

```python
# Test exchanger function directly
test_df = pd.DataFrame({
    "text": ["hello", "world"]
})

result_df = exchanger.func(test_df)

# Verify output schema matches
assert set(result_df.columns) == set(schema_out.names)

# Verify types
assert result_df["result"].dtype == float
```

---

## Best Practices

### 1. Specify Types Explicitly

```python
# Good: explicit types
@xo.udf.make_pandas_udf(
    schema=xo.schema({"col": str}),
    return_type=dt.float64,  # Explicit
    name="my_udf"
)
def my_func(df):
    return df["col"].apply(process).astype(float)

# Avoid: implicit types
@xo.udf.make_pandas_udf(schema=schema, return_type=dt.float64, name="udf")
def my_func(df):
    return df["col"].apply(process)  # May not be float
```

### 2. Test Before Serving

```python
# Good: test locally first
test_data = xo.memtable({"text": ["test"]}, schema=schema_in)
result = exchanger.func(test_data.execute())
assert "result" in result.columns

# Then serve
server = FlightServer(exchangers=[exchanger])
server.serve()

# Avoid: serve without testing
server = FlightServer(exchangers=[exchanger])
server.serve()  # May fail on first request
```

### 3. Use Stateful Classes for Complex UDFs

```python
# Good: stateful class
class ModelUDF:
    def __init__(self):
        self.model = load_model()

    def __call__(self, df):
        return self.model.predict(df)

udf = ModelUDF()

# Avoid: loading in function
def udf(df):
    model = load_model()  # Loads every call!
    return model.predict(df)
```

### 4. Add Logging to Exchangers

```python
# Good: log processing
def process(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Processing {len(df)} rows")
    result = transform(df)
    logger.info(f"Produced {len(result)} rows")
    return result

# Avoid: silent processing (hard to debug)
```

### 5. Handle Errors Gracefully

```python
# Good: error handling
def process(df: pd.DataFrame) -> pd.DataFrame:
    try:
        result = transform(df)
        return result
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        # Return empty result with correct schema
        return pd.DataFrame(columns=schema_out.names)

# Avoid: unhandled errors (crashes server)
```

---

## Troubleshooting

### Issue: UDF Type Mismatch

**Symptom**: `TypeError` or type conversion errors

**Solution:**
```python
# Ensure return type matches actual return
@xo.udf.make_pandas_udf(
    schema=schema,
    return_type=dt.float64,  # Must match
    name="udf"
)
def my_udf(df):
    result = process(df)
    return result.astype(float)  # Ensure float
```

### Issue: Exchanger Schema Mismatch

**Symptom**: Schema validation errors

**Check:**
```python
# Verify function returns correct columns
result = exchanger.func(test_df)
assert set(result.columns) == set(schema_out.names)

# Verify types
for col, dtype in schema_out.items():
    assert result[col].dtype == dtype.to_pandas()
```

### Issue: Flight Server Connection Error

**Symptom**: Cannot connect to server

**Check:**
1. Is server running?
2. Correct host/port?
3. Firewall rules?

**Solution:**
```python
# Server
server = FlightServer(
    exchangers=[exchanger],
    host="0.0.0.0",  # Listen on all interfaces
    port=8815
)

# Client
client = FlightClient(
    host="localhost",  # or server IP
    port=8815
)
```

### Issue: Performance Issues

**Symptom**: Slow UDF/exchanger execution

**Solutions:**
1. Batch processing instead of row-by-row
2. Load models once (stateful class)
3. Use vectorized operations
4. Profile function to find bottlenecks

```python
# Good: vectorized
def process(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(result=model.predict(df[features]))

# Avoid: row-by-row
def process(df: pd.DataFrame) -> pd.DataFrame:
    results = []
    for idx, row in df.iterrows():  # Slow!
        results.append(model.predict([row[features]]))
    return pd.DataFrame({"result": results})
```

---

## Summary

**UDF patterns:**
- `make_pandas_udf` - Pandas-based transformations
- `make_scalar_udf` - Scalar functions
- Specify schemas and return types explicitly
- Use stateful classes for models

**UDXF patterns:**
- `make_udxf` - Create exchangers
- Input/output schemas required
- Functions accept/return pandas DataFrames
- Composable with expressions

**Flight server patterns:**
- Serve multiple exchangers
- Arrow Flight protocol for efficiency
- Client/server architecture
- gRPC-based communication

**Best practices:**
- Test locally before serving
- Use stateful classes for heavy resources
- Add logging for debugging
- Handle errors gracefully
- Vectorize operations for performance
