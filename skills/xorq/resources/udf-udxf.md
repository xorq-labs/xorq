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

## User Defined Aggregate Functions (UDAFs)

### Overview

UDAFs enable custom aggregations using pandas logic within deferred execution. They reduce groups to single values (stats, models, complex aggregations) while keeping the pipeline deferred.

**⚠️ IMPORTANT: UDAFs only work on LOCAL backend!**

Cache remote data first before using UDAFs:

```python
from xorq.caching import ParquetCache

# PREFERRED: Use ParquetCache
local_data = remote_table.cache(ParquetCache.from_kwargs())

# OR simple cache
local_data = remote_table.cache()

# OR into_backend
local_data = remote_table.into_backend(xo.connect())
```

---

### Pattern: Statistical Aggregations Returning Structs

**Use case**: Compute multiple advanced statistics at once per group

```python
from xorq.expr.udf import agg
import xorq.expr.datatypes as dt
import pandas as pd

def calculate_advanced_stats(df):
    '''Compute multiple statistics at once'''
    return {
        'iqr': df['value'].quantile(0.75) - df['value'].quantile(0.25),
        'trimmed_mean': df['value'].iloc[
            int(len(df)*0.1):int(len(df)*0.9)
        ].mean(),
        'mode': df['value'].mode().iloc[0] if not df['value'].mode().empty else None,
        'cv': df['value'].std() / df['value'].mean(),  # Coefficient of variation
        'correlation': df['value'].corr(df['other_value'])
    }

# Create UDAF
schema = table.select(['value', 'other_value']).schema()
stats_udf = agg.pandas_df(
    fn=calculate_advanced_stats,
    schema=schema,
    return_type=dt.Struct({
        'iqr': dt.float64,
        'trimmed_mean': dt.float64,
        'mode': dt.float64,
        'cv': dt.float64,
        'correlation': dt.float64
    }),
    name="advanced_stats"
)

# Apply to groups
results = table.group_by(_.category).agg(
    stats=stats_udf.on_expr(table)
)

# Unpack struct to separate columns
results_unpacked = results.unpack('stats')
```

**Key points:**
- Function returns a dict matching the struct schema
- `schema` parameter defines input columns needed
- `return_type` must be a `Struct` for multiple outputs
- Use `.unpack()` to expand struct into columns

---

### Pattern: Model Training as Aggregation

**Use case**: Train one model per segment/group

```python
from xorq.expr.udf import agg
import xorq.expr.datatypes as dt
import pickle
from sklearn.ensemble import RandomForestRegressor

def train_segment_model(df):
    '''Train a model per group'''
    features = ['feature1', 'feature2', 'feature3']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(df[features], df['target'])
    return pickle.dumps(model)

# Define UDAF
model_udf = agg.pandas_df(
    fn=train_segment_model,
    schema=table.select(features + ['target']).schema(),
    return_type=dt.binary,
    name="train_model"
)

# Train one model per segment
models = table.group_by("segment").agg(
    model=model_udf.on_expr(table)
)

# Use models - unpickle and predict
@xo.udf.make_pandas_udf(
    schema=xo.schema({"model": dt.binary}),
    return_type=dt.float64,
    name="predict_with_model"
)
def predict_with_model(df):
    model = pickle.loads(df["model"].iloc[0])
    return pd.Series([model.predict(new_data)])

# Join models back to data and predict
predictions = (
    new_data
    .join(models, "segment")
    .mutate(prediction=predict_with_model.on_expr)
)
```

**Key points:**
- Use `pickle.dumps()` to serialize models to binary
- `return_type=dt.binary` for pickled objects
- Train once per group, reuse models for prediction
- Store models in a table for later use

---

### Pattern: Custom Window Aggregations

**Use case**: Complex rolling or cumulative aggregations not in ibis

```python
def rolling_quantile_range(df):
    '''Calculate rolling IQR (interquartile range)'''
    sorted_vals = df['value'].sort_values()
    q75 = sorted_vals.quantile(0.75)
    q25 = sorted_vals.quantile(0.25)
    return q75 - q25

quantile_range_udf = agg.pandas_df(
    fn=rolling_quantile_range,
    schema=table.select(['value']).schema(),
    return_type=dt.float64,
    name="quantile_range"
)

# Apply per time window
windowed_stats = (
    table
    .mutate(time_bucket=_.timestamp.truncate('1h'))
    .group_by(['category', 'time_bucket'])
    .agg(iqr=quantile_range_udf.on_expr(table))
)
```

---

### Pattern: Text Aggregations

**Use case**: Concatenate, summarize, or analyze text per group

```python
def aggregate_text(df):
    '''Combine text fields with stats'''
    texts = df['comment'].tolist()
    return {
        'combined_text': ' '.join(texts),
        'avg_length': df['comment'].str.len().mean(),
        'unique_words': len(set(' '.join(texts).split())),
        'sentiment_avg': analyze_sentiment(texts).mean()
    }

text_agg_udf = agg.pandas_df(
    fn=aggregate_text,
    schema=table.select(['comment']).schema(),
    return_type=dt.Struct({
        'combined_text': dt.string,
        'avg_length': dt.float64,
        'unique_words': dt.int64,
        'sentiment_avg': dt.float64
    }),
    name="text_aggregation"
)

# Aggregate comments per product
product_reviews = (
    reviews_table
    .group_by('product_id')
    .agg(text_stats=text_agg_udf.on_expr(reviews_table))
    .unpack('text_stats')
)
```

---

### Pattern: Custom Percentile Calculations

**Use case**: Weighted percentiles, trimmed stats, or custom distributions

```python
def weighted_percentiles(df):
    '''Calculate weighted percentiles'''
    # Sort by value
    sorted_df = df.sort_values('value')
    cumsum = sorted_df['weight'].cumsum()
    total_weight = sorted_df['weight'].sum()

    # Find weighted percentiles
    p50_idx = (cumsum >= total_weight * 0.5).idxmax()
    p90_idx = (cumsum >= total_weight * 0.9).idxmax()

    return {
        'p50': sorted_df.loc[p50_idx, 'value'],
        'p90': sorted_df.loc[p90_idx, 'value']
    }

weighted_pct_udf = agg.pandas_df(
    fn=weighted_percentiles,
    schema=table.select(['value', 'weight']).schema(),
    return_type=dt.Struct({
        'p50': dt.float64,
        'p90': dt.float64
    }),
    name="weighted_percentiles"
)

# Apply to groups
weighted_stats = (
    table
    .group_by('category')
    .agg(percentiles=weighted_pct_udf.on_expr(table))
    .unpack('percentiles')
)
```

---

### Best Practices for UDAFs

#### 1. Always Cache Remote Data First

```python
# Good: Cache first
local_data = remote_table.cache(ParquetCache.from_kwargs())
results = local_data.group_by('category').agg(
    custom_stat=my_udaf.on_expr(local_data)
)

# Bad: UDAF on remote data (will fail!)
results = remote_table.group_by('category').agg(
    custom_stat=my_udaf.on_expr(remote_table)
)
```

#### 2. Handle Empty Groups Gracefully

```python
def safe_aggregation(df):
    '''Handle empty groups'''
    if len(df) == 0:
        return {'result': None, 'count': 0}

    return {
        'result': df['value'].mean(),
        'count': len(df)
    }
```

#### 3. Vectorize Operations

```python
# Good: Vectorized pandas operations
def fast_agg(df):
    return df['value'].quantile([0.25, 0.75]).diff().iloc[-1]

# Avoid: Row-by-row iteration
def slow_agg(df):
    results = []
    for idx, row in df.iterrows():  # Slow!
        results.append(process(row))
    return sum(results)
```

#### 4. Use Appropriate Return Types

```python
# Single value: Use scalar type
return_type=dt.float64

# Multiple values: Use Struct
return_type=dt.Struct({
    'mean': dt.float64,
    'std': dt.float64
})

# Serialized objects: Use binary
return_type=dt.binary  # For pickled models, etc.
```

#### 5. Test UDAF Functions Independently

```python
# Test aggregation function with sample data
test_df = pd.DataFrame({
    'value': [1, 2, 3, 4, 5],
    'other_value': [5, 4, 3, 2, 1]
})

result = calculate_advanced_stats(test_df)
assert 'iqr' in result
assert isinstance(result['iqr'], float)
```

---

### Troubleshooting UDAFs

#### Issue: "UDAFs not supported on this backend"

**Solution:** Cache to local backend first
```python
local_data = remote_table.cache(ParquetCache.from_kwargs())
results = local_data.group_by('col').agg(udaf.on_expr(local_data))
```

#### Issue: Schema mismatch errors

**Solution:** Ensure input schema matches table schema
```python
# Verify schema includes all needed columns
schema = table.select(['col1', 'col2']).schema()
udaf = agg.pandas_df(fn=my_func, schema=schema, return_type=dt.float64)
```

#### Issue: Return type mismatch

**Solution:** Match return type to actual output
```python
# If returning dict, use Struct
return_type=dt.Struct({'key': dt.float64})

# If returning single value, use scalar type
return_type=dt.float64
```

---

### Summary

**UDAF key points:**
- Use `agg.pandas_df()` for custom aggregations
- **Only works on local backend** - cache remote data first
- Return single values (scalars, structs, binary for models)
- Keeps pipeline deferred - executes only when needed
- Perfect for: advanced stats, model training per group, custom metrics

**When to use UDAFs:**
- Aggregation not available in ibis
- Need pandas-specific logic (quantiles, complex rolling stats)
- Training models per group/segment
- Custom statistical measures

**When NOT to use UDAFs:**
- Simple aggregations (use ibis built-ins: `.mean()`, `.sum()`, etc.)
- Element-wise transformations (use regular UDFs instead)
- When working directly on remote backends without caching

---

## Storing Plots and Visualizations in UDAFs

### Overview

UDAFs can generate and store visualizations as binary data alongside statistics. This enables creating plots per group/segment while maintaining deferred execution. Plots are stored as PNG bytes using `dt.binary` return type.

**Key principle:** Generate plots in memory, serialize to bytes, return as binary data.

---

### Pattern: Basic Plot Storage

**Use case**: Generate distribution plots per category

```python
import xorq.api as xo
import xorq.expr.datatypes as dt
from xorq.expr.udf import agg
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt

def create_distribution_plot(df):
    """Generate distribution plot and return as binary"""
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create plot (histogram example)
    ax.hist(df['value'], bins=30, edgecolor='black')
    ax.set_title(f'Distribution for {df["category"].iloc[0]}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

    # Save to bytes buffer
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plot_bytes = buf.read()

    # Close figure to free memory (IMPORTANT!)
    plt.close(fig)

    return plot_bytes

# Create UDAF for plot generation
plot_udf = agg.pandas_df(
    fn=create_distribution_plot,
    schema=table.select(['value', 'category']).schema(),
    return_type=dt.binary,  # Binary type for images
    name="distribution_plot"
)

# Generate plots per category
plots = table.group_by('category').agg(
    plot=plot_udf.on_expr(table)
)

# Retrieve and display
result = plots.execute()
plot_bytes = result.iloc[0]['plot']

# Display in Jupyter
from IPython.display import Image, display
display(Image(plot_bytes))

# Or save to file
with open('category_plot.png', 'wb') as f:
    f.write(plot_bytes)
```

**Key points:**
- Use `BytesIO` buffer to capture plot in memory
- Always `plt.close(fig)` to prevent memory leaks
- Return type is `dt.binary` for image data
- Can display with IPython or save to file

---

### Pattern: Multi-Output with Plots and Statistics

**Use case**: Generate visualizations alongside statistical summaries

```python
def create_analysis_with_plot(df):
    """Generate both statistics and visualization"""
    # Calculate statistics
    stats = {
        'mean': df['value'].mean(),
        'std': df['value'].std(),
        'count': len(df),
        'q25': df['value'].quantile(0.25),
        'q75': df['value'].quantile(0.75)
    }

    # Create comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Histogram
    axes[0, 0].hist(df['value'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(stats['mean'], color='red', linestyle='--', label=f"Mean: {stats['mean']:.2f}")
    axes[0, 0].set_title('Distribution')
    axes[0, 0].legend()

    # Box plot
    axes[0, 1].boxplot(df['value'])
    axes[0, 1].set_title('Box Plot')
    axes[0, 1].set_ylabel('Value')

    # Q-Q plot for normality check
    from scipy import stats as scipy_stats
    scipy_stats.probplot(df['value'], dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot')

    # Cumulative distribution
    sorted_vals = np.sort(df['value'])
    axes[1, 1].plot(sorted_vals, np.linspace(0, 1, len(sorted_vals)))
    axes[1, 1].set_title('Cumulative Distribution')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot to bytes
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plot_bytes = buf.read()
    plt.close(fig)

    # Return structured output with stats and plot
    return {
        'mean': stats['mean'],
        'std': stats['std'],
        'count': stats['count'],
        'q25': stats['q25'],
        'q75': stats['q75'],
        'plot': plot_bytes
    }

# Create UDAF with structured return
analysis_udf = agg.pandas_df(
    fn=create_analysis_with_plot,
    schema=table.select(['value']).schema(),
    return_type=dt.Struct({
        'mean': dt.float64,
        'std': dt.float64,
        'count': dt.int64,
        'q25': dt.float64,
        'q75': dt.float64,
        'plot': dt.binary  # Binary for plot image
    }),
    name="analysis_with_plot"
)

# Apply and unpack results
results = (
    table
    .group_by('category')
    .agg(analysis=analysis_udf.on_expr(table))
    .unpack('analysis')  # Expands struct to separate columns
)
```

---

### Pattern: Model Training with Performance Plots

**Use case**: Train models and create performance visualizations per segment

```python
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def train_and_visualize(df):
    """Train model and create performance plots"""
    # Prepare data
    features = ['feature1', 'feature2', 'feature3']
    X = df[features]
    y = df['target']

    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Create performance visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Actual vs Predicted
    axes[0, 0].scatter(y_test, y_pred, alpha=0.5)
    axes[0, 0].plot([y_test.min(), y_test.max()],
                    [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual')
    axes[0, 0].set_ylabel('Predicted')
    axes[0, 0].set_title(f'Actual vs Predicted (R²={r2:.3f})')

    # Residuals histogram
    residuals = y_test - y_pred
    axes[0, 1].hist(residuals, bins=30, edgecolor='black')
    axes[0, 1].axvline(0, color='red', linestyle='--')
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_title(f'Residual Distribution (RMSE={rmse:.3f})')

    # Feature importance
    importances = model.feature_importances_
    axes[1, 0].bar(features, importances)
    axes[1, 0].set_xlabel('Features')
    axes[1, 0].set_ylabel('Importance')
    axes[1, 0].set_title('Feature Importance')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Residuals vs Predicted
    axes[1, 1].scatter(y_pred, residuals, alpha=0.5)
    axes[1, 1].axhline(0, color='red', linestyle='--')
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Residuals vs Predicted')

    plt.tight_layout()

    # Save plot
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plot_bytes = buf.read()
    plt.close(fig)

    # Return model, visualization, and metrics
    return {
        'model': pickle.dumps(model),
        'performance_plot': plot_bytes,
        'r2_score': r2,
        'rmse': rmse,
        'feature_importances': importances.tolist()
    }

# Create UDAF
model_viz_udf = agg.pandas_df(
    fn=train_and_visualize,
    schema=table.select(features + ['target']).schema(),
    return_type=dt.Struct({
        'model': dt.binary,
        'performance_plot': dt.binary,
        'r2_score': dt.float64,
        'rmse': dt.float64,
        'feature_importances': dt.array(dt.float64)
    }),
    name="model_with_viz"
)

# Train models with visualizations per segment
segment_models = (
    table
    .group_by('segment')
    .agg(results=model_viz_udf.on_expr(table))
    .unpack('results')
)
```

---

### Pattern: Time Series Visualization

**Use case**: Create time series plots with trends and seasonality

```python
def create_timeseries_plot(df):
    """Create comprehensive time series visualization"""
    # Sort by date
    df = df.sort_values('date')

    # Calculate rolling statistics
    df['ma_7'] = df['value'].rolling(7, center=True).mean()
    df['ma_30'] = df['value'].rolling(30, center=True).mean()
    df['std_7'] = df['value'].rolling(7, center=True).std()

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Main time series with moving averages
    axes[0].plot(df['date'], df['value'], alpha=0.5, label='Raw', linewidth=1)
    axes[0].plot(df['date'], df['ma_7'], label='7-day MA', linewidth=2)
    axes[0].plot(df['date'], df['ma_30'], label='30-day MA', linewidth=2)
    axes[0].set_ylabel('Value')
    axes[0].set_title(f'Time Series for {df["series_id"].iloc[0]}')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    # Volatility band
    axes[1].plot(df['date'], df['value'], alpha=0.3, color='blue')
    axes[1].fill_between(df['date'],
                         df['ma_7'] - 2*df['std_7'],
                         df['ma_7'] + 2*df['std_7'],
                         alpha=0.2, color='blue', label='±2 std')
    axes[1].plot(df['date'], df['ma_7'], color='red', label='7-day MA')
    axes[1].set_ylabel('Value with Volatility')
    axes[1].set_title('Volatility Bands')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    # Volume or secondary metric
    if 'volume' in df.columns:
        axes[2].bar(df['date'], df['volume'], alpha=0.7, width=0.8)
        axes[2].set_ylabel('Volume')
        axes[2].set_xlabel('Date')
        axes[2].set_title('Trading Volume')
        axes[2].grid(True, alpha=0.3)
    else:
        # Show daily changes
        df['daily_change'] = df['value'].pct_change() * 100
        axes[2].bar(df['date'], df['daily_change'],
                   color=['green' if x >= 0 else 'red' for x in df['daily_change']],
                   alpha=0.7)
        axes[2].set_ylabel('Daily Change (%)')
        axes[2].set_xlabel('Date')
        axes[2].set_title('Daily Percentage Change')
        axes[2].axhline(0, color='black', linewidth=0.5)
        axes[2].grid(True, alpha=0.3)

    # Format x-axis dates
    fig.autofmt_xdate()
    plt.tight_layout()

    # Save to bytes
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plot_bytes = buf.read()
    plt.close(fig)

    # Calculate summary stats
    return {
        'plot': plot_bytes,
        'start_date': df['date'].min().isoformat(),
        'end_date': df['date'].max().isoformat(),
        'total_points': len(df),
        'mean_value': df['value'].mean(),
        'volatility': df['value'].std()
    }

# Create UDAF for time series plots
ts_plot_udf = agg.pandas_df(
    fn=create_timeseries_plot,
    schema=table.select(['date', 'value', 'series_id']).schema(),
    return_type=dt.Struct({
        'plot': dt.binary,
        'start_date': dt.string,
        'end_date': dt.string,
        'total_points': dt.int64,
        'mean_value': dt.float64,
        'volatility': dt.float64
    }),
    name="timeseries_plot"
)

# Generate plots per time series
ts_results = (
    table
    .group_by('series_id')
    .agg(analysis=ts_plot_udf.on_expr(table))
    .unpack('analysis')
)
```

---

### Pattern: Correlation Matrix Heatmaps

**Use case**: Generate correlation heatmaps per group

```python
import seaborn as sns

def create_correlation_heatmap(df):
    """Generate correlation matrix heatmap"""
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": 0.8},
                ax=ax)
    ax.set_title(f'Correlation Matrix for {df["group_id"].iloc[0]}')

    plt.tight_layout()

    # Save to bytes
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plot_bytes = buf.read()
    plt.close(fig)

    return plot_bytes

# Create UDAF
corr_plot_udf = agg.pandas_df(
    fn=create_correlation_heatmap,
    schema=table.schema(),  # Use full schema
    return_type=dt.binary,
    name="correlation_heatmap"
)

# Generate heatmaps per group
heatmaps = table.group_by('group_id').agg(
    heatmap=corr_plot_udf.on_expr(table)
)
```

---

### Best Practices for Plot-Storing UDAFs

#### 1. Memory Management

```python
# Always close figures after saving
plt.close(fig)  # Critical to prevent memory leaks

# For many plots, consider clearing matplotlib cache
import matplotlib
matplotlib.pyplot.clf()
matplotlib.pyplot.cla()
matplotlib.pyplot.close('all')
```

#### 2. Plot Quality Settings

```python
# High quality for reports
fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')

# Lower quality for previews (smaller file size)
fig.savefig(buf, format='png', dpi=72)

# Use compression for large plots
fig.savefig(buf, format='png', dpi=100, optimize=True)
```

#### 3. Error Handling in Plot Generation

```python
def safe_plot_generation(df):
    """Handle errors gracefully"""
    try:
        if len(df) < 2:
            # Return empty plot or placeholder
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Insufficient data',
                   ha='center', va='center', fontsize=16)
            ax.set_title(f'No data for {df["category"].iloc[0]}')
        else:
            # Normal plot generation
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df['value'], bins=30)
            ax.set_title(f'Distribution for {df["category"].iloc[0]}')

        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plot_bytes = buf.read()
        plt.close(fig)
        return plot_bytes

    except Exception as e:
        # Return error placeholder
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f'Error: {str(e)}',
               ha='center', va='center', fontsize=12, color='red')
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plot_bytes = buf.read()
        plt.close(fig)
        return plot_bytes
```

#### 4. Retrieving and Using Stored Plots

```python
# Execute and get results
results = plots.execute()

# Save all plots to files
for idx, row in results.iterrows():
    category = row['category']
    plot_bytes = row['plot']

    # Save to file
    filename = f'plot_{category}.png'
    with open(filename, 'wb') as f:
        f.write(plot_bytes)

    print(f'Saved plot for {category} to {filename}')

# Display in Jupyter notebook
from IPython.display import Image, display
for idx, row in results.iterrows():
    print(f"Category: {row['category']}")
    display(Image(row['plot']))

# Convert to base64 for web display
import base64
plot_b64 = base64.b64encode(row['plot']).decode('utf-8')
html = f'<img src="data:image/png;base64,{plot_b64}"/>'
```

#### 5. Batch Processing Considerations

```python
def batch_plot_processor(df):
    """Process large groups efficiently"""
    # Limit plot complexity for large datasets
    sample_size = min(len(df), 10000)
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df

    # Use matplotlib's non-interactive backend
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend

    # Generate plot with sample
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df_sample['value'], bins=50)
    ax.set_title(f'Distribution (n={len(df)}, shown={len(df_sample)})')

    # Save with optimization
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, optimize=True)
    buf.seek(0)
    plot_bytes = buf.read()
    plt.close(fig)

    return plot_bytes
```

---

### Summary

**Key points for plot-storing UDAFs:**
- Use `dt.binary` return type for storing images
- Always use `BytesIO` buffer for in-memory generation
- **Always** close figures with `plt.close(fig)` to prevent memory leaks
- Cache remote data first - UDAFs only work on local backend
- Combine plots with statistics using `dt.Struct` return type
- Handle errors gracefully with placeholder images
- Consider memory and performance for large datasets

**Common use cases:**
- Distribution analysis per group
- Model performance visualization per segment
- Time series plots per entity
- Correlation heatmaps per category
- Quality control charts per batch

This approach keeps visualizations within the deferred xorq pipeline, enabling scalable analytics with integrated plotting.

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
