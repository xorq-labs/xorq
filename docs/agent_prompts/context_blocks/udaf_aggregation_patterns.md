CUSTOM AGGREGATIONS WITH UDAFs (agg.pandas_df):

⚠️ UDAFs only work on LOCAL backend! Cache remote data first:
```python
from xorq.caching import ParquetCache
local_data = remote_table.cache(ParquetCache.from_kwargs())  # PREFERRED
# OR simple cache: local_data = remote_table.cache()
# OR into_backend: local_data = remote_table.into_backend(xo.connect())
```

When you need aggregations not in ibis, use UDAFs:

```python
from xorq.expr.udf import agg
import xorq.expr.datatypes as dt
import pandas as pd
import pickle

# Pattern 1: Statistical aggregations returning structs
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
```

# Pattern 2: Model training as aggregation
def train_segment_model(df):
    '''Train a model per group'''
    model = RandomForestRegressor()
    model.fit(df[features], df['target'])
    return pickle.dumps(model)

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
```

KEY: UDAFs reduce groups → single values (stats, models, complex aggregations)
while keeping the pipeline deferred!

KEY POINT: UDAFs let you use ANY pandas aggregation logic within the deferred execution model!
