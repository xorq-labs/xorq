# Optimization Patterns in Xorq

This guide covers common optimization patterns using xorq's deferred expression system, including portfolio optimization, knapsack problems, and multi-criteria selection.

## Table of Contents
1. [Portfolio Optimization Pattern](#portfolio-optimization-pattern)
2. [Knapsack Problem Implementation](#knapsack-problem-implementation)
3. [Multi-Criteria Selection](#multi-criteria-selection)
4. [Performance Optimization Tips](#performance-optimization-tips)

## Portfolio Optimization Pattern

Portfolio optimization involves selecting a subset of items to maximize value while respecting constraints (budget, diversity, quality).

### Key Components

1. **Quality Filtering** - Remove items that don't meet minimum standards
2. **Efficiency Scoring** - Calculate value-per-cost metrics
3. **Diversity Constraints** - Ensure representation across categories
4. **Budget Constraints** - Stay within spending limits

### Basic Implementation

```python
import xorq.api as xo
from xorq.vendor import ibis
from xorq.caching import ParquetCache
from xorq.common.utils.ibis_utils import from_ibis

# 1. Filter by quality constraints
quality_items = source.filter([
    source.quality.isin(ACCEPTABLE_GRADES),
    source.price <= MAX_INDIVIDUAL_PRICE,
    source.price > 0
])

# 2. Add efficiency metrics
enriched = quality_items.mutate(
    efficiency=quality_items.value / quality_items.price,
    quality_score=ibis.cases(
        (quality_items.grade == "Premium", 5),
        (quality_items.grade == "Standard", 3),
        (quality_items.grade == "Basic", 1),
        else_=0
    )
)

# 3. Sort and select top candidates
candidates = enriched.order_by([
    enriched.efficiency.desc(),
    enriched.quality_score.desc()
]).limit(1000)

# 4. Cache for performance
cached = from_ibis(candidates).cache(ParquetCache.from_kwargs())
```

## Knapsack Problem Implementation

The knapsack problem seeks to maximize value within a weight/cost constraint. Since pure SQL can't implement dynamic programming efficiently, we use a greedy approximation or UDFs for exact solutions.

### Greedy Approximation (Pure Deferred)

```python
def greedy_knapsack(items_expr, budget):
    """
    Greedy knapsack using deferred expressions.
    Not optimal but works purely in SQL.
    """

    # Sort by efficiency (value/weight ratio)
    sorted_items = items_expr.order_by(
        (items_expr.value / items_expr.weight).desc()
    )

    # Add cumulative weight using window function
    with_cumsum = sorted_items.mutate(
        cumulative_weight=sorted_items.weight.sum().over(
            ibis.window(
                order_by=[(sorted_items.value / sorted_items.weight).desc()],
                preceding=0,
                following=0
            )
        )
    )

    # Select items that fit within capacity
    selected = with_cumsum.filter(
        with_cumsum.cumulative_weight <= budget
    )

    return selected
```

### Exact Solution with UDF

For exact knapsack solutions, use a UDF with dynamic programming:

```python
from xorq.expr.udf import agg
import xorq.expr.datatypes as dt

def create_knapsack_udf(capacity):
    """Create a UDAF for exact knapsack solution."""

    def knapsack_optimizer(df):
        import pandas as pd
        import numpy as np

        # Dynamic programming implementation
        n = len(df)
        W = capacity

        # Create DP table
        dp = np.zeros((n + 1, W + 1))

        # Fill DP table
        for i in range(1, n + 1):
            for w in range(1, W + 1):
                weight = df.iloc[i-1]['weight']
                value = df.iloc[i-1]['value']

                if weight <= w:
                    dp[i][w] = max(
                        value + dp[i-1][w-weight],
                        dp[i-1][w]
                    )
                else:
                    dp[i][w] = dp[i-1][w]

        # Backtrack to find selected items
        selected = []
        w = W
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                selected.append(i-1)
                w -= df.iloc[i-1]['weight']

        # Mark selected items
        df['selected'] = False
        df.loc[selected, 'selected'] = True

        return df[df['selected']].to_dict('records')

    return knapsack_optimizer

# Use the UDF in expression
knapsack_udf = agg.pandas_df(
    fn=create_knapsack_udf(BUDGET),
    schema=items.schema(),
    return_type=dt.binary,
    name="knapsack_optimize"
)

optimized = items.aggregate([
    knapsack_udf.on_expr(items).name('selected_items')
])
```

## Multi-Criteria Selection

When optimizing across multiple objectives with diversity requirements:

### Pattern 1: Category Diversity

Ensure at least one item from each required category:

```python
def ensure_category_diversity(expr, categories):
    """Select best item from each category, then optimize remainder."""

    # Phase 1: Mandatory selections
    mandatory = []
    for category in categories:
        best = (
            expr.filter(expr.category == category)
            .order_by(expr.efficiency.desc())
            .limit(1)
            .mutate(
                selection_phase=ibis.literal("mandatory"),
                selection_reason=ibis.literal(f"Best_{category}")
            )
        )
        mandatory.append(best)

    mandatory_selections = ibis.union(*mandatory)

    # Phase 2: Additional optimization
    remaining = expr.anti_join(
        mandatory_selections.select("item_id"),
        predicates=[expr.item_id == mandatory_selections.item_id]
    )

    additional = (
        remaining
        .order_by(remaining.efficiency.desc())
        .limit(50)
        .mutate(
            selection_phase=ibis.literal("optimization"),
            selection_reason=ibis.literal("High_efficiency")
        )
    )

    return ibis.union(mandatory_selections, additional)
```

### Pattern 2: Multi-Objective Scoring

Combine multiple objectives into a composite score:

```python
def multi_objective_score(expr, weights):
    """
    Create composite score from multiple objectives.

    weights = {
        'efficiency': 0.4,
        'quality': 0.3,
        'availability': 0.3
    }
    """

    # Normalize each metric to 0-1 scale
    normalized = expr.mutate(
        efficiency_norm=(
            (expr.efficiency - expr.efficiency.min()) /
            (expr.efficiency.max() - expr.efficiency.min())
        ),
        quality_norm=(
            (expr.quality - expr.quality.min()) /
            (expr.quality.max() - expr.quality.min())
        ),
        availability_norm=(
            (expr.availability - expr.availability.min()) /
            (expr.availability.max() - expr.availability.min())
        )
    )

    # Calculate weighted composite score
    scored = normalized.mutate(
        composite_score=(
            normalized.efficiency_norm * weights['efficiency'] +
            normalized.quality_norm * weights['quality'] +
            normalized.availability_norm * weights['availability']
        )
    )

    return scored.order_by(scored.composite_score.desc())
```

## Performance Optimization Tips

### 1. Strategic Caching

Cache after expensive operations:

```python
# Cache after complex filtering/joining
filtered = complex_expression.filter(many_conditions)
cached_filtered = from_ibis(filtered).cache(ParquetCache.from_kwargs())

# Continue with cached version
final = cached_filtered.mutate(...).select(...)
```

### 2. Limit Early

Reduce data volume before expensive operations:

```python
# Good: Limit before complex operations
top_candidates = (
    source
    .order_by(source.metric.desc())
    .limit(10000)  # Reduce to manageable size
    .mutate(complex_calculation=...)  # Now compute on smaller set
)

# Bad: Complex operations on full dataset
all_calculated = source.mutate(complex_calculation=...)
top = all_calculated.order_by(...).limit(10000)
```

### 3. Use Approximate Methods

For large-scale optimization, approximate methods are often sufficient:

```python
# Approximate top-k using sampling
sample_size = min(100000, source.count().execute())
sampled = source.sample(fraction=sample_size/total_count)

# Optimize on sample, then validate on full dataset
optimized_sample = optimize(sampled)
validation = source.filter(
    source.item_id.isin(optimized_sample.item_id)
)
```

### 4. Batch Processing

For very large portfolios, process in batches:

```python
def batch_optimize(source, batch_size=1000):
    """Process large dataset in batches."""

    batches = []
    for i in range(0, total_items, batch_size):
        batch = (
            source
            .limit(batch_size, offset=i)
            .pipe(optimize_batch)
        )
        batches.append(batch)

    # Combine and re-optimize top candidates
    combined = ibis.union(*batches)
    final = combined.order_by(combined.score.desc()).limit(100)

    return final
```

## Common Pitfalls and Solutions

### Issue: case() is deprecated
```python
# ❌ Wrong
score = ibis.case().when(expr.col == "A", 1).end()

# ✅ Correct
score = ibis.cases(
    (expr.col == "A", 1),
    (expr.col == "B", 2),
    else_=0
)
```

### Issue: Column reference errors
```python
# ❌ Wrong - using _ incorrectly
filtered = expr.filter(_.column > 0)

# ✅ Correct - reference via expr
filtered = expr.filter(expr.column > 0)

# ✅ Also correct - using xo._
filtered = expr.filter(xo._.column > 0)
```

### Issue: Schema mismatch in union
```python
# Ensure all branches have same columns
branch1 = expr1.mutate(extra_col=ibis.literal(None))
branch2 = expr2.mutate(extra_col=expr2.value)

# Now union works
combined = ibis.union(branch1, branch2)
```

## Real-World Example: Diamond Portfolio

See `examples/diamond_optimizer_simple.py` for a complete implementation that:

1. Filters diamonds by clarity and color grades
2. Calculates carat-per-dollar efficiency
3. Ensures at least one diamond from each cut category
4. Optimizes selection within $10,000 budget
5. Provides visualization and analytics

Key learnings from this implementation:
- Use `ibis.cases()` for grade mappings
- Cache candidates before optimization
- Implement diversity via union of category-best selections
- Use anti-join to exclude already-selected items
- Add analytics as final mutation step

## References

- [Expression API Documentation](expression-api.md)
- [ML Pipelines Guide](ml-pipelines.md)
- [Caching Strategies](caching.md)
- [Portfolio Optimization Template](../../examples/portfolio_optimization_template.py)