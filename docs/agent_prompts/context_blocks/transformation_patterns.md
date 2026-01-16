TRANSFORMATION PATTERNS:
- Use ibis/xorq expressions: `.filter()`, `.select()`, `.mutate()`, `.aggregate()`
- Chain small steps instead of one huge expression (easier to debug)
- Always print `table.schema()` before referencing columns
- Prefer `_.COLUMN` syntax and respect backend-specific casing
- For heavy transforms, push computation to the remote backend, then cache locally only if UDFs are required

## CONDITIONAL TRANSFORMATIONS

### Using ibis.case() (Preferred)

```python
from xorq.vendor import ibis

# ✅ CORRECT - Use chained .when() API
categorized = table.mutate(
    category=ibis.case()
        .when(_.value < 10, 'low')
        .when(_.value < 50, 'medium')
        .else_('high')
        .end()
)
```

### Error: SignatureValidationError with ibis.cases()

**Problem:** `ibis.cases()` (tuple-based API) doesn't accept `Deferred` objects (`_['col']`) in conditions.

```python
# ❌ WRONG - Deferred not allowed in tuple conditions
categorized = table.mutate(
    category=ibis.cases(
        (_['value'] < 10, 'low'),    # ❌ SignatureValidationError: not matching
        (_['value'] < 50, 'medium'),
        (ibis.literal(True), 'high')
    )
)
# Error: `value`: (_['value'] < 10) is not matching Any() then anything except a Deferred
```

**Fix Option A - Use ibis.case() chain API:**
```python
# ✅ CORRECT
categorized = table.mutate(
    category=ibis.case()
        .when(_['value'] < 10, 'low')
        .when(_['value'] < 50, 'medium')
        .else_('high')
        .end()
)
```

**Fix Option B - Execute first, classify in pandas:**
```python
# ✅ CORRECT for post-processing
results = table.execute()
results['category'] = results['value'].apply(
    lambda x: 'low' if x < 10 else ('medium' if x < 50 else 'high')
)
```

**Fix Option C - Use column.case() method:**
```python
# ✅ CORRECT - Use column method
categorized = table.mutate(
    category=_.value.case()
        .when(_.value < 10, 'low')
        .when(_.value < 50, 'medium')
        .else_('high')
        .end()
)
```

**Why it fails:** `ibis.cases()` expects concrete Column expressions, but `_['col']` is still a deferred selector at construction time. The chain API (`ibis.case().when()`) handles deferred resolution properly.

## ✅ FEATURE ENGINEERING - HIGHLY ENCOURAGED!

**Mathematical transformations are ENCOURAGED and IMPORTANT for ML!** Use them liberally to improve model performance.

### Mathematical Transformations (All Supported!)

#### Logarithmic Transformations
```python
# ✅ CORRECT: Use .ln() for natural log (NOT .log()!)
features = table.mutate(
    log_price=_.PRICE.ln(),           # Natural log (log base e)
    log_carat=_.CARAT.ln(),
    log1p_x=_.X.ln() + 1,             # log(x + 1) - handle zeros
    # For log base 10, use: _.PRICE.log10()
    # For log base 2, use: _.PRICE.log2()
)

# ❌ WRONG: Don't use .log() in Snowflake/Ibis
# log_price=_.PRICE.log()  # ❌ Doesn't work in Snowflake/Ibis!
```

**When to use log transforms:**
- Skewed distributions (prices, income, counts)
- Large value ranges (e.g., 100 to 1,000,000)
- Multiplicative relationships
- Heteroscedastic data (variance changes with magnitude)

#### Power Transformations
```python
# ✅ Square, cube, and higher powers
features = table.mutate(
    price_squared=_.PRICE ** 2,
    price_cubed=_.PRICE ** 3,
    carat_4th_power=_.CARAT ** 4,
)

# ✅ Square root
features = table.mutate(
    sqrt_price=_.PRICE.sqrt(),
    sqrt_carat=_.CARAT.sqrt(),
)

# ✅ Cube root and other roots
features = table.mutate(
    cube_root=_.VOLUME ** (1/3),
    fourth_root=_.X ** 0.25,
)
```

**When to use power transforms:**
- Square root: Reduce right skew (moderate)
- Square/cube: Create polynomial features
- Fractional powers: Tune transformation strength

#### Reciprocal Transformations
```python
# ✅ Inverse relationships
features = table.mutate(
    inverse_price=1.0 / _.PRICE,
    inverse_carat=1.0 / _.CARAT,
    # Safe version with null handling
    safe_inverse=ibis.case()
        .when(_.PRICE == 0, None)
        .else_(1.0 / _.PRICE)
        .end()
)
```

#### Absolute Value & Sign
```python
features = table.mutate(
    abs_change=_.CHANGE.abs(),
    sign_indicator=_.CHANGE.sign(),  # -1, 0, or 1
)
```

### Polynomial Features (ENCOURAGED!)
```python
# ✅ Create interaction and polynomial features
poly_features = table.mutate(
    # Interactions
    carat_x_depth=_.CARAT * _.DEPTH,
    price_x_carat=_.PRICE * _.CARAT,

    # Polynomials
    carat_squared=_.CARAT ** 2,
    carat_cubed=_.CARAT ** 3,
    depth_squared=_.DEPTH ** 2,

    # Complex interactions
    carat_sq_x_depth=_.CARAT ** 2 * _.DEPTH,
)
```

### Scaling & Normalization (In Expressions!)
```python
# ✅ Standardization (z-score)
standardized = table.mutate(
    price_zscore=(_.PRICE - _.PRICE.mean()) / _.PRICE.std(),
    carat_zscore=(_.CARAT - _.CARAT.mean()) / _.CARAT.std(),
)

# ✅ Min-Max normalization
normalized = table.mutate(
    price_normalized=(_.PRICE - _.PRICE.min()) / (_.PRICE.max() - _.PRICE.min()),
    carat_0_to_1=(_.CARAT - _.CARAT.min()) / (_.CARAT.max() - _.CARAT.min()),
)

# ✅ Robust scaling (using percentiles)
robust = table.mutate(
    # Scale by IQR (Interquartile Range)
    price_robust=(_.PRICE - _.PRICE.quantile(0.5)) /
                 (_.PRICE.quantile(0.75) - _.PRICE.quantile(0.25))
)
```

### Trigonometric Transformations
```python
from xorq.vendor import ibis

features = table.mutate(
    sin_angle=_.ANGLE.sin(),
    cos_angle=_.ANGLE.cos(),
    tan_angle=_.ANGLE.tan(),
    # Convert degrees to radians first if needed
    radians=_.DEGREES * 3.14159 / 180,
)
```

### Exponential Transformations
```python
features = table.mutate(
    exp_value=_.VALUE.exp(),  # e^x
    # For other bases, use: base ** _.VALUE
    power_of_10=10 ** _.VALUE,
)
```

### Rounding & Binning
```python
features = table.mutate(
    # Rounding
    price_rounded=_.PRICE.round(2),
    carat_rounded=_.CARAT.round(1),

    # Floor and ceiling
    price_floor=_.PRICE.floor(),
    carat_ceil=_.CARAT.ceil(),

    # Truncation
    price_int=_.PRICE.cast('int64'),
)
```

### Date/Time Features
```python
date_features = table.mutate(
    year=_.DATE.year(),
    month=_.DATE.month(),
    day=_.DATE.day(),
    day_of_week=_.DATE.day_of_week(),
    day_of_year=_.DATE.day_of_year(),
    quarter=_.DATE.quarter(),

    # Time differences
    days_since=(_.DATE - _.START_DATE).cast('int32'),
    months_since=_.DATE.month() - _.START_DATE.month(),
)
```

### String Transformations
```python
text_features = table.mutate(
    # Length
    text_length=_.TEXT.length(),

    # Case transformations
    upper_text=_.TEXT.upper(),
    lower_text=_.TEXT.lower(),

    # Substring extraction
    first_char=_.TEXT.substr(0, 1),
    last_3_chars=_.TEXT.substr(-3),

    # Pattern matching
    contains_keyword=_.TEXT.contains('important').cast('int'),
    starts_with_a=_.TEXT.startswith('A').cast('int'),
)
```

## COMMON TRANSFORMATION PATTERNS

### Binning continuous values:
```python
binned = table.mutate(
    size_category=ibis.case()
        .when(_.CARAT < 0.5, 'small')
        .when(_.CARAT < 1.5, 'medium')
        .when(_.CARAT < 3.0, 'large')
        .else_('very_large')
        .end()
)
```

### Creating flags/indicators:
```python
flagged = table.mutate(
    is_premium=(_.PRICE > _.PRICE.mean()).cast('int'),
    is_high_quality=(_.CUT == 'Ideal').cast('int')
)
```

### Computing ratios safely:
```python
ratios = table.mutate(
    price_per_carat=_.PRICE / _.CARAT,
    # Handle division by zero
    safe_ratio=ibis.case()
        .when(_.CARAT == 0, None)
        .else_(_.PRICE / _.CARAT)
        .end()
)
```
