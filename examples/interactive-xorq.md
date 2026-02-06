# Interactive Data Science with xorq Caching

This walkthrough shows how xorq's input-addressed caching turns a
build-from-scratch pipeline into an instant-feedback loop -- the way data
scientists actually work.

The script `interactive_caching_example.py` builds a classification pipeline
on the [bank marketing dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
(~11K rows, 16 features, binary target).  The pipeline has three cached stages:

1. **Data loading** — read CSV, encode target column
2. **Model fitting** — preprocess + train an sklearn classifier
3. **Prediction** — score the held-out test set

When you re-run after a change, only the stages whose inputs changed are
recomputed.  Everything upstream of the change is loaded from the local
Parquet cache in milliseconds.

---

## Setup

```bash
cd examples
source ../.venv/bin/activate   # or: uv sync --extra examples && source ../.venv/bin/activate
```

---

## Step 1 — Cold run (nothing cached)

```bash
python interactive_caching_example.py --clear
```

Expected output:

```
  Cleared cache at interactive-cache/

  Running WITH caching
  Classifier: GradientBoostingClassifier
  Features:   7 numeric + 9 categorical

  Pipeline executed in ~2.0s  (3339 test rows)
  Accuracy: 0.8503   ROC AUC: 0.8508

  Cache: warm (interactive-cache/)
```

Everything ran from scratch: CSV parsing, feature encoding, model training,
prediction.  The results are now saved in `interactive-cache/`.

---

## Step 2 — Warm run (everything cached)

Run the **exact same command** again, without `--clear`:

```bash
python interactive_caching_example.py
```

Expected output:

```
  Running WITH caching
  Classifier: GradientBoostingClassifier
  Features:   7 numeric + 9 categorical

  Pipeline executed in ~0.2s  (3339 test rows)
  Accuracy: 0.8503   ROC AUC: 0.8508

  Cache: warm (interactive-cache/)
```

**~10x faster.**  xorq recognized that every expression in the pipeline has
the same input hash as the cached version and returned the Parquet files
directly -- no CSV parsing, no model fitting, no prediction.

---

## Step 3 — Compare: run without caching

```bash
python interactive_caching_example.py --no-cache
```

Expected output:

```
  Running WITHOUT caching
  Classifier: GradientBoostingClassifier
  Features:   7 numeric + 9 categorical

  Pipeline executed in ~2.0s  (3339 test rows)
  Accuracy: 0.8503   ROC AUC: 0.8508
```

Back to full computation time.  This is what every run looks like without
caching -- the entire pipeline rebuilds from scratch every time, even if
nothing changed.

---

## Step 4 — Swap the classifier

Open `interactive_caching_example.py` and change the classifier.  Comment out
GradientBoosting and uncomment RandomForest:

```python
# classifier = GradientBoostingClassifier(n_estimators=200, random_state=42)
classifier = RandomForestClassifier(n_estimators=200, random_state=42)
```

Run again:

```bash
python interactive_caching_example.py
```

Expected output:

```
  Running WITH caching
  Classifier: RandomForestClassifier
  Features:   7 numeric + 9 categorical

  Pipeline executed in ~1.5s  (3339 test rows)
  Accuracy: 0.8506   ROC AUC: 0.8515

  Cache: warm (interactive-cache/)
```

The data loading stage was still cached (same CSV, same encoding), so xorq
skipped that work.  Only the model fitting and prediction ran from scratch.

---

## Step 5 — Shrink the feature set

Now try a smaller feature set.  Uncomment the alternative feature lists:

```python
numeric_features = ["age", "balance", "duration"]  # smaller set — try it!
categorical_features = ["job", "marital", "education"]  # smaller set — try it!
```

```bash
python interactive_caching_example.py
```

The data loading is still cached, but the model must retrain because the
feature definitions changed.

---

## How it works

xorq's `ParquetCache` is **input-addressed**: the cache key is a
deterministic hash of the expression graph.  Two expressions with identical
logic and identical upstream data always produce the same hash -- and hit the
same cache entry.

```python
# This expression defines a cache boundary.
# Its hash captures: the CSV path, the .mutate() logic, and all upstream deps.
expr = deferred_read_csv(...).mutate(...).cache(cache=cache)

# Downstream expressions that depend on `expr` benefit from the cached result.
# If you change only the classifier, `expr` still has the same hash → cache hit.
```

When you change a downstream step (like the classifier), all upstream cached
steps are unaffected.  When you change an upstream step (like adding a new
feature to the data expression), its hash changes and it recomputes --
along with everything downstream.

This is the same principle behind build systems like Make and Bazel: only
rebuild what changed.

---

## Quick reference

| Command | What happens |
|---------|-------------|
| `python interactive_caching_example.py --clear` | Clear cache, run from scratch |
| `python interactive_caching_example.py` | Run with caching (uses existing cache) |
| `python interactive_caching_example.py --no-cache` | Run without caching (always recomputes) |

---

## Typical timings

| Scenario | Time |
|----------|------|
| Cold cache (first run) | ~2.0s |
| Warm cache (identical re-run) | ~0.2s |
| Classifier swap (data cached) | ~1.5s |
| No caching at all | ~2.0s |
