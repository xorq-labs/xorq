# Agent Quick Reference - Xorq Workflow Improvements

## 🚀 Quick Start: ML Task Checklist

When user asks for ML pipeline (regression, classification, etc.):

### 1. Get Context (FIRST!)
```bash
xorq agent prime
# Now shows task-specific guidance if ML detected
```

### 2. Check for Examples
```bash
find examples/ -name "*prediction*" -o -name "*ml*"
# If found, READ and ADAPT instead of building from scratch
```

### 3. Read Required Context Blocks
```bash
# For ALL ML tasks, read these:
cat python/xorq/agent/context_blocks/ml_struct_pattern.md
cat python/xorq/agent/context_blocks/sklearn_type_requirements.md
cat python/xorq/agent/prompts/ml_task_planning.md
```

### 4. Follow the Patterns

**Categorical Encoding** (copy this pattern):
```python
def encode_quality_columns(table):
    return table.mutate(
        score=(
            _.column.case()
            .when("value1", 0.0)  # Use 0.0 not 0
            .when("value2", 1.0)
            .else_(0.0)
            .end()
            .cast("float64")  # REQUIRED!
        ),
    )
```

**Prediction** (copy this pattern):
```python
predictions = (
    table
    .mutate(as_struct(name="original_row"))  # Pack ALL
    .pipe(fitted.predict)
    .drop("target")
    .unpack("original_row")  # Unpack ALL
    .mutate(predicted_price=_.predicted)
    .drop("predicted")
)
```

### 5. Validate Before Building
```bash
python python/xorq/agent/build_validator.py scripts/your_script.py
# Fix any errors before running xorq build
```

### 6. Build and Test
```bash
xorq build scripts/your_script.py -e expr
xorq run builds/<hash> --limit 10 -o /tmp/test.parquet
```

## ⚠️ Critical Rules (NEVER Violate)

1. **ALWAYS use float64 for sklearn features**
   - ❌ `.cast("int8")`
   - ✅ `.cast("float64")`

2. **ALWAYS use struct pattern for ML predictions**
   - ❌ Manual joins
   - ✅ `.mutate(as_struct(...)).pipe(...).unpack(...)`

3. **NEVER include categorical columns in feature_columns**
   - ❌ `feature_columns = ["carat", "cut", "color"]`
   - ✅ `feature_columns = ["carat", "cut_score", "color_score"]`

4. **ALWAYS check schema first**
   - ✅ `print(table.schema())`

5. **ALWAYS use xorq.vendor ibis**
   - ❌ `import ibis`
   - ✅ `from xorq.vendor import ibis`

## 🔴 When Build Fails

### Step 1: Match Error Pattern
```python
from xorq.agent.error_patterns import handle_build_error

error_msg = "<error from build>"
help_text = handle_build_error(error_msg)
if help_text:
    print(help_text)  # Shows cause, fix, examples
```

### Step 2: Common Errors Quick Fix

**"Duplicate column 'X'"**
- Cause: Categorical in feature_columns
- Fix: Remove from feature_columns, keep in table

**"Failed to coerce arguments"**
- Cause: Using int8/int16 instead of float64
- Fix: Change all `.cast("int8")` to `.cast("float64")`

**"Cannot add Field to projection"**
- Cause: Manual join attempt
- Fix: Use struct pattern

## 📊 Project-Specific Patterns

**Check if project has custom patterns**:
```bash
cat .xorq/PRIME.md
# If exists, USE THESE PATTERNS (overrides defaults)
```

## 🎯 Decision Tree

```
User asks for ML task
    ↓
Run: xorq agent prime
    ↓
Check: examples/ directory
    ↓
    ├─ Example found? → ADAPT IT
    └─ No example? → Use template
           ↓
       Read context blocks
           ↓
       Plan features (numeric vs categorical)
           ↓
       Write script following patterns
           ↓
       Validate: build_validator.py
           ↓
       Build: xorq build
           ↓
       Test: xorq run --limit 10
           ↓
       Catalog: xorq catalog add
```

## 📚 Quick File Reference

| Need | File |
|------|------|
| Struct pattern guide | `python/xorq/agent/context_blocks/ml_struct_pattern.md` |
| Type requirements | `python/xorq/agent/context_blocks/sklearn_type_requirements.md` |
| Task planning | `python/xorq/agent/prompts/ml_task_planning.md` |
| Working example | `examples/diamonds_price_prediction.py` |
| Project patterns | `.xorq/PRIME.md` (if exists) |
| Error help | `python/xorq/agent/error_patterns.py` |
| Validation | `python/xorq/agent/build_validator.py` |

## ⏱️ Time Saving Shortcuts

**DON'T**:
- ❌ Build custom pattern from scratch (20+ min)
- ❌ Try int8 and debug later (4+ iterations)
- ❌ Skip validation (likely rebuild)
- ❌ Ignore examples (reinventing wheel)

**DO**:
- ✅ Read context blocks first (2 min)
- ✅ Copy working patterns (1 min)
- ✅ Validate before building (30 sec)
- ✅ Use float64 from start (0 iterations)

**Result**: 5-10 min total vs 30+ min trial-and-error

## 🎓 Learning Path

**First time doing ML in xorq?** Read in order:
1. `ml_struct_pattern.md` - Understand the pattern
2. `sklearn_type_requirements.md` - Understand types
3. `examples/diamonds_price_prediction.py` - See it in action
4. `ml_task_planning.md` - Plan your task
5. Build with patterns → Success!

## 🔧 Troubleshooting

**Build keeps failing?**
1. Run validator: `python python/xorq/agent/build_validator.py <script>`
2. Check error pattern: `handle_build_error(error_msg)`
3. Re-read struct pattern docs
4. Compare with working example

**Not sure about pattern?**
1. Check `.xorq/PRIME.md` for project rules
2. Reference `ml_struct_pattern.md` for canonical pattern
3. Look at `examples/` for working code
4. Ask user for clarification

## ✅ Success Criteria

Before saying "done":
- [ ] Used struct pattern (if ML)
- [ ] All features are float64 (if sklearn)
- [ ] Validated script passed
- [ ] Build succeeded
- [ ] Test run (--limit 10) worked
- [ ] Cataloged with alias
- [ ] Committed (if requested)

**If any fail → NOT done yet!**
