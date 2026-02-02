#!/usr/bin/env python3
"""
Post-process quartodoc-generated files to remove broken links for undocumented types
and add content to empty pages that lack docstrings.

This script runs after quartodoc build to:
1. Convert broken links to plain text
2. Add manual content to empty pages
"""

import re
from pathlib import Path


# Types that are NOT documented but appear in type hints
# These should be converted to plain text instead of links
UNDOCUMENTED_TYPES = {
    # Base expression types (users use Table, Column, Scalar instead)
    # Order matters: match more specific patterns first
    r"\[ibis\]\(`ibis`\)\.\[Expr\]\(`ibis\.Expr`\)": "ibis.Expr",
    r"\[ir\]\(`[^`]*`\)\.\[Expr\]\(`[^`]*`\)": "ir.Expr",
    r"\[Expr\]\(`[^`]*`\)": "Expr",
    # Generic expression type (not documented)
    r"\[Expression\]\(`Expression`\)": "Expression",
    # Internal implementation types
    r"\[Cache\]\(`Cache`\)": "Cache",
    r"\[Backend\]\(`[^`]*Backend`\)": "Backend",
    r"\[SessionConfig\]\(`[^`]*SessionConfig`\)": "SessionConfig",
    r"\[Node\]\(`Node`\)": "Node",
    # Standard library types (keep as plain text for consistency)
    r"\[Path\]\(`pathlib\.Path`\)": "Path",
    r"\[tuple\]\(`tuple`\)": "tuple",
    r"\[bool\]\(`bool`\)": "bool",
    r"\[str\]\(`str`\)": "str",
    r"\[int\]\(`int`\)": "int",
    r"\[float\]\(`float`\)": "float",
    r"\[dict\]\(`dict`\)": "dict",
    r"\[list\]\(`list`\)": "list",
    r"\[callable\]\(`callable`\)": "callable",
    r"\[object\]\(`object`\)": "object",
    r"\[Any\]\(`typing\.Any`\)": "Any",
    # Standard library exceptions
    r"\[ValueError\]\(`ValueError`\)": "ValueError",
    # Standard library collections/typing (not documented)
    r"\[Iterable\]\(`collections\.abc\.Iterable`\)": "Iterable",
    r"\[Iterable\]\(`typing\.Iterable`\)": "Iterable",
    r"\[Mapping\]\(`collections\.abc\.Mapping`\)": "Mapping",
    r"\[Mapping\]\(`typing\.Mapping`\)": "Mapping",
    r"\[Literal\]\(`typing\.Literal`\)": "Literal",
    r"\[Callable\]\(`collections\.abc\.Callable`\)": "Callable",
    r"\[Callable\]\(`typing\.Callable`\)": "Callable",
    r"\[Sequence\]\(`collections\.abc\.Sequence`\)": "Sequence",
    r"\[Sequence\]\(`typing\.Sequence`\)": "Sequence",
    # Standard library I/O types
    r"\[TextIOWrapper\]\(`io\.TextIOWrapper`\)": "TextIOWrapper",
    # PyArrow types (not documented)
    r"\[RecordBatchReader\]\(`[^`]*RecordBatchReader`\)": "RecordBatchReader",
    # Internal types (not documented)
    r"\[Namespace\]\(`[^`]*Namespace`\)": "Namespace",
    r"\[Selector\]\(`[^`]*Selector`\)": "Selector",
    # Fix broken link in to_pyarrow_batches
    r"\[results\]\(`results`\)": "RecordBatchReader",
    # External library types (scikit-learn) - convert nested links to plain text
    # Match the nested link structure: [sklearn](`sklearn`).[pipeline](`sklearn.pipeline`).[Pipeline](`sklearn.pipeline.Pipeline`)
    r"\[sklearn\]\(`sklearn`\)\.\[pipeline\]\(`sklearn\.pipeline`\)\.\[Pipeline\]\(`sklearn\.pipeline\.Pipeline`\)": "sklearn.pipeline.Pipeline",
    # Also match if there are spaces or different formatting
    r"\[sklearn\]\(`sklearn`\)\s*\.\s*\[pipeline\]\(`sklearn\.pipeline`\)\s*\.\s*\[Pipeline\]\(`sklearn\.pipeline\.Pipeline`\)": "sklearn.pipeline.Pipeline",
    # Fix nested Expr in Sequence types
    r"\[Sequence\]\(`collections\.abc\.Sequence`\)\\\[\[ir\]\(`[^`]*`\)\.\[Expr\]\(`[^`]*`\)\]": r"[Sequence](`collections.abc.Sequence`)\[ir.Expr\]",
    # Fix incorrect Schema link format
    r"\[Schema\]\(`xorq\.vendor\.ibis\.Schema`\)": "Schema",
    r"\[SchemaLike\]\(`xorq\.vendor\.ibis\.expr\.schema\.SchemaLike`\)": "SchemaLike",
    # Fix Table links (Table IS documented, but the link format is wrong)
    # Match: [ir](`xorq.vendor.ibis.expr.types`).[Table](`xorq.vendor.ibis.expr.types.Table`)
    r"\[ir\]\(`xorq\.vendor\.ibis\.expr\.types`\)\.\[Table\]\(`xorq\.vendor\.ibis\.expr\.types\.Table`\)": "Table",
    r"\[Table\]\(`xorq\.vendor\.ibis\.expr\.types\.Table`\)": "Table",
    # Fix Scalar links (Scalar IS documented, but link format is wrong)
    r"\[ir\]\(`xorq\.vendor\.ibis\.expr\.types`\)\.\[Scalar\]\(`xorq\.vendor\.ibis\.expr\.types\.Scalar`\)": "Scalar",
    r"\[Scalar\]\(`xorq\.vendor\.ibis\.expr\.types\.Scalar`\)": "Scalar",
    # Fix broken ibis.IntergerColumn (typo - should be IntegerColumn, but link is broken)
    r"\[ibis\]\(`ibis`\)\.\[IntergerColumn\]\(`ibis\.IntergerColumn`\)": "IntegerColumn",
    # Fix empty link text patterns
    r"\[\]\(`typing\.Iterable`\)": "Iterable",
    r"\[\]\(`str`\)": "str",
    # Fix ir.Expr in type annotations (should be plain text)
    r"\bir\.Expr\b": "Expr",
    # Fix broken Window link (Window is not documented as a standalone page)
    r"\[Window\]\(`Window`\)": "Window",
    # Fix broken Schema links (Schema IS documented, but link format is wrong)
    r"\[Schema\]\(`Schema`\)": "Schema",
    # Fix broken DataType links (DataType IS documented, but link format is wrong)
    r"\[DataType\]\(`DataType`\)": "DataType",
    # Fix [Any](`Any`) links - should be plain text
    r"\[Any\]\(`Any`\)": "Any",
    # Fix all [ir](`xorq.vendor.ibis.expr.types`).[*Value](`...`) links - convert to plain text
    r"\[ir\]\(`xorq\.vendor\.ibis\.expr\.types`\)\.\[Value\]\(`xorq\.vendor\.ibis\.expr\.types\.Value`\)": "Value",
    r"\[ir\]\(`xorq\.vendor\.ibis\.expr\.types`\)\.\[BooleanValue\]\(`xorq\.vendor\.ibis\.expr\.types\.BooleanValue`\)": "BooleanValue",
    r"\[ir\]\(`xorq\.vendor\.ibis\.expr\.types`\)\.\[NumericValue\]\(`xorq\.vendor\.ibis\.expr\.types\.NumericValue`\)": "NumericValue",
    r"\[ir\]\(`xorq\.vendor\.ibis\.expr\.types`\)\.\[IntegerValue\]\(`xorq\.vendor\.ibis\.expr\.types\.IntegerValue`\)": "IntegerValue",
    r"\[ir\]\(`xorq\.vendor\.ibis\.expr\.types`\)\.\[BooleanScalar\]\(`xorq\.vendor\.ibis\.expr\.types\.BooleanScalar`\)": "BooleanScalar",
    r"\[ir\]\(`xorq\.vendor\.ibis\.expr\.types`\)\.\[BooleanColumn\]\(`xorq\.vendor\.ibis\.expr\.types\.BooleanColumn`\)": "BooleanColumn",
    r"\[ir\]\(`xorq\.vendor\.ibis\.expr\.types`\)\.\[Column\]\(`xorq\.vendor\.ibis\.expr\.types\.Column`\)": "Column",
    r"\[Column\]\(`xorq\.vendor\.ibis\.expr\.types\.generic\.Column`\)": "Column",
    r"\[Int64Column\]\(`Int64Column`\)": "Int64Column",
    r"\[ir\]\(`xorq\.vendor\.ibis\.expr\.types`\)\.\[IntervalScalar\]\(`xorq\.vendor\.ibis\.expr\.types\.IntervalScalar`\)": "IntervalScalar",
    r"\[ir\]\(`xorq\.vendor\.ibis\.expr\.types`\)\.\[Deferred\]\(`xorq\.vendor\.ibis\.expr\.types\.Deferred`\)": "Deferred",
    # Fix Table link in to_pyarrow return type
    r"\[Table\]\(`xorq\.vendor\.ibis\.expr\.api\.Table`\)": "Table",
    # Fix nested links in type annotations like Sequence\[[ir](`...`).[Value](`...`)\]
    r"Sequence\\\[\[ir\]\(`[^`]*`\)\.\[Value\]\(`[^`]*`\)\\\]": "Sequence[Value]",
    r"Sequence\\\[\[ir\]\(`[^`]*`\)\.\[BooleanValue\]\(`[^`]*`\)\\\]": "Sequence[BooleanValue]",
    r"Sequence\\\[\[ir\]\(`[^`]*`\)\.\[NumericValue\]\(`[^`]*`\)\\\]": "Sequence[NumericValue]",
    r"Sequence\\\[\[ir\]\(`[^`]*`\)\.\[Column\]\(`[^`]*`\)\\\]": "Sequence[Column]",
    # Fix Mapping\[[ir](`...`).[Value](`...`), Any\] patterns
    r"Mapping\\\[\[ir\]\(`[^`]*`\)\.\[Value\]\(`[^`]*`\),\s*Any\\\]": "Mapping[Value, Any]",
    r"Mapping\\\[\[ir\]\(`[^`]*`\)\.\[Scalar\]\(`[^`]*`\),\s*Any\\\]": "Mapping[Scalar, Any]",
    # Fix Callable patterns with nested links
    r"Callable\\\[\\\[\[ir\]\(`[^`]*`\)\.\[Value\]\(`[^`]*`\)\\\],\s*\[ir\]\(`[^`]*`\)\.\[Value\]\(`[^`]*`\)\\\]": "Callable[[Value], Value]",
    # Fix Iterable patterns with nested links
    r"Iterable\\\[\[ir\]\(`[^`]*`\)\.\[Value\]\(`[^`]*`\)\\\]": "Iterable[Value]",
    r"Iterable\\\[str\\\]": "Iterable[str]",
    # Fix [dt](`xorq.vendor.ibis.expr.datatypes`).[DataType](`...`) links
    r"\[dt\]\(`xorq\.vendor\.ibis\.expr\.datatypes`\)\.\[DataType\]\(`xorq\.vendor\.ibis\.expr\.datatypes\.DataType`\)": "DataType",
    # Fix [s](`xorq.vendor.ibis.selectors`).Selector links
    r"\[s\]\(`xorq\.vendor\.ibis\.selectors`\)\.\[Selector\]\(`[^`]*`\)": "Selector",
    # Fix [sg](`sqlglot`).[Dialect](`sqlglot.Dialect`) links
    r"\[sg\]\(`sqlglot`\)\.\[Dialect\]\(`sqlglot\.Dialect`\)": "sqlglot.Dialect",
    # Fix [sqlglot](`sqlglot`).[expressions](`sqlglot.expressions`).[ColumnDef](`...`) links
    r"\[sqlglot\]\(`sqlglot`\)\.\[expressions\]\(`sqlglot\.expressions`\)\.\[ColumnDef\]\(`sqlglot\.expressions\.ColumnDef`\)": "sqlglot.expressions.ColumnDef",
    # Fix [IfAnyAll](`xorq.vendor.ibis.selectors.IfAnyAll`) links
    r"\[IfAnyAll\]\(`xorq\.vendor\.ibis\.selectors\.IfAnyAll`\)": "IfAnyAll",
    # Fix [Deferred](`xorq.vendor.ibis.common.deferred.Deferred`) links
    r"\[Deferred\]\(`xorq\.vendor\.ibis\.common\.deferred\.Deferred`\)": "Deferred",
    # Fix [StringValue](`xorq.vendor.ibis.expr.types.strings.StringValue`) links
    r"\[StringValue\]\(`xorq\.vendor\.ibis\.expr\.types\.strings\.StringValue`\)": "StringValue",
    # Fix [IntegerValue](`xorq.vendor.ibis.expr.types.numeric.IntegerValue`) links
    r"\[IntegerValue\]\(`xorq\.vendor\.ibis\.expr\.types\.numeric\.IntegerValue`\)": "IntegerValue",
    # Fix [TimeValue](`xorq.vendor.ibis.expr.types.temporal.TimeValue`) links
    r"\[TimeValue\]\(`xorq\.vendor\.ibis\.expr\.types\.temporal\.TimeValue`\)": "TimeValue",
    # Fix [IntervalValue](`xorq.vendor.ibis.expr.types.temporal.IntervalValue`) links
    r"\[IntervalValue\]\(`xorq\.vendor\.ibis\.expr\.types\.temporal\.IntervalValue`\)": "IntervalValue",
    # Fix [DateValue](`xorq.vendor.ibis.expr.types.temporal.DateValue`) links
    r"\[DateValue\]\(`xorq\.vendor\.ibis\.expr\.types\.temporal\.DateValue`\)": "DateValue",
    # Fix [GroupedTable](`xorq.vendor.ibis.expr.types.groupby.GroupedTable`) links
    r"\[GroupedTable\]\(`xorq\.vendor\.ibis\.expr\.types\.groupby\.GroupedTable`\)": "GroupedTable",
    # Fix [Table](`xorq.vendor.ibis.expr.types.joins.Table`) links
    r"\[Table\]\(`xorq\.vendor\.ibis\.expr\.types\.joins\.Table`\)": "Table",
    # Fix [NumericScalar](`xorq.vendor.ibis.expr.types.numeric.NumericScalar`) links
    r"\[NumericScalar\]\(`xorq\.vendor\.ibis\.expr\.types\.numeric\.NumericScalar`\)": "NumericScalar",
    # Fix [IntegerColumn](`xorq.vendor.ibis.expr.types.numeric.IntegerColumn`) links
    r"\[IntegerColumn\]\(`xorq\.vendor\.ibis\.expr\.types\.numeric\.IntegerColumn`\)": "IntegerColumn",
    # Fix [NumericColumn](`xorq.vendor.ibis.expr.types.numeric.NumericColumn`) links
    r"\[NumericColumn\]\(`xorq\.vendor\.ibis\.expr\.types\.numeric\.NumericColumn`\)": "NumericColumn",
    # Fix [Schema](`xorq.vendor.ibis.expr.schema.Schema`) links
    r"\[Schema\]\(`xorq\.vendor\.ibis\.expr\.schema\.Schema`\)": "Schema",
    # Fix [Scalar](`xorq.vendor.ibis.expr.types.generic.Scalar`) links
    r"\[Scalar\]\(`xorq\.vendor\.ibis\.expr\.types\.generic\.Scalar`\)": "Scalar",
    r"\[Scalar\]\(`xorq\.vendor\.ibis\.expr\.types\.uuid\.Scalar`\)": "Scalar",
    # Fix [Table](`xorq.vendor.ibis.expr.types.relations.Table`) links
    r"\[Table\]\(`xorq\.vendor\.ibis\.expr\.types\.relations\.Table`\)": "Table",
    # Fix [Value](`xorq.vendor.ibis.expr.types.generic.Value`) links
    r"\[Value\]\(`xorq\.vendor\.ibis\.expr\.types\.generic\.Value`\)": "Value",
    # Fix [IntegerScalar](`xorq.vendor.ibis.expr.types.numeric.IntegerScalar`) links
    r"\[IntegerScalar\]\(`xorq\.vendor\.ibis\.expr\.types\.numeric\.IntegerScalar`\)": "IntegerScalar",
    # Fix datetime.date links
    r"\[datetime\]\(`datetime`\)\.\[date\]\(`datetime\.date`\)": "datetime.date",
    r"\[datetime\]\(`datetime`\)\.\[time\]\(`datetime\.time`\)": "datetime.time",
    # Fix [dt](`xorq.vendor.ibis.expr.datatypes`).[Date](`...`) links
    r"\[dt\]\(`xorq\.vendor\.ibis\.expr\.datatypes`\)\.\[Date\]\(`xorq\.vendor\.ibis\.expr\.datatypes\.Date`\)": "Date",
    r"\[dt\]\(`xorq\.vendor\.ibis\.expr\.datatypes`\)\.\[String\]\(`xorq\.vendor\.ibis\.expr\.datatypes\.String`\)": "String",
    r"\[dt\]\(`xorq\.vendor\.ibis\.expr\.datatypes`\)\.\[Time\]\(`xorq\.vendor\.ibis\.expr\.datatypes\.Time`\)": "Time",
    # Fix nested Value[Date] patterns
    r"\[Value\]\(`xorq\.vendor\.ibis\.expr\.types\.generic\.Value`\)\\\[\[dt\]\(`xorq\.vendor\.ibis\.expr\.datatypes`\)\.\[Date\]\(`xorq\.vendor\.ibis\.expr\.datatypes\.Date`\)\\\]": "Value[Date]",
    r"\[Value\]\(`xorq\.vendor\.ibis\.expr\.types\.generic\.Value`\)\\\[\[dt\]\(`xorq\.vendor\.ibis\.expr\.datatypes`\)\.\[String\]\(`xorq\.vendor\.ibis\.expr\.datatypes\.String`\)\\\]": "Value[String]",
    r"\[Value\]\(`xorq\.vendor\.ibis\.expr\.types\.generic\.Value`\)\\\[\[dt\]\(`xorq\.vendor\.ibis\.expr\.datatypes`\)\.\[Time\]\(`xorq\.vendor\.ibis\.expr\.datatypes\.Time`\)\\\]": "Value[Time]",
    # Fix [FittedPipeline](`xorq.expr.ml.pipeline_lib.FittedPipeline`) links
    r"\[FittedPipeline\]\(`xorq\.expr\.ml\.pipeline_lib\.FittedPipeline`\)": "FittedPipeline",
    # Fix [FittedStep](`xorq.expr.ml.pipeline_lib.FittedStep`) links
    r"\[FittedStep\]\(`xorq\.expr\.ml\.pipeline_lib\.FittedStep`\)": "FittedStep",
    # Fix [Step](`xorq.expr.ml.pipeline_lib.Step`) links
    r"\[Step\]\(`xorq\.expr\.ml\.pipeline_lib\.Step`\)": "Step",
    # Fix [Pipeline](`xorq.expr.ml.pipeline_lib.Pipeline`) links
    r"\[Pipeline\]\(`xorq\.expr\.ml\.pipeline_lib\.Pipeline`\)": "Pipeline",
    # Fix [BasicAuth](`xorq.flight.BasicAuth`) links
    r"\[BasicAuth\]\(`xorq\.flight\.BasicAuth`\)": "BasicAuth",
    # Fix [CacheStorage](`CacheStorage`) links
    r"\[CacheStorage\]\(`CacheStorage`\)": "CacheStorage",
    # Fix [JoinKind](`xorq.vendor.ibis.expr.operations.relations.JoinKind`) links
    r"\[JoinKind\]\(`xorq\.vendor\.ibis\.expr\.operations\.relations\.JoinKind`\)": "JoinKind",
    # Fix [re](`re`).[Pattern](`re.Pattern`) links
    r"\[re\]\(`re`\)\.\[Pattern\]\(`re\.Pattern`\)": "re.Pattern",
    # Fix [TimestampValue](`xorq.vendor.ibis.expr.types.temporal.TimestampValue`) links
    r"\[TimestampValue\]\(`xorq\.vendor\.ibis\.expr\.types\.temporal\.TimestampValue`\)": "TimestampValue",
    # Fix [BooleanValue](`xorq.vendor.ibis.expr.types.logical.BooleanValue`) links
    r"\[BooleanValue\]\(`xorq\.vendor\.ibis\.expr\.types\.logical\.BooleanValue`\)": "BooleanValue",
    # Fix [BinaryValue](`xorq.vendor.ibis.expr.types.binary.BinaryValue`) links
    r"\[BinaryValue\]\(`xorq\.vendor\.ibis\.expr\.types\.binary\.BinaryValue`\)": "BinaryValue",
    # Fix [ArrayValue](`xorq.vendor.ibis.expr.types.arrays.ArrayValue`) links
    r"\[ArrayValue\]\(`xorq\.vendor\.ibis\.expr\.types\.arrays\.ArrayValue`\)": "ArrayValue",
    # Fix [SimpleCaseBuilder](`SimpleCaseBuilder`) links
    r"\[SimpleCaseBuilder\]\(`SimpleCaseBuilder`\)": "SimpleCaseBuilder",
    # Fix [ArrayScalar](`xorq.vendor.ibis.expr.types.arrays.ArrayScalar`) links
    r"\[ArrayScalar\]\(`xorq\.vendor\.ibis\.expr\.types\.arrays\.ArrayScalar`\)": "ArrayScalar",
    # Fix [StringScalar](`xorq.vendor.ibis.expr.types.strings.StringScalar`) links
    r"\[StringScalar\]\(`xorq\.vendor\.ibis\.expr\.types\.strings\.StringScalar`\)": "StringScalar",
    # Fix [datetime](`datetime`).[datetime](`datetime.datetime`) links
    r"\[datetime\]\(`datetime`\)\.\[datetime\]\(`datetime\.datetime`\)": "datetime.datetime",
    # Fix [dt](`xorq.vendor.ibis.expr.datatypes`).[Timestamp](`...`) links
    r"\[dt\]\(`xorq\.vendor\.ibis\.expr\.datatypes`\)\.\[Timestamp\]\(`xorq\.vendor\.ibis\.expr\.datatypes\.Timestamp`\)": "Timestamp",
    # Fix [type](`type`) links
    r"\[type\]\(`type`\)": "type",
    # Fix [str](`xorq.vendor.ibis.expr.datatypes.str`) links
    r"\[str\]\(`xorq\.vendor\.ibis\.expr\.datatypes\.str`\)": "str",
    # Fix [bool](`xorq.vendor.ibis.expr.datatypes.bool`) links
    r"\[bool\]\(`xorq\.vendor\.ibis\.expr\.datatypes\.bool`\)": "bool",
    # Fix [Struct](`xorq.vendor.ibis.expr.datatypes.core.Struct`) links
    r"\[Struct\]\(`xorq\.vendor\.ibis\.expr\.datatypes\.core\.Struct`\)": "Struct",
    # Fix [Iterator](`typing.Iterator`) links
    r"\[Iterator\]\(`typing\.Iterator`\)": "Iterator",
    # Fix [Signature](`Signature`) links
    r"\[Signature\]\(`Signature`\)": "Signature",
    # Fix [ibis](`ibis`).[backends](`ibis.backends`).[BaseBackend](`ibis.backends.BaseBackend`) links
    r"\[ibis\]\(`ibis`\)\.\[backends\]\(`ibis\.backends`\)\.\[BaseBackend\]\(`ibis\.backends\.BaseBackend`\)": "Backend",
    # Fix empty link patterns like [](`True`) or [](`False`)
    r"\[\]\(`True`\)": "True",
    r"\[\]\(`False`\)": "False",
    r"\[\]\(`None`\)": "None",
}

# Files to process
REFERENCE_DIR = Path(__file__).parent / "reference"

# Manual content for pages that lack docstrings
# Format: filename -> content to append after signature
MANUAL_CONTENT = {
    "ls.qmd": r"""
Accessor for caching, backends, and metadata on an expression (expr.ls).

Use ``expr.ls`` on any expression to get a LETSQLAccessor. You do not
construct LETSQLAccessor directly.

## Attributes {.doc-section .doc-section-attributes}

| Name           | Type   | Description                                                               |
|----------------|--------|---------------------------------------------------------------------------|
| expr           | ir.Expr | The expression this accessor is attached to.                              |
| cached_nodes   | tuple  | All CachedNode nodes in the expression.                                   |
| tags           | tuple  | Alias for get_tags().                                                     |
| cache          | Cache \| None | Cache for this node if is_cached; else None.                              |
| caches         | tuple  | Cache for each cached node in the expression.                             |
| backends       | tuple  | All backends (sources) used by the expression.                            |
| is_multiengine | bool   | True if the expression uses more than one backend.                        |
| dts            | tuple  | DatabaseTable / SQLQueryResult nodes (excluding RemoteTable, CachedNode). |
| is_cached      | bool   | True if the root of this expression is a cached node.                     |
| has_cached     | bool   | True if the expression contains any cached node.                          |
| untagged       | ir.Expr | Expression with tag nodes removed (for hashing).                          |
| uncached       | ir.Expr | Expression with cache nodes replaced by their inputs.                     |
| uncached_one   | ir.Expr | If is_cached, the single uncached parent; else the expression.            |
| tokenized      | str    | Stable token for the expression (e.g. for caching).                       |
| cache_path     | Path \| None | Path to cache file if cached with ParquetStorage; else None.              |
| cached_dt      | Node \| None | Cached result (e.g. Read) if exists(); else None.                         |
""",
    "get_plans.qmd": r"""
Get execution plans for an expression.

Returns a dictionary mapping plan type (str) to plan string for each plan
returned by the backend's EXPLAIN command.

## Parameters {.doc-section .doc-section-parameters}

| Name | Type | Description       | Default   |
|------|------|-------------------|-----------|
| expr | Expr | The expression to get plans for. | _required_ |

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description                                                                 |
|--------|--------|-----------------------------------------------------------------------------|
|        | dict   | Mapping of plan type (str) to plan string for each plan returned by the backend's EXPLAIN. |
""",
    "connect.qmd": r"""

Create a xorq backend connection.

## Parameters {.doc-section .doc-section-parameters}

| Name          | Type           | Description                                 | Default   |
|---------------|----------------|---------------------------------------------|-----------|
| session_config | SessionConfig \| None | Optional session configuration. | `None`    |

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description                                  |
|--------|--------|----------------------------------------------|
|        | Backend | A xorq backend instance ready for use. |
""",
    "register.qmd": r"""
Register a data source as a table in the current backend.

## Parameters {.doc-section .doc-section-parameters}

| Name       | Type                                                          | Description                                                                 | Default   |
|-----------|---------------------------------------------------------------|-----------------------------------------------------------------------------|-----------|
| source     | str \| Path \| pa.Table \| pa.RecordBatch \| pa.Dataset \| pd.DataFrame | The data source to register. Can be a file path, URL, or in-memory data structure. | _required_ |
| table_name | str \| None                                                    | Optional name for the registered table. If not provided, a name will be inferred. | `None`    |
| **kwargs   | Any                                                           | Additional keyword arguments passed to the backend's register method.      |           |

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description                                  |
|--------|--------|----------------------------------------------|
|        | Table  | A table expression representing the registered data source. |
""",
    "build_column_trees.qmd": r"""

Builds a lineage tree for each column in the expression.

This function analyzes an expression and creates a tree structure showing the
lineage (dependencies) for each column. Each tree represents how a column is
derived from other columns and operations in the expression.

## Parameters {.doc-section .doc-section-parameters}

| Name | Type | Description                                 | Default   |
|------|------|---------------------------------------------|-----------|
| expr | Expr | The expression to build column trees for.   | _required_ |

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description                                  |
|--------|--------|----------------------------------------------|
|        | dict    | A dictionary mapping column names (str) to GenericNode objects. Each GenericNode represents the lineage tree for that column, showing how it's derived from other columns and operations. |

## Examples {.doc-section .doc-section-examples}

```python
import xorq.api as xo
from xorq.common.utils.lineage_utils import build_column_trees, build_tree

# Create an expression
expr = xo.read_csv("data.csv").filter(xo._.age > 18).select("name", "age")

# Build column trees
column_trees = build_column_trees(expr)

# Display the lineage for a specific column
if "name" in column_trees:
    tree = build_tree(column_trees["name"])
    print(tree)
```
""",
    "FittedStep.qmd": r"""
A fitted step in a machine learning pipeline.

A FittedStep represents a step that has been fitted on training data and can
be used for transformation or prediction on new data.

## Attributes {.doc-section .doc-section-attributes}

| Name     | Type   | Description                                                               |
|----------|--------|---------------------------------------------------------------------------|
| step     | Step   | The step that was fitted.                                                 |
| expr     | Expr   | The expression used for fitting.                                          |
| features | tuple  | Optional tuple of feature column names.                                  |
| target   | str \| None | Optional target column name.                                             |
| cache    | Cache \| None | Optional cache configuration for this step.                              |
""",
    "FloatingColumn.qmd": r"""
A column expression representing floating-point numeric values.

FloatingColumn is a specialized numeric column type for floating-point data.
It inherits from NumericColumn and FloatingValue.
""",
    "SourceCache.qmd": r"""
A cache that uses the source file's modification time as the cache key.

SourceCache is useful when you want to cache results based on when the
source data file was last modified. If the source file changes, the cache
will be invalidated automatically.
""",
    "SourceSnapshotCache.qmd": r"""
A cache that uses a snapshot of the source file as the cache key.

SourceSnapshotCache creates a snapshot-based cache, which is useful when
you want to cache results based on the exact state of the source data.
The cache will be invalidated if the source file changes.
""",
    "deferred_fit_transform.qmd": r"""
Create a deferred fit and transform operation on an expression.

This function creates a DeferredFitOther object that can fit a model on training
data and then transform data using the fitted model, all within a deferred
execution context.

## Parameters {.doc-section .doc-section-parameters}

| Name        | Type   | Description                                                                 | Default        |
|-------------|--------|-----------------------------------------------------------------------------|----------------|
| expr        | Expr   | The input expression (table) to fit and transform on.                       | _required_     |
| features    | tuple  | Column names or indices to use as features.                                | _required_     |
| fit         | callable | Function that fits a model. Should accept (df, target, features) and return a fitted model. | _required_     |
| other       | callable | Function that transforms data. Should accept (model, df) and return transformed data. | _required_     |
| return_type | DataType | The data type of the transformed output.                                   | _required_     |
| target      | str \| None | Optional target column name for supervised learning.                       | `None`         |
| name_infix  | str    | Infix to use in generated function names.                                  | `'transform'`  |
| cache       | Cache \| None | Optional cache configuration for the fitted model.                          | `None`         |

## Returns {.doc-section .doc-section-returns}

| Name   | Type            | Description                                  |
|--------|-----------------|----------------------------------------------|
|        | DeferredFitOther | A deferred fit/transform object that can be executed later. |
""",
    "deferred_fit_predict.qmd": r"""
Create a deferred fit and predict operation on an expression.

This function creates a DeferredFitOther object that can fit a model on training
data and then make predictions on new data, all within a deferred execution
context.

## Parameters {.doc-section .doc-section-parameters}

| Name        | Type   | Description                                                                 | Default        |
|-------------|--------|-----------------------------------------------------------------------------|----------------|
| expr        | Expr   | The input expression (table) to fit and predict on.                        | _required_     |
| target      | str    | Target column name for supervised learning.                                 | _required_     |
| features    | tuple  | Column names or indices to use as features.                                 | _required_     |
| cls         | type   | Scikit-learn estimator class to fit.                                        | _required_     |
| return_type | DataType | The data type of the predictions.                                          | _required_     |
| params      | tuple  | Optional parameters to pass to the estimator constructor.                   | `()`           |
| name_infix  | str    | Infix to use in generated function names.                                   | `'predict'`    |
| cache       | Cache \| None | Optional cache configuration for the fitted model.                          | `None`         |

## Returns {.doc-section .doc-section-returns}

| Name   | Type            | Description                                  |
|--------|-----------------|----------------------------------------------|
|        | DeferredFitOther | A deferred fit/predict object that can be executed later. |
""",
    "make_quickgrove_udf.qmd": r"""

Creates a user-defined function (UDF) that can be used in Ibis expressions
from a quickgrove gradient boosted decision trees model.

## Parameters {.doc-section .doc-section-parameters}

| Name          | Type                                                          | Description                                                                 | Default   |
|---------------|---------------------------------------------------------------|-----------------------------------------------------------------------------|-----------|
| model_or_path | str \| Path \| PyGradientBoostedDecisionTrees                 | Either an already-loaded quickgrove model or a path to a saved model file. | _required_ |
| model_name    | str \| None                                                    | Alternative model name, if the name extracted from the path is not a valid Python identifier. | `None`    |

## Returns {.doc-section .doc-section-returns}

| Name   | Type        | Description                                  |
|--------|-------------|----------------------------------------------|
|        | UDFWrapper  | A wrapper object containing the UDF function, model, and metadata. The UDF can be called on Ibis expressions using `.on_expr()`. |
""",
    "make_udxf.qmd": r"""
Create a User-Defined Exchange Function (UDXF) for Apache Arrow Flight.

Creates a UDXF class that can process pandas DataFrames via Arrow Flight protocol.
The returned class can be used with `flight_udxf()` to create distributed data
processing operations.

## Parameters {.doc-section .doc-section-parameters}

| Name            | Type                                    | Description                                                                 | Default   |
|-----------------|-----------------------------------------|-----------------------------------------------------------------------------|-----------|
| process_df      | callable                                | Function that takes a pandas DataFrame and returns a transformed DataFrame. | _required_ |
| maybe_schema_in | Schema \| callable                      | Input schema specification. Can be an Ibis Schema or a callable that validates the input schema. | _required_ |
| maybe_schema_out | Schema \| callable                      | Output schema specification. Can be an Ibis Schema or a callable that computes the output schema from the input schema. | _required_ |
| name            | str \| None                             | Name for the UDXF. If None, uses the function name from `process_df`.      | `None`    |
| description     | str \| None                             | Description of the UDXF. If None, uses the name.                            | `None`    |
| command         | str \| None                             | Unique command identifier. If None, generates from function tokenization.   | `None`    |
| do_wraps        | bool                                    | If True, wraps the function with error handling and streaming support.      | `True`    |

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description                                  |
|--------|--------|----------------------------------------------|
|        | type   | A UDXF class (subclass of AbstractExchanger) that can be instantiated and used with flight_udxf(). |

## Examples {.doc-section .doc-section-examples}

```python
from xorq.flight.exchanger import make_udxf
import xorq.api as xo

# Define processing function
def transform_data(df):
    return df.assign(new_col=df['old_col'] * 2)

# Create UDXF
udxf_class = make_udxf(
    process_df=transform_data,
    maybe_schema_in=xo.schema({"old_col": "int64"}),
    maybe_schema_out=xo.schema({"old_col": "int64", "new_col": "int64"}),
    name="DoubleColumn"
)

# Use with flight_udxf
from xorq.expr.relations import flight_udxf
udxf = flight_udxf(
    expr=input_expr,
    process_df=transform_data,
    maybe_schema_in=xo.schema({"old_col": "int64"}).to_pyarrow(),
    maybe_schema_out=xo.schema({"old_col": "int64", "new_col": "int64"}).to_pyarrow()
)
```
""",
    "build_tree.qmd": r"""
Build a visual tree representation of a lineage node.

Creates a Rich Tree object that can be displayed or printed to visualize
the structure of a GenericNode and its children.

## Parameters {.doc-section .doc-section-parameters}

| Name      | Type            | Description                                                                 | Default   |
|-----------|-----------------|-----------------------------------------------------------------------------|-----------|
| node      | GenericNode     | The root node to build the tree from.                                       | _required_ |
| palette   | ColorScheme \| None | Optional color scheme for styling the tree nodes.                          | `None`    |
| dedup     | bool            | If True, deduplicate identical nodes and show references.                  | `True`    |
| max_depth | int \| None     | Maximum depth to render. Nodes beyond this depth are shown as "...".        | `None`    |

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description                                  |
|--------|--------|----------------------------------------------|
|        | Tree   | A Rich Tree object representing the lineage structure. |
""",
    "print_tree.qmd": r"""
Print a visual tree representation of a lineage node.

Builds and prints a Rich Tree object to visualize the structure of a
GenericNode and its children.

## Parameters {.doc-section .doc-section-parameters}

| Name      | Type            | Description                                                                 | Default   |
|-----------|-----------------|-----------------------------------------------------------------------------|-----------|
| node      | GenericNode     | The root node to build the tree from.                                       | _required_ |
| palette   | ColorScheme \| None | Optional color scheme for styling the tree nodes.                          | `None`    |
| dedup     | bool            | If True, deduplicate identical nodes and show references.                  | `True`    |
| max_depth | int \| None     | Maximum depth to render. Nodes beyond this depth are shown as "...".        | `None`    |

## Returns {.doc-section .doc-section-returns}

| Name   | Type   | Description                                  |
|--------|--------|----------------------------------------------|
|        | None   | This function prints to stdout and returns None. |
""",
}


def fix_broken_links(content: str) -> str:
    """Replace broken links with plain text."""
    for pattern, replacement in UNDOCUMENTED_TYPES.items():
        content = re.sub(pattern, replacement, content)
    return content


def remove_raw_docstrings(content: str) -> str:
    """Remove raw docstring markers (:param, :return:) and their content."""
    lines = content.split("\n")
    cleaned_lines = []
    skip_until_section = False

    for line in lines:
        stripped = line.strip()
        # Detect raw docstring markers
        if re.match(r":param\s+|:return:", stripped):
            skip_until_section = True
            continue
        # Skip lines until we hit a section header (##) or blank line followed by section
        if skip_until_section:
            if stripped.startswith("##") or (
                not stripped
                and cleaned_lines
                and cleaned_lines[-1].strip().startswith("##")
            ):
                skip_until_section = False
            elif not stripped:
                # Blank line - check next few lines to see if we're starting a section
                continue
            else:
                continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def is_empty_page(content: str) -> bool:
    """Check if a page is essentially empty (only has title and signature) or has raw docstring content."""
    # Check for raw docstring markers (Google-style or unformatted)
    if re.search(r":param\s+|:return:", content):
        return True

    # Remove code blocks to check actual content
    content_without_code = re.sub(r"```.*?```", "", content, flags=re.DOTALL)
    lines = [line.strip() for line in content_without_code.split("\n") if line.strip()]
    # Empty if only has title (# heading) and maybe a blank line or two
    # Allow up to 3 non-empty lines (title + maybe some whitespace)
    return len(lines) <= 3


def fix_missing_parameter_types(content: str, filename: str) -> str:
    """Fix missing types in parameter tables based on source code."""
    # Dictionary mapping filenames to their fixes
    fixes_by_file = {
        # Core Operations - missing expr parameters
        "execute.qmd": [
            (
                r"(\| Name   \| Type.*?\| Description.*?\| Default   \|\n\|--------\|.*?\|.*?\|.*?\|.*?\|\n)(\| kwargs \|)",
                r"\1| expr   | Expr                | The expression to execute. | _required_ |\n\2",
            ),
        ],
        "to_csv.qmd": [
            # Remove duplicate expr entries (keep only first occurrence)
            (
                r"(\| expr     \| Expr[^\n]*\| _required_ \|\n)(\| expr     \| Expr[^\n]*\| _required_ \|\n)+",
                r"\1",
            ),
            (r"\| https.*?\|\n", ""),  # Remove https line
            # Add Returns section if missing
            (
                r"(## Examples)",
                r"## Returns {.doc-section .doc-section-returns}\n\n| Name   | Type | Description |\n|--------|------|-------------|\n|        | None | This function writes to a file and returns None. |\n\n\1",
            ),
        ],
        "to_json.qmd": [
            # Fix the entire parameters section
            (
                r"(\| Name     \| Type.*?\| Description.*?\| Default    \|\n\|----------\|.*?\|.*?\|.*?\|.*?\|\n)(\| path     \|.*?Delta Lake table.*?\|\n\| \*\*kwargs \|.*?\|\n\| https    \|)",
                r"\1| expr     | Expr                                                                          | The expression to execute and write to NDJSON.             | _required_ |\n| path     | str \| Path \| TextIOWrapper | A string, Path, or file handle to the NDJSON file. | _required_ |\n| params   | Mapping\[Scalar, Any\] \| None | Mapping of scalar parameter expressions to value. | `None`     |",
            ),
            # Add Returns section if missing
            (
                r"(## Examples)",
                r"## Returns {.doc-section .doc-section-returns}\n\n| Name   | Type | Description |\n|--------|------|-------------|\n|        | None | This function writes to a file and returns None. |\n\n\1",
            ),
        ],
        "deferred_read_parquet.qmd": [
            # Fix [Any](`Any`) link
            (r"\[Any\]\(`Any`\)", r"Any"),
            # Fix the malformed Returns section
            (r"(\*\*kwargs.*?)\s+Returns\s+-+\s+Expr\s+An expression", r"\1"),
            # Add proper Returns section
            (
                r"(## Examples|## Notes)",
                r"## Returns {.doc-section .doc-section-returns}\n\n| Name   | Type | Description |\n|--------|------|-------------|\n|        | Expr | An expression representing the deferred read operation. |\n\n\1",
            ),
        ],
        "deferred_read_csv.qmd": [
            # Fix [Any](`Any`) link
            (r"\[Any\]\(`Any`\)", r"Any"),
        ],
        "to_pyarrow.qmd": [
            # Fix link in return type
            (r"\[Table\]\(`xorq\.vendor\.ibis\.expr\.api\.Table`\)", r"Table"),
            (
                r"(\| Name   \| Type.*?\| Description.*?\| Default   \|\n\|--------\|.*?\|.*?\|.*?\|.*?\|\n)(\| kwargs \|)",
                r"\1| expr   | Expr                | The expression to execute. | _required_ |\n\2",
            ),
        ],
        "to_parquet.qmd": [
            (
                r"(\| Name     \| Type.*?\| Description.*?\| Default    \|\n\|----------\|.*?\|.*?\|.*?\|.*?\|\n)(\| path     \|)",
                r"\1| expr     | Expr                                                                                                                                           | The expression to execute and write to Parquet.                     | _required_ |\n\2",
            ),
        ],
        "to_pyarrow_batches.qmd": [
            (
                r"(\| Name       \| Type.*?\| Description.*?\| Default   \|\n\|------------\|.*?\|.*?\|.*?\|.*?\|\n)(\| chunk_size \|)",
                r"\1| expr       | Expr                | The expression to execute.                             | _required_ |\n\2",
            ),
        ],
        "to_sql.qmd": [
            (
                r"\| compiler \|                                                                                \|",
                r"| compiler | Any \| None                                                                     |",
            ),
        ],
        # Types and schemas
        "datatypes.qmd": [
            (r"\| value    \|        \|", r"| value    | Any \|"),
            (r"\| nullable \|        \|", r"| nullable | bool \|"),
        ],
        # Data Operations
        "window.qmd": [
            (r"\| preceding \| int \| None \|", r"| preceding | int \| None |"),
            (r"\| following \| int \| None \|", r"| following | int \| None |"),
            (
                r"\| group_by  \| Expr \| tuple \| None \|",
                r"| group_by  | Expr \| tuple \| None |",
            ),
            (
                r"\| order_by  \| Expr \| tuple \| None \|",
                r"| order_by  | Expr \| tuple \| None |",
            ),
            (r"\| rows      \| bool \| None \|", r"| rows      | bool \| None |"),
            (r"\| range     \| bool \| None \|", r"| range     | bool \| None |"),
            (r"\| between   \| tuple \| None \|", r"| between   | tuple \| None |"),
        ],
        # Flight Operations
        "FlightServer.qmd": [
            (
                r"\| flight_url        \| FlightUrl \| None \|",
                r"| flight_url        | FlightUrl \| None |",
            ),
            (r"\| tls_certificates  \| tuple \|", r"| tls_certificates  | tuple |"),
            (r"\| verify_client     \| bool \|", r"| verify_client     | bool |"),
            (
                r"\| root_certificates \| bytes \| None \|",
                r"| root_certificates | bytes \| None |",
            ),
            (
                r"\| auth              \| [BasicAuth](`xorq\.flight\.BasicAuth`) \|",
                r"| auth              | BasicAuth \| None |",
            ),
            (
                r"\| make_connection   \| callable \|",
                r"| make_connection   | callable |",
            ),
            (r"\| exchangers        \| tuple \|", r"| exchangers        | tuple |"),
        ],
        "FlightUrl.qmd": [
            (r"\| scheme   \| str \|", r"| scheme   | str |"),
            (r"\| host     \| str \|", r"| host     | str |"),
            (r"\| username \| str \| None \|", r"| username | str \| None |"),
            (r"\| password \| str \| None \|", r"| password | str \| None |"),
            (r"\| port     \| int \| None \|", r"| port     | int \| None |"),
            (r"\| path     \| str \|", r"| path     | str |"),
            (r"\| query    \| str \|", r"| query    | str |"),
            (r"\| fragment \| str \|", r"| fragment | str |"),
        ],
        # Catalog Operations
        "Build.qmd": [
            (r"\| build_id \|        \|", r"| build_id | str \| None \|"),
            (r"\| path     \|        \|", r"| path     | Path \| None \|"),
        ],
        "Alias.qmd": [
            (r"\| entry_id    \|        \|", r"| entry_id    | str \|"),
            (r"\| revision_id \|        \|", r"| revision_id | str \| None \|"),
            (r"\| updated_at  \|        \|", r"| updated_at  | datetime \| None \|"),
        ],
        "CatalogMetadata.qmd": [
            (r"\| catalog_id   \|        \|", r"| catalog_id   | str \|"),
            (r"\| created_at   \|        \|", r"| created_at   | datetime \|"),
            (r"\| updated_at   \|        \|", r"| updated_at   | datetime \|"),
            (r"\| tool_version \|        \|", r"| tool_version | str \|"),
        ],
        # Caching
        "ParquetCache.qmd": [
            (
                r"\| source \| [ibis](`ibis`)\.[backends](`ibis\.backends`)\.[BaseBackend](`ibis\.backends\.BaseBackend`) \|",
                r"| source | Backend \|",
            ),
            (
                r"\| path   \| [Path](`Path`)                                                                        \|",
                r"| path   | Path \|",
            ),
        ],
        "ParquetSnapshotCache.qmd": [
            (
                r"\| source \| [ibis](`ibis`)\.[backends](`ibis\.backends`)\.[BaseBackend](`ibis\.backends\.BaseBackend`) \|",
                r"| source | Backend \|",
            ),
            (
                r"\| path   \| [Path](`Path`)                                                                        \|",
                r"| path   | Path \|",
            ),
        ],
    }

    # Apply fixes for this file
    if filename in fixes_by_file:
        for pattern, replacement in fixes_by_file[filename]:
            content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    return content


def add_manual_content(file_path: Path, content: str) -> str:
    """Add manual content to an empty page if configured."""
    filename = file_path.name

    if filename not in MANUAL_CONTENT:
        return content

    # Check if manual content already exists (to avoid duplicates)
    manual_content_start = MANUAL_CONTENT[filename].strip().split("\n")[0]
    if manual_content_start in content:
        # Manual content already exists, don't add again
        return content

    if is_empty_page(content):
        # Check if content has raw docstring markers - if so, replace everything after signature
        has_raw_docstring = bool(re.search(r":param\s+|:return:", content))

        # Find where to insert content (after the closing ``` of signature)
        if "```" in content:
            # Split by code blocks to find the signature block
            parts = content.split("```")
            if len(parts) >= 3:
                # parts[0] = before first ```, parts[1] = code content, parts[2] = after closing ```
                # Find the end of the signature block (after second ```)
                signature_end = "```".join(parts[:3])
                rest = "```".join(parts[3:]) if len(parts) > 3 else ""

                if has_raw_docstring:
                    # Completely replace everything after signature with manual content
                    return signature_end + MANUAL_CONTENT[filename] + "\n"
                else:
                    # Remove any trailing minimal description
                    rest_lines = rest.strip().split("\n")
                    non_empty_lines = [
                        line
                        for line in rest_lines
                        if line.strip() and not line.strip().startswith("#")
                    ]
                    # If minimal content, replace with manual content
                    if len(non_empty_lines) <= 2:
                        return signature_end + MANUAL_CONTENT[filename] + "\n"
                    else:
                        # Keep existing content, just append manual content
                        return signature_end + rest + MANUAL_CONTENT[filename]
        else:
            # No code block, append at end
            return content.rstrip() + "\n" + MANUAL_CONTENT[filename]

    return content


def process_file(file_path: Path) -> bool:
    """Process a single .qmd file: fix broken links, remove raw docstrings, add manual content, and fix missing parameter types."""
    try:
        content = file_path.read_text(encoding="utf-8")
        original_content = content

        # Remove raw docstrings first (before checking if empty)
        content = remove_raw_docstrings(content)

        # Fix broken links
        content = fix_broken_links(content)

        # Fix missing parameter types
        filename = file_path.name
        content = fix_missing_parameter_types(content, filename)

        # Add manual content if page is empty
        content = add_manual_content(file_path, content)

        if content != original_content:
            file_path.write_text(content, encoding="utf-8")
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Process all .qmd files in the reference directory (recursively)."""
    if not REFERENCE_DIR.exists():
        print(f"Reference directory not found: {REFERENCE_DIR}")
        return

    qmd_files = list(REFERENCE_DIR.rglob("*.qmd"))
    if not qmd_files:
        print(f"No .qmd files found in {REFERENCE_DIR}")
        return

    fixed_count = 0
    for qmd_file in qmd_files:
        if process_file(qmd_file):
            fixed_count += 1
            rel_path = qmd_file.relative_to(REFERENCE_DIR)
            print(f"Fixed: {rel_path}")

    if fixed_count > 0:
        print(f"\n✓ Processed {fixed_count} file(s) (fixed links and/or added content)")
    else:
        print("✓ No changes needed")


if __name__ == "__main__":
    main()
