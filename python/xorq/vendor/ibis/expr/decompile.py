from __future__ import annotations

import collections
import functools
import io
import itertools

import xorq.vendor.ibis.expr.datatypes as dt
import xorq.vendor.ibis.expr.operations as ops
import xorq.vendor.ibis.expr.types as ir
from xorq.vendor import ibis
from xorq.vendor.ibis.common.graph import Graph
from xorq.vendor.ibis.expr.rewrites import simplify
from xorq.vendor.ibis.util import experimental


_method_overrides = {
    ops.CountDistinct: "nunique",
    ops.CountStar: "count",
    ops.EndsWith: "endswith",
    ops.ExtractDay: "day",
    ops.ExtractDayOfYear: "day_of_year",
    ops.ExtractEpochSeconds: "epoch_seconds",
    ops.ExtractHour: "hour",
    ops.ExtractMicrosecond: "microsecond",
    ops.ExtractMillisecond: "millisecond",
    ops.ExtractMinute: "minute",
    ops.ExtractMinute: "minute",
    ops.ExtractMonth: "month",
    ops.ExtractQuarter: "quarter",
    ops.ExtractSecond: "second",
    ops.ExtractWeekOfYear: "week_of_year",
    ops.ExtractYear: "year",
    ops.Intersection: "intersect",
    ops.IsNull: "isnull",
    ops.Lowercase: "lower",
    ops.RegexSearch: "re_search",
    ops.StartsWith: "startswith",
    ops.StringContains: "contains",
    ops.StringSQLILike: "ilike",
    ops.StringSQLLike: "like",
    ops.TimestampNow: "now",
}


def _to_snake_case(camel_case):
    """Convert a camelCase string to snake_case."""
    result = list(camel_case[:1].lower())
    for char in camel_case[1:]:
        if char.isupper():
            result.append("_")
        result.append(char.lower())
    return "".join(result)


def _get_method_name(op):
    typ = op.__class__
    try:
        return _method_overrides[typ]
    except KeyError:
        return _to_snake_case(typ.__name__)


def _maybe_add_parens(op, string):
    if isinstance(op, ops.Binary):
        return f"({string})"
    elif isinstance(string, CallStatement):
        return string.args
    else:
        return string


class CallStatement:
    def __init__(self, func, args):
        self.func = func
        self.args = args

    def __str__(self):
        return f"{self.func}({self.args})"


@functools.singledispatch
def translate(op, *args, **kwargs):
    """Translate an ibis operation into a Python expression."""
    raise NotImplementedError(op)


@translate.register(ops.Value)
def value(op, *args, **kwargs):
    method = _get_method_name(op)
    kwargs = [(k, v) for k, v in kwargs.items() if v is not None]

    if args and kwargs:
        raise NotImplementedError(
            f"decompile does not support ops with both positional and keyword "
            f"arguments: {type(op).__name__} args={args} kwargs={kwargs}"
        )
    elif args:
        this, *args = args
    elif kwargs:
        (_, this), *kwargs = kwargs
    else:
        # Zero-argument analytics (row_number, rank, etc.)
        return f"ibis.{method}()"

    # if there is a single keyword argument prefer to pass that as positional
    if not args and len(kwargs) == 1:
        args = [kwargs[0][1]]
        kwargs = []

    args = ", ".join(map(str, args))
    kwargs = ", ".join(f"{k}={v}" for k, v in kwargs)
    parameters = ", ".join(filter(None, [args, kwargs]))

    return f"{this}.{method}({parameters})"


@translate.register(ops.ScalarParameter)
def scalar_parameter(op, dtype, counter):
    return f"ibis.param({str(dtype)!r})"


@translate.register(ops.UnboundTable)
@translate.register(ops.DatabaseTable)
def table(op, schema, name, **kwargs):
    fields = dict(zip(schema.names, map(str, schema.types)))
    return f"ibis.table(name={name!r}, schema={fields})"


def _extract_udxf_user_fn(udxf_cls):
    """Extract the user function from a UDXF class's exchange_f chain.

    Returns the function object or None.
    """
    try:
        ef = udxf_cls.exchange_f
        inner_partial = ef.func
        process_batch_partial = inner_partial.args[0]
        return process_batch_partial.args[0]
    except (AttributeError, IndexError, TypeError):
        return None


def _extract_udxf_comment(udxf_cls):
    """Extract a descriptive comment block from a UDXF class.

    Tries to include the full source if the file still exists on disk.
    Otherwise, reconstructs a summary from the code object's metadata:
    local variable names, imported modules, called methods, and string
    constants — enough to understand what the function does.
    """
    if udxf_cls is None:
        return ""

    user_fn = _extract_udxf_user_fn(udxf_cls)
    if user_fn is None:
        return ""

    code = user_fn.__code__
    arg_names = code.co_varnames[: code.co_argcount]
    sig = f"def {user_fn.__qualname__}({', '.join(arg_names)}):"
    lines = [f"# {sig}"]
    lines.append(f"#   Source: {code.co_filename}:{code.co_firstlineno}")

    # Try to read the actual source from disk
    import os

    src_file = code.co_filename
    if os.path.isfile(src_file):
        try:
            with open(src_file) as f:
                src_lines = f.readlines()
            start = code.co_firstlineno - 1
            body_lines = []
            for line in src_lines[start:]:
                body_lines.append(line.rstrip())
                if len(body_lines) > 1 and line.strip() and not line[0].isspace():
                    body_lines.pop()
                    break
            if body_lines:
                lines.append("#")
                for bl in body_lines:
                    lines.append(f"#   {bl}")
                return "\n".join(lines)
        except OSError:
            pass

    # Source file gone — flag as error and reconstruct from bytecode
    lines.append(f"#   ERROR: source file not found: {src_file}")
    lines.append("#   The original Python file has been deleted or moved.")
    lines.append("#   Below is a reconstruction from bytecode metadata.")
    lines.append("#")

    # Local variables (skip the arg names)
    locals_ = list(code.co_varnames[code.co_argcount :])
    if locals_:
        lines.append(f"#   Local variables: {', '.join(locals_)}")

    # Imports and method calls
    names = list(code.co_names)
    imports = [n for n in names if n == n.lower() and n.isidentifier() and len(n) > 2]
    methods = [n for n in names if n not in imports]
    if imports:
        lines.append(f"#   Uses: {', '.join(imports)}")
    if methods:
        lines.append(f"#   Calls: {', '.join(methods)}")

    # String constants (skip short/internal ones)
    str_consts = [
        c
        for c in code.co_consts
        if isinstance(c, str) and len(c) > 3 and not c.startswith("/")
    ]
    if str_consts:
        lines.append("#")
        lines.append("#   Key string literals (hints at logic):")
        for s in str_consts[:15]:
            lines.append(f"#     {s!r}")

    # Nested functions
    nested = [c for c in code.co_consts if hasattr(c, "co_name")]
    if nested:
        lines.append("#")
        for nc in nested:
            nc_args = nc.co_varnames[: nc.co_argcount]
            lines.append(f"#   Inner function: {nc.co_name}({', '.join(nc_args)})")

    return "\n".join(lines)


def _register_xorq_relation_handlers():
    """Register translate handlers for custom xorq relation types.

    These types inherit from DatabaseTable but carry extra fields (parent,
    remote_expr, input_expr, read_kwargs) that the generic handler discards.

    Called lazily from decompile() to avoid circular import issues —
    xorq.expr.relations is not importable when decompile.py first loads.
    """
    if _register_xorq_relation_handlers._done:
        return
    _register_xorq_relation_handlers._done = True
    try:
        from xorq.expr.relations import CachedNode, FlightUDXF, Read, RemoteTable
    except ImportError:
        return

    @translate.register(CachedNode)
    def cached_node(op, schema, name, parent=None, cache=None, **kwargs):
        fields = dict(zip(schema.names, map(str, schema.types)))
        if parent is not None:
            return f"{parent}.cache(name={name!r})  # schema={fields}"
        return f"ibis.table(name={name!r}, schema={fields})  # cached"

    @translate.register(RemoteTable)
    def remote_table(op, schema, name, remote_expr=None, **kwargs):
        if remote_expr is not None:
            # remote_expr is an ibis Table — recursively decompile to
            # valid Python, assigning intermediate results
            inner = decompile(
                remote_expr, render_import=False, assign_result_to="remote_expr"
            ).strip()
            # Return multi-line: the inner code block + final expression
            return f"{inner}\nremote_expr"
        fields = dict(zip(schema.names, map(str, schema.types)))
        return f"ibis.table(name={name!r}, schema={fields})  # remote"

    @translate.register(Read)
    def read(op, schema, name, method_name=None, read_kwargs=None, **kwargs):
        fields = dict(zip(schema.names, map(str, schema.types)))
        if method_name and read_kwargs:
            kw_parts = []
            for k, v in read_kwargs:
                kw_parts.append(f"{k}={v!r}")
            kw_str = ", ".join(kw_parts)
            return f"con.{method_name}({kw_str})  # {name!r}"
        return f"ibis.table(name={name!r}, schema={fields})  # read"

    @translate.register(FlightUDXF)
    def flight_udxf(op, schema, name, input_expr=None, udxf=None, **kwargs):
        fields = dict(zip(schema.names, map(str, schema.types)))
        udxf_cls = op.udxf  # raw class, not translated
        udxf_name = (
            getattr(udxf_cls, "__name__", "unknown") if udxf_cls else "unknown"
        )

        fn_comment = _extract_udxf_comment(udxf_cls)
        lines = []
        if fn_comment:
            lines.append(fn_comment)

        if input_expr is not None:
            # input_expr is an ibis Table — recursively decompile it
            inner = decompile(
                input_expr, render_import=False, assign_result_to="input_data"
            ).strip()
            lines.append(inner)
            lines.append("")
            lines.append(
                f"# UDXF: {udxf_name} — takes a DataFrame, returns a DataFrame"
            )
            lines.append(f"# Output: {fields}")
            lines.append(f"input_data.pipe({udxf_name})")
        else:
            lines.append(
                f"ibis.table(name={name!r}, schema={fields})"
                f"  # flight_udxf({udxf_name!r})"
            )
        return "\n".join(lines)


_register_xorq_relation_handlers._done = False


def _try_unwrap(stmt):
    if len(stmt) == 1:
        return stmt[0]
    else:
        stmt = map(str, stmt)
        values = ", ".join(stmt)
        return f"[{values}]"


def _wrap_alias(values, rendered):
    result = []
    for k, v in values.items():
        text = rendered[k]
        if v.name != k:
            if isinstance(v, ops.Binary):
                text = f"({text}).name({k!r})"
            else:
                text = f"{text}.name({k!r})"
        result.append(text)
    return result


def _inline(args):
    return ", ".join(map(str, args))


@translate.register(ops.Project)
def project(op, parent, values):
    out = f"{parent}"
    if not values:
        return out

    values = _wrap_alias(op.values, values)
    return f"{out}.select({_inline(values)})"


@translate.register(ops.Filter)
def filter_(op, parent, predicates):
    out = f"{parent}"
    if predicates:
        out = f"{out}.filter({_inline(predicates)})"
    return out


@translate.register(ops.Sort)
def sort(op, parent, keys):
    out = f"{parent}"
    if keys:
        out = f"{out}.order_by({_inline(keys)})"
    return out


@translate.register(ops.Aggregate)
def aggregation(op, parent, groups, metrics):
    groups = _wrap_alias(op.groups, groups)
    metrics = _wrap_alias(op.metrics, metrics)
    if groups and metrics:
        return f"{parent}.aggregate([{_inline(metrics)}], by=[{_inline(groups)}])"
    elif metrics:
        return f"{parent}.aggregate([{_inline(metrics)}])"
    else:
        raise ValueError("No metrics to aggregate")


@translate.register(ops.Distinct)
def distinct(op, parent):
    return f"{parent}.distinct()"


@translate.register(ops.DropColumns)
def drop(op, parent, columns_to_drop):
    return f"{parent}.drop({_inline(map(repr, columns_to_drop))})"


@translate.register(ops.SelfReference)
def self_reference(op, parent, identifier):
    return f"{parent}.view()"


@translate.register(ops.JoinReference)
def join_reference(op, parent, identifier):
    return parent


@translate.register(ops.JoinLink)
def join_link(op, table, predicates, how):
    return f".{how}_join({table}, {_try_unwrap(predicates)})"


@translate.register(ops.JoinChain)
def join(op, first, rest, values):
    calls = "".join(rest)
    pieces = [f"{first}{calls}"]
    if values:
        values = _wrap_alias(op.values, values)
        pieces.append(f"select({_inline(values)})")
    result = ".".join(pieces)
    return result


@translate.register(ops.Set)
def union(op, left, right, distinct):
    method = _get_method_name(op)
    if distinct:
        return f"{left}.{method}({right}, distinct=True)"
    else:
        return f"{left}.{method}({right})"


@translate.register(ops.Limit)
def limit(op, parent, n, offset):
    if offset:
        return f"{parent}.limit({n}, {offset})"
    else:
        return f"{parent}.limit({n})"


@translate.register(ops.Field)
def table_column(op, rel, name):
    if name.isidentifier():
        return f"{rel}.{name}"
    return f"{rel}[{name!r}]"


@translate.register(ops.SortKey)
def sort_key(op, expr, ascending, nulls_first):
    method = "asc" if ascending else "desc"
    call = f"{expr}.{method}"
    if nulls_first:
        return f"{call}(nulls_first={nulls_first})"
    return f"{call}()"


@translate.register(ops.Reduction)
def reduction(op, arg, where, **kwargs):
    method = _get_method_name(op)
    return f"{arg}.{method}()"


@translate.register(ops.Alias)
def alias(op, arg, name):
    arg = _maybe_add_parens(op.arg, arg)
    return f"{arg}.name({name!r})"


@translate.register(ops.Constant)
def constant(op, **kwargs):
    method = _get_method_name(op)
    return f"ibis.{method}()"


@translate.register(ops.Literal)
def literal(op, value, dtype):
    inferred = ibis.literal(value)

    if isinstance(op.dtype, dt.Timestamp):
        return f'ibis.timestamp("{value}")'
    elif isinstance(op.dtype, dt.Date):
        return f"ibis.date({value!r})"
    elif isinstance(op.dtype, dt.Interval):
        return f"ibis.interval({value!r})"
    elif inferred.type() != op.dtype:
        return CallStatement("ibis.literal", f"{value!r}, {dtype}")
    else:
        # prefer plain python literal values if the inferred datatype is the same,
        # though this makes rendering method calls on literals more complicated
        return CallStatement("ibis.literal", repr(value))


@translate.register(ops.Cast)
def cast(op, arg, to):
    return f"{arg}.cast({str(to)!r})"


@translate.register(ops.Between)
def between(op, arg, lower_bound, upper_bound):
    return f"{arg}.between({lower_bound}, {upper_bound})"


@translate.register(ops.IfElse)
def ifelse(op, bool_expr, true_expr, false_null_expr):
    return f"{bool_expr}.ifelse({true_expr}, {false_null_expr})"


@translate.register(ops.SimpleCase)
@translate.register(ops.SearchedCase)
def switch_case(op, cases, results, default, base=None):
    out = f"{base}.case()" if base else "ibis.case()"

    for case, result in zip(cases, results):
        out = f"{out}.when({case}, {result})"

    if default is not None:
        out = f"{out}.else_({default})"

    return f"{out}.end()"


_infix_ops = {
    ops.Equals: "==",
    ops.NotEquals: "!=",
    ops.GreaterEqual: ">=",
    ops.Greater: ">",
    ops.LessEqual: "<=",
    ops.Less: "<",
    ops.And: "and",
    ops.Or: "or",
    ops.Add: "+",
    ops.Subtract: "-",
    ops.Multiply: "*",
    ops.Divide: "/",
    ops.Power: "**",
    ops.Modulus: "%",
    ops.TimestampAdd: "+",
    ops.TimestampSub: "-",
    ops.TimestampDiff: "-",
}


@translate.register(ops.Binary)
def binary(op, left, right):
    operator = _infix_ops[type(op)]
    left = _maybe_add_parens(op.left, left)
    right = _maybe_add_parens(op.right, right)
    return _maybe_add_parens(op, f"{left} {operator} {right}")


@translate.register(ops.InValues)
def isin(op, value, options):
    return f"{value}.isin(({', '.join([str(option) for option in options])}))"


class CodeContext:
    always_assign = (
        ops.ScalarParameter,
        ops.Aggregate,
        ops.PhysicalTable,
        ops.SelfReference,
    )

    always_ignore = (
        ops.JoinReference,
        ops.Field,
        dt.Primitive,
        dt.Variadic,
        dt.Temporal,
    )
    shorthands = {
        ops.Aggregate: "agg",
        ops.Literal: "lit",
        ops.ScalarParameter: "param",
        ops.Project: "p",
        ops.Relation: "r",
        ops.Filter: "f",
        ops.Sort: "s",
    }

    def __init__(self, assign_result_to="result"):
        self.assign_result_to = assign_result_to
        self._shorthand_counters = collections.defaultdict(itertools.count)

    def variable_for(self, node):
        klass = type(node)
        if isinstance(node, ops.Relation) and hasattr(node, "name"):
            name = node.name
        elif klass in self.shorthands:
            name = self.shorthands[klass]
        else:
            name = klass.__name__.lower()

        # increment repeated type names: table, table1, table2, ...
        nth = next(self._shorthand_counters[name]) or ""
        return f"{name}{nth}"

    def _split_block(self, code):
        """Split multi-line code into (preamble, final_expr).

        If code contains newlines, the last non-empty line is the
        expression; preceding lines are a preamble block (comments,
        intermediate assignments) that must be emitted before the
        assignment.
        """
        code_str = str(code)
        lines = code_str.split("\n")
        if len(lines) <= 1:
            return ("", code_str)
        # Find the last non-empty line as the expression
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines:
            return ("", code_str)
        final_expr = lines.pop()
        preamble = "\n".join(lines)
        if preamble:
            preamble += "\n"
        return (preamble, final_expr)

    def render(self, node, code, n_dependents):
        isroot = n_dependents == 0
        ignore = isinstance(node, self.always_ignore)
        assign = n_dependents > 1 or isinstance(node, self.always_assign)

        # depending on the conditions return with (output code, node result) pairs
        if not code:
            return (None, None)
        elif isroot:
            preamble, final_expr = self._split_block(code)
            if self.assign_result_to:
                out = f"{preamble}\n{self.assign_result_to} = {final_expr}\n"
            else:
                out = f"{preamble}{final_expr}"
            return (out, final_expr)
        elif ignore:
            return (None, code)
        elif assign:
            var = self.variable_for(node)
            preamble, final_expr = self._split_block(code)
            out = f"{preamble}{var} = {final_expr}\n"
            return (out, var)
        else:
            return (None, code)


@experimental
def decompile(
    expr: ir.Expr,
    render_import: bool = True,
    assign_result_to: str = "result",
    format: bool = False,
) -> str:
    """Decompile an ibis expression into Python source code.

    Parameters
    ----------
    expr
        node or expression to decompile
    render_import
        Whether to add `import ibis` to the result.
    assign_result_to
        Variable name to store the result at, pass None to avoid assignment.
    format
        Whether to format the generated code using black code formatter.

    Returns
    -------
    str
        Equivalent Python source code for `node`.

    """
    if not isinstance(expr, ir.Expr):
        raise TypeError(f"Expected ibis expression, got {type(expr).__name__}")

    _register_xorq_relation_handlers()

    node = expr.op()
    node = simplify(node)
    out = io.StringIO()
    ctx = CodeContext(assign_result_to=assign_result_to)
    dependents = Graph(node).invert()

    def fn(node, _, *args, **kwargs):
        code = translate(node, *args, **kwargs)
        n_dependents = len(dependents[node])

        code, result = ctx.render(node, code, n_dependents)
        if code:
            out.write(code)

        return result

    node.map(fn)

    result = out.getvalue()
    if render_import:
        result = f"import ibis\n\n\n{result}"

    if format:
        try:
            import black
        except ImportError:
            raise ImportError(
                "The 'format' option requires the 'black' package to be installed"
            )

        result = black.format_str(result, mode=black.FileMode())

    return result
