import functools
import operator

import xorq.expr.datatypes as dt
import xorq.vendor.ibis.expr.operations as ops


EXACT_IN_FLOAT64 = (
    dt.Float32,
    dt.Int8,
    dt.Int16,
    dt.Int32,
    dt.UInt8,
    dt.UInt16,
    dt.UInt32,
)


def combine(left, right, sign):
    if left is None or right is None:
        return None
    (left_const, left_terms), (right_const, right_terms) = left, right
    atoms = dict.fromkeys((*left_terms, *right_terms))
    return (
        left_const + sign * right_const,
        {
            atom: left_terms.get(atom, 0.0) + sign * right_terms.get(atom, 0.0)
            for atom in atoms
        },
    )


def scale(linear_form, factor):
    if linear_form is None:
        return None
    const, terms = linear_form
    return const * factor, {atom: coef * factor for atom, coef in terms.items()}


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def linear(value):
    """`value` as (const, {atom: coef}), or None when it is not linear.

    Atoms are float64 fields (or exact widenings to float64), which
    propagate nulls, and 0/1 indicators (one-hot columns), which never are
    null.
    """
    match value:
        case ops.Literal(value=number) if is_number(number):
            return float(number), {}
        # an exact widening of a field to float64 is an atom; a float64 or
        # indicator cast is transparent; any other cast may round or truncate
        case ops.Cast(arg=ops.Field() as field, to=to) if isinstance(
            to, dt.Float64
        ) and isinstance(field.dtype, EXACT_IN_FLOAT64):
            return 0.0, {value: 1.0}
        case ops.Cast(arg=arg, to=to) if isinstance(to, dt.Float64) and (
            isinstance(arg.dtype, dt.Float64) or isinstance(arg, ops.IfElse)
        ):
            return linear(arg)
        case ops.Add(left=left, right=right):
            return combine(linear(left), linear(right), 1)
        case ops.Subtract(left=left, right=right):
            return combine(linear(left), linear(right), -1)
        case (
            ops.Multiply(left=ops.Literal(value=factor), right=arg)
            | ops.Multiply(left=arg, right=ops.Literal(value=factor))
        ) if is_number(factor):
            return scale(linear(arg), factor)
        case ops.FloorDivide():
            return None
        case ops.Divide(left=arg, right=ops.Literal(value=divisor)) if (
            is_number(divisor) and divisor
        ):
            return scale(linear(arg), 1 / divisor)
        case ops.Negate(arg=arg):
            return scale(linear(arg), -1)
        # float64 only: integer arithmetic must stay exact, float32 rounds
        case ops.Field(dtype=dt.Float64()):
            return 0.0, {value: 1.0}
        case ops.IfElse(
            true_expr=ops.Literal(value=1), false_null_expr=ops.Literal(value=0)
        ):
            return 0.0, {value: 1.0}
        case _:
            return None


def distribute(metric, base):
    """A Sum or Mean of a linear form as per-atom reductions over `base`.

    Every atom is summed over one shared mask, the rows where no field atom
    is null: exactly the rows where the row-level value is not null.
    """
    match metric:
        # float metrics only: an integer sum must stay exact
        case ops.Mean(arg=arg, where=None) | ops.Sum(arg=arg, where=None) if (
            metric.dtype.is_floating() and not isinstance(arg, ops.Field)
        ):
            linear_form = linear(arg)
        case _:
            return None
    if linear_form is None:
        return None
    const, terms = linear_form
    fields = tuple(
        field.to_expr()
        for field in dict.fromkeys(
            field
            for atom in terms
            if not isinstance(atom, ops.IfElse)
            for field in atom.find(ops.Field, filter=ops.Value)
        )
    )
    mask = (
        functools.reduce(operator.and_, (field.notnull() for field in fields))
        if fields
        else None
    )
    count = base.to_expr().count(where=mask)
    total = sum(
        (coef * atom.to_expr().sum(where=mask) for atom, coef in terms.items()),
        const * count,
    )
    value = total / count if isinstance(metric, ops.Mean) else total
    return value.cast(metric.dtype).op()


# Project values that change the row count or are not one value per row:
# removing the Project under them would change what the aggregate reads.
NOT_ROW_PRESERVING = (ops.Unnest, ops.WindowFunction, ops.Analytic, ops.Reduction)


def preserves_rows(values):
    return all(
        value.shape.is_columnar()
        and not value.find(NOT_ROW_PRESERVING, filter=ops.Value)
        for value in values.values()
    )


def push_aggregate(node):
    """Aggregate over a scoring Project, rewritten over the Project's parent
    with linear Sum/Mean metrics distributed; None when nothing distributes
    or when a metric cannot move below the Project.
    """
    match node:
        case ops.Aggregate(
            parent=ops.Project(parent=base, values=values) as project
        ) if preserves_rows(values):
            pass
        case _:
            return None

    def dereference(value):
        fields = value.find(ops.Field, filter=ops.Value)
        return value.replace(
            {field: values[field.name] for field in fields if field.rel == project},
            filter=ops.Value,
        )

    def rebind(metric):
        match metric:
            case ops.CountStar(where=where):
                return ops.CountStar(base, where and dereference(where))
            # a reduction over the Project itself (other than CountStar) stays put
            case ops.Reduction() if not any(
                isinstance(arg, ops.Relation) for arg in metric.args
            ):
                return dereference(metric)
            case _:
                return None

    metrics = {name: rebind(metric) for name, metric in node.metrics.items()}
    if None in metrics.values():
        return None
    distributed = {name: distribute(metric, base) for name, metric in metrics.items()}
    if not any(distributed.values()):
        return None
    return ops.Aggregate(
        base,
        {name: dereference(group) for name, group in node.groups.items()},
        {name: distributed[name] or metric for name, metric in metrics.items()},
    )
