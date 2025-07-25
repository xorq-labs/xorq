from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from public import public

import xorq.vendor.ibis.expr.operations as ops
from xorq.vendor.ibis.common.deferred import Deferred, deferrable
from xorq.vendor.ibis.expr.types.generic import Column, Scalar, Value


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import xorq.common.exceptions as com
    import xorq.vendor.ibis.expr.types as ir
    from xorq.vendor.ibis.expr.types.typing import V


@public
class ArrayValue(Value):
    """An Array is a variable-length sequence of values of a single type.

    Examples
    --------
    >>> import ibis
    >>> ibis.options.interactive = True
    >>> ibis.memtable({"a": [[1, None, 3], [4], [], None]})
    ┏━━━━━━━━━━━━━━━━━━━━━━┓
    ┃ a                    ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━┩
    │ array<int64>         │
    ├──────────────────────┤
    │ [1, None, ... +1]    │
    │ [4]                  │
    │ []                   │
    │ NULL                 │
    └──────────────────────┘
    """

    def length(self) -> ir.IntegerValue:
        """Compute the length of an array.

        Returns
        -------
        IntegerValue
            The integer length of each element of `self`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"a": [[7, 42], [3], None]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ a                    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [7, 42]              │
        │ [3]                  │
        │ NULL                 │
        └──────────────────────┘
        >>> t.a.length()
        ┏━━━━━━━━━━━━━━━━┓
        ┃ ArrayLength(a) ┃
        ┡━━━━━━━━━━━━━━━━┩
        │ int64          │
        ├────────────────┤
        │              2 │
        │              1 │
        │           NULL │
        └────────────────┘
        """
        return ops.ArrayLength(self).to_expr()

    def __getitem__(self, index: int | ir.IntegerValue | slice) -> ir.Value:
        """Extract one or more elements of `self`.

        Parameters
        ----------
        index
            Index into `array`

        Returns
        -------
        Value
            - If `index` is an [](`int`) or
              [`IntegerValue`](./expression-numeric.qmd#ibis.expr.types.IntegerValue)
              then the return type is the element type of `self`.
            - If `index` is a [](`slice`) then the return type is the same
              type as the input.

        Examples
        --------
        Extract a single element

        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"a": [[7, 42], [3], None]})
        >>> t.a[0]
        ┏━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayIndex(a, 0) ┃
        ┡━━━━━━━━━━━━━━━━━━┩
        │ int64            │
        ├──────────────────┤
        │                7 │
        │                3 │
        │             NULL │
        └──────────────────┘

        Extract a range of elements

        >>> t = ibis.memtable({"a": [[7, 42, 72], [3] * 5, None]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ a                    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [7, 42, ... +1]      │
        │ [3, 3, ... +3]       │
        │ NULL                 │
        └──────────────────────┘
        >>> t.a[1:2]
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArraySlice(a, 1, 2)  ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [42]                 │
        │ [3]                  │
        │ NULL                 │
        └──────────────────────┘
        """
        if isinstance(index, slice):
            start = index.start
            stop = index.stop
            step = index.step

            if step is not None and step != 1:
                raise NotImplementedError("step can only be 1")

            op = ops.ArraySlice(self, start if start is not None else 0, stop)
        else:
            op = ops.ArrayIndex(self, index)
        return op.to_expr()

    def concat(self, other: ArrayValue, *args: ArrayValue) -> ArrayValue:
        """Concatenate this array with one or more arrays.

        Parameters
        ----------
        other
            Other array to concat with `self`
        args
            Other arrays to concat with `self`

        Returns
        -------
        ArrayValue
            `self` concatenated with `other` and `args`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"a": [[7], [3], None]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ a                    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [7]                  │
        │ [3]                  │
        │ NULL                 │
        └──────────────────────┘
        >>> t.a.concat(t.a)
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayConcat((a, a))  ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [7, 7]               │
        │ [3, 3]               │
        │ NULL                 │
        └──────────────────────┘
        >>> t.a.concat(ibis.literal([4], type="array<int64>"))
        ┏━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayConcat((a, (4,))) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>           │
        ├────────────────────────┤
        │ [7, 4]                 │
        │ [3, 4]                 │
        │ [4]                    │
        └────────────────────────┘

        `concat` is also available using the `+` operator

        >>> [1] + t.a
        ┏━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayConcat(((1,), a)) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>           │
        ├────────────────────────┤
        │ [1, 7]                 │
        │ [1, 3]                 │
        │ [1]                    │
        └────────────────────────┘
        >>> t.a + [1]
        ┏━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayConcat((a, (1,))) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>           │
        ├────────────────────────┤
        │ [7, 1]                 │
        │ [3, 1]                 │
        │ [1]                    │
        └────────────────────────┘
        """
        return ops.ArrayConcat((self, other, *args)).to_expr()

    def __add__(self, other: ArrayValue) -> ArrayValue:
        return self.concat(other)

    def __radd__(self, other: ArrayValue) -> ArrayValue:
        return ops.ArrayConcat((other, self)).to_expr()

    def repeat(self, n: int | ir.IntegerValue) -> ArrayValue:
        """Repeat this array `n` times.

        Parameters
        ----------
        n
            Number of times to repeat `self`.

        Returns
        -------
        ArrayValue
            `self` repeated `n` times

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"a": [[7], [3], None]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ a                    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [7]                  │
        │ [3]                  │
        │ NULL                 │
        └──────────────────────┘
        >>> t.a.repeat(2)
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayRepeat(a, 2)    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [7, 7]               │
        │ [3, 3]               │
        │ []                   │
        └──────────────────────┘

        `repeat` is also available using the `*` operator

        >>> 2 * t.a
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayRepeat(a, 2)    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [7, 7]               │
        │ [3, 3]               │
        │ []                   │
        └──────────────────────┘
        """
        return ops.ArrayRepeat(self, n).to_expr()

    __mul__ = __rmul__ = repeat

    def unnest(self) -> ir.Value:
        """Unnest an array into a column.

        ::: {.callout-note}
        ## Empty arrays and `NULL`s are dropped in the output.
        To preserve empty arrays as `NULL`s as well as existing `NULL` values,
        use [`Table.unnest`](./expression-tables.qmd#ibis.expr.types.relations.Table.unnest).
        :::

        Returns
        -------
        ir.Value
            Unnested array

        See Also
        --------
        [`Table.unnest`](./expression-tables.qmd#ibis.expr.types.relations.Table.unnest)

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"a": [[7, 42], [3, 3], None]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ a                    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [7, 42]              │
        │ [3, 3]               │
        │ NULL                 │
        └──────────────────────┘
        >>> t.a.unnest()
        ┏━━━━━━━┓
        ┃ a     ┃
        ┡━━━━━━━┩
        │ int64 │
        ├───────┤
        │     7 │
        │    42 │
        │     3 │
        │     3 │
        └───────┘
        """
        expr = ops.Unnest(self).to_expr()
        try:
            return expr.name(self.get_name())
        except com.ExpressionError:
            return expr

    def join(self, sep: str | ir.StringValue) -> ir.StringValue:
        """Join the elements of this array expression with `sep`.

        Parameters
        ----------
        sep
            Separator to use for joining array elements

        Returns
        -------
        StringValue
            Elements of `self` joined with `sep`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"arr": [["a", "b", "c"], None, [], ["b", None]]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ arr                  ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<string>        │
        ├──────────────────────┤
        │ ['a', 'b', ... +1]   │
        │ NULL                 │
        │ []                   │
        │ ['b', None]          │
        └──────────────────────┘
        >>> t.arr.join("|")
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayStringJoin(arr, '|') ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ string                    │
        ├───────────────────────────┤
        │ a|b|c                     │
        │ NULL                      │
        │ NULL                      │
        │ b                         │
        └───────────────────────────┘

        See Also
        --------
        [`StringValue.join`](./expression-strings.qmd#ibis.expr.types.strings.StringValue.join)
        """
        return ops.ArrayStringJoin(self, sep=sep).to_expr()

    def map(self, func: Deferred | Callable[[ir.Value], ir.Value]) -> ir.ArrayValue:
        """Apply a `func` or `Deferred` to each element of this array expression.

        Parameters
        ----------
        func
            Function or `Deferred` to apply to each element of this array.

        Returns
        -------
        ArrayValue
            `func` applied to every element of this array expression.

        Examples
        --------
        >>> import ibis
        >>> from ibis import _
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"a": [[1, None, 2], [4], []]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ a                    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [1, None, ... +1]    │
        │ [4]                  │
        │ []                   │
        └──────────────────────┘

        The most succinct way to use `map` is with `Deferred` expressions:

        >>> t.a.map((_ + 100).cast("float"))
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayMap(a, Cast(Add(_, 100), float64)) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<float64>                          │
        ├─────────────────────────────────────────┤
        │ [101.0, None, ... +1]                   │
        │ [104.0]                                 │
        │ []                                      │
        └─────────────────────────────────────────┘

        You can also use `map` with a lambda function:

        >>> t.a.map(lambda x: (x + 100).cast("float"))
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayMap(a, Cast(Add(x, 100), float64)) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<float64>                          │
        ├─────────────────────────────────────────┤
        │ [101.0, None, ... +1]                   │
        │ [104.0]                                 │
        │ []                                      │
        └─────────────────────────────────────────┘

        `.map()` also supports more complex callables like `functools.partial`
        and lambdas with closures

        >>> from functools import partial
        >>> def add(x, y):
        ...     return x + y
        >>> add2 = partial(add, y=2)
        >>> t.a.map(add2)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayMap(a, Add(x, 2)) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>           │
        ├────────────────────────┤
        │ [3, None, ... +1]      │
        │ [6]                    │
        │ []                     │
        └────────────────────────┘
        >>> y = 2
        >>> t.a.map(lambda x: x + y)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayMap(a, Add(x, 2)) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>           │
        ├────────────────────────┤
        │ [3, None, ... +1]      │
        │ [6]                    │
        │ []                     │
        └────────────────────────┘
        """
        if isinstance(func, Deferred):
            name = "_"
            resolve = func.resolve
        elif callable(func):
            name = next(iter(inspect.signature(func).parameters.keys()))
            resolve = func
        else:
            raise TypeError(
                f"`func` must be a Deferred or Callable, got `{type(func).__name__}`"
            )

        parameter = ops.Argument(
            name=name, shape=self.op().shape, dtype=self.type().value_type
        )
        body = resolve(parameter.to_expr())
        return ops.ArrayMap(self, param=parameter.param, body=body).to_expr()

    def filter(
        self, predicate: Deferred | Callable[[ir.Value], bool | ir.BooleanValue]
    ) -> ir.ArrayValue:
        """Filter array elements using `predicate` function or `Deferred`.

        Parameters
        ----------
        predicate
            Function or `Deferred` to use to filter array elements

        Returns
        -------
        ArrayValue
            Array elements filtered using `predicate`

        Examples
        --------
        >>> import ibis
        >>> from ibis import _
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"a": [[1, None, 2], [4], []]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ a                    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [1, None, ... +1]    │
        │ [4]                  │
        │ []                   │
        └──────────────────────┘

        The most succinct way to use `filter` is with `Deferred` expressions:

        >>> t.a.filter(_ > 1)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayFilter(a, Greater(_, 1)) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>                  │
        ├───────────────────────────────┤
        │ [2]                           │
        │ [4]                           │
        │ []                            │
        └───────────────────────────────┘

        You can also use `map` with a lambda function:

        >>> t.a.filter(lambda x: x > 1)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayFilter(a, Greater(x, 1)) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>                  │
        ├───────────────────────────────┤
        │ [2]                           │
        │ [4]                           │
        │ []                            │
        └───────────────────────────────┘

        `.filter()` also supports more complex callables like `functools.partial`
        and lambdas with closures

        >>> from functools import partial
        >>> def gt(x, y):
        ...     return x > y
        >>> gt1 = partial(gt, y=1)
        >>> t.a.filter(gt1)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayFilter(a, Greater(x, 1)) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>                  │
        ├───────────────────────────────┤
        │ [2]                           │
        │ [4]                           │
        │ []                            │
        └───────────────────────────────┘
        >>> y = 1
        >>> t.a.filter(lambda x: x > y)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayFilter(a, Greater(x, 1)) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>                  │
        ├───────────────────────────────┤
        │ [2]                           │
        │ [4]                           │
        │ []                            │
        └───────────────────────────────┘
        """
        if isinstance(predicate, Deferred):
            name = "_"
            resolve = predicate.resolve
        elif callable(predicate):
            name = next(iter(inspect.signature(predicate).parameters.keys()))
            resolve = predicate
        else:
            raise TypeError(
                f"`predicate` must be a Deferred or Callable, got `{type(predicate).__name__}`"
            )
        parameter = ops.Argument(
            name=name,
            shape=self.op().shape,
            dtype=self.type().value_type,
        )
        body = resolve(parameter.to_expr())
        return ops.ArrayFilter(self, param=parameter.param, body=body).to_expr()

    def contains(self, other: ir.Value) -> ir.BooleanValue:
        """Return whether the array contains `other`.

        Parameters
        ----------
        other
            Ibis expression to check for existence of in `self`

        Returns
        -------
        BooleanValue
            Whether `other` is contained in `self`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"arr": [[1], [], [42, 42], None]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ arr                  ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [1]                  │
        │ []                   │
        │ [42, 42]             │
        │ NULL                 │
        └──────────────────────┘
        >>> t.arr.contains(42)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayContains(arr, 42) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ boolean                │
        ├────────────────────────┤
        │ False                  │
        │ False                  │
        │ True                   │
        │ NULL                   │
        └────────────────────────┘
        >>> t.arr.contains(None)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayContains(arr, None) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ boolean                  │
        ├──────────────────────────┤
        │ NULL                     │
        │ NULL                     │
        │ NULL                     │
        │ NULL                     │
        └──────────────────────────┘
        """
        return ops.ArrayContains(self, other).to_expr()

    def index(self, other: ir.Value) -> ir.IntegerValue:
        """Return the position of `other` in an array.

        Parameters
        ----------
        other
            Ibis expression to existence of in `self`

        Returns
        -------
        BooleanValue
            The position of `other` in `self`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"arr": [[1], [], [42, 42], None]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ arr                  ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [1]                  │
        │ []                   │
        │ [42, 42]             │
        │ NULL                 │
        └──────────────────────┘
        >>> t.arr.index(42)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayPosition(arr, 42) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ int64                  │
        ├────────────────────────┤
        │                     -1 │
        │                     -1 │
        │                      0 │
        │                   NULL │
        └────────────────────────┘
        >>> t.arr.index(800)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayPosition(arr, 800) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ int64                   │
        ├─────────────────────────┤
        │                      -1 │
        │                      -1 │
        │                      -1 │
        │                    NULL │
        └─────────────────────────┘
        >>> t.arr.index(None)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayPosition(arr, None) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ int64                    │
        ├──────────────────────────┤
        │                     NULL │
        │                     NULL │
        │                     NULL │
        │                     NULL │
        └──────────────────────────┘
        """
        return ops.ArrayPosition(self, other).to_expr()

    def remove(self, other: ir.Value) -> ir.ArrayValue:
        """Remove `other` from `self`.

        Parameters
        ----------
        other
            Element to remove from `self`.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"arr": [[3, 2], [], [42, 2], [2, 2], None]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ arr                  ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [3, 2]               │
        │ []                   │
        │ [42, 2]              │
        │ [2, 2]               │
        │ NULL                 │
        └──────────────────────┘
        >>> t.arr.remove(2)
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayRemove(arr, 2)  ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [3]                  │
        │ []                   │
        │ [42]                 │
        │ []                   │
        │ NULL                 │
        └──────────────────────┘
        """
        return ops.ArrayRemove(self, other).to_expr()

    def unique(self) -> ir.ArrayValue:
        """Return the unique values in an array.

        ::: {.callout-note}
        ## Element ordering in array may not be retained.
        :::

        Returns
        -------
        ArrayValue
            Unique values in an array

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"arr": [[1, 3, 3], [], [42, 42, None], None]})
        >>> t.arr.unique()
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayDistinct(arr)   ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [3, 1]               │
        │ []                   │
        │ [42, None]           │
        │ NULL                 │
        └──────────────────────┘
        """
        return ops.ArrayDistinct(self).to_expr()

    def sort(self) -> ir.ArrayValue:
        """Sort the elements in an array.

        Returns
        -------
        ArrayValue
            Sorted values in an array

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"arr": [[3, 2], [], [42, 42], None]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ arr                  ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [3, 2]               │
        │ []                   │
        │ [42, 42]             │
        │ NULL                 │
        └──────────────────────┘
        >>> t.arr.sort()
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArraySort(arr)       ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ [2, 3]               │
        │ []                   │
        │ [42, 42]             │
        │ NULL                 │
        └──────────────────────┘
        """
        return ops.ArraySort(self).to_expr()

    def union(self, other: ir.ArrayValue) -> ir.ArrayValue:
        """Union two arrays.

        Parameters
        ----------
        other
            Another array to union with `self`

        Returns
        -------
        ArrayValue
            Unioned arrays

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"arr1": [[3, 2], [], None], "arr2": [[1, 3], [None], [5]]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ arr1                 ┃ arr2                 ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │ array<int64>         │
        ├──────────────────────┼──────────────────────┤
        │ [3, 2]               │ [1, 3]               │
        │ []                   │ [None]               │
        │ NULL                 │ [5]                  │
        └──────────────────────┴──────────────────────┘
        >>> t.arr1.union(t.arr2)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayUnion(arr1, arr2) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>           │
        ├────────────────────────┤
        │ [1, 2, ... +1]         │
        │ [None]                 │
        │ [5]                    │
        └────────────────────────┘
        >>> t.arr1.union(t.arr2).contains(3)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayContains(ArrayUnion(arr1, arr2), 3) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ boolean                                  │
        ├──────────────────────────────────────────┤
        │ True                                     │
        │ False                                    │
        │ False                                    │
        └──────────────────────────────────────────┘
        """
        return ops.ArrayUnion(self, other).to_expr()

    def intersect(self, other: ArrayValue) -> ArrayValue:
        """Intersect two arrays.

        Parameters
        ----------
        other
            Another array to intersect with `self`

        Returns
        -------
        ArrayValue
            Intersected arrays

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"arr1": [[3, 2], [], None], "arr2": [[1, 3], [None], [5]]})
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ arr1                 ┃ arr2                 ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │ array<int64>         │
        ├──────────────────────┼──────────────────────┤
        │ [3, 2]               │ [1, 3]               │
        │ []                   │ [None]               │
        │ NULL                 │ [5]                  │
        └──────────────────────┴──────────────────────┘
        >>> t.arr1.intersect(t.arr2)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayIntersect(arr1, arr2) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>               │
        ├────────────────────────────┤
        │ [3]                        │
        │ []                         │
        │ NULL                       │
        └────────────────────────────┘
        """
        return ops.ArrayIntersect(self, other).to_expr()

    def zip(self, other: ArrayValue, *others: ArrayValue) -> ArrayValue:
        """Zip two or more arrays together.

        Parameters
        ----------
        other
            Another array to zip with `self`
        others
            Additional arrays to zip with `self`

        Returns
        -------
        Array
            Array of structs where each struct field is an element of each input
            array. The fields are named `f1`, `f2`, `f3`, etc.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> ibis.options.repr.interactive.max_depth = 2
        >>> t = ibis.memtable(
        ...     {
        ...         "numbers": [[3, 2], [6, 7], [], None],
        ...         "strings": [["a", "c"], ["d"], [], ["x", "y"]],
        ...     }
        ... )
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ numbers              ┃ strings              ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │ array<string>        │
        ├──────────────────────┼──────────────────────┤
        │ [3, 2]               │ ['a', 'c']           │
        │ [6, 7]               │ ['d']                │
        │ []                   │ []                   │
        │ NULL                 │ ['x', 'y']           │
        └──────────────────────┴──────────────────────┘
        >>> t.numbers.zip(t.strings)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayZip((numbers, strings))                  ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<struct<f1: int64, f2: string>>          │
        ├───────────────────────────────────────────────┤
        │ [{'f1': 3, 'f2': 'a'}, {'f1': 2, 'f2': 'c'}]  │
        │ [{'f1': 6, 'f2': 'd'}, {'f1': 7, 'f2': None}] │
        │ []                                            │
        │ NULL                                          │
        └───────────────────────────────────────────────┘
        """

        return ops.ArrayZip((self, other, *others)).to_expr()

    def flatten(self) -> ir.ArrayValue:
        """Remove one level of nesting from an array expression.

        Returns
        -------
        ArrayValue
            Flattened array expression

        Examples
        --------
        >>> import ibis
        >>> import ibis.selectors as s
        >>> from ibis import _
        >>> ibis.options.interactive = True
        >>> schema = {
        ...     "empty": "array<array<int>>",
        ...     "happy": "array<array<string>>",
        ...     "nulls_only": "array<array<struct<a: array<string>>>>",
        ...     "mixed_nulls": "array<array<string>>",
        ... }
        >>> data = {
        ...     "empty": [[], [], []],
        ...     "happy": [[["abc"]], [["bcd"]], [["def"]]],
        ...     "nulls_only": [None, None, None],
        ...     "mixed_nulls": [[], None, [None]],
        ... }
        >>> import pyarrow as pa
        >>> t = ibis.memtable(
        ...     pa.Table.from_pydict(
        ...         data,
        ...         schema=ibis.schema(schema).to_pyarrow(),
        ...     )
        ... )
        >>> t
        ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━┓
        ┃ empty                ┃ happy                ┃ nulls_only ┃ … ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━┩
        │ array<array<int64>>  │ array<array<string>> │ array<arr… │ … │
        ├──────────────────────┼──────────────────────┼────────────┼───┤
        │ []                   │ [[...]]              │ NULL       │ … │
        │ []                   │ [[...]]              │ NULL       │ … │
        │ []                   │ [[...]]              │ NULL       │ … │
        └──────────────────────┴──────────────────────┴────────────┴───┘
        >>> t.empty.flatten()
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayFlatten(empty)  ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<int64>         │
        ├──────────────────────┤
        │ []                   │
        │ []                   │
        │ []                   │
        └──────────────────────┘
        >>> t.happy.flatten()
        ┏━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayFlatten(happy)  ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<string>        │
        ├──────────────────────┤
        │ ['abc']              │
        │ ['bcd']              │
        │ ['def']              │
        └──────────────────────┘
        >>> t.nulls_only.flatten()
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayFlatten(nulls_only) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<struct<a: array<s… │
        ├──────────────────────────┤
        │ NULL                     │
        │ NULL                     │
        │ NULL                     │
        └──────────────────────────┘
        >>> t.mixed_nulls.flatten()
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ ArrayFlatten(mixed_nulls) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ array<string>             │
        ├───────────────────────────┤
        │ []                        │
        │ NULL                      │
        │ []                        │
        └───────────────────────────┘
        >>> t.select(s.across(s.all(), _.flatten()))
        ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━┓
        ┃ empty                ┃ happy                ┃ nulls_only ┃ … ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━┩
        │ array<int64>         │ array<string>        │ array<str… │ … │
        ├──────────────────────┼──────────────────────┼────────────┼───┤
        │ []                   │ ['abc']              │ NULL       │ … │
        │ []                   │ ['bcd']              │ NULL       │ … │
        │ []                   │ ['def']              │ NULL       │ … │
        └──────────────────────┴──────────────────────┴────────────┴───┘
        """
        return ops.ArrayFlatten(self).to_expr()

    def anys(self) -> ir.BooleanValue:
        """Return whether any element in the array is true.

        Returns NULL if the array is empty or contains only NULLs.

        See Also
        --------
        [`BooleanColumn.any`](./expression-numeric.qmd#ibis.expr.types.logical.BooleanColumn.any)

        Returns
        -------
        BooleanValue
            Whether any element in the array is true

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable(
        ...     {
        ...         "arr": [
        ...             [True, False],
        ...             [False],
        ...             [True],
        ...             [None, False],
        ...             [None, True],
        ...             [None],
        ...             [],
        ...             None,
        ...         ]
        ...     }
        ... )
        >>> t.mutate(x=t.arr.anys())
        ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
        ┃ arr                  ┃ x       ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
        │ array<boolean>       │ boolean │
        ├──────────────────────┼─────────┤
        │ [True, False]        │ True    │
        │ [False]              │ False   │
        │ [True]               │ True    │
        │ [None, False]        │ False   │
        │ [None, True]         │ True    │
        │ [None]               │ NULL    │
        │ []                   │ NULL    │
        │ NULL                 │ NULL    │
        └──────────────────────┴─────────┘
        """
        return ops.ArrayAny(self).to_expr()

    def alls(self) -> ir.BooleanValue:
        """Return whether all elements (ignoring nulls) in the array are true.

        Returns NULL if the array is empty or contains only NULLs.

        See Also
        --------
        [`BooleanColumn.all`](./expression-numeric.qmd#ibis.expr.types.logical.BooleanColumn.all)

        Returns
        -------
        BooleanValue
            Whether all elements (ignoring nulls) in the array are true.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable(
        ...     {
        ...         "id": range(8),
        ...         "arr": [
        ...             [True, False],
        ...             [False],
        ...             [True],
        ...             [None, False],
        ...             [None, True],
        ...             [None],
        ...             [],
        ...             None,
        ...         ],
        ...     }
        ... )
        >>> t.mutate(x=t.arr.alls()).order_by("id")
        ┏━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
        ┃ id    ┃ arr                  ┃ x       ┃
        ┡━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
        │ int64 │ array<boolean>       │ boolean │
        ├───────┼──────────────────────┼─────────┤
        │     0 │ [True, False]        │ False   │
        │     1 │ [False]              │ False   │
        │     2 │ [True]               │ True    │
        │     3 │ [None, False]        │ False   │
        │     4 │ [None, True]         │ True    │
        │     5 │ [None]               │ NULL    │
        │     6 │ []                   │ NULL    │
        │     7 │ NULL                 │ NULL    │
        └───────┴──────────────────────┴─────────┘
        """
        return ops.ArrayAll(self).to_expr()

    def mins(self) -> ir.NumericValue:
        """Return the minimum value in the array.

        Returns NULL if the array is empty or contains only NULLs.

        See Also
        --------
        [`Column.min`](./expression-generic.qmd#ibis.expr.types.generic.Column.min)

        Returns
        -------
        Value
            Minimum value in the array

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"arr": [[1, 2, 3], [None, 6], [None], [], None]})
        >>> t.mutate(x=t.arr.mins())
        ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
        ┃ arr                  ┃ x     ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
        │ array<int64>         │ int64 │
        ├──────────────────────┼───────┤
        │ [1, 2, ... +1]       │     1 │
        │ [None, 6]            │     6 │
        │ [None]               │  NULL │
        │ []                   │  NULL │
        │ NULL                 │  NULL │
        └──────────────────────┴───────┘
        """
        return ops.ArrayMin(self).to_expr()

    def maxs(self) -> ir.NumericValue:
        """Return the maximum value in the array.

        Returns NULL if the array is empty or contains only NULLs.

        See Also
        --------
        [`Column.max`](./expression-generic.qmd#ibis.expr.types.generic.Column.max)

        Returns
        -------
        Value
            Maximum value in the array

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"arr": [[1, 2, 3], [None, 6], [None], [], None]})
        >>> t.mutate(x=t.arr.maxs())
        ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
        ┃ arr                  ┃ x     ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
        │ array<int64>         │ int64 │
        ├──────────────────────┼───────┤
        │ [1, 2, ... +1]       │     3 │
        │ [None, 6]            │     6 │
        │ [None]               │  NULL │
        │ []                   │  NULL │
        │ NULL                 │  NULL │
        └──────────────────────┴───────┘
        """
        return ops.ArrayMax(self).to_expr()

    def sums(self) -> ir.NumericValue:
        """Return the sum of the values in the array.

        Returns NULL if the array is empty or contains only NULLs.

        See Also
        --------
        [`NumericColumn.sum`](./expression-numeric.qmd#ibis.expr.types.numeric.NumericColumn.sum)

        Returns
        -------
        Value
            Sum of the values in the array

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"arr": [[1, 2, 3], [None, 6], [None], [], None]})
        >>> t.mutate(x=t.arr.sums())
        ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
        ┃ arr                  ┃ x     ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
        │ array<int64>         │ int64 │
        ├──────────────────────┼───────┤
        │ [1, 2, ... +1]       │     6 │
        │ [None, 6]            │     6 │
        │ [None]               │  NULL │
        │ []                   │  NULL │
        │ NULL                 │  NULL │
        └──────────────────────┴───────┘
        """
        return ops.ArraySum(self).to_expr()

    def means(self) -> ir.FloatingValue:
        """Return the mean of the values in the array.

        Returns NULL if the array is empty or contains only NULLs.

        See Also
        --------
        [`NumericColumn.mean`](./expression-numeric.qmd#ibis.expr.types.numeric.NumericColumn.mean)

        Returns
        -------
        Value
            Mean of the values in the array

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable({"arr": [[1, 2, 3], [None, 6], [None], [], None]})
        >>> t.mutate(x=t.arr.means())
        ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
        ┃ arr                  ┃ x       ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
        │ array<int64>         │ float64 │
        ├──────────────────────┼─────────┤
        │ [1, 2, ... +1]       │     2.0 │
        │ [None, 6]            │     6.0 │
        │ [None]               │    NULL │
        │ []                   │    NULL │
        │ NULL                 │    NULL │
        └──────────────────────┴─────────┘
        """
        return ops.ArrayMean(self).to_expr()


@public
class ArrayScalar(Scalar, ArrayValue):
    pass


@public
class ArrayColumn(Column, ArrayValue):
    def __getitem__(self, index: int | ir.IntegerValue | slice) -> ir.Column:
        return ArrayValue.__getitem__(self, index)


@public
@deferrable
def array(values: Iterable[V]) -> ArrayValue:
    """Create an array expression.

    If any values are [column expressions](../concepts/datatypes.qmd) the
    result will be a column. Otherwise the result will be a
    [scalar](../concepts/datatypes.qmd).

    Parameters
    ----------
    values
        An iterable of Ibis expressions or Python literals

    Returns
    -------
    ArrayValue

    Examples
    --------
    Create an array scalar from scalar values

    >>> import ibis
    >>> ibis.options.interactive = True
    >>> ibis.array([1.0, None])
    ┌─────────────┐
    │ [1.0, None] │
    └─────────────┘

    Create an array from column and scalar expressions

    >>> t = ibis.memtable({"a": [1, 2, 3], "b": [4, 5, 6]})
    >>> ibis.array([t.a, 42, ibis.literal(None)])
    ┏━━━━━━━━━━━━━━━━━━━━━━┓
    ┃ Array((a, 42, None)) ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━┩
    │ array<int64>         │
    ├──────────────────────┤
    │ [1, 42, ... +1]      │
    │ [2, 42, ... +1]      │
    │ [3, 42, ... +1]      │
    └──────────────────────┘

    >>> ibis.array([t.a, 42 + ibis.literal(5)])
    ┏━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃ Array((a, Add(5, 42))) ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━┩
    │ array<int64>           │
    ├────────────────────────┤
    │ [1, 47]                │
    │ [2, 47]                │
    │ [3, 47]                │
    └────────────────────────┘
    """
    return ops.Array(tuple(values)).to_expr()
