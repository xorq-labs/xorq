from __future__ import annotations

import contextlib
import os
import webbrowser
from typing import TYPE_CHECKING, Any, NoReturn

from attr import (
    field,
    frozen,
)
from attr.validators import (
    instance_of,
)
from public import public

import xorq.vendor.ibis.expr.operations as ops
from xorq.common.exceptions import TranslationError, XorqError
from xorq.vendor import ibis
from xorq.vendor.ibis.common.annotations import ValidationError
from xorq.vendor.ibis.common.grounds import Immutable
from xorq.vendor.ibis.common.patterns import Coercible, CoercionError
from xorq.vendor.ibis.common.typing import get_defining_scope
from xorq.vendor.ibis.config import options as opts
from xorq.vendor.ibis.expr.format import pretty
from xorq.vendor.ibis.util import deprecated, experimental


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from rich.console import Console, RenderableType

    import xorq.vendor.ibis.expr.types as ir
    from xorq.vendor.ibis.backends import BaseBackend
    from xorq.vendor.ibis.expr.visualize import (
        EdgeAttributeGetter,
        NodeAttributeGetter,
    )


try:
    from rich.jupyter import JupyterMixin
except ImportError:

    class _FixedTextJupyterMixin:
        """No-op when rich is not installed."""
else:

    class _FixedTextJupyterMixin(JupyterMixin):
        """JupyterMixin adds a spurious newline to text, this fixes the issue."""

        def _repr_mimebundle_(self, *args, **kwargs):
            try:
                bundle = super()._repr_mimebundle_(*args, **kwargs)
            except Exception:  # noqa: BLE001
                return None
            else:
                bundle["text/plain"] = bundle["text/plain"].rstrip()
                return bundle


def _capture_rich_renderable(renderable: RenderableType) -> str:
    from rich.console import Console

    console = Console(force_terminal=False)
    with console.capture() as capture:
        console.print(renderable)
    return capture.get().rstrip()


@public
class Expr(Immutable, Coercible):
    """Base expression class."""

    __slots__ = ("_arg",)
    _arg: ops.Node

    def _noninteractive_repr(self) -> str:
        if ibis.options.repr.show_variables:
            scope = get_defining_scope(self, types=Expr)
        else:
            scope = None
        return pretty(self.op(), scope=scope)

    def __rich_console__(self, console: Console, options):
        from rich.text import Text

        if console.is_jupyter:
            # Rich infers a console width in jupyter notebooks, but since
            # notebooks can use horizontal scroll bars we don't want to apply a
            # limit here. Since rich requires an integer for max_width, we
            # choose an arbitrarily large integer bound. Note that we need to
            # handle this here rather than in `to_rich`, as this setting
            # also needs to be forwarded to `console.render`.
            options = options.update(max_width=1_000_000)
            console_width = None
        else:
            console_width = options.max_width

        try:
            if opts.interactive:
                from xorq.vendor.ibis.expr.types.pretty import to_rich

                rich_object = to_rich(self, console_width=console_width)
            else:
                rich_object = Text(self._noninteractive_repr())
        except TranslationError as e:
            lines = [
                "Translation to backend failed",
                f"Error message: {e!r}",
                "Expression repr follows:",
                self._noninteractive_repr(),
            ]
            return Text("\n".join(lines))
        return console.render(rich_object, options=options)

    def __repr__(self):
        import xorq as xo

        if xo.options.interactive:
            return _capture_rich_renderable(self)
        else:
            return self._noninteractive_repr()

    def __init__(self, arg: ops.Node) -> None:
        object.__setattr__(self, "_arg", arg)

    def __iter__(self) -> NoReturn:
        raise TypeError(f"{self.__class__.__name__!r} object is not iterable")

    @classmethod
    def __coerce__(cls, value):
        if isinstance(value, cls):
            return value
        elif isinstance(value, ops.Node):
            return value.to_expr()
        else:
            raise CoercionError("Unable to coerce value to an expression")

    def __reduce__(self):
        return (self.__class__, (self._arg,))

    def __hash__(self):
        return hash((self.__class__, self._arg))

    def equals(self, other):
        """Return whether this expression is _structurally_ equivalent to `other`.

        If you want to produce an equality expression, use `==` syntax.

        Parameters
        ----------
        other
            Another expression

        Examples
        --------
        >>> import xorq as xo
        >>> t1 = xo.table(dict(a="int"), name="t")
        >>> t2 = xo.table(dict(a="int"), name="t")
        >>> t1.equals(t2)
        True
        >>> v = xo.table(dict(a="string"), name="v")
        >>> t1.equals(v)
        False
        """
        if not isinstance(other, Expr):
            raise TypeError(
                f"invalid equality comparison between Expr and {type(other)}"
            )
        return self._arg.equals(other._arg)

    def __bool__(self) -> bool:
        raise ValueError("The truth value of an Ibis expression is not defined")

    __nonzero__ = __bool__

    @deprecated(
        instead="remove any usage of `has_name`, since it is always `True`",
        as_of="9.4",
        removed_in="10.0",
    )
    def has_name(self):
        """Check whether this expression has an explicit name."""
        return hasattr(self._arg, "name")

    def get_name(self):
        """Return the name of this expression."""
        return self._arg.name

    def _repr_png_(self) -> bytes | None:
        if opts.interactive or not opts.graphviz_repr:
            return None
        try:
            import xorq.vendor.ibis.expr.visualize as viz
        except ImportError:
            return None
        else:
            # Something may go wrong, and we can't error in the notebook
            # so fallback to the default text representation.
            with contextlib.suppress(Exception):
                return viz.to_graph(self).pipe(format="png")

    def visualize(
        self,
        format: str = "svg",
        *,
        label_edges: bool = False,
        verbose: bool = False,
        node_attr: Mapping[str, str] | None = None,
        node_attr_getter: NodeAttributeGetter | None = None,
        edge_attr: Mapping[str, str] | None = None,
        edge_attr_getter: EdgeAttributeGetter | None = None,
    ) -> None:
        """Visualize an expression as a GraphViz graph in the browser.

        Parameters
        ----------
        format
            Image output format. These are specified by the `graphviz` Python
            library.
        label_edges
            Show operation input names as edge labels
        verbose
            Print the graphviz DOT code to stderr if [](`True`)
        node_attr
            Mapping of `(attribute, value)` pairs set for all nodes.
            Options are specified by the `graphviz` Python library.
        node_attr_getter
            Callback taking a node and returning a mapping of `(attribute, value)` pairs
            for that node. Options are specified by the `graphviz` Python library.
        edge_attr
            Mapping of `(attribute, value)` pairs set for all edges.
            Options are specified by the `graphviz` Python library.
        edge_attr_getter
            Callback taking two adjacent nodes and returning a mapping of `(attribute, value)` pairs
            for the edge between those nodes. Options are specified by the `graphviz` Python library.

        Examples
        --------
        Open the visualization of an expression in default browser:

        >>> import xorq as xo
        >>> import xorq.vendor.ibis.expr.operations as ops
        >>> left = ibis.table(dict(a="int64", b="string"), name="left")
        >>> right = ibis.table(dict(b="string", c="int64", d="string"), name="right")
        >>> expr = left.inner_join(right, "b").select(left.a, b=right.c, c=right.d)
        >>> expr.visualize(
        ...     format="svg",
        ...     label_edges=True,
        ...     node_attr={"fontname": "Roboto Mono", "fontsize": "10"},
        ...     node_attr_getter=lambda node: isinstance(node, ops.Field) and {"shape": "oval"},
        ...     edge_attr={"fontsize": "8"},
        ...     edge_attr_getter=lambda u, v: isinstance(u, ops.Field) and {"color": "red"},
        ... )  # quartodoc: +SKIP # doctest: +SKIP

        Raises
        ------
        ImportError
            If `graphviz` is not installed.
        """
        import xorq.vendor.ibis.expr.visualize as viz

        path = viz.draw(
            viz.to_graph(
                self,
                node_attr=node_attr,
                node_attr_getter=node_attr_getter,
                edge_attr=edge_attr,
                edge_attr_getter=edge_attr_getter,
                label_edges=label_edges,
            ),
            format=format,
            verbose=verbose,
        )
        webbrowser.open(f"file://{os.path.abspath(path)}")

    def pipe(self, f, *args: Any, **kwargs: Any) -> Expr:
        """Compose `f` with `self`.

        Parameters
        ----------
        f
            If the expression needs to be passed as anything other than the
            first argument to the function, pass a tuple with the argument
            name. For example, (f, 'data') if the function f expects a 'data'
            keyword
        args
            Positional arguments to `f`
        kwargs
            Keyword arguments to `f`

        Examples
        --------
        >>> import xorq as xo
        >>> xo.options.interactive = False
        >>> t = xo.table([("a", "int64"), ("b", "string")], name="t")
        >>> f = lambda a: (a + 1).name("a")
        >>> g = lambda a: (a * 2).name("a")
        >>> result1 = t.a.pipe(f).pipe(g)
        >>> result1
        r0 := UnboundTable: t
          a int64
          b string
        <BLANKLINE>
        a: r0.a + 1 * 2

        >>> result2 = g(f(t.a))  # equivalent to the above
        >>> result1.equals(result2)
        True

        Returns
        -------
        Expr
            Result type of passed function
        """
        if isinstance(f, tuple):
            f, data_keyword = f
            kwargs = kwargs.copy()
            kwargs[data_keyword] = self
            return f(*args, **kwargs)
        else:
            return f(self, *args, **kwargs)

    def op(self) -> ops.Node:
        return self._arg

    def _find_backends(self) -> tuple[list[BaseBackend], bool]:
        """Return the possible backends for an expression.

        Returns
        -------
        list[BaseBackend]
            A list of the backends found.
        """

        import xorq.expr.relations as rel

        backends = set()
        has_unbound = False
        node_types = (
            ops.UnboundTable,
            ops.DatabaseTable,
            ops.SQLQueryResult,
            rel.CachedNode,
            rel.Read,
        )
        for table in self.op().find(node_types):
            if isinstance(table, ops.UnboundTable):
                has_unbound = True
            else:
                backends.add(table.source)

        return list(backends), has_unbound

    def _find_backend_original(self, *, use_default: bool = False) -> BaseBackend:
        """Find the backend attached to an expression.

        Parameters
        ----------
        use_default
            If [](`True`) and the default backend isn't set, initialize the
            default backend and use that. This should only be set to `True` for
            `.execute()`. For other contexts such as compilation, this option
            doesn't make sense so the default value is [](`False`).

        Returns
        -------
        BaseBackend
            A backend that is attached to the expression
        """
        from xorq.config import _backend_init

        backends, has_unbound = self._find_backends()

        if not backends:
            if has_unbound:
                raise XorqError(
                    "Expression contains unbound tables and therefore cannot "
                    "be executed. Use `<backend>.execute(expr)` to execute "
                    "against an explicit backend, or rebuild the expression "
                    "using bound tables instead."
                )
            default = _backend_init() if use_default else None
            if default is None:
                raise XorqError(
                    "Expression depends on no backends, and found no default"
                )
            return default

        if len(backends) > 1:
            raise XorqError("Multiple backends found for this expression")

        return backends[0]

    def _find_backend(self, *, use_default=True):
        from xorq.config import _backend_init

        try:
            current_backend = self._find_backend_original(use_default=use_default)
        except XorqError as e:
            if "Multiple backends found" in e.args[0]:
                current_backend = _backend_init()
            else:
                raise e
        return current_backend

    def into_backend(self, con, name=None):
        """
        Converts the Expr to a table in the given backend `con` with an optional table name `name`.

        The table is backed by a PyArrow RecordBatchReader, the RecordBatchReader is teed
        so it can safely be reaused without spilling to disk.

        Parameters
        ----------
        con
            The backend where the table should be created
        name
            The name of the table

        Examples
        -------
        >>> import xorq as xo
        >>> from xorq import _
        >>> xo.options.interactive = True
        >>> ls_con = xo.connect()
        >>> pg_con = xo.postgres.connect_examples()
        >>> t = pg_con.table("batting").into_backend(ls_con, "ls_batting")
        >>> expr = (
        ...     t.join(t, "playerID")
        ...     .order_by("playerID", "yearID")
        ...     .limit(15)
        ...     .select(player_id="playerID", year_id="yearID_right")
        ... )
        >>> expr
        ┏━━━━━━━━━━━┳━━━━━━━━━┓
        ┃ player_id ┃ year_id ┃
        ┡━━━━━━━━━━━╇━━━━━━━━━┩
        │ string    │ int64   │
        ├───────────┼─────────┤
        │ aardsda01 │    2015 │
        │ aardsda01 │    2007 │
        │ aardsda01 │    2006 │
        │ aardsda01 │    2009 │
        │ aardsda01 │    2008 │
        │ aardsda01 │    2010 │
        │ aardsda01 │    2004 │
        │ aardsda01 │    2013 │
        │ aardsda01 │    2012 │
        │ aardsda01 │    2006 │
        │ …         │       … │
        └───────────┴─────────┘
        """

        from xorq.expr.relations import RemoteTable

        return RemoteTable.from_expr(con=con, expr=self, name=name).to_expr()

    def compile(
        self,
        limit: int | None = None,
        params: Mapping[ir.Value, Any] | None = None,
        pretty: bool = False,
    ):
        """Compile to an execution target.

        Parameters
        ----------
        limit
            An integer to effect a specific row limit. A value of `None` means
            "no limit". The default is in `ibis/config.py`.
        params
            Mapping of scalar parameter expressions to value
        pretty
            In case of SQL backends, return a pretty formatted SQL query.
        """
        return self._find_backend().compile(
            self, limit=limit, params=params, pretty=pretty
        )

    def execute(self: ir.Expr, **kwargs: Any):
        """Execute an expression against its backend if one exists.

        Parameters
        ----------
        kwargs
            Keyword arguments

        Examples
        --------
        >>> import xorq as xo
        >>> t = xo.examples.penguins.fetch()
        >>> t.execute()
               species     island  bill_length_mm  ...  body_mass_g     sex  year
        0       Adelie  Torgersen            39.1  ...       3750.0    male  2007
        1       Adelie  Torgersen            39.5  ...       3800.0  female  2007
        2       Adelie  Torgersen            40.3  ...       3250.0  female  2007
        3       Adelie  Torgersen             NaN  ...          NaN    None  2007
        4       Adelie  Torgersen            36.7  ...       3450.0  female  2007
        ..         ...        ...             ...  ...          ...     ...   ...
        339  Chinstrap      Dream            55.8  ...       4000.0    male  2009
        340  Chinstrap      Dream            43.5  ...       3400.0  female  2009
        341  Chinstrap      Dream            49.6  ...       3775.0    male  2009
        342  Chinstrap      Dream            50.8  ...       4100.0    male  2009
        343  Chinstrap      Dream            50.2  ...       3775.0  female  2009
        [344 rows x 8 columns]

        Scalar parameters can be supplied dynamically during execution.
        >>> species = xo.param("string")
        >>> expr = t.filter(t.species == species).order_by(t.bill_length_mm)
        >>> expr.execute(limit=3, params={species: "Gentoo"})
          species  island  bill_length_mm  ...  body_mass_g     sex  year
        0  Gentoo  Biscoe            40.9  ...         4650  female  2007
        1  Gentoo  Biscoe            41.7  ...         4700  female  2009
        2  Gentoo  Biscoe            42.0  ...         4150  female  2007
        <BLANKLINE>
        [3 rows x 8 columns]
        """
        from xorq.expr.api import execute

        return execute(self, **kwargs)

    def to_pyarrow_batches(
        self: ir.Expr,
        *,
        chunk_size: int = 1_000_000,
        **kwargs: Any,
    ):
        """Execute expression and return a RecordBatchReader.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        chunk_size
            Maximum number of rows in each returned record batch.
        kwargs
            Keyword arguments

        Returns
        -------
        results
            RecordBatchReader
        """
        from xorq.expr.api import to_pyarrow_batches

        return to_pyarrow_batches(self, chunk_size=chunk_size, **kwargs)

    def to_pyarrow(self: ir.Expr, **kwargs: Any):
        """Execute expression and return results in as a pyarrow table.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        kwargs
            Keyword arguments

        Returns
        -------
        Table
            A pyarrow table holding the results of the executed expression.
        """

        from xorq.expr.api import to_pyarrow

        return to_pyarrow(self, **kwargs)

    def to_parquet(
        self: ir.Expr,
        path: str | Path,
        params: Mapping[ir.Scalar, Any] | None = None,
        **kwargs: Any,
    ):
        """Write the results of executing the given expression to a parquet file.

        This method is eager and will execute the associated expression
        immediately.

        See https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetWriter.html for details.

        Parameters
        ----------
        path
            A string or Path where the Parquet file will be written.
        params
            Mapping of scalar parameter expressions to value.
        **kwargs
            Additional keyword arguments passed to pyarrow.parquet.ParquetWriter

        Examples
        --------
        Write out an expression to a single parquet file.

        >>> import ibis
        >>> import tempfile
        >>> penguins = ibis.examples.penguins.fetch()
        >>> penguins.to_parquet(tempfile.mktemp())
        """
        from xorq.expr.api import to_parquet

        return to_parquet(self, path=path, params=params, **kwargs)

    @experimental
    def to_csv(
        self,
        path: str | Path,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Write the results of executing the given expression to a CSV file.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        path
            The data source. A string or Path to the CSV file.
        params
            Mapping of scalar parameter expressions to value.
        **kwargs
            Additional keyword arguments passed to pyarrow.csv.CSVWriter

        https://arrow.apache.org/docs/python/generated/pyarrow.csv.CSVWriter.html
        """
        from xorq.expr.api import to_csv

        return to_csv(self, path=path, params=params, **kwargs)

    @experimental
    def to_json(
        self,
        path: str | Path,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Write the results of `expr` to a NDJSON file.

        This method is eager and will execute the associated expression
        immediately.

        Parameters
        ----------
        path
            The data source. A string or Path to the Delta Lake table.
        **kwargs
            Additional, backend-specific keyword arguments.

        https://github.com/ndjson/ndjson-spec
        """
        from xorq.expr.api import to_json

        return to_json(self, path=path, params=params)

    def unbind(self) -> ir.Table:
        """Return an expression built on `UnboundTable` instead of backend-specific objects."""
        from xorq.vendor.ibis.expr.rewrites import _, d, p

        rule = p.DatabaseTable >> d.UnboundTable(
            name=_.name, schema=_.schema, namespace=_.namespace
        )
        return self.op().replace(rule).to_expr()

    def as_table(self) -> ir.Table:
        """Convert an expression to a table."""
        raise NotImplementedError(
            f"{type(self)} expressions cannot be converted into tables"
        )

    def as_scalar(self) -> ir.Scalar:
        """Convert an expression to a scalar."""
        raise NotImplementedError(
            f"{type(self)} expressions cannot be converted into scalars"
        )

    @property
    def ls(self):
        return LETSQLAccessor(self)


@frozen
class LETSQLAccessor:
    expr = field(validator=instance_of(Expr))
    node_types = (ops.DatabaseTable, ops.SQLQueryResult)

    @property
    def op(self):
        return self.expr.op()

    @property
    def cached_nodes(self):
        from xorq.common.utils.graph_utils import walk_nodes
        from xorq.expr.relations import (
            CachedNode,
        )

        return walk_nodes((CachedNode,), self.expr)

    @property
    def storage(self):
        if self.is_cached:
            return self.op.storage
        else:
            return None

    @property
    def storages(self):
        return tuple(node.storage for node in self.cached_nodes)

    @property
    def backends(self):
        from xorq.common.utils.graph_utils import (
            find_all_sources,
        )

        return find_all_sources(self.expr)

    @property
    def is_multiengine(self):
        (_, *rest) = set(self.backends)
        return bool(rest)

    @property
    def dts(self):
        from xorq.common.utils.graph_utils import (
            walk_nodes,
        )
        from xorq.expr.relations import (
            RemoteTable,
        )

        return tuple(
            el
            for el in walk_nodes(self.node_types, self.expr)
            if not isinstance(el, RemoteTable)
        )

    @property
    def is_cached(self):
        from xorq.expr.relations import (
            CachedNode,
        )

        return isinstance(self.op, CachedNode)

    @property
    def has_cached(self):
        return bool(self.cached_nodes)

    @property
    def uncached(self):
        from xorq.expr.relations import (
            legacy_replace_cache_table,
        )

        if self.has_cached:
            op = self.expr.op()
            return op.map_clear(legacy_replace_cache_table).to_expr()
        else:
            return self.expr

    @property
    def uncached_one(self):
        if self.is_cached:
            from xorq.expr.relations import RemoteTable

            parent = self.expr.op().parent
            if isinstance(parent.op(), RemoteTable):
                return parent.op().remote_expr
            else:
                return parent
        else:
            return self.expr

    def get_key(self):
        if self.is_cached:
            return self.storage.get_key(self.uncached_one)
        else:
            return None

    def get_keys(self):
        if self.has_cached and self.cached_nodes[0].to_expr().ls.exists():
            # FIXME: yield storage with key
            return tuple(op.to_expr().ls.get_key() for op in self.cached_nodes)
        else:
            return None

    def exists(self):
        if self.is_cached:
            cn = self.op
            return cn.storage.exists(cn.parent)
        else:
            return None


def _binop(op_class: type[ops.Binary], left: ir.Value, right: ir.Value) -> ir.Value:
    """Try to construct a binary operation.

    Parameters
    ----------
    op_class
        The `ops.Binary` subclass for the operation
    left
        Left operand
    right
        Right operand

    Returns
    -------
    ir.Value
        A value expression

    Examples
    --------
    >>> import xorq as xo
    >>> import xorq.vendor.ibis.expr.operations as ops
    >>> expr = _binop(ops.TimeAdd, ibis.time("01:00"), ibis.interval(hours=1))
    >>> expr
    TimeAdd(datetime.time(1, 0), 1h): datetime.time(1, 0) + 1 h
    >>> _binop(ops.TimeAdd, 1, ibis.interval(hours=1))
    TimeAdd(datetime.time(0, 0, 1), 1h): datetime.time(0, 0, 1) + 1 h
    """
    try:
        node = op_class(left, right)
    except (ValidationError, NotImplementedError):
        return NotImplemented
    else:
        return node.to_expr()


def _is_null_literal(value: Any) -> bool:
    """Detect whether `value` will be treated by ibis as a null literal."""
    if value is None:
        return True
    if isinstance(value, Expr):
        op = value.op()
        return isinstance(op, ops.Literal) and op.value is None
    return False
