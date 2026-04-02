from __future__ import annotations

import contextlib
import functools
import os
import webbrowser
from functools import cached_property
from typing import TYPE_CHECKING, Any, NoReturn, Optional

import toolz
from attr import (
    field,
    frozen,
)
from attr.validators import (
    deep_iterable,
    instance_of,
    optional,
)
from public import public

import xorq.vendor.ibis.expr.operations as ops
from xorq.common.exceptions import TranslationError, XorqError
from xorq.common.utils.func_utils import return_constant
from xorq.ibis_yaml.enums import ExprKind
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
    from xorq.common.utils.lineage_utils import LineageDAG
    from xorq.vendor.ibis import Schema
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
        from xorq.config import options

        if options.interactive:
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
        >>> import xorq.api as xo
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

        >>> import xorq.api as xo
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
        >>> import xorq.api as xo
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
        from xorq.common.utils.graph_utils import get_ordered_unique_sources

        node_types = (
            ops.UnboundTable,
            ops.DatabaseTable,
            ops.SQLQueryResult,
            rel.CachedNode,
            rel.Read,
        )
        found = self.op().find(node_types)
        bound = tuple(op for op in found if not isinstance(op, ops.UnboundTable))
        backends = tuple(get_ordered_unique_sources(bound))
        has_unbound = any(isinstance(op, ops.UnboundTable) for op in found)
        return backends, has_unbound

    def _find_backend(self, *, use_default=True) -> BaseBackend:
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

        match backends:
            case []:
                if has_unbound:
                    raise XorqError(
                        "Expression contains unbound tables and therefore cannot "
                        "be executed. Use `<backend>.execute(expr)` to execute "
                        "against an explicit backend, or rebuild the expression "
                        "using bound tables instead."
                    )
                elif use_default:
                    backend = _backend_init()
                    return backend
                else:
                    raise XorqError(
                        "Expression depends on no backends, and found no default"
                    )
            case [backend]:
                return backend
            case _:
                raise XorqError("Multiple backends found for this expression")

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
        >>> import xorq.api as xo
        >>> from xorq.api import _
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
        >>> import xorq.api as xo
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
        return LETSQLAccessor(self.op())


def _extract_unbound_node(expr):
    from xorq.common.utils.graph_utils import walk_nodes  # noqa: PLC0415

    unbound_node, *rest = walk_nodes(ops.UnboundTable, expr) or (None,)
    if rest:
        raise ValueError("Expected at most one UnboundTable")
    return unbound_node


def _extract_is_source(expr):
    from xorq.expr.relations import CachedNode, Read  # noqa: PLC0415

    source_nodes = (ops.DatabaseTable, Read, ops.InMemoryTable, CachedNode)
    root = expr.ls.unwrapped
    return isinstance(root, source_nodes)


def _extract_catalog_tag_nodes(expr):
    from xorq.catalog.bind import CatalogTag  # noqa: PLC0415
    from xorq.common.utils.graph_utils import walk_nodes  # noqa: PLC0415
    from xorq.expr.relations import HashingTag  # noqa: PLC0415

    return tuple(
        ht
        for ht in (walk_nodes(HashingTag, expr) or ())
        if ht.metadata.get("tag") in frozenset(CatalogTag)
    )


def _extract_sources(catalog_tag_nodes):
    from xorq.catalog.bind import CatalogTag  # noqa: PLC0415

    return tuple(
        {
            "entry_name": ht.metadata.get("entry_name"),
            "alias": ht.metadata.get("alias"),
            "kind": ht.metadata.get("kind"),
        }
        for ht in catalog_tag_nodes
        if ht.metadata.get("tag") in (CatalogTag.SOURCE, CatalogTag.TRANSFORM)
    )


def _extract_builders(expr):
    from xorq.common.utils.graph_utils import walk_nodes  # noqa: PLC0415
    from xorq.expr.ml.enums import FittedPipelineTagKey  # noqa: PLC0415
    from xorq.expr.relations import HashingTag, Tag  # noqa: PLC0415
    from xorq.vendor.ibis.common.collections import FrozenOrderedDict  # noqa: PLC0415

    tag_nodes = walk_nodes((Tag, HashingTag), expr)
    if not tag_nodes:
        return ()

    builders = []
    for tag_node in tag_nodes:
        tag_name = tag_node.metadata.get("tag")
        # ML pipeline tags — inline dict extraction
        if tag_name in tuple(FittedPipelineTagKey):
            if FittedPipelineTagKey.ALL_STEPS in tag_node.metadata:
                tag_key = tag_name
                steps_info = tuple(
                    {"name": d["name"], "estimator": d["typ"].__name__}
                    for step_items in tag_node.metadata.get(
                        FittedPipelineTagKey.ALL_STEPS, ()
                    )
                    for d in (dict(step_items),)
                )
                builders.append(
                    FrozenOrderedDict(
                        {
                            "type": "fitted_pipeline",
                            "description": f"{tag_key}, {len(steps_info)} steps",
                            "is_predict": tag_key
                            in (
                                str(FittedPipelineTagKey.PREDICT),
                                str(FittedPipelineTagKey.PREDICT_PROBA),
                                str(FittedPipelineTagKey.DECISION_FUNCTION),
                            ),
                            "steps": steps_info,
                        }
                    )
                )
                continue
        # BSL tags — provenance dict
        if tag_name == "bsl":
            meta = tag_node.metadata
            # dims/measures live on the SemanticTableOp — which may be nested
            # inside SemanticAggregateOp -> SemanticGroupByOp -> SemanticTableOp
            table_meta = meta
            while table_meta.get("bsl_op_type") != "SemanticTableOp":
                source = table_meta.get("source")
                if source is None:
                    break
                table_meta = dict(source) if isinstance(source, tuple) else source
            dims = tuple(d[0] for d in table_meta.get("dimensions", ()))
            measures = tuple(m[0] for m in table_meta.get("measures", ()))
            builders.append(
                FrozenOrderedDict(
                    {
                        "type": "semantic_model",
                        "description": f"{len(dims)} dims, {len(measures)} measures",
                        "dimensions": dims,
                        "measures": measures,
                    }
                )
            )
            continue
    return tuple(builders)


def _extract_kind(unbound_node, catalog_tag_nodes, is_source, has_builders=False):
    # Priority: UnboundExpr (incomplete/has placeholder) > ExprBuilder (has
    # builder tags) > Composed (has catalog HashingTag nodes) > Source
    # (plain table) > Expr (everything else).
    match (unbound_node, has_builders, bool(catalog_tag_nodes), is_source):
        case (node, _, _, _) if node is not None:
            return ExprKind.UnboundExpr
        case (_, True, _, _):
            return ExprKind.ExprBuilder
        case (_, _, True, _):
            return ExprKind.Composed
        case (_, _, _, True):
            return ExprKind.Source
        case _:
            return ExprKind.Expr


def _validate_lineage(instance, attribute, value):
    """optional(instance_of(LineageDAG)) but with a deferred import to break
    the cycle: core.py → lineage_utils → rel → backends → … → core.py"""
    from xorq.common.utils.lineage_utils import LineageDAG  # noqa: PLC0415

    match value:
        case None | LineageDAG():
            return
        case _:
            raise TypeError(
                f"'lineage' must be a LineageDAG or None, got {type(value).__name__}"
            )


def _parse_lineage(raw):
    """Convert JSON-deserialized lineage dict into a LineageDAG, or None."""
    from xorq.common.utils.lineage_utils import LineageDAG  # noqa: PLC0415

    match raw:
        case None:
            return None
        case LineageDAG():
            return raw
        case dict():
            return LineageDAG.from_dict(raw)
        case _:
            raise TypeError(
                f"Expected dict, LineageDAG, or None for lineage, got {type(raw).__name__}"
            )


@frozen
class ExprMetadata:
    kind: ExprKind = field(validator=instance_of(ExprKind))
    schema_out: Schema = field(validator=instance_of(ibis.expr.schema.Schema))
    schema_in: Optional[Schema] = field(
        default=None, validator=optional(instance_of(ibis.expr.schema.Schema))
    )
    root_tag: Optional[str] = field(default=None)
    parquet_cache_paths: tuple[str, ...] = field(factory=tuple)
    composed_from: tuple = field(
        factory=tuple, validator=deep_iterable(instance_of(dict))
    )
    params: tuple = field(factory=tuple)
    sql_queries: tuple[tuple[str, str, str], ...] = field(factory=tuple)
    lineage: Optional[LineageDAG] = field(default=None, validator=_validate_lineage)
    builders: tuple = field(factory=tuple)

    @classmethod
    def from_dict(cls, data):
        schema_in_raw = data.get("schema_in")
        return cls(
            kind=ExprKind(data["kind"]),
            schema_out=ibis.Schema.from_tuples(
                [(k, v) for k, v in data["schema_out"].items()]
            ),
            schema_in=(
                ibis.Schema.from_tuples([(k, v) for k, v in schema_in_raw.items()])
                if schema_in_raw
                else None
            ),
            root_tag=data.get("root_tag"),
            parquet_cache_paths=tuple(data.get("parquet_cache_paths") or ()),
            composed_from=tuple(data.get("composed_from") or data.get("sources") or ()),
            params=tuple(data.get("params") or ()),
            sql_queries=tuple(tuple(q) for q in data.get("sql_queries", ())),
            lineage=_parse_lineage(data.get("lineage")),
            builders=tuple(data.get("builders", ())),
        )

    @classmethod
    def from_expr(cls, expr):
        from xorq.caching import ParquetSnapshotCache  # noqa: PLC0415
        from xorq.common.utils.graph_utils import (  # noqa: PLC0415
            validate_params,
            walk_nodes,
        )
        from xorq.expr.operations import _MISSING, NamedScalarParameter  # noqa: PLC0415
        from xorq.expr.relations import CachedNode  # noqa: PLC0415

        validate_params(expr)

        unbound_node = _extract_unbound_node(expr)
        is_source = _extract_is_source(expr)
        catalog_tag_nodes = _extract_catalog_tag_nodes(expr)

        tags = expr.ls.tags
        root_tag = tags[0].tag if tags else None

        cached_nodes = walk_nodes((CachedNode,), expr)
        parquet_cache_paths = tuple(
            str(cn.cache.storage.get_path(cn.cache.calc_key(cn.parent)))
            for cn in cached_nodes
            if isinstance(cn.cache, ParquetSnapshotCache)
        )

        named_params = tuple(
            {
                "param_name": node.label,
                "type": str(node.dtype),
                **({"default": node.default} if node.default is not _MISSING else {}),
            }
            for node in walk_nodes(NamedScalarParameter, expr)
        )

        builders = _extract_builders(expr)

        return cls(
            kind=_extract_kind(
                unbound_node,
                catalog_tag_nodes,
                is_source,
                has_builders=bool(builders),
            ),
            schema_in=unbound_node.schema if unbound_node else None,
            schema_out=expr.as_table().schema(),
            root_tag=root_tag,
            parquet_cache_paths=parquet_cache_paths,
            composed_from=_extract_sources(catalog_tag_nodes),
            params=named_params,
            builders=builders,
        )

    def to_dict(self):
        return {
            key: value
            for key, value in (
                ("kind", str(self.kind)),
                (
                    "schema_in",
                    toolz.valmap(str, self.schema_in) if self.schema_in else None,
                ),
                ("schema_out", toolz.valmap(str, self.schema_out)),
                ("root_tag", self.root_tag),
                ("parquet_cache_paths", list(self.parquet_cache_paths) or None),
                ("params", self.params or None),
                (
                    "composed_from",
                    list(self.composed_from) if self.composed_from else None,
                ),
<<<<<<< HEAD
                (
                    "sql_queries",
                    [list(q) for q in self.sql_queries] if self.sql_queries else None,
                ),
                (
                    "lineage",
                    self.lineage.to_dict() if self.lineage else None,
                ),
                ("builders", list(self.builders) if self.builders else None),
            )
            if value is not None
        }


@frozen
class LETSQLAccessor:
    op = field(validator=instance_of(ops.Node))
    node_types = (ops.DatabaseTable, ops.SQLQueryResult)

    @property
    def expr(self):
        return self.op.to_expr()

    @cached_property
    def metadata(self):
        return ExprMetadata.from_expr(self.expr)

    @property
    def kind(self) -> ExprKind:
        return self.metadata.kind

    @property
    def is_source(self):
        return _extract_is_source(self.expr)

    @property
    def composed_from(self):
        return self.metadata.composed_from

    @property
    def cached_nodes(self):
        from xorq.common.utils.graph_utils import walk_nodes
        from xorq.expr.relations import (
            CachedNode,
        )

        return walk_nodes((CachedNode,), self.expr)

    @property
    def tags(self):
        return self.get_tags()

    def get_tags(self, predicate=return_constant(True)):
        from xorq.common.utils.graph_utils import walk_nodes
        from xorq.expr.relations import Tag

        return tuple(
            node for node in walk_nodes((Tag,), self.expr) if predicate(node.metadata)
        )

    @property
    def cache(self):
        if self.is_cached:
            return self.op.cache
        else:
            return None

    @property
    def caches(self):
        return tuple(node.cache for node in self.cached_nodes)

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
        from xorq.expr.relations import CachedNode, RemoteTable

        return tuple(
            el
            for el in walk_nodes(self.node_types, self.expr)
            if not isinstance(
                el,
                (
                    RemoteTable,
                    CachedNode,
                ),
            )
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

    @cached_property
    def unwrapped(self):
        """Unwrap Tag and HashingTag layers to get the underlying op."""
        from xorq.expr.relations import HashingTag, Tag

        root = self.op
        while isinstance(root, (Tag, HashingTag)):
            root = root.parent
        return root

    @property
    @functools.cache
    def untagged(self):
        from xorq.expr.api import (
            _remove_non_hashing_tag_nodes,
        )

        return _remove_non_hashing_tag_nodes(self.expr)

    @property
    def fused(self):
        """Strip catalog-created RemoteTable + HashingTag wrappers."""
        from xorq.catalog.bind import fuse_catalog_source

        return fuse_catalog_source(self.expr)

    @property
    def uncached(self):
        from xorq.expr.relations import CachedNode, RemoteTable, replace_cache_table

        if self.has_cached:
            op = self.expr.op()
            return op.replace(
                replace_cache_table, filter=(RemoteTable, CachedNode)
            ).to_expr()
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

    @property
    def tokenized(self):
        from xorq.common.utils.dask_normalize.dask_normalize_utils import (
            patched_tokenize,
        )

        # NOTE: this should almost certainly not be functools.cache'd: it can obscure filesystem / source table changes within the same process run
        return patched_tokenize(self.expr)

    def get_cache_path(self):
        if self.is_cached and hasattr(self.cache.storage, "get_path"):
            cn = self.op
            return cn.cache.storage.get_path(cn.cache.calc_key(cn.parent))
        else:
            return None

    def get_cache_paths(self):
        if self.has_cached:
            return tuple(
                filter(
                    None, (op.to_expr().ls.get_cache_path() for op in self.cached_nodes)
                )
            )
        else:
            return None

    @property
    def cache_path(self):
        return self.get_cache_path()

    @property
    def cached_dt(self):
        if self.exists():
            return self.cache.get(self.uncached_one)
        else:
            return None

    def get_key(self):
        if self.is_cached:
            return self.cache.calc_key(self.uncached_one)
        else:
            return None

    def get_keys(self):
        if self.has_cached and self.cached_nodes[0].to_expr().ls.exists():
            # FIXME: yield cache with key
            return tuple(op.to_expr().ls.get_key() for op in self.cached_nodes)
        else:
            return None

    def exists(self):
        if self.is_cached:
            cn = self.op
            return cn.cache.exists(cn.parent)
        else:
            return None

    @property
    def pipelines(self):
        from xorq.expr.ml.pipeline_lib import (
            Pipeline,
            get_sklearn_pipeline_tags,
            pipeline_tag_to_pipeline,
        )

        return tuple(
            Pipeline.from_instance(pipeline_tag_to_pipeline(pipeline_tag))
            for pipeline_tag in get_sklearn_pipeline_tags(self.expr)
        )

    @property
    def pipeline(self):
        from xorq.expr.ml.pipeline_lib import Pipeline, get_outermost_pipeline

        return Pipeline.from_instance(get_outermost_pipeline(self.expr))


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
    >>> import xorq.api as xo
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
