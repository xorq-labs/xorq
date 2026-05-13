"""Project-wide deterministic hashing for xorq, built on xorq_dasher.

This module exposes the canonical ``HASHER`` instance and the ``tokenize`` /
``normalize`` helpers that the rest of xorq uses for cache keys, build
hashes, deterministic names, and lineage.

``DEFAULT_HASHER`` from xorq_dasher already covers ibis/xorq expression
types, Python callables (FunctionType/MethodType/CodeType/CellType/classmethod
/staticmethod), and the common builtins/numpy/pandas/pyarrow/sklearn rules.
This module adds the few gap rules that exist in xorq's legacy
dask-normalize code but are not yet in xorq_dasher 0.1.0:
``functools._lru_cache_wrapper``, ``property``, ``toolz.functoolz.Compose``,
``toolz.curry`` (both stock and the xorq ``toolz_utils`` variant),
``toolz.functoolz.excepts``, and ``operator.methodcaller``.
"""

from __future__ import annotations

import contextvars
import functools
import operator
from ctypes import POINTER, Structure, c_size_t, c_void_p, cast, py_object

import toolz
from xorq_dasher import DEFAULT_HASHER, Hasher, fqn
from xorq_dasher.rules.functions import normalize_function


# Active hasher for transitive tokenize calls (e.g. _parent_token inside the
# opaque-placeholder replacer). Snapshot strategy sets this so its data-blind
# rules propagate into recursive parent normalization. Unset → use global
# HASHER, the data-sensitive default.
_current_hasher: contextvars.ContextVar[Hasher | None] = contextvars.ContextVar(
    "_xorq_current_hasher", default=None
)


_PYOBJECT_HEAD = [("ob_refcnt", c_size_t), ("ob_type", c_void_p)]


def _ctypes_field(fields, field, obj):
    cls = type(
        "ctypes-hack",
        (Structure,),
        {"_fields_": _PYOBJECT_HEAD + [(f, c_void_p) for f in fields]},
    )
    inst = cast(c_void_p(id(obj)), POINTER(cls)).contents
    return cast(getattr(inst, field), py_object).value


def normalize_attrs(obj):
    """Stable normalization for any ``attrs.frozen`` object.

    Used by classes that previously aliased ``__dask_tokenize__ = normalize_attrs``.
    """
    return tuple(sorted(obj.__getstate__().items()))


def normalize_lru_cache(func):
    inner = func
    while hasattr(inner, "__wrapped__"):
        inner = inner.__wrapped__
    return normalize_function(inner)


def normalize_property(prop):
    return ("property", prop.fget, prop.fset, prop.fdel)


def normalize_toolz_compose(composed):
    return ("toolz.Compose", composed.first, composed.funcs)


def normalize_toolz_curry(curried):
    from xorq.common.utils.inspect_utils import get_partial_arguments  # noqa: PLC0415

    partial_arguments = get_partial_arguments(
        curried.func, *curried.args, **curried.keywords
    )
    return ("toolz.curry", curried.func, tuple(sorted(partial_arguments.items())))


def normalize_toolz_excepts(f):
    return ("toolz.excepts", f.exc, f.func)


def normalize_methodcaller(obj):
    fields = ("name", "args", "kwargs")
    return ("operator.methodcaller", *(_ctypes_field(fields, f, obj) for f in fields))


def normalize_functools_partial(p):
    """``functools.partial`` is callable; capture func + args + sorted kwargs."""
    return (
        "functools.partial",
        p.func,
        tuple(p.args),
        tuple(sorted(p.keywords.items())),
    )


def normalize_builtin_callable(func):
    """Builtin C functions / methods (e.g. ``json.dumps``)."""
    return (
        "builtins.builtin",
        getattr(func, "__module__", None),
        getattr(func, "__qualname__", getattr(func, "__name__", repr(func))),
    )


def normalize_slice(s):
    return ("slice", s.start, s.stop, s.step)


def normalize_ibis_schema(schema):
    """Schema normalizer that preserves ibis type identity.

    xorq_dasher 0.1.0's rule uses ``schema.to_pandas()`` which collapses
    decimal/array/struct/map to ``dtype('O')`` — two semantically distinct
    schemas with the same column names but different "complex" ibis dtypes
    would collide. Round-tripping through ``str(dtype)`` preserves full ibis
    type info (precision, parameterization, etc.).
    """
    return ("ibis.Schema", tuple((name, str(dtype)) for name, dtype in schema.items()))


def normalize_numpy_dtype(dtype):
    return ("numpy.dtype", str(dtype), dtype.kind, dtype.itemsize)


def normalize_pandas_series(series):
    import pyarrow as pa  # noqa: PLC0415

    return (
        "pandas.Series",
        series.name,
        str(series.dtype),
        pa.Array.from_pandas(series).to_pylist(),
    )


def normalize_pandas_dataframe(df):
    import pyarrow as pa  # noqa: PLC0415

    table = pa.Table.from_pandas(df)
    return (
        "pandas.DataFrame",
        tuple(df.columns),
        tuple(str(t) for t in df.dtypes),
        table,
    )


def _normalize_read_xorq(read):
    """xorq-flavored Read normalizer.

    xorq stores the read path under the canonical ``hash_path`` key (defer_utils
    renames backend-specific kwargs), so the dasher 0.1.0 Read rule (which looks
    up ``path/paths/source/source_list``) does not match. This restores the
    legacy xorq behavior covering http(s), cloud, build-bundle relative, and
    local-filesystem paths.
    """
    import pathlib  # noqa: PLC0415

    read_kwargs = dict(read.read_kwargs)
    path = read_kwargs["hash_path"]
    if isinstance(path, (list, tuple)):
        path = path[0] if len(path) == 1 else path
    if isinstance(path, (str, pathlib.Path)):
        path = str(path)
        if path.startswith(("http://", "https://", "s3://", "gs://", "gcs://")):
            # Remote paths: defer to the legacy stat helper if available.
            from xorq.expr import api  # noqa: PLC0415

            if path.startswith(("http://", "https://")):
                import urllib.request  # noqa: PLC0415

                req = urllib.request.Request(
                    path, method="HEAD", headers={"User-Agent": "xorq-cache"}
                )
                resp = urllib.request.urlopen(req, timeout=10)
                headers = resp.info()
                tpls = (
                    ("url", path),
                    *(
                        (k, headers.get(k))
                        for k in ("Last-Modified", "Content-Length", "Content-Type")
                    ),
                )
            else:
                meta = api.get_object_metadata(
                    path, **{k: v for k, v in read_kwargs.items() if k != "hash_path"}
                )
                tpls = tuple(
                    (k, meta.get(k))
                    for k in ("location", "last_modified", "size", "e_tag", "version")
                )
        elif not pathlib.Path(path).is_absolute() and path == read_kwargs.get(
            "read_path"
        ):
            # Build-bundled Read: relative read_path is already a content hash.
            tpls = (("build-relative-path", path),)
        elif (p := pathlib.Path(path)).exists():
            tpls = read.normalize_method(p)
        else:
            raise NotImplementedError(f'Don\'t know how to deal with path "{path}"')
    else:
        raise NotImplementedError(f'Don\'t know how to deal with path "{path}"')
    tpls += tuple(
        (k, v) for k, v in read.read_kwargs if k in ("mode", "schema", "temporary")
    )
    return ("xorq.Read", read.schema, tpls)


_DATAFUSION_PATH_RE = None  # populated lazily on first use


# Catalog-extract tempdir prefix. ``xorq.catalog.expr_utils.load_expr_from_zip``
# extracts each load into a fresh ``tempfile.mkdtemp(prefix="xorq-catalog-")``,
# so the absolute paths embedded in DataFusion/DuckDB plan strings differ per
# load and would otherwise leak per-load randomness into the DT token. The
# regex strips everything up to and including the first ``xorq-catalog-<…>/``
# segment, leaving the build-hashed suffix (which IS content-addressed and
# stable across reloads). Owned by ADR-0007.
_CATALOG_EXTRACT_DIR_RE = None


def _canonicalize_catalog_path(s):
    """Strip the ``xorq-catalog-<random>/`` tempdir prefix if present.

    Returns the canonicalized path AND whether canonicalization actually
    fired — callers should skip the ``stat`` step on canonicalized paths
    because the canonicalized form is now relative and the per-load tempfile
    has a fresh inode/mtime that would defeat catalog-reload stability.
    """
    import re  # noqa: PLC0415

    global _CATALOG_EXTRACT_DIR_RE
    if _CATALOG_EXTRACT_DIR_RE is None:
        _CATALOG_EXTRACT_DIR_RE = re.compile(r".*?/xorq-catalog-[^/]+/")
    canonical = _CATALOG_EXTRACT_DIR_RE.sub("", s)
    return canonical, canonical != s


def _normalize_path_stat(path, **kwargs):
    """Stable metadata for a path: HTTP HEAD, cloud metadata, or local stat."""
    import pathlib  # noqa: PLC0415

    if isinstance(path, str) and path.startswith(("http://", "https://")):
        import urllib.request  # noqa: PLC0415

        req = urllib.request.Request(
            path, method="HEAD", headers={"User-Agent": "xorq-cache"}
        )
        resp = urllib.request.urlopen(req, timeout=10)
        headers = resp.info()
        return (
            ("url", path),
            *(
                (k, headers.get(k))
                for k in ("Last-Modified", "Content-Length", "Content-Type")
            ),
        )
    if isinstance(path, str) and path.startswith(("s3://", "gs://", "gcs://")):
        from xorq.expr import api  # noqa: PLC0415

        meta = api.get_object_metadata(path, **kwargs)
        return tuple(
            (k, meta.get(k))
            for k in ("location", "last_modified", "size", "e_tag", "version")
        )
    p = pathlib.Path(path)
    if p.exists():
        # noqa: PLC0415 -- lazy import to avoid circulars during module bootstrap
        from xorq.common.utils import defer_utils  # noqa: PLC0415

        return defer_utils.normalize_read_path_stat(p)
    raise FileNotFoundError(f"local path does not exist: {path!r}")


def _extract_duckdb_file_paths(sql_ddl):
    """Extract file paths from a DuckDB DDL's read_parquet/read_csv literals.

    Paths are canonicalized via :func:`_canonicalize_catalog_path` so loads of
    a catalog zip into different tempdirs produce stable tokens.
    """
    import pathlib  # noqa: PLC0415

    import sqlglot as sg  # noqa: PLC0415

    tree = sg.parse_one(sql_ddl, dialect="duckdb")

    def canon(raw):
        if raw.startswith(("http://", "https://", "s3://", "gs://", "gcs://")):
            return raw
        p = pathlib.Path(raw)
        absolute = str(p if p.is_absolute() else pathlib.Path("/") / raw)
        canonical, _ = _canonicalize_catalog_path(absolute)
        return canonical

    parquet_paths = tuple(
        canon(lit.this)
        for func in tree.find_all(sg.exp.ReadParquet)
        for lit in func.find_all(sg.exp.Literal)
        if lit.is_string
    )
    # ReadCSV's func.expressions hold keyword args whose string literals are
    # not paths; restrict to func.this (the path argument).
    csv_paths = tuple(
        canon(lit.this)
        for func in tree.find_all(sg.exp.ReadCSV)
        if func.this is not None
        for lit in func.this.find_all(sg.exp.Literal)
        if lit.is_string
    )
    return parquet_paths + csv_paths


def _normalize_duckdb_databasetable_xorq(dt):
    """DuckDB DT normalizer with catalog-extract path canonicalization.

    Dasher 0.1.0's ``normalize_duckdb_file_databasetable`` returns the raw
    DDL string, which embeds the absolute path DuckDB sees — for tables
    rehydrated from a catalog zip, that path lives under a per-load
    ``xorq-catalog-<random>/`` tempdir and leaks into the token. Parse paths
    out, canonicalize, then stat-or-pass-through (see :func:`_stat_or_canonical`).
    """
    import re  # noqa: PLC0415

    import sqlglot as sg  # noqa: PLC0415
    from xorq_dasher.rules.expr import (  # noqa: PLC0415
        normalize_memory_databasetable,
    )

    name = sg.table(dt.name, quoted=dt.source.compiler.quoted).sql(
        dialect=dt.source.name
    )
    ((_, plan),) = dt.source.raw_sql(f"EXPLAIN SELECT * FROM {name}").fetchall()
    scan_line = plan.split("\n")[1]
    scan_kind = re.match(r"\s*│\s*(\w+)\s*│\s*", scan_line).group(1)
    if scan_kind in ("ARROW_SCAN", "PANDAS_SCAN"):
        return normalize_memory_databasetable(dt)
    if scan_kind in ("READ_PARQUET", "READ_CSV", "SEQ_SCAN"):
        sql_name = sg.exp.convert(dt.name).sql(dialect=dt.source.name)
        (sql_ddl,) = dt.source.con.sql(
            f"select sql from duckdb_views() where view_name = {sql_name} "
            f"UNION select sql from duckdb_tables() where table_name = {sql_name}"
        ).fetchone()
        paths = _extract_duckdb_file_paths(sql_ddl)
        if paths:
            file_metadata = tuple((p, _stat_or_canonical(p)) for p in sorted(paths))
            return (
                "ibis.DatabaseTable.duckdb.file",
                dt.schema.to_pandas(),
                file_metadata,
            )
        # Fallback to the raw-DDL form when we can't parse paths (preserves
        # dasher 0.1.0 behavior).
        return ("ibis.DatabaseTable.duckdb.file", dt.schema.to_pandas(), sql_ddl)
    raise NotImplementedError(scan_line)


def _stat_or_canonical(path):
    """Token for an extracted file path.

    Paths that were canonicalized (relative after stripping the
    ``xorq-catalog-…/`` prefix) live in a tempdir whose inode/mtime differ
    per reload — stat'ing them would defeat catalog-reload stability. The
    canonical path already carries the build-hashed prefix, which is
    content-addressed, so the canonical string alone is a stable token.

    Non-canonical (absolute) paths point at user-managed files; stat them
    to keep ``ModificationTimeStrategy`` cache invalidation working (see
    ``test_parquet_cache_storage``).
    """
    import pathlib  # noqa: PLC0415

    if isinstance(path, str) and (
        path.startswith(("http://", "https://", "s3://", "gs://", "gcs://"))
        or pathlib.Path(path).is_absolute()
    ):
        return _normalize_path_stat(path)
    return ("canonical-build-path", path)


def _extract_datafusion_plan_paths(ep_str):
    """Extract file paths from a DataFusion execution plan's ``file_groups``.

    DataFusion's plan repr strips the leading ``/`` from absolute local paths;
    we restore it. Catalog-extract tempdir prefixes
    (``…/xorq-catalog-<random>/``) are then stripped via
    :func:`_canonicalize_catalog_path` so two ``load_expr_from_zip`` calls on
    the same zip produce equal tokens (ADR-0007).
    """
    import itertools  # noqa: PLC0415
    import pathlib  # noqa: PLC0415
    import re  # noqa: PLC0415

    import yaml12  # noqa: PLC0415

    file_groups_match = re.search(r"file_groups=(\{[^}]*\})", ep_str)
    if not file_groups_match:
        return ()
    parsed = yaml12.parse_yaml(file_groups_match.group(1))
    (groups,) = parsed.values()
    out = []
    for raw in itertools.chain.from_iterable(groups):
        if raw.startswith(("http://", "https://", "s3://", "gs://", "gcs://")):
            out.append(raw)
            continue
        p = pathlib.Path(raw)
        absolute = str(p if p.is_absolute() else pathlib.Path("/") / raw)
        canonical, _ = _canonicalize_catalog_path(absolute)
        out.append(canonical)
    return tuple(out)


def _normalize_datafusion_databasetable_xorq(dt):
    """Datafusion DT normalizer that stats Parquet/CSV files for content sensitivity.

    Dasher 0.1.0's rule returns just ``(schema, ep_str)`` for parquet/csv-backed
    tables; ep_str captures the path but no mtime/size, so file edits don't
    invalidate ``ModificationTimeStrategy`` cache keys (the test in
    ``test_parquet_cache_storage``). Mirror the legacy xorq behavior: extract
    file paths from the plan and stat them.
    """
    import re  # noqa: PLC0415

    from xorq_dasher.rules.expr import (  # noqa: PLC0415
        normalize_memory_databasetable,
    )

    table = dt.source.con.table(dt.name)
    ep_str = str(table.execution_plan())
    is_file = ep_str.startswith(("ParquetExec:", "CsvExec:")) or re.match(
        r"DataSourceExec:.+file_type=(csv|parquet)", ep_str
    )
    if is_file:
        paths = _extract_datafusion_plan_paths(ep_str)
        if paths:
            file_metadata = tuple((p, _stat_or_canonical(p)) for p in sorted(paths))
            return (
                "ibis.DatabaseTable.datafusion.file",
                dt.schema.to_pandas(),
                file_metadata,
            )
        raise ValueError(
            f"no parquet/csv paths extractable from execution plan: {ep_str!r}"
        )
    if ep_str.startswith(("MemoryExec:", "DataSourceExec:")):
        return normalize_memory_databasetable(dt)
    if "PyRecordBatchProviderExec" in ep_str:
        return (
            "ibis.DatabaseTable.datafusion.recordbatch",
            dt.schema.to_pandas(),
            dt.name,
        )
    if ep_str.startswith("EmptyExec"):
        raise ValueError("No data to cache")
    raise ValueError(f"unrecognized DataFusion execution plan: {ep_str!r}")


def _rename_unbound_xorq(op, prefix="static"):
    """Rewrite UnboundTable nodes to sequential placeholder names.

    Equivalent of ``xorq_dasher.rules.expr._rename_unbound`` but with a correct
    op.replace callback signature: dasher 0.1.0's version uses ``**kwargs``
    which captures nothing when ibis passes the rewritten-children dict as a
    positional, then crashes when ``__recreate__({})`` is called on ops with
    required fields (e.g. ``Field`` needs ``rel`` and ``name``).
    """
    import itertools  # noqa: PLC0415

    from xorq.vendor.ibis.expr.operations.relations import UnboundTable  # noqa: PLC0415

    count = itertools.count()

    def rename(node, _kwargs=None, **_kw):
        if isinstance(node, UnboundTable):
            return node.copy(name=f"{prefix}-{next(count)}")
        if _kwargs:
            return node.__recreate__(_kwargs)
        return node

    return op.replace(rename)


def _stable_opaque_name(prefix, *parts):
    """Build a deterministic placeholder name from xxhash of structural parts.

    xorq_dasher 0.1.0's ``_opaque_to_placeholder`` uses ``id(node)`` for some
    leaf names, which breaks across catalog reloads (different Python object
    identities for semantically-identical Reads). This helper keys on a
    content-stable hash of the supplied parts instead.
    """
    import xxhash  # noqa: PLC0415

    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return f"{prefix}-{xxhash.xxh128(payload).hexdigest()[:16]}"


def _parent_token(thing):
    """Tokenize an opaque sub-expression's parent / inner expr structurally.

    Used to fold the inner expression's identity into the placeholder name so
    two opaque wrappers with the same schema/cache-type/etc. but different
    inner expressions do not collide. Accepts either Op or Expr; falls back
    to repr-hash if neither is recognized so the function never raises in
    pathological op trees.

    Uses ``_current_hasher`` when set (so snapshot tokenize propagates its
    data-blind rules into recursive parent normalization); otherwise falls
    back to the global HASHER.
    """
    try:
        if hasattr(thing, "to_expr") and not hasattr(thing, "op"):
            thing = thing.to_expr()
        hasher = _current_hasher.get() or HASHER
        return hasher.tokenize(thing)
    except Exception:
        import xxhash  # noqa: PLC0415

        return xxhash.xxh128(repr(thing).encode("utf-8")).hexdigest()


def _xorq_opaque_to_placeholder(node, _kwargs=None, **_kw):
    """Replace opaque leaf nodes with UnboundTable placeholders.

    Mirrors xorq_dasher.rules.expr._opaque_to_placeholder but
    (a) uses content-stable hashes instead of ``id()`` so tokenize is
    reproducible across catalog reloads, and
    (b) folds the *parent/inner* expression's structural token into each
    placeholder name so wrappers with identical schema but distinct inner
    expressions do not collide.

    Callable from both ibis ``op.replace`` (positional ``(node, kwargs)``)
    and xorq's ``replace_nodes`` (same shape); for non-opaque nodes with
    rewritten children, ``_kwargs`` is the children-dict and we recreate.
    """
    import xorq.expr.operations as xops  # noqa: PLC0415
    from xorq.expr import api  # noqa: PLC0415
    from xorq.expr.relations import (  # noqa: PLC0415
        CachedNode,
        FlightExpr,
        FlightUDXF,
        HashingTag,
        Read,
        RemoteTable,
    )

    match node:
        case CachedNode():
            name = _stable_opaque_name(
                "cached",
                node.schema,
                type(node.cache).__name__,
                _parent_token(node.parent),
            )
        case Read():
            read_kwargs = dict(node.read_kwargs)
            anchor = read_kwargs.get("read_path") or read_kwargs.get("hash_path")
            name = _stable_opaque_name("read", node.schema, anchor)
        case RemoteTable():
            name = _stable_opaque_name(
                "remote",
                node.schema,
                _parent_token(node.remote_expr),
                getattr(node.source, "name", ""),
            )
        case FlightExpr():
            # unbound_expr names are user-chosen and may differ between two
            # FlightExprs that should hash identically (see
            # test_flight_expr_name_doesnt_matter). Canonicalize via
            # _rename_unbound_xorq before folding into the placeholder name.
            name = _stable_opaque_name(
                "flight-expr",
                node.schema,
                _parent_token(node.input_expr),
                _parent_token(_rename_unbound_xorq(node.unbound_expr.op()).to_expr()),
            )
        case FlightUDXF():
            name = _stable_opaque_name(
                "flight-udxf",
                node.schema,
                _parent_token(node.input_expr),
                _parent_token(getattr(node.udxf, "exchange_f", None)),
            )
        case HashingTag():
            name = _stable_opaque_name(
                "tag",
                node.schema,
                node.metadata,
                _parent_token(node.parent),
            )
        case xops.NamedScalarParameter():
            # Replace with a literal of the same dtype so SQL compilation
            # works without a translation rule for NamedScalarParameter, and
            # use a content-stable name so two builds with the same param
            # produce identical placeholders.
            anchor = _stable_opaque_name(
                "param", node.label, str(node.dtype), str(node.default)
            )
            return api.literal(value=None, type=node.dtype).name(anchor).op()
        case _:
            if _kwargs:
                return node.__recreate__(_kwargs)
            return node
    return api.table(node.schema, name=name).op()


def _normalize_expr_xorq(expr):
    """Deterministic Expr normalizer; replaces dasher's id()-based version."""
    from xorq_dasher.rules.expr import normalize_inmemorytable  # noqa: PLC0415

    from xorq.common.utils.graph_utils import replace_nodes, walk_nodes  # noqa: PLC0415
    from xorq.expr.api import get_compiler, to_sql  # noqa: PLC0415
    from xorq.expr.relations import CachedNode, Read  # noqa: PLC0415
    from xorq.vendor.ibis.expr.operations.relations import (  # noqa: PLC0415
        DatabaseTable,
        InMemoryTable,
    )
    from xorq.vendor.ibis.expr.operations.udf import AggUDF, ScalarUDF  # noqa: PLC0415

    op = expr.op()
    compiler = get_compiler(expr)
    # Use replace_nodes (not op.replace) so the opaque-placeholder rewrite
    # descends into Any-typed sub-expressions (RemoteTable.remote_expr,
    # CachedNode.parent, FlightExpr/UDXF.input_expr,
    # ExprScalarUDF.computed_kwargs_expr). Without this, inner opaque nodes
    # keep their gen_name()-randomized names and leak randomness into SQL.
    rewritten = replace_nodes(_xorq_opaque_to_placeholder, op)
    sql = str(to_sql(rewritten.to_expr().unbind(), compiler=compiler))
    # walk_nodes descends through the same Any-typed boundaries, so leaves
    # reachable only through opaque sub-expressions still contribute their
    # data identity to the hash.
    reads = tuple(walk_nodes(Read, op))
    dts = tuple(
        n
        for n in walk_nodes(DatabaseTable, op)
        if not isinstance(n, (CachedNode, Read))
    )
    udfs = tuple(walk_nodes((AggUDF, ScalarUDF), op))
    mems = tuple(walk_nodes(InMemoryTable, op))
    return (
        "ibis.Expr",
        sql,
        reads,
        dts,
        udfs,
        tuple(normalize_inmemorytable(m) for m in mems),
    )


def _databasetable_dispatcher(dt):
    """Dispatch DatabaseTable subclasses to their specific normalizers.

    xorq_dasher 0.1.0's normalize_databasetable does not handle the
    ``xorq_datafusion`` backend name (only ``xorq``) and its DatabaseTable
    rule outranks the more-specific Read/CachedNode/RemoteTable rules in
    MRO-with-earliest-match-wins lookup. This wrapper restores the
    most-specific-wins behavior xorq depends on.
    """
    from xorq_dasher.rules.expr import (  # noqa: PLC0415
        normalize_cached_node,
        normalize_databasetable,
        normalize_remote_table,
    )

    from xorq.expr.relations import (  # noqa: PLC0415
        CachedNode,
        FlightExpr,
        FlightUDXF,
        Read,
        RemoteTable,
    )

    if isinstance(dt, Read):
        return _normalize_read_xorq(dt)
    if isinstance(dt, CachedNode):
        return normalize_cached_node(dt)
    if isinstance(dt, RemoteTable):
        return normalize_remote_table(dt)
    # FlightExpr/FlightUDXF carry input_expr / make_connection that the plain
    # datafusion path would silently flatten away. Inline the dasher 0.1.0
    # logic but use ``_rename_unbound_xorq`` (whose op.replace callback signs
    # ``(node, _kwargs)`` correctly — dasher 0.1.0's ``_rename_unbound`` uses
    # ``**kwargs`` and crashes recreating ops with required positional fields
    # like ``Field``).
    if isinstance(dt, FlightExpr):
        return (
            "xorq.FlightExpr",
            dt.input_expr,
            _rename_unbound_xorq(dt.unbound_expr.op()).to_expr(),
            dt.make_connection,
        )
    if isinstance(dt, FlightUDXF):
        return (
            "xorq.FlightUDXF",
            dt.input_expr,
            getattr(dt.udxf, "exchange_f", None),
            dt.make_connection,
        )
    # For datafusion-backed file tables, dasher's normalize_datafusion_
    # databasetable stops at ep_str — which captures the path but no stat —
    # so file edits don't invalidate the cache key. _normalize_datafusion_
    # databasetable_xorq stats the underlying files to restore mtime sensitivity.
    if dt.source.name in ("datafusion", "xorq_datafusion"):
        return _normalize_datafusion_databasetable_xorq(dt)
    if dt.source.name == "duckdb":
        return _normalize_duckdb_databasetable_xorq(dt)
    return normalize_databasetable(dt)


def _build_extra_rules():
    import types as _types  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    from xorq.expr.relations import Read  # noqa: PLC0415
    from xorq.vendor.ibis.expr.operations.relations import (  # noqa: PLC0415
        DatabaseTable,
        Schema,
    )
    from xorq.vendor.ibis.expr.types import Expr  # noqa: PLC0415

    rules = [
        (fqn(functools._lru_cache_wrapper), normalize_lru_cache),
        (fqn(functools.partial), normalize_functools_partial),
        (fqn(_types.BuiltinFunctionType), normalize_builtin_callable),
        (fqn(_types.BuiltinMethodType), normalize_builtin_callable),
        (fqn(slice), normalize_slice),
        (fqn(property), normalize_property),
        (fqn(toolz.functoolz.Compose), normalize_toolz_compose),
        (fqn(toolz.curry), normalize_toolz_curry),
        (fqn(toolz.functoolz.excepts), normalize_toolz_excepts),
        (fqn(operator.methodcaller), normalize_methodcaller),
        (fqn(DatabaseTable), _databasetable_dispatcher),
        (fqn(Read), _normalize_read_xorq),
        (fqn(Expr), _normalize_expr_xorq),
        (fqn(Schema), normalize_ibis_schema),
        (fqn(np.dtype), normalize_numpy_dtype),
        (fqn(pd.Series), normalize_pandas_series),
        (fqn(pd.DataFrame), normalize_pandas_dataframe),
    ]
    try:
        from xorq.common.utils.toolz_utils import curry as xo_curry  # noqa: PLC0415

        rules.append((fqn(xo_curry), normalize_toolz_curry))
    except ImportError:
        pass
    return tuple(rules)


HASHER: Hasher = DEFAULT_HASHER.override(*_build_extra_rules())


def tokenize(*objs) -> str:
    """Return a deterministic hex digest for one or more objects."""
    return HASHER.tokenize(*objs)


def normalize(obj):
    """Return the primitive-tuple normalization of an object."""
    return HASHER.normalize(obj)


def snapshot_hasher(*extra_rules) -> Hasher:
    """Return a Hasher with snapshot-specific overrides layered on top of HASHER.

    Used by ``SnapshotStrategy`` to swap in backend / DatabaseTable / Read
    normalizers for the duration of a single key calculation.
    """
    return HASHER.override(*extra_rules)


__all__ = [
    "HASHER",
    "Hasher",
    "fqn",
    "tokenize",
    "normalize",
    "normalize_attrs",
    "snapshot_hasher",
]
