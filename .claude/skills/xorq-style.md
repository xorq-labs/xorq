# Xorq Coding Style

This skill defines the coding philosophy, patterns, and conventions for the xorq codebase. xorq is a multi-engine data processing library built on Ibis and DataFusion. The codebase blends rigorous engineering with idiomatic functional programming: immutable data, declarative composition, and exhaustive dispatch are the foundation; OOP hierarchies serve only as the scaffolding for these patterns.

---

## 1. Immutability as Architecture

**`@frozen` is the default.** Every class that represents data, configuration, or a computation node must be `@frozen` from `attrs`. Not `@define`, not `@dataclass`, not a plain class with `__init__`.

```python
from attr import field, frozen
from attr.validators import instance_of, optional

@frozen
class ParquetStorage(CacheStorage):
    source = field(
        validator=instance_of(ibis.backends.BaseBackend),
        factory=_backend_init,
    )
    relative_path = field(
        validator=instance_of(Path),
        factory=functools.partial(options.get, "cache.default_relative_path"),
        converter=Path,
    )
    base_path = field(
        validator=optional(instance_of(Path)),
        default=None,
        converter=if_not_none(Path),
    )
```

**Tuple before list, tuple-of-two-tuples before dict.**

```python
from toolz import compose
from xorq.common.utils.attr_utils import validate_kwargs_tuple

# Point-free converter: dict -> sorted items -> tuple
convert_sorted_kwargs_tuple = compose(tuple, sorted, dict.items, dict)

# On frozen classes, params are stored as tuple-of-two-tuples
params_tuple = field(
    validator=validate_kwargs_tuple,
    converter=compose(freeze, convert_sorted_kwargs_tuple),
    factory=tuple,
)

# Convert back to dict only at the moment of use
@property
def instance(self):
    return self.typ(**dict(self.params_tuple))
```

**`freeze()` recursively converts mutable structures.** Dicts become `FrozenOrderedDict`, lists become `tuple`, nested structures stay frozen throughout.

```python
from xorq.ibis_yaml.utils import freeze

def freeze(obj):
    if isinstance(obj, dict):
        return FrozenOrderedDict({k: freeze(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return tuple(freeze(x) for x in obj)
    elif isinstance(obj, tuple):
        return tuple(freeze(x) for x in obj)
    return obj
```

**`@property` + `@functools.cache` on frozen classes.** This is the xorq pattern for lazy, memoized computed attributes. Because the instance is frozen (hashable), `functools.cache` works. This replaces mutable instance attributes entirely.

```python
@frozen(eq=False)
class Popened:
    args = field(...)
    kwargs_tuple = field(...)
    deferred = field(validator=instance_of(bool), default=True)

    @property
    @functools.cache
    def popen(self):
        return non_blocking_subprocess_run(self.args, **self.kwargs)

    @property
    @functools.cache
    def stdout_peeker(self):
        return Peeker(self.popen.stdout) if self.popen.stdout else None
```

**Validators enforce invariants at construction time, not at call sites.**

```python
from attr.validators import instance_of, optional, deep_iterable

features = field(
    validator=optional(deep_iterable(instance_of(str), instance_of(tuple))),
    default=None,
    converter=tuple,
)
```

**`__attrs_post_init__` for validation and assertions, never for mutation.**

```python
@frozen
class CatalogEntry:
    name = field(validator=instance_of(str))
    catalog = field(validator=instance_of(Catalog))
    require_exists = field(validator=instance_of(bool), default=True)

    def __attrs_post_init__(self):
        self.assert_consistency()
        if self.require_exists:
            assert self.exists()
```

---

## 2. Functional Composition Over Imperative Control Flow

**`@toolz.curry` for all reusable functions that benefit from partial application.**

```python
@toolz.curry
def fit_sklearn(df, target=None, *, cls, params):
    obj = cls(**dict(params))
    obj.fit(df, target)
    return obj

@toolz.curry
def predict_sklearn(model, df):
    return model.predict(df)

# Partial application: create a fitter for a specific model class
fitter = fit_sklearn(cls=LinearRegression, params=())
```

**`compose()` for point-free pipelines.** Read right-to-left.

```python
from toolz import compose

convert_sorted_kwargs_tuple = compose(tuple, sorted, dict.items, dict)

datetime_from_unix_nano_str = compose(
    datetime.datetime.fromtimestamp,
    toolz.curried.flip(operator.truediv)(1e9),
    int,
)
```

**`toolz.excepts()` for error-safe function application.**

```python
try_decode_ascii = toolz.excepts(
    AttributeError, operator.methodcaller("decode", "ascii")
)

get_compiler = toolz.excepts(
    (XorqError, AttributeError),
    lambda e: e._find_backend(use_default=True).compiler,
    lambda _: Backend.compiler,
)
```

**Comprehensions are the default iteration pattern.** Use loops only when debugging demands breakpoints.

```python
# Preferred: comprehension
projections = [
    pr for p in projection.projections()
    if (pr := convert(p.to_variant(), catalog=catalog)) is not None
]

# Preferred: generator expression in tuple
children = tuple(
    _build_column_tree(to_node(child))
    for child in gen_children_of(node)
)

# Preferred: nested comprehension with tuple result
return (line,) + tuple(
    grandchild_line
    for i, child in enumerate(self.children)
    for grandchild_line in child._lines(
        child_prefix, i == len(self.children) - 1, False
    )
)
```

**`if_not_none` curried pattern for conditional application.**

```python
@toolz.curry
def if_not_none(f, value):
    return value if value is None else f(value)

# Usage as a converter
base_path = field(
    validator=optional(instance_of(Path)),
    default=None,
    converter=if_not_none(Path),
)
```

**`functools.reduce` with `operator` for combining values.**

```python
pred = reduce(operator.and_, predicates)
```

**`map()`, `partial()`, `chain.from_iterable()` for streaming transforms.**

```python
gen = map(
    partial(pa.RecordBatch.from_pandas, preserve_index=False),
    chain.from_iterable(
        pd.read_csv(path, dtype=dtype, chunksize=chunksize, **kwargs)
        for path in paths
    ),
)
```

---

## 3. Structural Pattern Matching as Primary Dispatch

**`match/case` with final `case _` for exhaustive coverage.** Every match block must end with a catchall that either returns a default or raises.

```python
def arbitrate_transform_predict(transform, predict):
    match (transform, predict):
        case [None, None]:
            raise ValueError
        case [other, None]:
            return other, "transform"
        case [None, other]:
            return other, ResponseMethod.PREDICT
        case [other0, other1]:
            raise ValueError(other0, other1)
        case _:
            raise ValueError
```

**Pattern matching on tuples of booleans for multi-flag dispatch.**

```python
match (structer.is_series, structer.is_kv_encoded, structer.struct_has_kv_fields):
    case (True, True, _):
        return cls(fit=fit_sklearn_series(...), other=transform_sklearn_series_kv(...), ...)
    case (False, True, _):
        return cls(fit=fit_sklearn_args(...), other=kv_encode_output, ...)
    case (False, False, True):
        return cls(fit=fit_sklearn_args(...), other=structer.get_convert_struct_with_kv(), ...)
    case _:
        return cls(fit=fit_sklearn_args(...), other=transform_sklearn_struct(...), ...)
```

**Walrus operator (`:=`) in match subjects for bind-and-dispatch.**

```python
match obj := self.deferred_model.execute():
    case pd.DataFrame():
        ((obj,),) = obj.values
    case bytes():
        pass
    case _:
        raise ValueError
```

**Object attribute patterns with guards.**

```python
match instance:
    case object(k=k):
        return n_features_in if k == "all" else min(k, n_features_in)
    case object(percentile=percentile):
        return max(1, int(n_features_in * percentile / 100))
    case object(n_features_to_select=n):
        match n:
            case None:
                return max(1, n_features_in // 2)
            case float() if 0 < n < 1:
                return max(1, int(n_features_in * n))
            case _:
                return min(n, n_features_in)
```

**Type-based dispatch for domain objects.**

```python
match model:
    case ClusterMixin():
        return make_scorer(adjusted_rand_score)
    case ClassifierMixin():
        return make_scorer(accuracy_score)
    case RegressorMixin():
        return make_scorer(r2_score)
    case _:
        raise ValueError(
            f"Cannot determine default scorer for model type {type(model).__name__}."
        )
```

**Domain-specific graph traversal with pattern matching.**

```python
def gen_children_of(node: Node) -> tuple[Node, ...]:
    match node:
        case ops.Field():
            gen = (to_node(node.rel),)
        case rel.RemoteTable():
            gen = (to_node(node.remote_expr),)
        case rel.CachedNode():
            gen = (to_node(node.parent),)
        case rel.Read():
            gen = ()
        case _:
            raw_children = getattr(node, "__children__", ())
            gen = tuple(to_node(child) for child in raw_children if isinstance(child, Node))
```

---

## 4. Single Dispatch + Cache for Extensible Translation

**`@functools.singledispatch` for type-based handler registration.** New operations register via decorator, no central switch needed.

```python
@functools.singledispatch
def convert(step, catalog, *args):
    raise TypeError(type(step))

@convert.register(Projection)
def convert_projection(projection, catalog):
    ...

@convert.register(Filter)
def convert_filter(_filter, catalog):
    ...
```

**Stack `@functools.cache` or `@functools.lru_cache` on singledispatch.**

```python
@functools.cache
@functools.singledispatch
def translate_from_yaml(yaml_dict: dict, context: TranslationContext) -> Any:
    ...

@functools.lru_cache(maxsize=None, typed=True)
@functools.singledispatch
def translate_to_yaml(op: Any, context: TranslationContext) -> dict:
    raise NotImplementedError(f"No translation rule for {type(op)}")
```

**Curried decorator pattern for automatic registry enrollment.**

```python
@toolz.curry
def convert_to_ref(which, wrapped):
    @functools.wraps(wrapped)
    def wrapper(op, context):
        frozen = wrapped(op, context)
        if context is None:
            return frozen
        return context.register(which, op, frozen)
    return wrapper

convert_to_dtype_ref = convert_to_ref(RegistryEnum.dtypes)
convert_to_node_ref = convert_to_ref(RegistryEnum.nodes)
```

**Handler registration via decorator for YAML deserialization.**

```python
def register_from_yaml_handler(*op_names: str):
    def decorator(func):
        for name in op_names:
            FROM_YAML_HANDLERS[name] = func
        return func
    return decorator

@register_from_yaml_handler("Literal")
def _literal_from_yaml(yaml_dict: dict, context: TranslationContext) -> ir.Expr:
    value = yaml_dict["value"]
    dtype = context.translate_from_yaml(yaml_dict["type"])
    return ibis.literal(value, type=dtype)
```

**`@singledispatch` for extensible formatting.**

```python
@singledispatch
def format_node(node: Node) -> str:
    return node.__class__.__name__

@format_node.register
def _(node: ops.Field) -> str:
    return f"Field:{node.name}"
```

---

## 5. Frozen Abstract Base Classes

**`@frozen` + `@abstractmethod` for abstract immutable interfaces.**

```python
@frozen
class CacheStorage:
    @abstractmethod
    def exists(self, key):
        pass

    @abstractmethod
    def get(self, key):
        pass

    @abstractmethod
    def put(self, key, value):
        pass

    @abstractmethod
    def drop(self, key):
        pass
```

**Concrete subclasses inherit `@frozen` and add validated fields.**

```python
@frozen
class ParquetTTLStorage(ParquetStorage):
    ttl = field(
        validator=instance_of(datetime.timedelta),
        default=datetime.timedelta(days=1),
    )

    def exists(self, key):
        path = self.get_path(key)
        return path.exists() and self.satisfies_ttl(path)
```

**`__dask_tokenize__` on frozen classes for distributed hashing.**

```python
@frozen
class CacheStrategy:
    @abstractmethod
    def calc_key(self, expr):
        pass

    def __dask_tokenize__(self):
        return (type(self).__name__,)
```

**Class-level constants (not instance fields) for type-level configuration.**

```python
@public
@frozen
class ParquetCache(Cache):
    strategy_typ = ModificationTimeStrategy
    storage_typ = ParquetStorage

@public
@frozen
class ParquetSnapshotCache(Cache):
    strategy_typ = SnapshotStrategy
    storage_typ = ParquetStorage
```

---

## 6. Expression Trees as Immutable DAGs

**`__recreate__` for immutable node transformation.** Never mutate a node; always produce a new one.

```python
def recursive_update(obj, replacements):
    if isinstance(obj, Node):
        if obj in replacements:
            return replacements[obj]
        return obj.__recreate__({
            name: recursive_update(arg, replacements)
            for name, arg in zip(obj.argnames, obj.args)
        })
    elif isinstance(obj, tuple):
        return tuple(recursive_update(el, replacements) for el in obj)
    return obj
```

**Higher-order functions returning node transformers.**

```python
def replace_source_factory(source: Any):
    def replace_source(node, _, **kwargs):
        if "source" in kwargs:
            kwargs["source"] = source
        return node.__recreate__(kwargs)
    return replace_source
```

**Composable transformation pipeline.** Each stage receives an expression, returns a (possibly modified) expression. Stages are independently testable.

```python
def _transform_expr(expr, **kwargs):
    expr = _remove_tag_nodes(expr)
    expr = _register_and_transform_cache_tables(expr)
    expr, created = register_and_transform_remote_tables(expr, **kwargs)
    expr, dt_to_read = _transform_deferred_reads(expr)
    return (expr, created)
```

**`op.replace(fn)` for tree-wide rewrites.**

```python
def _register_and_transform_cache_tables(expr):
    def fn(node, kwargs):
        if kwargs:
            node = node.__recreate__(kwargs)
        if isinstance(node, CachedNode):
            uncached, cache = node.parent, node.cache
            node = cache.set_default(uncached, uncached.op())
        return node
    op = expr.op()
    out = op.replace(fn)
    return out.to_expr()
```

---

## 7. Curried Factories and Declarative Construction

**Curried functions as field values in frozen classes.** This enables declarative pipeline composition where the operations themselves are partially-applied functions.

```python
@toolz.curry
def deferred_fit_predict(
    expr, target, features, cls, return_type, params=(), name_infix=ResponseMethod.PREDICT, cache=None,
):
    return DeferredFitOther(
        expr=expr,
        target=target,
        features=features,
        fit=fit_sklearn(cls=cls, params=params),
        other=predict_sklearn,
        return_type=return_type,
        name_infix=name_infix,
        cache=cache,
    )
```

**`@classmethod` factories for complex construction.**

```python
class RemoteTable(DatabaseTableView):
    remote_expr: Expr = None

    @classmethod
    def from_expr(cls, con, expr, name=None):
        name = name or gen_name()
        return cls(
            name=name,
            schema=expr.schema(),
            source=con,
            remote_expr=expr,
        )
```

**Lazy type registration via `Dispatch` with `register_lazy`.**

```python
registry = Dispatch()

@registry.register_lazy("sklearn")
def lazy_register_sklearn():
    registry.register(LinearRegression, return_constant(dt.float))
    registry.register(LogisticRegression, get_target_type)
    registry.register(ClassifierMixin, get_target_type)
```

---

## 8. Context Managers, Observability, and Resource Safety

**`@contextmanager` + `@curry` for parameterized resource management.**

```python
@contextmanager
@curry
def commit_context(repo, message):
    yield repo.index
    repo.index.commit(message)
```

**`@tracer.start_as_current_span("name")` on all major operations.**

```python
@tracer.start_as_current_span("execute")
def execute(expr: ir.Expr, **kwargs: Any):
    ...

@tracer.start_as_current_span("_transform_expr")
def _transform_expr(expr, **kwargs):
    ...
```

**`rbr_wrapper(reader, clean_up)` for cleanup-on-exhaust.**

```python
def to_pyarrow_batches(expr, *, chunk_size=1_000_000, **kwargs):
    reader = con.to_pyarrow_batches(expr, chunk_size=chunk_size, **kwargs)

    def clean_up():
        for table_name, conn in created.items():
            try:
                conn.drop_table(table_name, force=True)
            except Exception:
                conn.drop_view(table_name)

    return rbr_wrapper(reader, clean_up)
```

**`with self.lock:` for thread-safe server operations.**

```python
def do_get(self, context, rec, ticket):
    kwargs = loads(ticket.ticket)
    expr = kwargs.pop("expr")
    with self.lock:
        rbr = self._conn.to_pyarrow_batches(expr)
    return pyarrow.flight.RecordBatchStream(rbr)
```

**Nested context managers for temporary normalization.**

```python
@contextlib.contextmanager
def normalization_context(self, expr):
    typs = map(type, expr.ls.backends)
    with patch_normalize_token(*typs, f=self.normalize_backend):
        with patch_normalize_token(ops.DatabaseTable, f=self.normalize_databasetable):
            with patch_normalize_token(Read, f=self.cached_normalize_read):
                yield
```

---

## 9. Testing Conventions

**`__init__.py` is required in every test directory.** Without it, pytest collection will fail with `ImportError`.

**`pytest.param(..., id="descriptive-name")` for all parametrized cases.**

```python
@pytest.mark.parametrize(
    "connection,port",
    [
        pytest.param("duckdb", 5005, id="duckdb"),
        pytest.param("datafusion", 5005, id="datafusion"),
        pytest.param("sqlite", 5005, id="sqlite"),
        pytest.param("", 5005, id="xorq"),
    ],
)
def test_port_in_use(connection, port):
    ...
```

**Session-scoped fixtures for expensive resources, function-scoped for stateful ones.**

```python
@pytest.fixture(scope="session")
def parquet_dir(root_dir):
    return root_dir / "ci" / "ibis-testing-data" / "parquet"

@pytest.fixture(scope="function")
def pg():
    conn = xo.postgres.connect_env()
    remove_unexpected_tables(conn)
    yield conn
    remove_unexpected_tables(conn)
```

**Fixture chaining:** `con` -> `table` -> `df` for progressive data access.

```python
@pytest.fixture(scope="session")
def con(data_dir, ddl_file):
    conn = xo.connect()
    conn.read_parquet(data_dir / "parquet" / "functional_alltypes.parquet", "functional_alltypes")
    return conn

@pytest.fixture(scope="session")
def alltypes(con):
    return con.table("functional_alltypes")

@pytest.fixture(scope="session")
def df(alltypes):
    return alltypes.execute()
```

**Backend-specific conftest.py overrides.** Each backend directory has its own `conftest.py` that customizes fixtures for that engine.

**Custom assertion utilities with sensible defaults.**

```python
def assert_frame_equal(left, right, *args, **kwargs):
    left = left.reset_index(drop=True)
    right = right.reset_index(drop=True)
    kwargs.setdefault("check_dtype", True)
    tm.assert_frame_equal(left, right, *args, **kwargs)
```

---

## 10. Import Organization and API Surface

**`from __future__ import annotations` at the top of every file.**

**Import order (enforced by ruff):**
1. `from __future__ import annotations`
2. Standard library (`functools`, `pathlib`, `typing`)
3. Third-party (`pyarrow`, `toolz`, `opentelemetry`)
4. First-party (`xorq.*`)
5. Local/relative

**Two blank lines after all imports.**

**`TYPE_CHECKING` blocks for expensive or circular imports.**

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping
    import pandas as pd
```

**`__getattr__` for lazy backend loading.**

```python
def __getattr__(name):
    from xorq.vendor import ibis
    backend = load_backend(name) or ibis.load_backend(name)
    setattr(sys.modules[__name__], name, backend)
    return backend
```

**`@public` decorator + explicit `__all__` for API surface control.**

```python
@public
@frozen
class ParquetCache(Cache):
    ...

__all__ = [
    "ParquetCache",
    "ParquetSnapshotCache",
    *api.__all__,
]
```

**Union types with `|` syntax (PEP 604) and `Mapping` from collections.abc.**

```python
def create_table(
    self,
    name: str,
    obj: ir.Table | pa.Table | pa.RecordBatchReader | None = None,
    *,
    schema: sch.SchemaLike | None = None,
    database: str | None = None,
) -> ir.Table:
```

---

## 11. Anti-Patterns to Avoid

- **Never add packages that shadow existing ones** and then patch modules to restore the shadowed package's functionality
- **No mutable state on instances.** Use `@property` + `@functools.cache` instead of setting attributes
- **No bare `if/elif/else` chains** where `match/case` with a final `case _` is viable
- **No `for` loops** where a comprehension or `map()`/`reduce()` suffices
- **No `dict` where a `tuple` of two-tuples** or `FrozenOrderedDict` works for immutable data
- **No `@dataclass`.** Use `@frozen` from `attrs`
- **No `list` where `tuple` suffices.** Default to `tuple`
- **No `Union[X, Y]`.** Use `X | Y` syntax
- **No comments that restate code.** Code should be self-documenting via type annotations, descriptive names, and structural clarity

---

## 12. Commit and Development Conventions

- **Conventional commits:** `type(scope): message` (e.g., `feat(catalog): add new catalog`, `fix(cache): handle TTL expiry`)
- **Spike branches** for exploratory development, followed by refactoring and PR partitioning
- **Agent-generated code must be fully understood** by the code owner before PR submission
- **Tests should be aggressively refactored** by humans to ensure feature clarity
- **Package management:** Use `uv` for environment management
- **CI testing:** Tests run via `pytest` with backend-specific markers (`-m duckdb`, `-m postgres`)
