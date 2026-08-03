"""Golden-token contract tests for xorq's canonical hashing.

These tokens are the *definition* of hash identity for in-memory table data
(``NORMALIZATION_VERSION`` in ``xorq.common.utils.dasher._canonical``). Any
code change that alters them silently renames every existing build artifact
and invalidates every cache entry fleet-wide — so it must fail here, in the
PR that causes it, not in production.

If a test in this module fails:

* You changed normalization on purpose → bump ``NORMALIZATION_VERSION``,
  regenerate the goldens below, and call out the cache-invalidation event in
  the release notes.
* You didn't → a dependency broke a stability assumption. The failure
  message prints the hash-relevant dependency versions and the normalized
  tuple; compare against a passing environment to localize the divergent
  component.

Deliberately excluded from the goldens: UDF tokens (cloudpickle/bytecode are
Python-version-coupled) and generated-SQL surfaces (sqlglot-coupled) — those
are over-discrimination surfaces, not per-environment contracts. The
canonical digests here were verified byte-identical across pyarrow
18.0.0–25.0.0 (probe script on issue #2191).
"""

from __future__ import annotations

import datetime as dt
import decimal
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import xorq_dasher
from xorq_dasher.rules.expr import (
    normalize_memory_databasetable as dasher_normalize_memory_databasetable,
)

import xorq
import xorq.api as xo
from xorq.backends.pandas import Backend as PandasBackend
from xorq.backends.sqlite import Backend as SqliteBackend
from xorq.common.utils.dasher import _relations, normalize, tokenize
from xorq.common.utils.dasher._canonical import (
    NORMALIZATION_VERSION,
    canonical_column_digest,
    normalize_inmemorytable_canonical,
    normalize_pyarrow_table_canonical,
)
from xorq.common.utils.dasher._relations import _dispatch_databasetable
from xorq.vendor.ibis.expr.types.core import Expr


def _env_versions() -> str:
    mods = (pa, pd, np, xorq_dasher, xorq)
    return ", ".join(f"{m.__name__}=={m.__version__}" for m in mods)


def _contract_message(name: str, obj: object) -> str:
    return (
        f"hash contract broken for {name!r} "
        f"(NORMALIZATION_VERSION={NORMALIZATION_VERSION}; {_env_versions()}).\n"
        f"Either bump NORMALIZATION_VERSION and regenerate goldens "
        f"(deliberate), or diff this normalized form against a passing "
        f"environment to find the divergent component:\n{normalize(obj)!r}"
    )


def _fixture_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "i": [1, 2, None],
            "f": [1.5, float("nan"), -0.0],
            "s": ["a", None, "é中文"],
            "t": pd.to_datetime(["2020-01-01", "2021-06-15", "NaT"]).tz_localize("UTC"),
        }
    )


def _fixture_columns() -> dict[str, pa.Array | pa.ChunkedArray]:
    D = decimal.Decimal
    utc = dt.timezone.utc
    return {
        "int64_nulls": pa.array([1, None, 2**62], type=pa.int64()),
        "float64_edges": pa.array(
            [0.0, -0.0, float("nan"), float("inf"), None], type=pa.float64()
        ),
        "bool_nulls": pa.array([True, None, False]),
        "decimal128": pa.array(
            [D("1.230000000"), None, D("-99.999999999")], type=pa.decimal128(38, 9)
        ),
        "string": pa.array(["apple", "banana", "apple", None, "", "é中文"]),
        "timestamp_tz": pa.array(
            [dt.datetime(2020, 1, 1, tzinfo=utc), None],
            type=pa.timestamp("us", tz="UTC"),
        ),
        "list_int": pa.array([[1, 2, None], None, []], type=pa.list_(pa.int64())),
        "struct": pa.array(
            [{"a": 1, "b": "x"}, None, {"a": None, "b": ""}],
            type=pa.struct([("a", pa.int64()), ("b", pa.string())]),
        ),
        "map": pa.array(
            [[("k1", 1), ("k2", 2)], None, []], type=pa.map_(pa.string(), pa.int64())
        ),
        "dictionary_multichunk": pa.chunked_array(
            [
                pa.array(["apple", "banana", "apple"]).dictionary_encode(),
                pa.array(["cherry", "banana", None]).dictionary_encode(),
            ]
        ),
        # interior dictionary layer: erased by ``col.cast``'s nested decode,
        # the most cast-machinery-dependent canonicalization branch
        "nested_dict_in_list": pa.array(
            [["a", "b"], None, ["a"]], type=pa.list_(pa.string())
        ).cast(pa.list_(pa.dictionary(pa.int32(), pa.string()))),
        # view type: canonicalized to large_string (long value forces an
        # out-of-line buffer, the layout-dependent case)
        "string_view": pa.array(
            ["apple", None, "", "é中文", "x" * 100], type=pa.string_view()
        ),
        "empty_string": pa.array([], type=pa.string()),
    }


GOLDEN_COLUMN_DIGESTS = {
    "int64_nulls": "f7e0cccc78c213f89bac19e52fb2f67d",
    "float64_edges": "8a2d78480547a06c5665712722a97e30",
    "bool_nulls": "448934082adc04da19a2274aab5d3b69",
    "decimal128": "27f431c699954a75aa6a9ff7a5be1881",
    "string": "8174a8d746db7b57a1d0721bf83487cd",
    "timestamp_tz": "0a0e7066f0f567772d2185e5e7eff7d4",
    "list_int": "26e34968d249b6e878408d1c638b1066",
    "struct": "d7c505e657beab1d3b433ee2633c9131",
    "map": "d4a434fe254471ca2176cd8b7ae9120e",
    "dictionary_multichunk": "b50c865c81720aa1354729bda834915d",
    "nested_dict_in_list": "5ea7e62c328b1dc0af45bf7307b544c0",
    "string_view": "a8d8f0321b2ca4aa7ca2b1768f9ee97e",
    "empty_string": "1ff18035104640f53e6bbf8ca6bb9aa1",
}

GOLDEN_TOKENS = {
    "memtable": "97a59f9c8d55dc45684174488aa34f1e",
    "pandas_dataframe": "eef21642b55d3160e3f85e18cee39c99",
    "pyarrow_table": "798314b6d6e801a854c99cfa5fb8ca70",
}


@pytest.mark.parametrize("name", sorted(GOLDEN_COLUMN_DIGESTS))
def test_golden_column_digest(name: str) -> None:
    col = _fixture_columns()[name]
    actual = canonical_column_digest(col)
    assert actual == GOLDEN_COLUMN_DIGESTS[name], _contract_message(name, col)


def test_golden_memtable_token() -> None:
    op = xo.memtable(_fixture_df(), name="name").op()
    actual = tokenize(normalize_inmemorytable_canonical(op))
    assert actual == GOLDEN_TOKENS["memtable"], _contract_message("memtable", op)


def test_golden_pandas_dataframe_token() -> None:
    df = _fixture_df()
    assert tokenize(df) == GOLDEN_TOKENS["pandas_dataframe"], _contract_message(
        "pandas_dataframe", df
    )


def test_golden_pyarrow_table_token() -> None:
    table = pa.Table.from_pandas(_fixture_df())
    assert tokenize(table) == GOLDEN_TOKENS["pyarrow_table"], _contract_message(
        "pyarrow_table", table
    )


def test_digest_erases_chunk_layout() -> None:
    single = pa.array([1, 2, None, 4, 5], type=pa.int64())
    chunked = pa.chunked_array(
        [pa.array([1, 2, None], type=pa.int64()), pa.array([4, 5], type=pa.int64())]
    )
    assert canonical_column_digest(chunked) == canonical_column_digest(single)


SLICE_CASES = {
    # fixed-width: IPC truncation alone happens to be exact
    "int64": pa.array(list(range(100)), type=pa.int64()),
    # bitmap-packed: slice offsets are not byte-aligned
    "bool_nulls": pa.array([True, None, False] * 10),
    # var-length: slice offsets are NOT rebased by the IPC writer — only
    # explicit compaction (concat_arrays) makes sliced == fresh
    "string": pa.array([f"s{i}" * (i % 3 + 1) for i in range(30)]),
    "list_int": pa.array(
        [[i, None, i * 2] if i % 3 else None for i in range(30)],
        type=pa.list_(pa.int64()),
    ),
    # view: serializes its physical buffer layout verbatim — only the cast
    # to large_string erases it (concat_arrays does NOT compact views)
    "string_view": pa.array(
        [f"s{i}" * (i % 5 + 1) for i in range(30)], type=pa.string_view()
    ),
}


@pytest.mark.parametrize("name", sorted(SLICE_CASES))
def test_digest_erases_slicing(name: str) -> None:
    big = SLICE_CASES[name]
    sliced = big.slice(3, 10)
    fresh = pa.array(sliced.to_pylist(), type=big.type)
    assert canonical_column_digest(sliced) == canonical_column_digest(fresh)


def test_digest_erases_offset_width() -> None:
    # string vs large_string (and list vs large_list) differ only in offset
    # width — a physical encoding. The canonical form widens to large_*, so
    # both the digests and the full table normalizations must agree.
    strings = ["apple", None, "", "é中文"]
    assert canonical_column_digest(pa.array(strings, type=pa.string())) == (
        canonical_column_digest(pa.array(strings, type=pa.large_string()))
    )
    lists = [[1, None], None, []]
    assert canonical_column_digest(pa.array(lists, type=pa.list_(pa.int64()))) == (
        canonical_column_digest(pa.array(lists, type=pa.large_list(pa.int64())))
    )
    assert normalize_pyarrow_table_canonical(
        pa.table({"c": pa.array(strings, type=pa.string())})
    ) == normalize_pyarrow_table_canonical(
        pa.table({"c": pa.array(strings, type=pa.large_string())})
    )


def test_digest_erases_view_encoding() -> None:
    # string_view/binary_view serialize their physical buffer layout (views
    # + variadic data buffers), which varies with how the array was built —
    # the canonical form rewrites them to large_string/large_binary, so all
    # three offset representations of the same values must agree, digests
    # and full-table normalizations alike.
    strings = ["apple", None, "", "é中文", "x" * 100]
    want = canonical_column_digest(pa.array(strings, type=pa.large_string()))
    assert canonical_column_digest(pa.array(strings, type=pa.string_view())) == want
    bins = [b"\x00\x01" * 20, None, b""]
    assert canonical_column_digest(pa.array(bins, type=pa.binary_view())) == (
        canonical_column_digest(pa.array(bins, type=pa.large_binary()))
    )
    assert normalize_pyarrow_table_canonical(
        pa.table({"c": pa.array(strings, type=pa.string_view())})
    ) == normalize_pyarrow_table_canonical(
        pa.table({"c": pa.array(strings, type=pa.large_string())})
    )


def test_digest_refuses_list_view_types() -> None:
    # pyarrow's list_view -> list cast emits invalid arrays (offsets fail
    # validate(full=True); seen on 18 and 21), so list-view columns cannot
    # be canonicalized — and hashing them raw would serialize physical
    # buffer layout. Refuse loudly, at any nesting depth.
    lv = pa.array([[1, 2, None], None, []], type=pa.list_view(pa.int64()))
    with pytest.raises(NotImplementedError, match="list-view"):
        canonical_column_digest(lv)
    nested = pa.array([], type=pa.struct([("x", pa.large_list_view(pa.int64()))]))
    with pytest.raises(NotImplementedError, match="list-view"):
        canonical_column_digest(nested)


def test_digest_erases_dictionary_encoding() -> None:
    plain = pa.array(["apple", "banana", "apple", "cherry", None])
    assert canonical_column_digest(plain.dictionary_encode()) == (
        canonical_column_digest(plain)
    )


def test_digest_discriminates_dictionary_values() -> None:
    # Two dictionary arrays with identical indices but different dictionary
    # values MUST hash differently (an un-decoded batch message serializes
    # indices only — the exact under-discrimination this module exists to
    # prevent).
    a = pa.array(["x", "y", "x"]).dictionary_encode()
    b = pa.array(["p", "q", "p"]).dictionary_encode()
    assert canonical_column_digest(a) != canonical_column_digest(b)


def test_digest_discriminates_signedness() -> None:
    # Same buffer bytes, different logical type: identity must come from the
    # schema component, not the value bytes alone.
    signed = pa.array([1, -128], type=pa.int8())
    unsigned = signed.view(pa.uint8())  # bit-identical buffers, values [1, 128]
    op_s = xo.memtable(pa.table({"c": signed})).op()
    op_u = xo.memtable(pa.table({"c": unsigned})).op()
    assert tokenize(normalize_inmemorytable_canonical(op_s)) != tokenize(
        normalize_inmemorytable_canonical(op_u)
    )


def test_memtable_normalization_does_not_execute_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The pa20↔21 regression came from hashing a backend-executed batch
    # stream. Identity must come from the stored proxy data only
    # (``op.data.to_pyarrow`` is a proxy format conversion, not execution).
    # NOTE: this guard is enumerative, not exhaustive — it proves these
    # specific Expr entry points are unused, not that no execution of any
    # kind occurs. A regression that reached a backend directly (e.g. via
    # ``dt.source``) would not trip it.
    def _forbidden(self: Expr, *args: object, **kwargs: object) -> None:
        raise AssertionError(
            "normalize_inmemorytable_canonical must not execute the expression"
        )

    for entry_point in ("to_pyarrow_batches", "to_pyarrow", "execute"):
        monkeypatch.setattr(Expr, entry_point, _forbidden)
    op = xo.memtable(_fixture_df(), name="name").op()
    tokenize(normalize_inmemorytable_canonical(op))


def test_digest_refuses_unerasable_nested_dictionary() -> None:
    # ``_canonical_type`` does not rewrite union types, so a
    # dictionary nested inside one cannot be decoded away — and serializing
    # it would drop the dictionary values from the digest. Refusing loudly
    # beats silently under-discriminating.
    dict_arr = pa.array(["x", "y"]).dictionary_encode()
    int_arr = pa.array([1, 2], type=pa.int64())
    type_ids = pa.array([0, 1], type=pa.int8())
    union = pa.UnionArray.from_sparse(type_ids, [dict_arr, int_arr])
    with pytest.raises(NotImplementedError, match="dictionary"):
        canonical_column_digest(union)


def test_zero_column_table_discriminates_row_count() -> None:
    # With no columns there are no digests to carry the row count, so the
    # normalization carries ``num_rows`` explicitly.
    two = pa.table({"a": [1, 2]}).drop_columns(["a"])
    three = pa.table({"a": [1, 2, 3]}).drop_columns(["a"])
    assert two.num_columns == 0 and two.num_rows == 2
    assert normalize_pyarrow_table_canonical(two) != (
        normalize_pyarrow_table_canonical(three)
    )


def test_pandas_backend_databasetable_routes_to_canonical() -> None:
    # xorq_dasher's DatabaseTable dispatch would hash a pandas-backend
    # table's to_pyarrow_batches() IPC stream — the pyarrow-version-coupled
    # form (#2191). The dispatcher must route it to the canonical rule.
    con = PandasBackend().connect({"t": _fixture_df()})
    normalized = _dispatch_databasetable(con.table("t").op())
    assert normalized[:2] == ("xorq.MemoryDatabaseTable", NORMALIZATION_VERSION)


def test_sqlite_memory_databasetable_routes_to_canonical() -> None:
    # Same as above for in-memory sqlite (xorq_dasher's sqlite rule falls
    # back to the batch-stream form when is_in_memory()).
    con = SqliteBackend().connect()
    assert con.is_in_memory()
    con.create_table("t", pd.DataFrame({"a": [1, 2, 3]}))
    normalized = _dispatch_databasetable(con.table("t").op())
    assert normalized[:2] == ("xorq.MemoryDatabaseTable", NORMALIZATION_VERSION)


def test_dasher_memory_rule_tag_still_matches_safety_net() -> None:
    # The fall-through safety net in _dispatch_databasetable recognizes
    # dasher's memory rule by its tuple tag ("ibis.MemoryDatabaseTable", ...).
    # The fallthrough test below proves the net catches that tag, but only
    # this test proves dasher still EMITS it — if a dasher release renames
    # the tag, the net dies silently and #2191 revives; fail here instead.
    con = PandasBackend().connect({"t": _fixture_df()})
    result = dasher_normalize_memory_databasetable(con.table("t").op())
    assert result[:1] == ("ibis.MemoryDatabaseTable",), (
        "xorq_dasher renamed its memory-rule tag "
        f"(got {result[:1]!r}; {_env_versions()}); update the safety net in "
        "_relations._dispatch_databasetable to match"
    )


def test_dispatcher_fallthrough_net_reroutes_dasher_memory_rule(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    # If xorq_dasher's fall-through dispatch ever resolves a table to its
    # batch-stream memory rule (e.g. a future/renamed backend — dasher still
    # maps a backend *named* "xorq" to it), the dispatcher must re-route to
    # the canonical form rather than let #2191 silently revive.
    con = SqliteBackend().connect(f"{tmp_path}/t.db")
    assert not con.is_in_memory()
    con.create_table("t", pd.DataFrame({"a": [1, 2, 3]}))
    dt = con.table("t").op()
    monkeypatch.setattr(
        _relations,
        "normalize_databasetable",
        lambda dt: ("ibis.MemoryDatabaseTable", "version-coupled-form"),
    )
    normalized = _dispatch_databasetable(dt)
    assert normalized[:2] == ("xorq.MemoryDatabaseTable", NORMALIZATION_VERSION)


def test_probe_script_matches_module() -> None:
    # scripts/canonical_digest_xver_probe.py is a hand-maintained standalone
    # replica of the module's digest logic (it must import nothing from
    # xorq so it runs in bare venvs across pyarrow versions). Any edit to
    # one that misses the other silently invalidates the cross-version
    # verification story — pin them to each other over the probe's corpus.
    probe_path = (
        Path(__file__).parents[5] / "scripts" / "canonical_digest_xver_probe.py"
    )
    if not probe_path.exists():
        pytest.skip("probe script not present (installed-package test run)")
    spec = importlib.util.spec_from_file_location(
        "canonical_digest_xver_probe", probe_path
    )
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    for name, col in probe.corpus().items():
        assert probe.canonical_column_digest(col) == canonical_column_digest(col), (
            f"probe script and _canonical module disagree on corpus column {name!r}: "
            f"their digest logic has drifted apart"
        )


def test_refuses_extension_types() -> None:
    # Extension type parameters live in __arrow_ext_serialize__ bytes
    # (third-party, no cross-version stability contract) and str(type)
    # omits them — pandas period columns of freq 'M' vs 'D' with equal
    # ordinals produce IDENTICAL storage and str(type), so hashing the
    # storage would collide logically different tables. Refuse loudly.
    monthly = pa.Table.from_pandas(
        pd.DataFrame({"p": pd.PeriodIndex.from_ordinals([0, 1], freq="M")})
    )
    with pytest.raises(NotImplementedError, match="extension"):
        normalize_pyarrow_table_canonical(monthly)
    # nested extensions must be found too
    period_type = monthly.schema.field("p").type
    nested = pa.struct([("x", period_type)])
    with pytest.raises(NotImplementedError, match="extension"):
        canonical_column_digest(pa.array([], type=nested))
