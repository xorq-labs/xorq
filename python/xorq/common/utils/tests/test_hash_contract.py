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

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import xorq_dasher

import xorq
import xorq.api as xo
from xorq.common.utils.dasher import normalize, tokenize
from xorq.common.utils.dasher._canonical import (
    NORMALIZATION_VERSION,
    canonical_column_digest,
    normalize_inmemorytable_canonical,
)
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
        "empty_string": pa.array([], type=pa.string()),
    }


GOLDEN_COLUMN_DIGESTS = {
    "int64_nulls": "f7e0cccc78c213f89bac19e52fb2f67d",
    "float64_edges": "8a2d78480547a06c5665712722a97e30",
    "bool_nulls": "448934082adc04da19a2274aab5d3b69",
    "decimal128": "27f431c699954a75aa6a9ff7a5be1881",
    "string": "e6e303236c8b24e8bb47ba4e568ddb8f",
    "timestamp_tz": "0a0e7066f0f567772d2185e5e7eff7d4",
    "list_int": "f8e26be3b81d390d0b214c95d8eab034",
    "struct": "9ad3a5c7685e160e7892d9603985e3d0",
    "map": "2f994665818074e02eb7d7d29b9a0c59",
    "dictionary_multichunk": "7f6586a4862513f7cd033019b8af5815",
    "empty_string": "95f33ef94af5d33af33b4f80eb67ad76",
}

GOLDEN_TOKENS = {
    "memtable": "c3a49075e3d5f635d0eb5b999bad74ca",
    "pandas_dataframe": "5a2b5b1d7e148d299ff2bd592a2e7714",
    "pyarrow_table": "f38b0e88acf5958f4955e9c8ec917a2e",
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


def test_digest_erases_slicing() -> None:
    big = pa.array(list(range(100)), type=pa.int64())
    direct = pa.array(list(range(3, 13)), type=pa.int64())
    assert canonical_column_digest(big.slice(3, 10)) == canonical_column_digest(direct)


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
    # stream. Identity must come from the stored proxy data only.
    def _forbidden(self: Expr, *args: object, **kwargs: object) -> None:
        raise AssertionError(
            "normalize_inmemorytable_canonical must not execute the expression"
        )

    monkeypatch.setattr(Expr, "to_pyarrow_batches", _forbidden)
    op = xo.memtable(_fixture_df(), name="name").op()
    tokenize(normalize_inmemorytable_canonical(op))
