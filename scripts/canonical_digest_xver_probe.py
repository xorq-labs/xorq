"""Cross-pyarrow-version probe for the canonical column digest.

Standalone replica of ``xorq.common.utils.dasher._canonical`` digest logic
(no xorq import, so it runs in a bare venv with just pyarrow+xxhash+pandas).
Writes one ``name<TAB>canonical-type<TAB>digest`` line per corpus column to
stdout; run under multiple pyarrow versions and diff the outputs — they must
be byte-identical. The canonical-type column guards the ``str(type)`` schema
surface of the normalized tuple: a pyarrow type-repr change across versions
would drift tokens even with stable digests.

    for v in 18.0.0 20.0.0 21.0.0 25.0.0; do
      uv venv /tmp/pa$v -q && uv pip install -p /tmp/pa$v -q pyarrow==$v xxhash pandas
      /tmp/pa$v/bin/python scripts/canonical_digest_xver_probe.py > /tmp/pa$v.out
    done
    diff /tmp/pa18.0.0.out /tmp/pa21.0.0.out  # etc.
"""

from __future__ import annotations

import datetime as dt
import decimal
import sys

import pyarrow as pa
import xxhash


def _canonical_type(typ: pa.DataType) -> pa.DataType:
    if pa.types.is_dictionary(typ):
        return _canonical_type(typ.value_type)
    if pa.types.is_string(typ) or pa.types.is_string_view(typ):
        return pa.large_string()
    if pa.types.is_binary(typ) or pa.types.is_binary_view(typ):
        return pa.large_binary()
    if pa.types.is_list(typ) or pa.types.is_large_list(typ):
        return pa.large_list(_canonical_type(typ.value_type))
    if pa.types.is_fixed_size_list(typ):
        return pa.list_(_canonical_type(typ.value_type), typ.list_size)
    if pa.types.is_struct(typ):
        return pa.struct(
            [
                (typ.field(i).name, _canonical_type(typ.field(i).type))
                for i in range(typ.num_fields)
            ]
        )
    if pa.types.is_map(typ):
        return pa.map_(
            _canonical_type(typ.key_type),
            _canonical_type(typ.item_type),
        )
    return typ


def canonical_column_digest(col: pa.Array | pa.ChunkedArray) -> str:
    canonical_type = _canonical_type(col.type)
    if canonical_type != col.type:
        col = col.cast(canonical_type)
    if isinstance(col, pa.ChunkedArray):
        arr = col.combine_chunks()
    else:
        arr = pa.concat_arrays([col])
    batch = pa.RecordBatch.from_arrays([arr], names=["c"])
    return xxhash.xxh128(batch.serialize().to_pybytes()).hexdigest()


def corpus() -> dict[str, pa.Array | pa.ChunkedArray]:
    D = decimal.Decimal
    utc = dt.timezone.utc
    cols = {
        "int64_nulls": pa.array([1, None, 2**62], type=pa.int64()),
        "float64_edges": pa.array(
            [0.0, -0.0, float("nan"), float("inf"), None], type=pa.float64()
        ),
        "bool_nulls": pa.array([True, None, False]),
        "decimal128": pa.array(
            [D("1.230000000"), None, D("-99.999999999")], type=pa.decimal128(38, 9)
        ),
        "string": pa.array(["apple", "banana", "apple", None, "", "é中文"]),
        "large_string": pa.array(["apple", None, ""], type=pa.large_string()),
        "binary": pa.array([b"\x00\x01", None, b""], type=pa.binary()),
        "timestamp_tz": pa.array(
            [dt.datetime(2020, 1, 1, tzinfo=utc), None],
            type=pa.timestamp("us", tz="UTC"),
        ),
        "date32": pa.array([dt.date(2020, 1, 1), None]),
        "duration": pa.array([dt.timedelta(seconds=1), None]),
        "float32": pa.array([1.5, None], type=pa.float32()),
        "list_int": pa.array([[1, 2, None], None, []], type=pa.list_(pa.int64())),
        "list_string": pa.array([["a", None], None, []], type=pa.list_(pa.string())),
        "fixed_size_list": pa.array(
            [[1, 2], None, [3, None]], type=pa.list_(pa.int64(), 2)
        ),
        "fixed_size_list_string": pa.array(
            [["a", "bb"], None, ["c", None]], type=pa.list_(pa.string(), 2)
        ),
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
        "nested_dict_in_list": pa.array(
            [["a", "b"], None, ["a"]], type=pa.list_(pa.string())
        ).cast(pa.list_(pa.dictionary(pa.int32(), pa.string()))),
        "empty_string": pa.array([], type=pa.string()),
        # view types: physical buffer layout is erased by canonicalizing to
        # large_string/large_binary (long values force out-of-line buffers)
        "string_view": pa.array(
            ["apple", None, "", "é中文", "x" * 100], type=pa.string_view()
        ),
        "binary_view": pa.array([b"\x00\x01" * 20, None, b""], type=pa.binary_view()),
    }
    # physical-layout variants: sliced and chunked forms of var-length data
    s = pa.array([f"s{i}" * (i % 3 + 1) for i in range(30)])
    cols["string_sliced"] = s.slice(3, 10)
    cols["string_chunked"] = pa.chunked_array([s.slice(0, 11), s.slice(11)])
    b = pa.array([True, None, False] * 10)
    cols["bool_sliced_unaligned"] = b.slice(3, 10)
    li = pa.array(
        [[i, None, i * 2] if i % 3 else None for i in range(30)],
        type=pa.list_(pa.int64()),
    )
    cols["list_sliced"] = li.slice(3, 10)
    sv = pa.array([f"s{i}" * (i % 5 + 1) for i in range(30)], type=pa.string_view())
    cols["string_view_sliced"] = sv.slice(3, 10)
    cols["string_view_chunked"] = pa.chunked_array([sv.slice(0, 11), sv.slice(11)])
    return cols


def main() -> None:
    sys.stdout.write(f"# pyarrow=={pa.__version__}\n")
    for name, col in corpus().items():
        canonical_type = _canonical_type(col.type)
        sys.stdout.write(
            f"{name}\t{canonical_type!s}\t{canonical_column_digest(col)}\n"
        )


if __name__ == "__main__":
    main()
