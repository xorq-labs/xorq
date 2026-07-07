from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd

import xorq.api as xo


if TYPE_CHECKING:
    from pytest_snapshot.plugin import Snapshot

    from xorq.backends.sqlite import Backend


def test_hash(sqlite_con: Backend, df: pd.DataFrame) -> None:
    t = sqlite_con.create_table("test", df, overwrite=True)
    expr = t.mutate(my_hash=t.c.hash())

    assert not expr.execute().empty


def test_hash_pinned_values(sqlite_con: Backend, snapshot: Snapshot) -> None:
    # Pin the sqlite Hash op (ibis_hash_32 udf, a 32-bit blake2b digest) so an
    # unintended algorithm/encoding change is caught. The 1e20 float pins the
    # scientific-notation repr ("1e+20") the hash is taken over. Regenerate
    # deliberately with --snapshot-update; a diff here is a behavioral break.
    t = xo.memtable(
        {"i": [0, 1, 123], "f": [0.0, 1.5, 1e20], "s": ["", "a", "1"]}
    ).into_backend(sqlite_con)
    df = t.mutate(hi=t.i.hash(), hf=t.f.hash(), hs=t.s.hash()).execute()
    mapping = {
        **{f"int:{v}": int(h) for v, h in zip(df["i"], df["hi"])},
        **{f"float:{v}": int(h) for v, h in zip(df["f"], df["hf"])},
        **{f"str:{v!r}": int(h) for v, h in zip(df["s"], df["hs"])},
    }
    actual = json.dumps(mapping, indent=2, sort_keys=True) + "\n"
    snapshot.assert_match(actual, "ibis_hash_32.json")


def test_hash_distinguishes_types(sqlite_con: Backend) -> None:
    # Values with the same string form but different types must not collide:
    # int 1, float 1.0 and str "1" all hash to distinct buckets.
    t = xo.memtable({"i": [1], "f": [1.0], "s": ["1"]}).into_backend(sqlite_con)
    df = t.mutate(hi=t.i.hash(), hf=t.f.hash(), hs=t.s.hash()).execute()
    hashes = {int(df["hi"][0]), int(df["hf"][0]), int(df["hs"][0])}
    assert len(hashes) == 3


def test_hash_null_propagates(sqlite_con: Backend) -> None:
    t = sqlite_con.create_table(
        "test_hash_null", xo.memtable(pd.DataFrame({"c": ["a", None]})), overwrite=True
    )
    df = t.mutate(h=t.c.hash()).execute()
    assert df.loc[df["c"].isna(), "h"].isna().all()
    assert df.loc[df["c"].notna(), "h"].notna().all()
