from __future__ import annotations

import pytest

import xorq.api as xo
from xorq.common.utils.graph_utils import find_all_sources
from xorq.expr.relations import TeeNode
from xorq.ibis_yaml.compiler import (
    YamlExpressionTranslator,
    build_expr,
    load_expr,
)
from xorq.writes import (
    BackendWriteThrough,
    ParquetWriteThrough,
    ThreadedBackendWriteThrough,
    WriteMode,
    WritePrimaryWriteThrough,
)


@pytest.fixture(scope="session")
def awards_players() -> object:
    # datafusion-backed: tee() needs a backend that allows concurrent reader
    # pulls, else execute() deadlocks on the single connection (ADR-0014).
    con = xo.connect()
    parquet_path = xo.config.options.pins.get_path("awards_players")
    return con.read_parquet(parquet_path, table_name="awards_players")


def _tee_op(expr: object) -> TeeNode:
    op = expr.op()
    assert isinstance(op, TeeNode)
    return op


def _profiles(expr: object) -> dict:
    return {con._profile.hash_name: con for con in find_all_sources(expr)}


def test_parquet_tee_roundtrip(awards_players: object, tmp_path: object) -> None:
    expr = awards_players.tee(ParquetWriteThrough(path=tmp_path / "out.parquet"))

    profiles = _profiles(expr)
    yaml_dict = YamlExpressionTranslator.to_yaml(expr, profiles)
    roundtrip = YamlExpressionTranslator.from_yaml(yaml_dict, profiles)

    op = _tee_op(roundtrip)
    assert isinstance(op.writer, ParquetWriteThrough)
    assert op.writer.path == (tmp_path / "out.parquet")
    assert op.writer.mode is WriteMode.CREATE
    assert op.drain is True
    # byte-stable round-trip (determinism contract)
    assert (
        YamlExpressionTranslator.to_yaml(roundtrip, _profiles(roundtrip)) == yaml_dict
    )
    # a tee is a transparent pass-through: same rows as the parent
    assert xo.execute(roundtrip).equals(xo.execute(awards_players))
    assert (tmp_path / "out.parquet").exists()


def test_parquet_tee_preserves_mode_and_drain(
    awards_players: object, tmp_path: object
) -> None:
    expr = awards_players.tee(
        ParquetWriteThrough(path=tmp_path / "out.parquet", mode="append"),
        drain=False,
    )

    profiles = _profiles(expr)
    yaml_dict = YamlExpressionTranslator.to_yaml(expr, profiles)
    op = _tee_op(YamlExpressionTranslator.from_yaml(yaml_dict, profiles))

    assert op.writer.mode is WriteMode.APPEND
    assert op.drain is False


def test_backend_tee_roundtrip(awards_players: object) -> None:
    target = xo.duckdb.connect()
    writer = BackendWriteThrough(target, table_name="sink")
    expr = awards_players.tee(writer)

    profiles = _profiles(expr)
    yaml_dict = YamlExpressionTranslator.to_yaml(expr, profiles)
    roundtrip = YamlExpressionTranslator.from_yaml(yaml_dict, profiles)

    op = _tee_op(roundtrip)
    assert type(op.writer) is BackendWriteThrough
    assert op.writer.table_name == "sink"
    assert op.writer.con is target
    assert xo.execute(roundtrip).equals(xo.execute(awards_players))


def test_threaded_backend_tee_roundtrip(awards_players: object) -> None:
    target = xo.duckdb.connect()
    expr = awards_players.tee(target, table_name="sink")
    assert isinstance(_tee_op(expr).writer, ThreadedBackendWriteThrough)

    profiles = _profiles(expr)
    yaml_dict = YamlExpressionTranslator.to_yaml(expr, profiles)
    roundtrip = YamlExpressionTranslator.from_yaml(yaml_dict, profiles)

    op = _tee_op(roundtrip)
    assert type(op.writer) is ThreadedBackendWriteThrough
    assert op.writer.table_name == "sink"
    assert op.writer.con is target


def test_write_primary_tee_roundtrip(awards_players: object, tmp_path: object) -> None:
    inner = ParquetWriteThrough(path=tmp_path / "out.parquet")
    expr = awards_players.tee(WritePrimaryWriteThrough(inner=inner))

    profiles = _profiles(expr)
    yaml_dict = YamlExpressionTranslator.to_yaml(expr, profiles)
    op = _tee_op(YamlExpressionTranslator.from_yaml(yaml_dict, profiles))

    assert isinstance(op.writer, WritePrimaryWriteThrough)
    assert isinstance(op.writer.inner, ParquetWriteThrough)
    assert op.writer.inner.path == (tmp_path / "out.parquet")


def test_tee_build_load_roundtrip(awards_players: object, builds_dir: object) -> None:
    target = xo.duckdb.connect()
    expr = awards_players.tee(target, table_name="sink")

    path = build_expr(expr, builds_dir=builds_dir)
    loaded = load_expr(path)
    assert isinstance(_tee_op(loaded).writer, ThreadedBackendWriteThrough)
