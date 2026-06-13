from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

import xorq.api as xo
import xorq.expr.datatypes as dt
from xorq.caching import ParquetSnapshotCache
from xorq.cli import cli
from xorq.common.utils.dasher import tokenize
from xorq.common.utils.defer_utils import deferred_read_parquet
from xorq.common.utils.graph_utils import walk_nodes
from xorq.expr.ml import (
    Pipeline,
    deferred_fit_predict_sklearn,
    deferred_fit_transform,
)
from xorq.expr.pin_lib import (
    PinInfo,
    pin_caches,
    pin_infos,
    pinned_tag_nodes,
    verify_pinned,
)
from xorq.expr.relations import CachedNode, Read
from xorq.expr.udf import ExprScalarUDF
from xorq.ibis_yaml.compiler import build_expr, load_expr
from xorq.vendor.ibis.expr.operations.relations import InMemoryTable


sk_linear_model = pytest.importorskip("sklearn.linear_model")
sk_preprocessing = pytest.importorskip("sklearn.preprocessing")
sklearn_pipeline = pytest.importorskip("sklearn.pipeline")


deferred_linear_regression = deferred_fit_predict_sklearn(
    cls=sk_linear_model.LinearRegression, return_type=dt.float64
)


def make_data():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    # y = 1 * x_0 + 2 * x_1 + 3
    y = np.dot(X, np.array([1, 2])) + 3
    df = pd.DataFrame(np.hstack((X, y[:, np.newaxis]))).rename(
        columns=lambda x: chr(x + ord("a"))
    )
    (*features, target) = df.columns
    return (df, features, target)


@pytest.fixture(scope="module")
def train_path(tmp_path_factory):
    (df, _, _) = make_data()
    path = tmp_path_factory.mktemp("train") / "train.parquet"
    df.to_parquet(path)
    return path


@pytest.fixture
def cache(tmp_path):
    return ParquetSnapshotCache.from_kwargs(base_path=tmp_path / "cache")


@pytest.fixture
def predict_expr(train_path, cache):
    (_, features, target) = make_data()
    con = xo.connect()
    t = deferred_read_parquet(train_path, con, "train")
    instance = deferred_linear_regression(t, target, features, cache=cache)
    return t.mutate(predicted=instance.deferred_other.on_expr(t))


@pytest.fixture
def uncached_predict_expr(train_path):
    (_, features, target) = make_data()
    con = xo.connect()
    t = deferred_read_parquet(train_path, con, "train")
    instance = deferred_linear_regression(t, target, features)
    return t.mutate(predicted=instance.deferred_other.on_expr(t))


def pinned_model_read(node):
    (tag,) = pinned_tag_nodes(node.computed_kwargs_expr)
    read = tag.parent
    assert isinstance(read, Read)
    return read


def test_pin_predictions_match(predict_expr):
    expected = predict_expr.execute()
    pinned = pin_caches(predict_expr)
    pd.testing.assert_frame_equal(expected, pinned.execute())


def test_pin_swaps_cache_for_tagged_read(predict_expr):
    pinned = pin_caches(predict_expr)
    (node,) = walk_nodes(ExprScalarUDF, pinned)
    read = pinned_model_read(node)
    assert dict(read.read_kwargs).get("relocate")
    # the fit (and its cache) are gone from the pinned expression
    assert not walk_nodes(CachedNode, pinned)


def test_pin_uncached_fit_is_noop(uncached_predict_expr):
    # pin never stages a cache: an uncached fit is left as compute
    pinned = pin_caches(uncached_predict_expr)
    assert not pinned_tag_nodes(pinned)
    assert tokenize(pinned) == tokenize(uncached_predict_expr)


def test_pin_does_not_retrain(train_path, cache):
    fit_count = {"n": 0}

    def counting_fit(df, target):
        fit_count["n"] += 1
        model = sk_linear_model.LinearRegression()
        model.fit(df, target)
        return model

    (_, features, target) = make_data()
    con = xo.connect()
    t = deferred_read_parquet(train_path, con, "train")
    instance = deferred_fit_transform(
        expr=t,
        features=list(features),
        fit=counting_fit,
        other=lambda model, df: model.predict(df),
        return_type=dt.float64,
        target=target,
        name_infix="predict",
        cache=cache,
    )
    expr = t.mutate(predicted=instance.deferred_other.on_expr(t))

    pinned = pin_caches(expr)
    n_after_pin = fit_count["n"]
    assert n_after_pin == 1
    for _ in range(3):
        pinned.execute()
    assert fit_count["n"] == n_after_pin


def test_pinned_model_content_contributes_to_token(train_path, tmp_path):
    # two pins of a structurally identical expression that pin different
    # model bytes must tokenize differently
    calls = []

    def stamped_fit(df, target):
        model = sk_linear_model.LinearRegression()
        model.fit(df, target)
        model.stamp_ = len(calls)
        calls.append(None)
        return model

    def make_expr(cache):
        (_, features, target) = make_data()
        con = xo.connect()
        t = deferred_read_parquet(train_path, con, "train")
        instance = deferred_fit_transform(
            expr=t,
            features=list(features),
            fit=stamped_fit,
            other=lambda model, df: model.predict(df),
            return_type=dt.float64,
            target=target,
            name_infix="predict",
            cache=cache,
        )
        return t.mutate(predicted=instance.deferred_other.on_expr(t))

    pinned_a = pin_caches(
        make_expr(ParquetSnapshotCache.from_kwargs(base_path=tmp_path / "a"))
    )
    pinned_b = pin_caches(
        make_expr(ParquetSnapshotCache.from_kwargs(base_path=tmp_path / "b"))
    )
    (info_a,) = pin_infos(pinned_a)
    (info_b,) = pin_infos(pinned_b)
    assert info_a.content_token != info_b.content_token
    assert tokenize(pinned_a) != tokenize(pinned_b)


def test_pin_infos(predict_expr):
    (info,) = pin_infos(pin_caches(predict_expr))
    assert info.content_token
    assert info.source_token and info.cache_key
    assert not pin_infos(predict_expr)


def test_verify_pinned(predict_expr):
    (result,) = verify_pinned(pin_caches(predict_expr))
    assert result.ok


def test_pin_survives_extra_tag(predict_expr):
    # an extra tag atop the pinned read (e.g. a pipeline step tag) must not
    # hide the pin from introspection
    pinned = pin_caches(predict_expr)
    (expected_info,) = pin_infos(pinned)
    (node,) = walk_nodes(ExprScalarUDF, pinned)
    retagged = node.with_computed_kwargs_expr(
        node.computed_kwargs_expr.tag("user_note", reason="approved")
    )
    (tag_node,) = pinned_tag_nodes(retagged.computed_kwargs_expr)
    assert PinInfo.from_tag_node(tag_node) == expected_info


def test_pin_build_roundtrip(predict_expr, tmp_path):
    expected = predict_expr.execute()
    pinned = pin_caches(predict_expr)
    build_path = build_expr(pinned, builds_dir=tmp_path / "builds")
    # the pinned model parquet is packed inside the artifact, byte-for-byte
    # identical to the cache file it came from (copied, not re-encoded)
    (packed_file,) = (build_path / "reads").glob("*.parquet")
    (info,) = pin_infos(pinned)
    (cache_file,) = (tmp_path / "cache").rglob(f"{info.cache_key}.parquet")
    assert packed_file.read_bytes() == cache_file.read_bytes()

    loaded = load_expr(build_path)
    # loading stays lazy: the model arrives as a registered parquet table
    # (metadata only), not in-memory data, and the pin tag survives
    (node,) = walk_nodes(ExprScalarUDF, loaded)
    assert pinned_tag_nodes(node.computed_kwargs_expr)
    assert not walk_nodes(InMemoryTable, node.computed_kwargs_expr)
    pd.testing.assert_frame_equal(expected, loaded.execute())
    (result,) = verify_pinned(loaded, reads_dir=build_path / "reads")
    assert result.ok


def test_pin_build_relocated_read_survives_source_removal(predict_expr, tmp_path):
    expected = predict_expr.execute()
    pinned = pin_caches(predict_expr)
    build_path = build_expr(pinned, builds_dir=tmp_path / "builds")
    # the build must not depend on the cache dir the pin read through
    (info,) = pin_infos(pinned)
    for cached in tmp_path.rglob(f"{info.cache_key}.parquet"):
        if build_path not in cached.parents:
            cached.unlink()
    moved = tmp_path / "moved"
    shutil.move(build_path, moved)
    loaded = load_expr(moved)
    pd.testing.assert_frame_equal(expected, loaded.execute())


def test_pin_no_relocate_reads_from_cache(predict_expr, tmp_path):
    expected = predict_expr.execute()
    pinned = pin_caches(predict_expr, relocate=False)
    (node,) = walk_nodes(ExprScalarUDF, pinned)
    read = pinned_model_read(node)
    assert "relocate" not in dict(read.read_kwargs)
    pd.testing.assert_frame_equal(expected, pinned.execute())
    build_path = build_expr(pinned, builds_dir=tmp_path / "builds")
    # without relocation the model stays at the cache path
    assert not (build_path / "reads").exists()
    loaded = load_expr(build_path)
    pd.testing.assert_frame_equal(expected, loaded.execute())


def test_pin_build_deterministic(predict_expr, tmp_path):
    pinned = pin_caches(predict_expr)
    first = build_expr(pinned, builds_dir=tmp_path / "builds")
    second = build_expr(pin_caches(predict_expr), builds_dir=tmp_path / "builds")
    assert first == second


def test_pin_registered_table_build_roundtrip(tmp_path, cache):
    (df, features, target) = make_data()
    con = xo.connect()
    t = con.register(df, "t")
    instance = deferred_linear_regression(t, target, features, cache=cache)
    expr = t.mutate(predicted=instance.deferred_other.on_expr(t))
    expected = expr.execute()

    pinned = pin_caches(expr)
    build_path = build_expr(pinned, builds_dir=tmp_path / "builds")
    loaded = load_expr(build_path)
    pd.testing.assert_frame_equal(expected, loaded.execute())
    (result,) = verify_pinned(loaded, reads_dir=build_path / "reads")
    assert result.ok


def test_pin_pipeline_multi_step(train_path, tmp_path, cache):
    (_, features, target) = make_data()
    con = xo.connect()
    t = deferred_read_parquet(train_path, con, "train")
    pipeline = Pipeline.from_instance(
        sklearn_pipeline.Pipeline(
            [
                ("scaler", sk_preprocessing.StandardScaler()),
                ("lr", sk_linear_model.LinearRegression()),
            ]
        )
    )
    fitted = pipeline.fit(t, features=list(features), target=target, cache=cache)
    expr = fitted.predict(t)
    expected = expr.execute()

    pinned = pin_caches(expr)
    pd.testing.assert_frame_equal(expected, pinned.execute())
    assert len(pin_infos(pinned)) == 2

    build_path = build_expr(pinned, builds_dir=tmp_path / "builds")
    loaded = load_expr(build_path)
    pd.testing.assert_frame_equal(expected, loaded.execute())


def test_pin_cli_round_trip(predict_expr, tmp_path):
    builds_dir = tmp_path / "builds"
    plain_build = build_expr(predict_expr, builds_dir=builds_dir)
    runner = CliRunner()

    def emitted_build_path(result):
        # status lines go to stderr but CliRunner merges streams; the build
        # path is the only line that starts with the builds dir
        return next(
            line
            for line in result.output.strip().splitlines()
            if line.startswith(str(builds_dir))
        )

    verify_missing = runner.invoke(cli, ["pin", str(plain_build), "--verify"])
    assert verify_missing.exit_code != 0
    assert "no pinned caches" in verify_missing.output

    pinned_run = runner.invoke(
        cli, ["pin", str(plain_build), "--builds-dir", str(builds_dir)]
    )
    assert pinned_run.exit_code == 0, pinned_run.output
    pinned_build = emitted_build_path(pinned_run)

    verified = runner.invoke(cli, ["pin", pinned_build, "--verify"])
    assert verified.exit_code == 0, verified.output
    assert "ok" in verified.output

    # pinning an already-pinned build is a no-op: it echoes the input build
    # instead of writing an equivalent but differently-hashed artifact
    repinned = runner.invoke(
        cli, ["pin", pinned_build, "--builds-dir", str(builds_dir)]
    )
    assert repinned.exit_code == 0, repinned.output
    assert "already pinned" in repinned.output
    assert emitted_build_path(repinned) == pinned_build

    # re-pinning with a recorded cache key (as printed by --verify) exits
    # cleanly with a clear message, not a raw traceback
    recorded_key = verified.output.strip().splitlines()[0].split(":")[0]
    repin_key = runner.invoke(
        cli,
        ["pin", pinned_build, "--key", recorded_key, "--builds-dir", str(builds_dir)],
    )
    assert repin_key.exit_code != 0
    assert "already pinned" in repin_key.output


def test_unpin_cli_round_trip(predict_expr, tmp_path):
    builds_dir = tmp_path / "builds"
    expected = predict_expr.execute()
    pinned = pin_caches(predict_expr)
    pinned_build = build_expr(pinned, builds_dir=builds_dir)
    runner = CliRunner()

    # unpinning a build with no pins fails loudly
    plain_build = build_expr(predict_expr, builds_dir=builds_dir)
    missing = runner.invoke(cli, ["unpin", str(plain_build)])
    assert missing.exit_code != 0
    assert "no pinned caches" in missing.output

    unpinned = runner.invoke(
        cli, ["unpin", str(pinned_build), "--builds-dir", str(builds_dir)]
    )
    assert unpinned.exit_code == 0, unpinned.output
    assert "Unpinned 1 cache(s)" in unpinned.output
    unpinned_build = next(
        line
        for line in unpinned.output.strip().splitlines()
        if line.startswith(str(builds_dir))
    )
    # the unpinned build recomputes and matches the original predictions
    loaded = load_expr(unpinned_build)
    assert not loaded.ls.pinned_tags
    assert loaded.ls.cached_nodes
    pd.testing.assert_frame_equal(expected, loaded.execute())
