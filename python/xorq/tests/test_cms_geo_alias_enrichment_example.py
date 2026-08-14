"""Check the claims ``examples/cms_geo_alias_enrichment.py`` makes.

The example's story is one sequence -- publish, consume, consume, rewrite a
source, enrich, consume, consume -- so it runs once per module and each test
interrogates a different step of the result. Evidence is the cache directory,
not ``ls.get_key()``: see the note in the example about why the pre-transform
key does not match what a catalog-rooted expression actually writes under.
"""

from __future__ import annotations

import importlib.util
import pathlib
from types import ModuleType

import pytest

import xorq.api as xo


EXAMPLE = pathlib.Path(__file__).parents[3] / "examples" / "cms_geo_alias_enrichment.py"


@pytest.fixture(scope="module")
def module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("cms_geo_alias_example", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def demo_dir(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    return tmp_path_factory.mktemp("cms-geo-alias")


@pytest.fixture(scope="module")
def results(module: ModuleType, demo_dir: pathlib.Path) -> tuple[dict, ...]:
    """The five consumer runs, in order, sharing one catalog and one cache."""
    return module.run_all(demo_dir, echo=lambda *args, **kwargs: None)


def test_cold_run_computes(results: tuple[dict, ...]) -> None:
    (cold, *_rest) = results
    assert cold["verdict"] == "miss"


def test_unchanged_alias_hits(results: tuple[dict, ...]) -> None:
    (cold, warm, *_rest) = results
    assert warm["verdict"] == "hit"
    assert warm["entry"] == cold["entry"]


def test_snapshot_cache_ignores_a_source_rewrite(results: tuple[dict, ...]) -> None:
    """The point of a snapshot cache: new bytes upstream do not force a recompute."""
    (cold, _warm, stale, *_rest) = results
    assert stale["verdict"] == "hit"
    # medicare.parquet had its payments multiplied by 1000 before this run
    assert stale["frame"].tot_mdcr_stdzd_amt.max() == pytest.approx(
        cold["frame"].tot_mdcr_stdzd_amt.max()
    )


def test_moving_the_alias_busts_the_cache(results: tuple[dict, ...]) -> None:
    (cold, _warm, _stale, busted, _settled) = results
    assert busted["verdict"] == "miss"
    assert busted["entry"] != cold["entry"]


def test_the_bust_recomputes_rather_than_reshapes(results: tuple[dict, ...]) -> None:
    """A miss means real work: the rewritten source lands on the recompute."""
    (_cold, _warm, stale, busted, _settled) = results
    assert busted["frame"].tot_mdcr_stdzd_amt.max() == pytest.approx(
        stale["frame"].tot_mdcr_stdzd_amt.max() * 1000
    )


def test_v2_adds_columns_the_consumer_never_named(
    module: ModuleType, results: tuple[dict, ...]
) -> None:
    (cold, _warm, _stale, busted, _settled) = results
    added = set(busted["frame"].columns) - set(cold["frame"].columns)
    assert added == set(module.cms_geo_sample.PLACES_MEASURES)
    assert not set(cold["frame"].columns) - set(busted["frame"].columns)


def test_the_new_version_caches_too(results: tuple[dict, ...]) -> None:
    (*_rest, busted, settled) = results
    assert settled["verdict"] == "hit"
    assert settled["entry"] == busted["entry"]


def test_each_version_gets_its_own_cache_file(
    module: ModuleType, demo_dir: pathlib.Path, results: tuple[dict, ...]
) -> None:
    """Two aliased versions consumed means two cached answers, not an overwrite."""
    cache_dir = module.demo_paths(demo_dir)["cache"]
    assert len(module.cache_state(cache_dir)) == 2


def test_v1_stays_addressable_after_the_alias_moves(
    module: ModuleType, demo_dir: pathlib.Path, results: tuple[dict, ...]
) -> None:
    """Publishing v2 must not disturb v1: entries are immutable, aliases move."""
    (cold, *_rest) = results
    catalog = module.open_catalog(demo_dir)
    assert cold["entry"] in catalog.list()
    v1 = xo.execute(catalog.load(cold["entry"]))
    assert not set(module.cms_geo_sample.PLACES_MEASURES) & set(v1.columns)


def test_the_alias_resolves_to_v2(
    module: ModuleType, demo_dir: pathlib.Path, results: tuple[dict, ...]
) -> None:
    (*_rest, settled) = results
    catalog = module.open_catalog(demo_dir)
    entry = catalog.get_catalog_entry(module.METADATA_ALIAS, maybe_alias=True)
    assert entry.name == settled["entry"]
    v2 = xo.execute(catalog.load(module.METADATA_ALIAS))
    assert set(module.cms_geo_sample.PLACES_MEASURES) <= set(v2.columns)


def test_the_metadata_versions_differ_only_by_added_columns(
    module: ModuleType, demo_dir: pathlib.Path
) -> None:
    """v2 is v1 joined with new columns -- v1's columns all survive."""
    data_dir = module.demo_paths(demo_dir)["data"]
    v1 = module.build_metadata_v1(data_dir, xo.connect()).schema()
    v2 = module.build_metadata_v2(data_dir, xo.connect()).schema()
    assert set(v1.names) <= set(v2.names)
    assert set(v2.names) - set(v1.names) == set(module.cms_geo_sample.PLACES_MEASURES)


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        pytest.param({}, {"a": 1}, "miss", id="new-file"),
        pytest.param({"a": 1}, {"a": 1}, "hit", id="untouched"),
        pytest.param({"a": 1}, {"a": 2}, "rewrote", id="same-key-rewritten"),
        pytest.param({"a": 1}, {"a": 1, "b": 1}, "miss", id="second-file"),
    ],
)
def test_classify_cache_change(
    module: ModuleType, before: dict, after: dict, expected: str
) -> None:
    assert module.classify_cache_change(before, after) == expected


def test_the_example_asserts_its_own_invariants(
    module: ModuleType, results: tuple[dict, ...]
) -> None:
    """``assert_invariants`` is what the example checks under pytest; run it here too."""
    module.assert_invariants(results)


def test_repointing_to_a_same_schema_version_still_busts(
    module: ModuleType, tmp_path: pathlib.Path
) -> None:
    """The bust tracks the pointer, not the shape.

    The example's v2 adds columns, so on its own it cannot distinguish "the
    alias moved" from "the schema changed". Here the new version has v1's exact
    schema and only different numbers, and the consumer still recomputes --
    which is what makes the alias, and not the column list, the thing the
    snapshot key is following.
    """
    demo_dir = tmp_path / "same-schema"
    paths = module.demo_paths(demo_dir)
    module.publish(demo_dir, echo=lambda *args, **kwargs: None)

    cold = module.consume(demo_dir, echo=lambda *args, **kwargs: None)
    warm = module.consume(demo_dir, echo=lambda *args, **kwargs: None)
    assert (cold["verdict"], warm["verdict"]) == ("miss", "hit")

    # same columns, same dtypes, different incomes
    revised = module.cms_geo_sample.frame("acs").assign(
        median_household_income=lambda df: df.median_household_income + 1.0
    )
    revised.to_parquet(paths["data"] / "acs.parquet", index=False)

    catalog = module.open_catalog(demo_dir)
    entry = catalog.add(
        module.build_metadata_v1(paths["data"], xo.connect()),
        project_path=module.PROJECT_PATH,
    )
    assert entry.name != cold["entry"], "new bytes must be a new entry"
    catalog.add_alias(entry.name, module.METADATA_ALIAS)

    busted = module.consume(demo_dir, echo=lambda *args, **kwargs: None)
    assert busted["verdict"] == "miss"
    assert list(busted["frame"].columns) == list(cold["frame"].columns)
    assert busted["frame"].median_household_income.max() == pytest.approx(
        cold["frame"].median_household_income.max() + 1.0
    )
