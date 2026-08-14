"""An aliased metadata table, enriched in place, busting a consumer's cache.

Two things people usually build by hand, and one that falls out for free.

**A metadata table with versions.** ZIP-level enrichment metadata -- which
county a ZIP belongs to, what that county's income and population are, later
how sick its residents are -- built from the sources
``examples/cms_geo_join.py`` fetches. Publishing it to a catalog freezes the
bytes: an entry is content-addressed and immutable. Enriching it means joining
new columns onto the previous version and publishing *that* as a new entry.
Nothing is overwritten, so nothing that referenced v1 breaks.

**An alias as the pointer.** Consumers do not name a version, they name
``cms_zip_metadata``. Enrichment ends with the alias moved from v1 to v2; the
consumer's source code never changes. Ordinarily this is where the trouble
starts -- the pointer moves, the consumer keeps serving whatever it cached, and
the staleness surfaces weeks later as a number nobody can explain.

**Cache invalidation that tracks the pointer, not the bytes.** The consumer
merges the alias into its provider table behind a ``ParquetSnapshotCache``.
That cache is deliberately *data*-insensitive: rewrite the provider parquet
underneath it and it keeps serving the cached answer, which is the point --
snapshot caches exist so that churn upstream doesn't force recomputation. But
the key is computed over expression *structure*, and a catalog entry loaded by
alias carries its entry name in a ``HashingTag``. Move the alias and the
structure changes; the key changes with it; the merge recomputes and picks up
the new columns. One pointer write is the whole invalidation mechanism::

    publish v1 ── alias ──> v1        consumer merge: MISS, then HIT, HIT, HIT
    publish v2 ── alias ──> v2        consumer merge: MISS  (same source code)

It is the pointer being followed and not the column list: re-pointing the alias
at a version with v1's *exact* schema and different numbers busts the cache just
the same (``test_repointing_to_a_same_schema_version_still_busts``).

Usage -- either the whole story at once::

    python examples/cms_geo_alias_enrichment.py all

or one step at a time, so you can watch the cache directory between them::

    python examples/cms_geo_alias_enrichment.py publish
    python examples/cms_geo_alias_enrichment.py consume
    python examples/cms_geo_alias_enrichment.py consume      # HIT
    python examples/cms_geo_alias_enrichment.py enrich
    python examples/cms_geo_alias_enrichment.py consume      # MISS: alias moved
    python examples/cms_geo_alias_enrichment.py clean

State (catalog, sample parquet, cache) lives under
``~/.cache/xorq/cms-geo-alias`` unless ``CMS_ALIAS_DEMO_DIR`` says otherwise.

The six sources are sampled offline in ``libs/cms_geo_sample.py`` -- same
columns, same dtypes, same real-world quirks (a split ZIP, a hospital-only ZIP,
a provider with no Medicare row) as the live fetchers produce. That keeps the
demo deterministic and credential-free; ``cms_geo_join.fetch_sources()`` is a
drop-in replacement for the reads here when you want the real thing.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import click

import xorq.api as xo
from xorq.api import _
from xorq.caching import ParquetSnapshotCache
from xorq.common.utils.env_utils import EnvConfigable
from xorq.common.utils.import_utils import import_python


cms_geo_sample = import_python(Path(__file__).parent / "libs" / "cms_geo_sample.py")

env_config = EnvConfigable.subclass_from_kwargs(CMS_ALIAS_DEMO_DIR="").from_env()

DEMO_DIR = Path(
    env_config["CMS_ALIAS_DEMO_DIR"]
    or Path.home() / ".cache" / "xorq" / "cms-geo-alias"
).resolve()

# the name consumers know; the version it points at is the catalog's business
METADATA_ALIAS = "cms_zip_metadata"

# ``catalog.add`` builds a wheel from the project's pyproject.toml. Passing the
# path explicitly means the demo works from any cwd, not just inside the repo.
PROJECT_PATH = Path(__file__).parents[1]


# ---------------------------------------------------------------------------
# on-disk layout -- three directories, each one a step in the story
# ---------------------------------------------------------------------------


def demo_paths(demo_dir: Path) -> dict[str, Path]:
    return {
        "data": demo_dir / "data",
        "catalog": demo_dir / "catalog",
        "cache": demo_dir / "cache",
    }


def open_catalog(demo_dir: Path, init: bool = False) -> Any:
    return xo.catalog.Catalog.from_repo_path(demo_paths(demo_dir)["catalog"], init=init)


def read(name: str, data_dir: Path, con: Any) -> xo.Expr:
    return xo.deferred_read_parquet(
        data_dir / f"{name}.parquet", con=con, table_name=name
    )


# ---------------------------------------------------------------------------
# the metadata table, in two versions
# ---------------------------------------------------------------------------


def build_metadata_v1(data_dir: Path, con: Any) -> xo.Expr:
    """ZIP -> county crosswalk, carrying that county's ACS demographics.

    A ZIP split across counties appears once per county, so this is a weighted
    many-to-many and both allocation ratios have to survive: providers are
    businesses and belong on ``bus_ratio``, anything denominated in people
    belongs on ``res_ratio``. USPS "unique" ZIPs assigned to a single hospital
    carry ``res_ratio`` 0 and ``bus_ratio`` 1.0, so filtering on ``res_ratio``
    would silently drop the largest providers in the state.
    """
    hud = (
        read("hud", data_dir, con)
        .select(
            zip5="zip",
            county_fips="county_fips",
            res_ratio="res_ratio",
            bus_ratio="bus_ratio",
            tot_ratio="tot_ratio",
        )
        .filter(_.tot_ratio > 0)
    )
    acs = read("acs", data_dir, con)
    return hud.join(acs, "county_fips")


def build_metadata_v2(data_dir: Path, con: Any) -> xo.Expr:
    """v1 with CDC PLACES health measures joined on -- five new columns.

    This is the shape an enrichment takes: the previous version is the left
    side, untouched, and the update is columns arriving beside it. PLACES is
    served long (one row per ZCTA x measure), so it is pivoted first; ZCTA and
    ZIP5 are not the same geography, but for this purpose they line up.
    """
    places = (
        read("places", data_dir, con)
        .rename(zcta="locationname")
        .pivot_wider(
            names_from="measureid",
            values_from="data_value",
            names=list(cms_geo_sample.PLACES_MEASURES),
        )
    )
    v1 = build_metadata_v1(data_dir, con)
    return v1.join(places, places.zcta == _.zip5, how="left").drop("zcta")


# ---------------------------------------------------------------------------
# the consumer -- identical source code on both sides of the enrichment
# ---------------------------------------------------------------------------


def build_provider_merge(catalog: Any, data_dir: Path, cache_dir: Path) -> xo.Expr:
    """Medicare x NPPES x taxonomy, merged with whatever the alias points at.

    ``catalog.load(METADATA_ALIAS)`` is the only reference to the metadata
    table anywhere in here: no version, no schema, no column list. Everything
    downstream of the merge is cached under a snapshot key, which is why moving
    the alias -- and only moving the alias -- recomputes it.
    """
    con = xo.connect()
    medicare = read("medicare", data_dir, con).select(
        npi="rndrng_npi",
        medicare_specialty="rndrng_prvdr_type",
        tot_benes="tot_benes",
        tot_mdcr_stdzd_amt="tot_mdcr_stdzd_amt",
    )
    nppes = read("nppes", data_dir, con).select(
        "npi",
        "taxonomy_code",
        provider_name=_.org_name.coalesce(_.first_name.concat(" ", _.last_name)),
        is_organization=_.entity_type_code == 2,
        practice_city=_.practice_city,
        zip5=_.practice_postal_code[:5],
    )
    nucc = read("nucc", data_dir, con).select(
        taxonomy_code="Code",
        taxonomy_grouping="Grouping",
        taxonomy_classification="Classification",
    )
    metadata = catalog.load(METADATA_ALIAS, con=con)
    return (
        medicare.join(nppes, "npi")
        .join(nucc, "taxonomy_code", how="left")
        .join(metadata, "zip5", how="left")
        .drop("taxonomy_code_right", "zip5_right")
        .cache(ParquetSnapshotCache.from_kwargs(source=con, relative_path=cache_dir))
    )


def build_density_expr(merged: xo.Expr) -> xo.Expr:
    """Provider density per 100k by county x specialty -- a downstream reader.

    Reads only columns v1 already had, so it keeps working across the
    enrichment. Providers are allocated by the HUD *business* ratio while the
    population denominator is residential; that mismatch is what "providers per
    resident" means.
    """
    return (
        merged.filter(_.taxonomy_classification.notnull())
        .group_by(["county_fips", "county_name", "taxonomy_classification"])
        .agg(
            providers=_.bus_ratio.sum(),
            medicare_payments=(_.tot_mdcr_stdzd_amt * _.bus_ratio).sum(),
            population=_.population.max(),
        )
        .mutate(providers_per_100k=_.providers / _.population * 100_000)
        .order_by(_.providers_per_100k.desc())
    )


# ---------------------------------------------------------------------------
# cache observation -- the cache directory is the evidence
# ---------------------------------------------------------------------------

# ``expr.ls.get_key()`` is not the evidence to use here. It is computed before
# the transform passes run, and for an expression rooted in a catalog-loaded
# RemoteTable it does not match the key the cache actually writes under. The
# files on disk do not lie: a miss adds one, a hit adds none.


def cache_state(cache_dir: Path) -> dict[str, int]:
    """Name -> mtime for every cache file, or empty if nothing is cached yet."""
    if not cache_dir.exists():
        return {}
    return {p.name: p.stat().st_mtime_ns for p in sorted(cache_dir.glob("*.parquet"))}


def classify_cache_change(before: dict[str, int], after: dict[str, int]) -> str:
    """``"miss"`` if a new cache file appeared, ``"hit"`` if none did."""
    if set(after) - set(before):
        return "miss"
    if any(before[name] != after[name] for name in before if name in after):
        return "rewrote"
    return "hit"


# ---------------------------------------------------------------------------
# the steps
# ---------------------------------------------------------------------------


def publish(demo_dir: Path, echo: Any = click.echo) -> Any:
    """Write the sample sources, init the catalog, publish v1, alias it."""
    paths = demo_paths(demo_dir)
    cms_geo_sample.write_parquet(paths["data"])
    catalog = open_catalog(demo_dir, init=True)
    entry = catalog.add(
        build_metadata_v1(paths["data"], xo.connect()),
        aliases=(METADATA_ALIAS,),
        project_path=PROJECT_PATH,
    )
    echo(f"published v1 {entry.name}")
    echo(f"  {METADATA_ALIAS} -> {entry.name}")
    return entry


def enrich(demo_dir: Path, echo: Any = click.echo) -> Any:
    """Publish v2 -- v1 with the PLACES columns joined on -- and move the alias.

    ``add_alias`` overwrites, so this is one pointer write. v1 stays in the
    catalog, still fetchable by name, still byte-identical.
    """
    paths = demo_paths(demo_dir)
    catalog = open_catalog(demo_dir)
    previous = catalog.get_catalog_entry(METADATA_ALIAS, maybe_alias=True)
    entry = catalog.add(
        build_metadata_v2(paths["data"], xo.connect()),
        project_path=PROJECT_PATH,
    )
    catalog.add_alias(entry.name, METADATA_ALIAS)
    echo(f"published v2 {entry.name}")
    echo(f"  {METADATA_ALIAS} -> {entry.name} (was {previous.name})")
    echo(f"  v1 still in catalog: {previous.name in catalog.list()}")
    return entry


def consume(demo_dir: Path, limit: int = 10, echo: Any = click.echo) -> dict:
    """Execute the consumer's merge and report what the cache did."""
    paths = demo_paths(demo_dir)
    catalog = open_catalog(demo_dir)
    target = catalog.get_catalog_entry(METADATA_ALIAS, maybe_alias=True)

    before = cache_state(paths["cache"])
    merged = build_provider_merge(catalog, paths["data"], paths["cache"])
    frame = xo.execute(merged)
    verdict = classify_cache_change(before, cache_state(paths["cache"]))

    echo(f"{METADATA_ALIAS} -> {target.name}: cache {verdict.upper()}")
    echo(f"  merged columns ({len(frame.columns)}): {', '.join(frame.columns)}")
    echo(frame.head(limit).to_string())
    echo(xo.execute(build_density_expr(merged).limit(limit)).to_string())
    return {"verdict": verdict, "entry": target.name, "frame": frame}


def clean(demo_dir: Path, echo: Any = click.echo) -> None:
    shutil.rmtree(demo_dir, ignore_errors=True)
    echo(f"removed {demo_dir}")


def run_all(demo_dir: Path, echo: Any = click.echo) -> tuple[dict, ...]:
    """publish -> consume -> consume -> enrich -> consume -> consume."""

    def step(label: str) -> None:
        echo(f"\n=== {label} " + "=" * max(0, 60 - len(label)))

    step("publish v1")
    publish(demo_dir, echo=echo)

    step("consume (cold)")
    cold = consume(demo_dir, echo=echo)

    step("consume again -- nothing moved")
    warm = consume(demo_dir, echo=echo)

    step("rewrite a source parquet under the consumer")
    # A snapshot cache keys on structure, not bytes: this must NOT invalidate.
    # Blowing up the payment amounts makes a recompute impossible to miss.
    data_dir = demo_paths(demo_dir)["data"]
    inflated = cms_geo_sample.frame("medicare").assign(
        tot_mdcr_stdzd_amt=lambda df: df.tot_mdcr_stdzd_amt * 1000
    )
    inflated.to_parquet(data_dir / "medicare.parquet", index=False)
    echo("  medicare.parquet payments x1000")
    stale = consume(demo_dir, echo=echo)

    step("enrich: publish v2, move the alias")
    enrich(demo_dir, echo=echo)

    step("consume -- same source code, moved pointer")
    busted = consume(demo_dir, echo=echo)

    step("consume again -- v2 is now the cached answer")
    settled = consume(demo_dir, echo=echo)

    echo("\n" + "=" * 62)
    for label, result in (
        ("cold", cold),
        ("unchanged", warm),
        ("source rewritten", stale),
        ("alias moved", busted),
        ("unchanged", settled),
    ):
        echo(f"  {label:>18}: {result['verdict'].upper():<5} -> {result['entry']}")
    return (cold, warm, stale, busted, settled)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group()
@click.option(
    "--demo-dir",
    default=str(DEMO_DIR),
    show_default=True,
    type=click.Path(path_type=Path),
)
@click.pass_context
def cli(ctx: click.Context, demo_dir: Path) -> None:
    ctx.obj = Path(demo_dir).resolve()


@cli.command(name="publish")
@click.pass_obj
def publish_command(demo_dir: Path) -> None:
    """Write the sample sources and publish v1 of the metadata table."""
    publish(demo_dir)


@cli.command(name="consume")
@click.option("--limit", default=10, show_default=True)
@click.pass_obj
def consume_command(demo_dir: Path, limit: int) -> None:
    """Merge the aliased metadata into the provider table."""
    consume(demo_dir, limit=limit)


@cli.command(name="enrich")
@click.pass_obj
def enrich_command(demo_dir: Path) -> None:
    """Publish v2 with new columns and re-point the alias at it."""
    enrich(demo_dir)


@cli.command(name="clean")
@click.pass_obj
def clean_command(demo_dir: Path) -> None:
    """Remove the demo's catalog, sample data, and cache."""
    clean(demo_dir)


@cli.command(name="all")
@click.pass_obj
def all_command(demo_dir: Path) -> None:
    """Run the whole story in one go, in a fresh demo directory."""
    clean(demo_dir)
    run_all(demo_dir)


def assert_invariants(results: tuple[dict, ...]) -> None:
    """The claims this example makes, checked."""
    (cold, warm, stale, busted, settled) = results
    assert cold["verdict"] == "miss", "a cold cache must compute"
    assert warm["verdict"] == "hit", "an unchanged alias must not recompute"
    assert stale["verdict"] == "hit", "a snapshot cache must ignore source bytes"
    assert busted["verdict"] == "miss", "a moved alias must bust the snapshot cache"
    assert settled["verdict"] == "hit", "the new version must then cache too"

    assert cold["entry"] == warm["entry"] == stale["entry"]
    assert busted["entry"] != cold["entry"], "the alias must resolve to v2"
    assert busted["entry"] == settled["entry"]

    new_columns = set(busted["frame"].columns) - set(cold["frame"].columns)
    assert new_columns == set(cms_geo_sample.PLACES_MEASURES), (
        f"v2 must add the PLACES measures, got {sorted(new_columns)}"
    )
    # the recompute also picked up the source rewrite it had been ignoring
    served_stale = stale["frame"].tot_mdcr_stdzd_amt.max()
    served_fresh = busted["frame"].tot_mdcr_stdzd_amt.max()
    assert served_fresh == served_stale * 1000, (
        f"the busted merge must re-read the rewritten source: "
        f"{served_stale} -> {served_fresh}"
    )


def main(argv: list[str] | None = None) -> None:
    cli.main(args=argv if argv is not None else sys.argv[1:], standalone_mode=False)


# ``xorq build`` imports an example as "__main__", so the usual guard alone
# would run the whole demo at build time; only act when this file really is the
# entry point. Under pytest the demo runs against a throwaway directory so it
# never touches whatever state a reader has on disk.
if __name__ == "__main__" and Path(sys.argv[0]).resolve() == Path(__file__).resolve():
    main()
elif __name__ == "__pytest_main__":
    with tempfile.TemporaryDirectory() as tmp:
        assert_invariants(run_all(Path(tmp)))
    pytest_examples_passed = True
