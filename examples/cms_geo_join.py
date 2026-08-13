"""Joins six open CMS / Census / CDC datasets into one provider-level table.

Traditional approach: You would write a bespoke download script per source --
paging one API, unzipping another, remembering which files you already have --
then load the results into a warehouse and join them there. The ingest lives
outside the query, so nothing about where a column came from is visible in the
expression, and re-running means re-downloading.

With xorq: static file URLs are read directly by the engine, and the API
endpoints become ``flight_udxf`` fetchers (the same pattern as
``libs/hackernews_lib.py``), so fetching *is* part of the expression graph.
Every source is wrapped in a ParquetCache keyed by expression hash, placed on
the *raw* fetched table rather than on a projection of it -- so the first run
fetches and every later run replays from parquet, and reshaping a source
downstream reuses the cache instead of refetching.

    Medicare by-Provider ──(NPI)── NPPES ──(taxonomy code)── NUCC taxonomy
                                     │
                              zip5 = left(postal_code, 5)
                                     │
                     ┌───────────────┴────────────────┐
            HUD ZIP→county crosswalk            zip5 ≈ ZCTA
            (weighted by BUS_RATIO)                   │
                     │                          CDC PLACES
              ACS county demographics           (pivoted wide)

Only NUCC is read straight from its URL. The rest are dynamic endpoints that
answer a HEAD with ``Content-Length: 0``, which the engine's HTTP object store
reads as an empty file *without erroring* -- so they must be fetched, not read.

Credentials, both free, come from the environment like every other setting here
-- ``env_config`` declares them, so nothing else has to go looking:

* ``HUD_TOKEN``       -- https://www.huduser.gov/hudapi/public/register
* ``CENSUS_API_KEY``  -- https://api.census.gov/data/key_signup.html
  (ACS stopped serving keyless requests; it now returns a "Missing Key" HTML
  page with a 200 status, so we fail loudly rather than parse it)

Usage::

    python examples/cms_geo_join.py --state MA
    xorq build examples/cms_geo_join.py -e expr

NPPES has no server-side filter, so its fetcher pulls the full ~1 GB national
ZIP once and filters it locally; the parquet cache means that happens only on a
cold cache. The archive and the caches live under
``~/.cache/xorq/cms-geo`` unless ``CMS_GEO_ARCHIVE_DIR`` says otherwise.
"""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path
from typing import Any

import click
import pandas as pd
import requests

import xorq.api as xo
from xorq.api import _
from xorq.caching import ParquetCache
from xorq.common.exceptions import XorqError
from xorq.common.utils.env_utils import EnvConfigable
from xorq.common.utils.toolz_utils import curry
from xorq.vendor import ibis


env_config = EnvConfigable.subclass_from_kwargs(
    "HUD_TOKEN",
    "CENSUS_API_KEY",
    CMS_GEO_STATE="MA",
    CMS_GEO_ARCHIVE_DIR="",
).from_env()

STATE = env_config["CMS_GEO_STATE"]
# defaults outside the repo: this holds a 1.1 GB archive plus the caches, and
# the repo's .gitignore ignores itself, so an in-tree default would show up as
# untracked junk in everyone else's working copy
ARCHIVE_DIR = Path(
    env_config["CMS_GEO_ARCHIVE_DIR"] or Path.home() / ".cache" / "xorq" / "cms-geo"
).resolve()

# ParquetCache resolves a *relative* path under ~/.cache/xorq, so pass an
# absolute one to keep the caches beside the NPPES archive they derive from.
CACHE_DIR = ARCHIVE_DIR / "cache"

NUCC_CSV_URL = "https://www.nucc.org/images/stories/CSV/nucc_taxonomy_250.csv"
NPPES_ZIP_URL = (
    "https://download.cms.gov/nppes/NPPES_Data_Dissemination_August_2026_V2.zip"
)
MEDICARE_API = (
    "https://data.cms.gov/data-api/v1/dataset/8889d81e-2ee7-448f-8713-f071038289b5/data"
)
PLACES_API = "https://data.cdc.gov/resource/qnzd-25i4.csv"
CENSUS_API = "https://api.census.gov/data/2022/acs/acs5"
HUD_API = "https://www.huduser.gov/hudapi/public/usps"

# HUD crosswalk type 2 == ZIP -> county.
HUD_ZIP_COUNTY = 2

# the credentials the fetchers need, and where a reader gets one, free
CREDENTIAL_SIGNUPS = {
    "HUD_TOKEN": "https://www.huduser.gov/hudapi/public/register",
    "CENSUS_API_KEY": "https://api.census.gov/data/key_signup.html",
}

PLACES_MEASURES = ("CHD", "DIABETES", "OBESITY", "ACCESS2", "CHECKUP")

ACS_VARIABLES = {
    "B19013_001E": "median_household_income",
    "B01003_001E": "population",
    "B27010_033E": "uninsured_18_34",
}

NPPES_COLUMNS = {
    "NPI": "npi",
    "Entity Type Code": "entity_type_code",
    "Provider Organization Name (Legal Business Name)": "org_name",
    "Provider Last Name (Legal Name)": "last_name",
    "Provider First Name": "first_name",
    "Provider Business Practice Location Address City Name": "practice_city",
    "Provider Business Practice Location Address State Name": "practice_state",
    "Provider Business Practice Location Address Postal Code": "practice_postal_code",
    "Healthcare Provider Taxonomy Code_1": "taxonomy_code",
    "Provider Enumeration Date": "enumeration_date",
}

STATE_FIPS = {
    "AL": "01",
    "AK": "02",
    "AZ": "04",
    "AR": "05",
    "CA": "06",
    "CO": "08",
    "CT": "09",
    "DE": "10",
    "DC": "11",
    "FL": "12",
    "GA": "13",
    "HI": "15",
    "ID": "16",
    "IL": "17",
    "IN": "18",
    "IA": "19",
    "KS": "20",
    "KY": "21",
    "LA": "22",
    "ME": "23",
    "MD": "24",
    "MA": "25",
    "MI": "26",
    "MN": "27",
    "MS": "28",
    "MO": "29",
    "MT": "30",
    "NE": "31",
    "NV": "32",
    "NH": "33",
    "NJ": "34",
    "NM": "35",
    "NY": "36",
    "NC": "37",
    "ND": "38",
    "OH": "39",
    "OK": "40",
    "OR": "41",
    "PA": "42",
    "RI": "44",
    "SC": "45",
    "SD": "46",
    "TN": "47",
    "TX": "48",
    "UT": "49",
    "VT": "50",
    "VA": "51",
    "WA": "53",
    "WV": "54",
    "WI": "55",
    "WY": "56",
}


# ---------------------------------------------------------------------------
# schemas: one `in` schema for the fetcher parameters, one `out` per source
# ---------------------------------------------------------------------------

state_schema_in = xo.schema({"state": "string"})
empty_schema_in = xo.schema({"unit": "int64"})

NUCC_SCHEMA = ibis.schema(
    {
        "Code": "string",
        "Grouping": "string",
        "Classification": "string",
        "Specialization": "string",
        "Definition": "string",
        "Notes": "string",
        "Display Name": "string",
        "Section": "string",
    }
)

nppes_schema_out = xo.schema(
    {
        "npi": "string",
        "entity_type_code": "float64",
        "org_name": "string",
        "last_name": "string",
        "first_name": "string",
        "practice_city": "string",
        "practice_state": "string",
        "practice_postal_code": "string",
        "taxonomy_code": "string",
        "enumeration_date": "string",
    }
)

medicare_schema_out = xo.schema(
    {
        "rndrng_npi": "string",
        "rndrng_prvdr_state_abrvtn": "string",
        "rndrng_prvdr_zip5": "string",
        "rndrng_prvdr_type": "string",
        "tot_benes": "float64",
        "tot_srvcs": "float64",
        "tot_mdcr_pymt_amt": "float64",
        "tot_mdcr_stdzd_amt": "float64",
    }
)

hud_schema_out = xo.schema(
    {
        "zip": "string",
        "county_fips": "string",
        "res_ratio": "float64",
        "bus_ratio": "float64",
        "oth_ratio": "float64",
        "tot_ratio": "float64",
    }
)

acs_schema_out = xo.schema(
    {
        "county_fips": "string",
        "county_name": "string",
        "median_household_income": "float64",
        "population": "float64",
        "uninsured_18_34": "float64",
    }
)

places_schema_out = xo.schema(
    {"locationname": "string", "measureid": "string", "data_value": "float64"}
)


# ---------------------------------------------------------------------------
# fetchers -- plain functions that hit an endpoint and return a DataFrame,
# wrapped as flight_udxf exchangers below (see libs/hackernews_lib.py)
# ---------------------------------------------------------------------------


# Everything raised below must be an `Exception` -- never a `SystemExit`. The
# fetchers run inside a `flight_udxf` exchanger, on a Flight server thread
# rather than the caller's. `make_udxf` wraps the exchange in
# `excepts_print_exc(..., Exception)` (python/xorq/flight/exchanger.py), so an
# `Exception` gets its traceback printed and ends the exchange. A `SystemExit`
# is not an `Exception`, so it escapes that wrapper and kills the handler
# outright -- and because the client only queues its end-of-stream sentinel
# after a clean read (python/xorq/flight/client.py), the consumer then blocks in
# `queue.get()` forever, with the main thread parked in DataFusion where it
# cannot even take a KeyboardInterrupt.


def require_env(name: str) -> str:
    """Look up a credential in the environment, via ``env_config``."""
    value = env_config.get(name)
    if not value:
        raise XorqError(
            f"{name} is not set -- register (free) at {CREDENTIAL_SIGNUPS[name]}"
        )
    return value


def get_json(url: str, **kwargs: Any) -> Any:
    resp = requests.get(url, timeout=300, **kwargs)
    resp.raise_for_status()
    return resp.json()


def conform(frame: pd.DataFrame, schema: ibis.Schema) -> pd.DataFrame:
    """Order columns to match ``schema`` and coerce its float columns.

    A UDXF hands its output straight to Arrow, so column order and dtypes have
    to line up with ``maybe_schema_out`` exactly.
    """
    missing = set(schema.names) - set(frame.columns)
    if missing:
        raise XorqError(f"fetcher output missing columns: {sorted(missing)}")
    frame = frame[list(schema.names)].copy()
    for name, dtype in schema.items():
        if dtype.is_floating():
            # .astype("float64") is not redundant: to_numeric infers int64 for
            # an all-integer column (ACS population, Medicare counts), which
            # mismatches the declared float64 and makes the exchange return
            # zero rows rather than raising
            frame[name] = pd.to_numeric(frame[name], errors="coerce").astype("float64")
        else:
            frame[name] = frame[name].astype("string").astype(object)
    return frame


def fetch_nppes(state: str) -> pd.DataFrame:
    """Download the national NPPES ZIP once, filter to ``state``.

    CMS publishes only a full national snapshot -- no server-side filter -- so
    this pays ~1 GB compressed / ~11.6 GB expanded the first time. The archive
    is kept on disk so clearing the parquet cache does not re-download it.
    """
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archive = ARCHIVE_DIR / Path(NPPES_ZIP_URL).name
    if not archive.exists():
        # rename only on success -- an interrupted transfer must not leave a
        # truncated zip that the next run treats as complete
        partial = archive.with_name(archive.name + ".partial")
        with requests.get(NPPES_ZIP_URL, stream=True, timeout=600) as response:
            response.raise_for_status()
            expected = int(response.headers.get("Content-Length", 0))
            with partial.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=1 << 22):
                    fh.write(chunk)
        written = partial.stat().st_size
        if expected and written != expected:
            partial.unlink()
            raise XorqError(
                f"NPPES download truncated: got {written:,} of {expected:,} bytes"
            )
        partial.rename(archive)

    with zipfile.ZipFile(archive) as zf:
        # the bundle ships npidata_pfile_<dates>.csv alongside a tiny
        # npidata_pfile_<dates>_fileheader.csv -- match case-insensitively so
        # the header companion never sneaks through
        (member,) = (
            n
            for n in zf.namelist()
            if n.lower().startswith("npidata_pfile_")
            and n.lower().endswith(".csv")
            and "fileheader" not in n.lower()
        )
        with zf.open(member) as fh:
            chunks = pd.read_csv(
                fh, usecols=list(NPPES_COLUMNS), dtype=str, chunksize=250_000
            )
            kept = [
                chunk.rename(columns=NPPES_COLUMNS).loc[
                    lambda df: df.practice_state.eq(state)
                ]
                for chunk in chunks
            ]
    return pd.concat(kept, ignore_index=True)


def fetch_medicare(state: str) -> pd.DataFrame:
    """Page the CMS data-api until it returns a short page."""
    rows, offset, size = [], 0, 5000
    while True:
        page = get_json(
            MEDICARE_API,
            params={
                "filter[Rndrng_Prvdr_State_Abrvtn]": state,
                "size": size,
                "offset": offset,
            },
        )
        rows.extend(page)
        if len(page) < size:
            break
        offset += size
    return pd.DataFrame(rows).rename(columns=str.lower)


def fetch_hud(state: str) -> pd.DataFrame:
    """ZIP -> county crosswalk with allocation ratios, via the HUD USPS API."""
    token = require_env("HUD_TOKEN")
    response = requests.get(
        HUD_API,
        params={"type": HUD_ZIP_COUNTY, "query": state},
        headers={"Authorization": f"Bearer {token}"},
        timeout=300,
    )
    if response.status_code == 401:
        raise XorqError("HUD_TOKEN was rejected (401 Unauthenticated)")
    response.raise_for_status()
    results = response.json().get("data", {}).get("results")
    if not results:
        raise XorqError(f"unexpected HUD response shape: {response.text[:400]}")
    return pd.DataFrame(results).rename(columns={"geoid": "county_fips"})


def fetch_acs(state: str) -> pd.DataFrame:
    key = require_env("CENSUS_API_KEY")
    response = requests.get(
        CENSUS_API,
        params={
            "get": ",".join(("NAME", *ACS_VARIABLES)),
            "for": "county:*",
            "in": f"state:{STATE_FIPS[state]}",
            "key": key,
        },
        timeout=180,
    )
    response.raise_for_status()
    if not response.text.lstrip().startswith("["):
        raise XorqError(
            "Census returned HTML instead of JSON (usually a bad or missing "
            f"CENSUS_API_KEY): {response.text[:200]!r}"
        )
    header, *rows = response.json()
    frame = pd.DataFrame(rows, columns=header).rename(columns=ACS_VARIABLES)
    return frame.assign(county_fips=frame.state + frame.county).rename(
        columns={"NAME": "county_name"}
    )


def fetch_places() -> pd.DataFrame:
    """CDC PLACES, ZCTA level, long format -- one row per (ZCTA, measure)."""
    measures = ",".join(f"'{m}'" for m in PLACES_MEASURES)
    response = requests.get(
        PLACES_API,
        params={
            "$select": "locationname,measureid,data_value",
            "$where": f"measureid in({measures})",
            "$limit": 500_000,
        },
        timeout=300,
    )
    response.raise_for_status()
    return pd.read_csv(io.BytesIO(response.content), dtype={"locationname": str})


# --- process_df adapters: (params frame) -> (results frame) ----------------


@curry
def by_state(df: pd.DataFrame, fetch: Any, schema: ibis.Schema) -> pd.DataFrame:
    frames = [fetch(state) for state in df.state]
    return conform(pd.concat(frames, ignore_index=True), schema)


@curry
def no_params(df: pd.DataFrame, fetch: Any, schema: ibis.Schema) -> pd.DataFrame:
    return conform(fetch(), schema)


# Both names matter, and for different reasons. `name` is the RemoteTable that
# `into_backend` puts in front of the fetcher; the build serializes it under a
# content-derived name, so its spelling never reaches the hash. `inner_name` is
# the FlightUDXF node itself, and *that* one is written to expr.yaml verbatim --
# left unset it defaults to `gen_name()`, a fresh uuid4 per process, so two
# builds of an unchanged expression land in different `builds/` directories and
# every cache key downstream of the fetcher moves with it. Keep these lowercase
# for the same reason the memtables below are.
nppes_fetcher = xo.expr.relations.flight_udxf(
    process_df=by_state(fetch=fetch_nppes, schema=nppes_schema_out),
    maybe_schema_in=state_schema_in,
    maybe_schema_out=nppes_schema_out,
    name="NppesFetcher",
    inner_name="cms_geo_nppes_udxf",
)

medicare_fetcher = xo.expr.relations.flight_udxf(
    process_df=by_state(fetch=fetch_medicare, schema=medicare_schema_out),
    maybe_schema_in=state_schema_in,
    maybe_schema_out=medicare_schema_out,
    name="MedicareFetcher",
    inner_name="cms_geo_medicare_udxf",
)

hud_fetcher = xo.expr.relations.flight_udxf(
    process_df=by_state(fetch=fetch_hud, schema=hud_schema_out),
    maybe_schema_in=state_schema_in,
    maybe_schema_out=hud_schema_out,
    name="HudCrosswalkFetcher",
    inner_name="cms_geo_hud_udxf",
)

acs_fetcher = xo.expr.relations.flight_udxf(
    process_df=by_state(fetch=fetch_acs, schema=acs_schema_out),
    maybe_schema_in=state_schema_in,
    maybe_schema_out=acs_schema_out,
    name="AcsFetcher",
    inner_name="cms_geo_acs_udxf",
)

places_fetcher = xo.expr.relations.flight_udxf(
    process_df=no_params(fetch=fetch_places, schema=places_schema_out),
    maybe_schema_in=empty_schema_in,
    maybe_schema_out=places_schema_out,
    name="PlacesFetcher",
    inner_name="cms_geo_places_udxf",
)


# ---------------------------------------------------------------------------
# the expression
# ---------------------------------------------------------------------------


def fetch_sources(state: str, con: Any) -> dict[str, ibis.Table]:
    """The six raw source tables: five fetchers plus one static-file URL read."""
    # Name these, and keep the names lowercase. An unnamed memtable gets a
    # fresh random name per call, which changes the expression hash, which
    # changes every downstream cache key -- the caches would be rewritten every
    # run and never read. But the engine lowercases unquoted identifiers, so a
    # name with any uppercase in it (`..._MA`) registers under one spelling and
    # is looked up under another: "table '...' not found" at execution.
    params = xo.memtable([{"state": state}], name=f"cms_geo_params_{state.lower()}")
    unit = xo.memtable([{"unit": 1}], name="cms_geo_unit")
    return {
        "nppes": nppes_fetcher(params),
        "medicare": medicare_fetcher(params),
        "hud": hud_fetcher(params),
        "acs": acs_fetcher(params),
        "places": places_fetcher(unit),
        # the one source that is a plain static file: read straight from its URL
        "nucc": xo.deferred_read_csv(
            NUCC_CSV_URL, con=con, table_name="nucc", schema=NUCC_SCHEMA
        ),
    }


def build_expr(state: str, raw: dict[str, ibis.Table] | None = None) -> ibis.Table:
    """Wire the six sources into one provider-level table.

    ``raw`` overrides the source tables -- pass equivalents built from
    memtables to exercise the join without touching the network.
    """
    con = xo.connect()
    cache = ParquetCache.from_kwargs(source=con, relative_path=CACHE_DIR)
    raw = raw if raw is not None else fetch_sources(state, con)

    nppes = (
        raw["nppes"]
        .cache(cache)
        .select(
            "npi",
            "taxonomy_code",
            "enumeration_date",
            provider_name=_.org_name.coalesce(_.first_name.concat(" ", _.last_name)),
            is_organization=_.entity_type_code == 2,
            practice_city=_.practice_city,
            zip5=_.practice_postal_code[:5],
        )
    )

    medicare = (
        raw["medicare"]
        .cache(cache)
        .select(
            npi="rndrng_npi",
            medicare_specialty="rndrng_prvdr_type",
            tot_benes="tot_benes",
            tot_srvcs="tot_srvcs",
            tot_mdcr_pymt_amt="tot_mdcr_pymt_amt",
            tot_mdcr_stdzd_amt="tot_mdcr_stdzd_amt",
        )
    )

    nucc = (
        raw["nucc"]
        .cache(cache)
        .select(
            taxonomy_code="Code",
            taxonomy_grouping="Grouping",
            taxonomy_classification="Classification",
            taxonomy_specialization="Specialization",
        )
    )

    hud = (
        raw["hud"]
        .cache(cache)
        # A ZIP split across counties appears once per county, so this join is
        # a weighted many-to-many. Keep both allocation ratios: providers are
        # businesses and must be allocated by bus_ratio, while anything
        # denominated in people belongs on res_ratio. USPS "unique" ZIPs
        # assigned to a single hospital (01655 UMass Memorial, 01805 Lahey,
        # 01199 Baystate) carry res_ratio == 0 and bus_ratio == 1.0, so
        # filtering or weighting on res_ratio silently drops the largest
        # providers in the state.
        .select(
            zip5="zip",
            county_fips="county_fips",
            res_ratio="res_ratio",
            bus_ratio="bus_ratio",
            tot_ratio="tot_ratio",
        )
        .filter(_.tot_ratio > 0)
    )

    acs = raw["acs"].cache(cache)

    # PLACES arrives long (one row per ZCTA x measure); pivot to one row per ZCTA
    places = (
        raw["places"]
        .cache(cache)
        .rename(zcta="locationname")
        .pivot_wider(
            names_from="measureid",
            values_from="data_value",
            names=list(PLACES_MEASURES),
        )
    )

    return (
        medicare.join(nppes, "npi")
        .join(nucc, "taxonomy_code", how="left")
        .join(hud, "zip5", how="left")
        .join(acs, "county_fips", how="left")
        .join(places, places.zcta == _.zip5, how="left")
        # equality joins keep both sides of the key; the _right copies are
        # redundant and `zcta` is just zip5 under another name
        .drop("zcta", "taxonomy_code_right", "zip5_right", "county_fips_right")
        .mutate(state=ibis.literal(state))
    )


def build_density_expr(joined: ibis.Table) -> ibis.Table:
    """Provider density per 100k by county x specialty, vs. local health burden.

    Providers are allocated to counties by the HUD *business* ratio, so a ZIP
    straddling a county line contributes fractionally to each, and a
    hospital-only ZIP with no residents still counts its providers. The
    population denominator comes from ACS, which is residential by nature --
    that mismatch is intentional: it is what "providers per resident" means.
    """
    return (
        joined.filter(_.taxonomy_classification.notnull())
        .group_by(["county_fips", "county_name", "taxonomy_classification"])
        .agg(
            providers=_.bus_ratio.sum(),
            medicare_payments=(_.tot_mdcr_stdzd_amt * _.bus_ratio).sum(),
            beneficiaries=(_.tot_benes * _.bus_ratio).sum(),
            population=_.population.max(),
            median_household_income=_.median_household_income.max(),
            pct_chd=_.CHD.mean(),
            pct_diabetes=_.DIABETES.mean(),
            pct_uninsured=_.ACCESS2.mean(),
        )
        .mutate(providers_per_100k=_.providers / _.population * 100_000)
        .order_by(_.providers_per_100k.desc())
    )


joined = build_expr(STATE)
expr = joined.cache(
    ParquetCache.from_kwargs(source=xo.connect(), relative_path=CACHE_DIR)
)
density = build_density_expr(joined)


def parse_args(override: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default=STATE)
    parser.add_argument("--limit", type=int, default=20)
    (args, *_rest) = parser.parse_known_args(override)
    return args


def main() -> None:
    args = parse_args()
    built = build_expr(args.state)
    click.echo(built.schema())
    click.echo(xo.execute(built.limit(args.limit)).to_string())
    click.echo(xo.execute(build_density_expr(built).limit(args.limit)).to_string())


# `xorq build` imports this module as "__main__" (cli.py:276), so the usual
# guard alone would re-execute the whole pipeline at build time. Only run the
# preview when this file really is the entry point.
if __name__ == "__main__" and Path(sys.argv[0]).resolve() == Path(__file__).resolve():
    main()
elif __name__ == "__pytest_main__":
    main()
    pytest_examples_passed = True
