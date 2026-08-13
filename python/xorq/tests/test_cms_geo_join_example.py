"""Exercise examples/cms_geo_join.py's join without network or credentials.

``build_expr`` takes its six raw source tables as an optional argument, so the
fetchers are replaced with memtables carrying the same schemas. Everything
under test -- projections, the weighted crosswalk join, the PLACES pivot, the
rollup -- is the code the real run uses.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib
from types import ModuleType

import pandas as pd
import pytest

import xorq.api as xo
from xorq.common.exceptions import XorqError


EXAMPLE = pathlib.Path(__file__).parents[3] / "examples" / "cms_geo_join.py"

# 02139 is split across two counties so the ratio weighting has something to
# do; 01655 is a USPS "unique" ZIP (a single hospital) with no residents at
# all -- res_ratio 0, bus_ratio 1 -- which is how real hospital ZIPs look.
HUD_ROWS = (
    # zip, county, res_ratio, bus_ratio
    ("02138", "25017", 0.75, 0.80),
    ("02139", "25017", 0.60, 0.50),
    ("02139", "25025", 0.40, 0.50),
    ("01609", "25027", 1.00, 1.00),
    ("01655", "25027", 0.00, 1.00),
)

NPPES_ROWS = (
    # npi, entity, org, last, first, city, state, postal, taxonomy, enum_date
    (
        "1000000001",
        1.0,
        None,
        "Bruns",
        "Alex",
        "Cambridge",
        "MA",
        "021384102",
        "207RC0000X",
        "2005-05-23",
    ),
    (
        "1000000002",
        1.0,
        None,
        "Okafor",
        "Sam",
        "Cambridge",
        "MA",
        "02139",
        "207RC0000X",
        "2008-01-14",
    ),
    (
        "1000000003",
        1.0,
        None,
        "Vance",
        "Jo",
        "Worcester",
        "MA",
        "016091234",
        "207R00000X",
        "2011-09-02",
    ),
    (
        "1000000004",
        2.0,
        "Bay Cardiology",
        None,
        None,
        "Cambridge",
        "MA",
        "02138",
        "207RC0000X",
        "2014-04-01",
    ),
    # practices in a hospital-only ZIP: no residents, but very much a provider
    (
        "1000000005",
        1.0,
        None,
        "Reyes",
        "Kim",
        "Worcester",
        "MA",
        "01655",
        "207R00000X",
        "2016-06-06",
    ),
    # no Medicare row -> must not appear (inner join on npi)
    (
        "1000000009",
        1.0,
        None,
        "Ghost",
        "Pat",
        "Cambridge",
        "MA",
        "02138",
        "207R00000X",
        "2019-01-01",
    ),
)

MEDICARE_ROWS = (
    ("1000000001", "MA", "02138", "Cardiology", 300.0, 900.0, 50000.0, 48000.0),
    ("1000000002", "MA", "02139", "Cardiology", 200.0, 500.0, 30000.0, 29000.0),
    ("1000000003", "MA", "01609", "Internal Medicine", 150.0, 400.0, 20000.0, 19500.0),
    ("1000000004", "MA", "02138", "Cardiology", 100.0, 250.0, 15000.0, 14000.0),
    (
        "1000000005",
        "MA",
        "01655",
        "Internal Medicine",
        900.0,
        2500.0,
        220000.0,
        210000.0,
    ),
)

NUCC_ROWS = (
    (
        "207RC0000X",
        "Allopathic & Osteopathic Physicians",
        "Internal Medicine",
        "Cardiovascular Disease",
    ),
    ("207R00000X", "Allopathic & Osteopathic Physicians", "Internal Medicine", ""),
)

ACS_ROWS = (
    ("25017", "Middlesex County, Massachusetts", 120000.0, 1_600_000.0, 5000.0),
    ("25025", "Suffolk County, Massachusetts", 85000.0, 800_000.0, 9000.0),
    ("25027", "Worcester County, Massachusetts", 90000.0, 860_000.0, 7000.0),
)

PLACES_ROWS = tuple(
    (zcta, measure, value)
    for zcta, values in {
        "02138": {
            "CHD": 3.1,
            "DIABETES": 6.0,
            "OBESITY": 20.0,
            "ACCESS2": 4.0,
            "CHECKUP": 70.0,
        },
        "02139": {
            "CHD": 3.5,
            "DIABETES": 7.2,
            "OBESITY": 22.5,
            "ACCESS2": 5.1,
            "CHECKUP": 72.0,
        },
        "01609": {
            "CHD": 5.9,
            "DIABETES": 11.4,
            "OBESITY": 31.0,
            "ACCESS2": 8.8,
            "CHECKUP": 66.0,
        },
        "01655": {
            "CHD": 4.4,
            "DIABETES": 9.1,
            "OBESITY": 27.0,
            "ACCESS2": 6.2,
            "CHECKUP": 70.0,
        },
    }.items()
    for measure, value in values.items()
)


@pytest.fixture(scope="module")
def module() -> ModuleType:
    """Import the example without running it or touching the network."""
    spec = importlib.util.spec_from_file_location("cms_geo_join_example", EXAMPLE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def frame(rows: tuple, schema: object) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(schema.names))


@pytest.fixture(scope="module")
def raw(module: ModuleType) -> dict:
    """The six source tables the fetchers would otherwise produce."""
    nucc = pd.DataFrame(
        list(NUCC_ROWS),
        columns=["Code", "Grouping", "Classification", "Specialization"],
    ).assign(Definition="", Notes="", **{"Display Name": "", "Section": ""})
    return {
        "nppes": xo.memtable(frame(NPPES_ROWS, module.nppes_schema_out)),
        "medicare": xo.memtable(frame(MEDICARE_ROWS, module.medicare_schema_out)),
        "hud": xo.memtable(
            frame(
                tuple(r + (0.0, 1.0) for r in HUD_ROWS),
                module.hud_schema_out,
            )
        ),
        "acs": xo.memtable(frame(ACS_ROWS, module.acs_schema_out)),
        "places": xo.memtable(frame(PLACES_ROWS, module.places_schema_out)),
        "nucc": xo.memtable(nucc[list(module.NUCC_SCHEMA.names)]),
    }


@pytest.fixture(scope="module")
def joined(module: ModuleType, raw: dict) -> pd.DataFrame:
    return xo.execute(module.build_expr("MA", raw=raw))


def test_join_keeps_only_providers_with_medicare_rows(joined: pd.DataFrame) -> None:
    # 1000000009 has an NPPES row but no Medicare row
    assert set(joined.npi) == {
        "1000000001",
        "1000000002",
        "1000000003",
        "1000000004",
        "1000000005",
    }


def test_split_zip_fans_out_to_both_counties(joined: pd.DataFrame) -> None:
    zip_02139 = joined.loc[joined.zip5.eq("02139")]
    assert set(zip_02139.county_fips) == {"25017", "25025"}
    assert sorted(zip_02139.res_ratio) == [0.40, 0.60]
    assert sorted(zip_02139.bus_ratio) == [0.50, 0.50]


def test_zip_plus_four_is_truncated_to_zip5(joined: pd.DataFrame) -> None:
    # NPPES stores 021384102; the crosswalk and PLACES are keyed on 02138
    row = joined.loc[joined.npi.eq("1000000001")].iloc[0]
    assert row.zip5 == "02138"
    assert row.county_fips == "25017"


def test_taxonomy_and_acs_and_places_all_land(joined: pd.DataFrame) -> None:
    row = joined.loc[joined.npi.eq("1000000003")].iloc[0]
    assert row.taxonomy_classification == "Internal Medicine"
    assert row.county_name.startswith("Worcester")
    assert row.median_household_income == 90000.0
    assert row.DIABETES == pytest.approx(11.4)  # pivoted out of the long format


def test_organizations_are_flagged(joined: pd.DataFrame) -> None:
    orgs = joined.loc[joined.is_organization]
    assert set(orgs.npi) == {"1000000004"}
    assert orgs.iloc[0].provider_name == "Bay Cardiology"


def test_hospital_only_zip_survives_the_crosswalk(joined: pd.DataFrame) -> None:
    """USPS unique ZIPs have res_ratio 0; filtering on it drops whole hospitals."""
    row = joined.loc[joined.npi.eq("1000000005")].iloc[0]
    assert row.county_fips == "25027"
    assert row.res_ratio == 0.0
    assert row.bus_ratio == 1.0


def test_density_weights_providers_by_business_ratio(
    module: ModuleType, raw: dict
) -> None:
    density = xo.execute(module.build_density_expr(module.build_expr("MA", raw=raw)))
    row = density.loc[
        density.county_fips.eq("25017")
        & density.taxonomy_classification.eq("Internal Medicine")
    ].iloc[0]
    # 02138 x 2 providers @ bus 0.80 + 02139 x 1 provider @ bus 0.50 = 2.10
    assert row.providers == pytest.approx(2 * 0.80 + 0.50)
    assert row.providers_per_100k == pytest.approx(2.10 / 1_600_000 * 100_000)
    assert row.medicare_payments == pytest.approx(
        48000 * 0.80 + 14000 * 0.80 + 29000 * 0.50
    )


def test_density_counts_the_hospital_zip_provider(
    module: ModuleType, raw: dict
) -> None:
    density = xo.execute(module.build_density_expr(module.build_expr("MA", raw=raw)))
    row = density.loc[
        density.county_fips.eq("25027")
        & density.taxonomy_classification.eq("Internal Medicine")
    ].iloc[0]
    # 01609 @ bus 1.0 plus the hospital-only 01655 @ bus 1.0 -- weighting on
    # res_ratio would have yielded 1.0 and silently lost the hospital
    assert row.providers == pytest.approx(2.0)
    assert row.medicare_payments == pytest.approx(19500 + 210000)


def test_runs_on_the_xorq_backend(module: ModuleType, raw: dict) -> None:
    assert module.build_expr("MA", raw=raw)._find_backend().name == "xorq_datafusion"


def test_only_nucc_is_read_from_a_url(module: ModuleType) -> None:
    """Dynamic endpoints answer HEAD with Content-Length 0 and read as empty."""
    con = xo.connect()
    sources = module.fetch_sources("MA", con)

    nucc = sources["nucc"].op()
    assert type(nucc).__name__ == "Read"
    assert nucc.source is con
    assert dict(nucc.read_kwargs)["hash_path"] == module.NUCC_CSV_URL

    for name in ("nppes", "medicare", "hud", "acs", "places"):
        op = sources[name].op()
        assert type(op).__name__ == "RemoteTable", name
        assert "FlightUDXF" in repr(op.remote_expr), name


def test_nothing_raises_systemexit(module: ModuleType) -> None:
    """A ``SystemExit`` from a fetcher wedges the run, uninterruptibly.

    The fetchers execute on a Flight server thread, and ``make_udxf`` wraps the
    exchange in ``excepts_print_exc(..., Exception)``. ``SystemExit`` is not an
    ``Exception``, so it escapes that wrapper, the client never queues its
    end-of-stream sentinel, and the consumer blocks in ``queue.get()`` forever
    while the main thread sits in DataFusion ignoring Ctrl-C.
    """
    raised = {
        node.exc.func.id
        for node in ast.walk(ast.parse(EXAMPLE.read_text()))
        if isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
    }
    assert "SystemExit" not in raised
    assert raised == {"XorqError"}


def test_credentials_come_from_the_environment(module: ModuleType) -> None:
    """No env-file parsing: ``env_config`` declares them, so the env supplies them."""
    assert set(module.CREDENTIAL_SIGNUPS) <= set(module.env_config.varnames)
    assert not hasattr(module, "secret_env")
    assert not hasattr(module, "ENVRCS_DIR")


def test_require_env_raises_a_catchable_error(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "env_config", module.env_config.clone(HUD_TOKEN=""))
    with pytest.raises(XorqError, match="HUD_TOKEN is not set") as excinfo:
        module.require_env("HUD_TOKEN")
    assert module.CREDENTIAL_SIGNUPS["HUD_TOKEN"] in str(excinfo.value)


def test_require_credentials_checks_every_credential_up_front(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Up front and in this thread: a fetcher that raises mid-exchange yields
    zero rows, and the ParquetCache stores those as the answer."""
    monkeypatch.setattr(
        module, "env_config", module.env_config.clone(HUD_TOKEN="", CENSUS_API_KEY="k")
    )
    with pytest.raises(XorqError, match="HUD_TOKEN"):
        module.require_credentials()

    monkeypatch.setattr(
        module, "env_config", module.env_config.clone(HUD_TOKEN="t", CENSUS_API_KEY="")
    )
    with pytest.raises(XorqError, match="CENSUS_API_KEY"):
        module.require_credentials()

    monkeypatch.setattr(
        module, "env_config", module.env_config.clone(HUD_TOKEN="t", CENSUS_API_KEY="k")
    )
    assert module.require_credentials() is None


def test_caches_sit_on_the_raw_sources(module: ModuleType, raw: dict) -> None:
    """A cache on a projection refetches whenever the projection changes.

    Placed on the raw table instead, every CachedNode carries that source's
    fetched schema, so reshaping downstream replays from parquet.
    """
    expr = module.build_expr("MA", raw=raw)
    cached = {
        tuple(node.schema.names)
        for node in expr.op().find(lambda n: type(n).__name__ == "CachedNode")
    }
    expected = {
        tuple(schema.names)
        for schema in (
            module.nppes_schema_out,
            module.medicare_schema_out,
            module.hud_schema_out,
            module.acs_schema_out,
            module.places_schema_out,
            module.NUCC_SCHEMA,
        )
    }
    assert cached == expected
