"""Offline sample of the six sources ``examples/cms_geo_join.py`` fetches.

Each frame here has the same columns and dtypes as the corresponding fetcher's
*output*, before any projection -- ``zip`` not ``zip5``, ``rndrng_npi`` not
``npi``, PLACES still long -- so an expression written against this sample is
the same expression that runs against the live endpoints. Swap
``sample_sources()`` for ``cms_geo_join.fetch_sources()`` and nothing
downstream changes.

The rows are small but not toy-shaped; they carry the quirks that make the
crosswalk join worth writing carefully:

* ``02139`` is split across two counties, so the ratio weighting has work to do
* ``01655`` is a USPS "unique" ZIP -- one hospital, no residents: ``res_ratio``
  0, ``bus_ratio`` 1.0
* NPI ``1000000009`` has an NPPES row but no Medicare row, so an inner join on
  NPI must drop it
* ``021384102`` is ZIP+4, so the crosswalk key has to be a 5-character prefix
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# Column order matters: it is the fetchers' ``maybe_schema_out`` order, and a
# UDXF hands its frame straight to Arrow. Floats are declared so an all-integer
# column (population, Medicare counts) does not land as int64.
NPPES_COLUMNS = (
    "npi",
    "entity_type_code",
    "org_name",
    "last_name",
    "first_name",
    "practice_city",
    "practice_state",
    "practice_postal_code",
    "taxonomy_code",
    "enumeration_date",
)
NPPES_FLOATS = ("entity_type_code",)
NPPES_ROWS = (
    # a ZIP+4 postal code: the crosswalk key is its first five characters
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
    # an organization rather than an individual: entity_type_code 2
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
    # no Medicare row -> must not survive the inner join on npi
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

MEDICARE_COLUMNS = (
    "rndrng_npi",
    "rndrng_prvdr_state_abrvtn",
    "rndrng_prvdr_zip5",
    "rndrng_prvdr_type",
    "tot_benes",
    "tot_srvcs",
    "tot_mdcr_pymt_amt",
    "tot_mdcr_stdzd_amt",
)
MEDICARE_FLOATS = (
    "tot_benes",
    "tot_srvcs",
    "tot_mdcr_pymt_amt",
    "tot_mdcr_stdzd_amt",
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

HUD_COLUMNS = (
    "zip",
    "county_fips",
    "res_ratio",
    "bus_ratio",
    "oth_ratio",
    "tot_ratio",
)
HUD_FLOATS = ("res_ratio", "bus_ratio", "oth_ratio", "tot_ratio")
HUD_ROWS = (
    ("02138", "25017", 0.75, 0.80, 0.0, 1.0),
    # split across two counties: both rows are real, and both must survive
    ("02139", "25017", 0.60, 0.50, 0.0, 1.0),
    ("02139", "25025", 0.40, 0.50, 0.0, 1.0),
    ("01609", "25027", 1.00, 1.00, 0.0, 1.0),
    # USPS "unique" ZIP: a single hospital, zero residents
    ("01655", "25027", 0.00, 1.00, 0.0, 1.0),
)

ACS_COLUMNS = (
    "county_fips",
    "county_name",
    "median_household_income",
    "population",
    "uninsured_18_34",
)
ACS_FLOATS = ("median_household_income", "population", "uninsured_18_34")
ACS_ROWS = (
    ("25017", "Middlesex County, Massachusetts", 120000.0, 1_600_000.0, 5000.0),
    ("25025", "Suffolk County, Massachusetts", 85000.0, 800_000.0, 9000.0),
    ("25027", "Worcester County, Massachusetts", 90000.0, 860_000.0, 7000.0),
)

PLACES_COLUMNS = ("locationname", "measureid", "data_value")
PLACES_FLOATS = ("data_value",)
PLACES_MEASURES = ("CHD", "DIABETES", "OBESITY", "ACCESS2", "CHECKUP")
# long format, exactly as the CDC serves it: one row per (ZCTA, measure)
PLACES_ROWS = tuple(
    (zcta, measure, value)
    for zcta, values in {
        "02138": (3.1, 6.0, 20.0, 4.0, 70.0),
        "02139": (3.5, 7.2, 22.5, 5.1, 72.0),
        "01609": (5.9, 11.4, 31.0, 8.8, 66.0),
        "01655": (4.4, 9.1, 27.0, 6.2, 70.0),
    }.items()
    for measure, value in zip(PLACES_MEASURES, values)
)

NUCC_COLUMNS = (
    "Code",
    "Grouping",
    "Classification",
    "Specialization",
    "Definition",
    "Notes",
    "Display Name",
    "Section",
)
NUCC_FLOATS = ()
NUCC_ROWS = (
    (
        "207RC0000X",
        "Allopathic & Osteopathic Physicians",
        "Internal Medicine",
        "Cardiovascular Disease",
        "",
        "",
        "",
        "",
    ),
    (
        "207R00000X",
        "Allopathic & Osteopathic Physicians",
        "Internal Medicine",
        "",
        "",
        "",
        "",
        "",
    ),
)

SOURCES = {
    "nppes": (NPPES_COLUMNS, NPPES_ROWS, NPPES_FLOATS),
    "medicare": (MEDICARE_COLUMNS, MEDICARE_ROWS, MEDICARE_FLOATS),
    "hud": (HUD_COLUMNS, HUD_ROWS, HUD_FLOATS),
    "acs": (ACS_COLUMNS, ACS_ROWS, ACS_FLOATS),
    "places": (PLACES_COLUMNS, PLACES_ROWS, PLACES_FLOATS),
    "nucc": (NUCC_COLUMNS, NUCC_ROWS, NUCC_FLOATS),
}


def frame(name: str) -> pd.DataFrame:
    """One source as a DataFrame with the fetcher's columns and dtypes."""
    (columns, rows, floats) = SOURCES[name]
    df = pd.DataFrame(list(rows), columns=list(columns))
    return df.assign(
        **{
            col: (
                df[col].astype("float64")
                if col in floats
                else df[col].astype("string").astype(object)
            )
            for col in columns
        }
    )


def write_parquet(data_dir: Path) -> dict[str, Path]:
    """Write every source to ``data_dir`` as parquet; return name -> path."""
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in SOURCES:
        path = data_dir.joinpath(f"{name}.parquet")
        frame(name).to_parquet(path, index=False)
        paths[name] = path
    return paths
